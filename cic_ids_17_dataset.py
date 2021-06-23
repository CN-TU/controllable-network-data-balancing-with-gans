import pandas as pd
import torch
import joblib
import numpy as np
import collections

from pathlib import Path
from torch.utils import data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import utils


def load_and_preprocess_dataset(data_folder_path, keep_benign=False):
    """
    Loads and preprocesses the CIC-IDS 2017 dataset:
        - concatenates all csv files
        - fixes column names
        - resets the index (i.e, from 1 to n_rows)
        - handles missing values in `Flow Bytes/s`
        - drops ["Flow ID", "Source IP", "Destination IP", "Protocol", "Timestamp"]

    Args:
        data_folder_path: String. Folder path where .csv files reside.
        keep_benign: Bool.

    Returns: pandas.DataFrame

    """
    # load dataset
    p = Path(data_folder_path).glob('**/*')
    files = sorted([x for x in p if x.is_file()])

    dfs = []
    for file in files:
        if file.name == "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv":
            # there are 288602 empty rows at the end of
            # Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
            # do not read those
            dfs.append(pd.read_csv(file, encoding="ISO-8859-1", nrows=170366))
        else:
            dfs.append(pd.read_csv(file, encoding="ISO-8859-1"))

    df = pd.concat(dfs, axis=0)

    # preprocess
    df.columns = [col.strip() for col in df.columns]
    df["Flow Bytes/s"] = df["Flow Bytes/s"].fillna(value=0.0)
    # remove benign flows
    # LabelEncoder orders the labels alphabetically
    # to have BENIGN as the very last label, add 'z' in the front
    df.Label = df.Label.replace({"BENIGN": "zBENIGN"})
    if not keep_benign:
        df = df[df.Label != "zBENIGN"]
    drop_cols = ["Flow ID", "Source IP", "Destination IP", "Protocol", "Timestamp"]
    df = df.drop(drop_cols, axis=1)
    df.reset_index(drop=True, inplace=True)
    return df


def get_linear_vector(df, df_means, label, feature, steps=3):
    """
    Get linear interval for a given feature of a given class (with n intervals).

    Args:
        df: pd.DataFrame. Complete Dataframe.
        df_means: pd.DataFrame. Same dataframe but grouped by "Label" column and averaged per feature.
        label: Str. Label column.
        feature: Str. Feature column.
        steps: Int. Defines how big the interval will be. (Currently only steps=3 is supported though.)

    """
    _max = df[feature].max()
    _min = df[feature].min()
    interval = (_max - _min) / steps
    t1, t2 = _min + interval, _max - interval
    rep = [int(df_means[feature][label] < t1),
           int(t1 < df_means[feature][label] < t2),
           int(df_means[feature][label] > t2)]
    return np.argmax(rep), rep


def get_percentile_vector(df, df_means, label, feature):
    """
    Get percentile interval for a given feature of a given class feature (33%-median and 66%).

    Args:
        df: pd.DataFrame. Complete Dataframe.
        df_means: pd.DataFrame. Same dataframe but grouped by "Label" column and averaged per feature.
        label: Str. Label column.
        feature: Str. Feature column.

    """
    t1, t2 = df[feature].quantile(0.33), df[feature].quantile(0.66)
    rep = [int(df_means[feature][label] < t1),
           int(t1 < df_means[feature][label] < t2),
           int(df_means[feature][label] > t2)]
    return np.argmax(rep), rep


def compute_static_condition_vectors(df, df_means, relevant_features, vector_type="percentile"):
    """
    Constructs the condition vector per attack type.

    Args:
        df: pd.DataFrame. Complete Dataframe.
        df_means: pd.DataFrame. Same dataframe but grouped by "Label" column and averaged per feature.
        relevant_features: List. Features to use.
        vector_type: Str. Indicates what method to use to compute condition vector.
    Returns: representation, levels

    """
    df_mean_condition = df_means[relevant_features]
    representations, levels = collections.defaultdict(list), collections.defaultdict(list)
    for attack in df_mean_condition.index:
        for feature in relevant_features:
            if feature in ['PSH Flag Count', 'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count']:
                val = int(df_mean_condition[feature][attack])
                representations[attack].append([val])
                levels[attack].append(val)
            elif feature == 'Destination Port':
                port_num = int(df_mean_condition[feature][attack])
                port_bin = utils.convert_port_to_binary(port_num)
                representations[attack].append(port_bin)
                levels[attack].append(port_num)
            else:
                if vector_type == "percentile":
                    level, representation = get_percentile_vector(df, df_mean_condition, attack, feature)
                elif vector_type == "linear":
                    level, representation = get_linear_vector(df, df_mean_condition, attack, feature)
                else:
                    raise ValueError("Supported vector types are ['percentile', 'linear']")
                representations[attack].append(representation)
                levels[attack].append(level)
        # flatten representation
        representations[attack] = [val for rep in representations[attack] for val in rep]
    return representations, levels


def compute_dynamic_condition_vectors(df, relevant_features, quantiles=(0.33, 0.66)):
    """
    Computes the condition vectors/levels for each flow individually.

    Args:
        df: pd.DataFrame. The complete (or train/test slice) dataset.
        relevant_features: List. Features to use.
        quantiles: Tuple. Contains the values for first and second quantile. By default 33 and 66the percentile,
                   such that it is split into 3 equi-sized buckets.

    Returns: np.array, np.array. The condition vectors, the condition levels
    """
    flag_cols = [col for col in relevant_features if "Flag" in col]
    port_col = "Destination Port"
    cols_to_dummy = [col for col in relevant_features if col not in flag_cols and col != port_col]
    df_condition = df[relevant_features]

    # compute bucket for each feature based on quantile values
    quantiles = [0, *quantiles, 1.0]
    df_condition.loc[:, cols_to_dummy] = df_condition.loc[:, cols_to_dummy].apply(
        lambda x: pd.qcut(x, q=quantiles, labels=False)
    )

    df_levels = df_condition

    # turn bucket values in one-hot encoded features (except for port and flag columns)
    df_condition = pd.get_dummies(df_condition, columns=cols_to_dummy)

    # reorder cols (.get_dummies() appends to the end)
    reorder_cols = [col for col in df_condition.columns if col not in flag_cols] + flag_cols
    df_condition = df_condition[reorder_cols]

    # handle "Destination Port"
    # covert port to 16 bit array and add to df
    df_condition[port_col] = df_condition[port_col].apply(utils.convert_port_to_binary)
    # make new port df to separate arrays into columns
    df_port = pd.DataFrame(df_condition[port_col].tolist(),
                           columns=[port_col + f"_{i}" for i in range(16)],
                           index=df_condition.index)
    # concatenate dfs
    df_condition = pd.concat([df_port, df_condition.drop(port_col, axis=1)], axis=1)

    return df_condition.values, df_levels.values


def compute_dynamic_condition_vector_dict(condition_vectors, labels, max_per_class=10000, num_labels=14):
    """
    Constructs a dictionary that contains class-condition_vector/level key-value pairs.
    For each class we select max_per_class condition vectors.
    If there are more than max_per_class condition vectors/levels,
    max_per_class vectors are drawn at random from the population. If there are less of them are selected.
    Args:
        condition_vectors: np.array. Either condition vectors or levels.
        labels: np.array.
        max_per_class: Int.
        num_labels: Int.

    Returns: Dict

    """
    condition_vector_dict = {}
    for label in range(num_labels):
        idx = np.where(labels == label)[0]
        if len(idx) > max_per_class:
            idx = np.random.choice(idx, max_per_class)
        else:
            idx = idx[:max_per_class]
        condition_vector_dict[label] = condition_vectors[idx]
    return condition_vector_dict


def generate_train_test_split(data_folder_path, write_path="./data/cic-ids-2017_splits",
                              test_size=0.05, seed=0, stratify=False, scale=False, keep_benign=False,
                              write_class_means=False, write_static_condition_vectors=False,
                              write_dynamic_condition_vectors=False):
    """
    Generate a train-test-split of the CIC-IDS dataset:
        - loads and preprocess dataset
        - encodes the `Label` col
        - removes rows with `inf` values in `Flow Bytes/s` and `Flow Packets/s`
        - scales the numeric columns in the dataframe using MinMaxScaler
        - split df into train and test set given `test_size` and `seed`.
        - saves the generate split + class array to `write_path`
        - (optionally) keeps benign flows
        - (optionally) saves the class means
        - (optionally) saves the static condition vectors
        - (optionally) saves the dynamic condition vectors.

    Args:
        data_folder_path: String. Folder path where .csv files reside.
        write_path: String. Folder path to save files to.
        test_size: Float. Proportion of test samples
        seed: Int. For reproducibility.
        stratify: Bool. Whether to preserve the original distribution of labels in train/test.
        scale: Bool. Whether to scale numeric columns.
        keep_benign: Bool.
        write_class_means: Bool.
        write_static_condition_vectors: Bool. Only executed if write_class_means is True and scale is False.
            This saves a dictionary containing for each class a static vector representation constructed from
            a selection of features in the dataset. For each selected feature, we compute if the value lies in
            the low/mid/high quantile. The result is a 3-dim vector per features. We encode 11 features this way,
            and also add the port, resulting in a 34-dim vector per class.
        write_dynamic_condition_vectors: Bool.

    """
    labels = utils.get_label_names()
    condition_vector_features = utils.get_condition_vector_names()

    write_path = Path(write_path) / f"seed_{seed}"
    if not write_path.exists():
        write_path.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing data...")
    df = load_and_preprocess_dataset(data_folder_path, keep_benign=keep_benign)

    # encode target col
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    df.Label = label_encoder.transform(df.Label)

    # scale columns
    df = df[np.isfinite(df).all(1)]
    X = df.drop("Label", axis=1).values
    y = df.Label.values
    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, write_path / 'min_max_scaler.gz')

    # split train_test
    print("Generating split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=seed
    )

    print(f"Saving split to {write_path}...")
    suffix = "_scaled" if scale else ""
    torch.save({"features": X_train, "labels": y_train}, write_path / f"train_dataset{suffix}.pt")
    torch.save({"features": X_test, "labels": y_test}, write_path / f"test_dataset{suffix}.pt")
    # save labels and scaler to inverse-transform data
    joblib.dump(label_encoder, write_path / 'label_encoder.gz')
    torch.save(label_encoder.classes_, write_path / "class_names.pt")
    torch.save(df.columns, write_path / "column_names.pt")

    if write_class_means:
        df_reconstruct = pd.DataFrame(np.append(X, y.reshape(-1, 1), axis=1), columns=df.columns)
        df_mean = df_reconstruct.groupby("Label").mean()
        torch.save(df_mean, write_path / f"class_means{suffix}.pt")
        if write_static_condition_vectors:
            assert not scale
            representations, levels = compute_static_condition_vectors(df_reconstruct, df_mean,
                                                                       condition_vector_features)
            torch.save(representations, write_path / f"static_condition_vectors.pt")
            torch.save(levels, write_path / f"static_condition_levels.pt")
            torch.save(condition_vector_features, write_path / f"condition_vector_names.pt")

    if write_dynamic_condition_vectors:
        assert not scale
        df_train_reconstruct = pd.DataFrame(np.append(X_train, y_train.reshape(-1, 1), axis=1), columns=df.columns)
        df_test_reconstruct = pd.DataFrame(np.append(X_test, y_test.reshape(-1, 1), axis=1), columns=df.columns)
        train_dynamic_condition_vectors, train_dynamic_condition_levels = compute_dynamic_condition_vectors(
            df_train_reconstruct,
            condition_vector_features,
        )
        test_dynamic_condition_vectors, _ = compute_dynamic_condition_vectors(
            df_test_reconstruct,
            condition_vector_features,

        )
        # we also want to construct the dynamic_condition_vector_dict --> e.g. 5000 condition vectors per class.
        # y_train should be in the same order as train_dynamic_condition_vectors still
        dynamic_condition_vector_dict = compute_dynamic_condition_vector_dict(train_dynamic_condition_vectors,
                                                                              df_train_reconstruct.Label)
        dynamic_condition_level_dict = compute_dynamic_condition_vector_dict(train_dynamic_condition_levels,
                                                                             df_train_reconstruct.Label)

        torch.save(train_dynamic_condition_vectors, write_path / f"train_dynamic_condition_vectors.pt")
        torch.save(test_dynamic_condition_vectors, write_path / f"test_dynamic_condition_vectors.pt")
        torch.save(dynamic_condition_vector_dict, write_path / f"dynamic_condition_vector_dict.pt")
        torch.save(dynamic_condition_level_dict, write_path / f"dynamic_condition_level_dict.pt")


class CIC17Dataset(data.Dataset):

    def __init__(self, file_path, is_scaled=False,
                 use_static_condition_vectors=False,
                 use_dynamic_condition_vectors=False,
                 is_test=False):
        assert not (use_static_condition_vectors and use_dynamic_condition_vectors)
        folder_path = Path(file_path).parent
        dataset = torch.load(file_path)
        self.X = dataset["features"]
        self.y = dataset["labels"]
        self.column_names = torch.load(folder_path / "column_names.pt")
        self.class_names = torch.load(folder_path / "class_names.pt")
        self.label_encoder = joblib.load(folder_path / "label_encoder.gz")
        self.static_condition_vectors = torch.load(folder_path / "static_condition_vectors.pt")
        self.static_condition_levels = torch.load(folder_path / "static_condition_levels.pt")
        self.condition_vector_names = torch.load(folder_path / "condition_vector_names.pt")
        self.dynamic_condition_vectors = torch.load(
            folder_path / f"{'test' if is_test else 'train'}_dynamic_condition_vectors.pt"
        )
        self.dynamic_condition_vector_dict = torch.load(folder_path / "dynamic_condition_vector_dict.pt")
        self.dynamic_condition_level_dict = torch.load(folder_path / "dynamic_condition_level_dict.pt")
        self.use_static_condition_vectors = use_static_condition_vectors
        self.use_dynamic_condition_vectors = use_dynamic_condition_vectors
        if is_scaled:
            self.scaler = joblib.load(folder_path / "min_max_scaler.gz")
            self.class_means = torch.load(folder_path / "class_means_scaled.pt")
        else:
            self.class_means = torch.load(folder_path / "class_means.pt")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label = self.y[idx]
        # torch.Dataset cannot handle 'None' values, hence just empty list
        condition_vector = []
        if self.use_static_condition_vectors:
            condition_vector = torch.Tensor(self.static_condition_vectors[label])
        elif self.use_dynamic_condition_vectors:
            condition_vector = self.dynamic_condition_vectors[idx]
        return [self.X[idx], label, condition_vector]


if __name__ == '__main__':
    # --------------------------------- Data generation  ---------------------------------

    # unscaled dataset
    generate_train_test_split("./data/cic-ids-2017/TrafficLabelling",
                              stratify=True, scale=False, write_class_means=True,
                              write_static_condition_vectors=True, write_dynamic_condition_vectors=True)

    # scaled dataset
    generate_train_test_split("./data/cic-ids-2017/TrafficLabelling",
                              stratify=True, scale=True, write_class_means=True)

    # only required for training the classifier
    generate_train_test_split("./data/cic-ids-2017/TrafficLabelling",
                              write_path="./data/cic-ids-2017_splits_with_benign",
                              stratify=True, scale=False, keep_benign=True, write_class_means=True)

    # ---------------------------------  Sanity checks  ----------------------------------

    # train_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/train_dataset_scaled.pt", is_scaled=True)
    # test_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/test_dataset_scaled.pt", is_scaled=True)
    #
    # #  1. Label distribution checks
    # print(len(train_dataset))  # 528728
    # print(len(test_dataset))  # 27828
    # train_label_counts = collections.Counter(train_dataset.y)
    # test_label_counts = collections.Counter(test_dataset.y)
    # print({label: round(count / len(train_dataset), 5) for label, count in train_label_counts.most_common()})
    # print({label: round(count / len(test_dataset), 5) for label, count in test_label_counts.most_common()})
    # # {3: 0.41348, 9: 0.28533, 1: 0.23003, 2: 0.01849, 6: 0.01426, 10: 0.0106, 5: 0.01041, 4: 0.00988, 0: 0.00351,
    # #  11: 0.00271, 13: 0.00117, 8: 6e-05, 12: 4e-05, 7: 2e-05}
    # # {3: 0.41347, 9: 0.28532, 1: 0.23002, 2: 0.01851, 6: 0.01427, 10: 0.0106, 5: 0.01042, 4: 0.00988, 0: 0.00352,
    # #  11: 0.0027, 13: 0.00119, 8: 7e-05, 12: 4e-05}
    #
    # # 2. PyTorch data loaders check
    # train_loader = data.DataLoader(train_dataset, batch_size=128)
    # test_loader = data.DataLoader(test_dataset, batch_size=128)
    # batch = next(iter(train_loader))
    # print(batch[0].shape, batch[1].shape)
    #
    # # 3. Inverse transform labels checks
    # class_names = train_dataset.class_names
    # label_encoder = train_dataset.label_encoder
    # print("\nClasses: ", class_names)
    # print("Transformed labels: ", *zip(train_dataset.y[:10],
    #                                    label_encoder.inverse_transform(train_dataset.y[:10])))
    #
    # # 4. inverse transform scaling checks
    # scaler = train_dataset.scaler
    # train_dataset_unscaled = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/train_dataset.pt")
    # X_unscaled = scaler.inverse_transform(train_dataset.X)
    # print("\nInverse scaled X: ", X_unscaled[0][:10])
    # print("Original unscaled X: ", train_dataset_unscaled.X[0][:10])
    # print("Equal: ", np.array_equal(X_unscaled, train_dataset_unscaled.X))
    # # tiny numeric differences are expected
    # print("All close: ", np.allclose(X_unscaled, train_dataset_unscaled.X))
    #
    # # 5. validate static condition vectors
    # train_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/train_dataset_scaled.pt", is_scaled=True,
    #                              use_static_condition_vectors=True)
    #
    # print(train_dataset.class_names, train_dataset.condition_vector_names)
    # print(train_dataset.static_condition_vectors, train_dataset.static_condition_levels)
    # levels = collections.defaultdict(list)
    # for label, level in train_dataset.static_condition_levels.items():
    #     levels[tuple(level)].append(label)
    # print("Duplicates:", not len(levels) == len(train_dataset.static_condition_levels))
    # # 'DoS GoldenEye' 'DoS Hulk' have duplicate representations, unfortunately.
    # print(levels)
    #
    # # 6. validate dynamic condition vectors
    # train_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/train_dataset_scaled.pt", is_scaled=True,
    #                              use_dynamic_condition_vectors=True)
    # test_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/test_dataset_scaled.pt", is_scaled=True,
    #                             use_dynamic_condition_vectors=True, is_test=True)
    # print(train_dataset.dynamic_condition_vectors)
    # print(test_dataset.dynamic_condition_vectors)
    # # dynamic condition vector dict
    # print(train_dataset.dynamic_condition_vector_dict)
    # for label, vectors in train_dataset.dynamic_condition_vector_dict.items():
    #     print(label, vectors.shape)
    # # dynamic condition vector levels
    # print(train_dataset.dynamic_condition_leve_dict)
    # for label, vectors in train_dataset.dynamic_condition_level_dict.items():
    #     print(label, vectors.shape)
