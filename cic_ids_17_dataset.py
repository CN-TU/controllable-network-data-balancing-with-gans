import pandas as pd
import torch
import joblib
import numpy as np

from pathlib import Path
from collections import Counter
from torch.utils import data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


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
    Get percentile interval for a given feature of a given class feature (35%-median and 65%).

    Args:
        df: pd.DataFrame. Complete Dataframe.
        df_means: pd.DataFrame. Same dataframe but grouped by "Label" column and averaged per feature.
        label: Str. Label column.
        feature: Str. Feature column.

    """
    t1, t2 = df[feature].quantile(0.35), df[feature].quantile(0.65)
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
    representations, levels = {}, {}
    for attack in df_mean_condition.index:
        # We cannot just use the real Port number, as it will be of a different scale than all other inputs.
        port = int(df_mean_condition['Destination Port'][attack])
        representations[attack] = [[port]]
        levels[attack] = [port]
        for feature in relevant_features[1:]:  # ignore destination port
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


def generate_train_test_split(data_folder_path, write_path="./data/cic-ids-2017_splits",
                              test_size=0.05, seed=0, stratify=False, scale=False, keep_benign=False,
                              write_class_means=False, write_static_condition_vectors=False):
    """
    Generate a train-test-split of the CIC-IDS dataset:
        - loads and preprocess dataset
        - encodes the `Label` col
        - removes rows with `inf` values in `Flow Bytes/s` and `Flow Packets/s`
        - scales the numeric columns in the dataframe using MinMaxScaler
        - split df into train and test set given `test_size` and `seed`.
        - saves the generate split + class array to `write_path`

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

    """
    labels = ['Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris',
              'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack \x96 Brute Force',
              'Web Attack \x96 Sql Injection', 'Web Attack \x96 XSS', 'zBENIGN']

    # as defined by Fares
    condition_vector_features = ['Destination Port', 'Flow Duration', 'Total Backward Packets', 'Total Fwd Packets',
                                 'Packet Length Mean', 'Flow Bytes/s', 'Flow IAT Min', 'Flow IAT Max', 'PSH Flag Count',
                                 'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count']

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
        if write_static_condition_vectors and not scale:
            representations, levels = compute_static_condition_vectors(df_reconstruct, df_mean,
                                                                       condition_vector_features)
            torch.save(representations, write_path / f"static_condition_vectors.pt")
            torch.save(levels, write_path / f"static_condition_levels.pt")
            torch.save(condition_vector_features, write_path / f"condition_vector_names.pt")


class CIC17Dataset(data.Dataset):

    def __init__(self, file_path, is_scaled=False):
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
        if is_scaled:
            self.scaler = joblib.load(folder_path / "min_max_scaler.gz")
            self.class_means = torch.load(folder_path / "class_means_scaled.pt")
        else:
            self.class_means = torch.load(folder_path / "class_means.pt")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


if __name__ == '__main__':
    # --------------------------------- Data generation  ---------------------------------
    # generate_train_test_split("./data/cic-ids-2017/TrafficLabelling",
    #                           stratify=True, scale=False, write_class_means=True, write_static_condition_vectors=True)
    # generate_train_test_split("./data/cic-ids-2017/TrafficLabelling",
    #                           stratify=True, scale=True, write_class_means=True)
    # generate_train_test_split("./data/cic-ids-2017/TrafficLabelling",
    #                           write_path="./data/cic-ids-2017_splits_with_benign",
    #                           stratify=True, scale=False, keep_benign=True, write_class_means=True)

    # --------------------------------- Sanity checks  ---------------------------------
    train_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/train_dataset_scaled.pt", is_scaled=True)
    test_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/test_dataset_scaled.pt", is_scaled=True)

    #  1. Label distribution checks
    print(len(train_dataset))  # 528728
    print(len(test_dataset))  # 27828
    train_label_counts = Counter(train_dataset.y)
    test_label_counts = Counter(test_dataset.y)
    print({label: round(count / len(train_dataset), 5) for label, count in train_label_counts.most_common()})
    print({label: round(count / len(test_dataset), 5) for label, count in test_label_counts.most_common()})
    # {3: 0.41348, 9: 0.28533, 1: 0.23003, 2: 0.01849, 6: 0.01426, 10: 0.0106, 5: 0.01041, 4: 0.00988, 0: 0.00351,
    #  11: 0.00271, 13: 0.00117, 8: 6e-05, 12: 4e-05, 7: 2e-05}
    # {3: 0.41347, 9: 0.28532, 1: 0.23002, 2: 0.01851, 6: 0.01427, 10: 0.0106, 5: 0.01042, 4: 0.00988, 0: 0.00352,
    #  11: 0.0027, 13: 0.00119, 8: 7e-05, 12: 4e-05}

    # 2. PyTorch data loaders check
    train_loader = data.DataLoader(train_dataset, batch_size=128)
    test_loader = data.DataLoader(test_dataset, batch_size=128)
    batch = next(iter(train_loader))
    print(batch[0].shape, batch[1].shape)

    # 3. Inverse transform labels checks
    class_names = train_dataset.class_names
    label_encoder = train_dataset.label_encoder
    print("\nClasses: ", class_names)
    print("Transformed labels: ", *zip(train_dataset.y[:10],
                                       label_encoder.inverse_transform(train_dataset.y[:10])))

    # 4. inverse transform scaling checks
    scaler = train_dataset.scaler
    train_dataset_unscaled = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/train_dataset.pt")
    X_unscaled = scaler.inverse_transform(train_dataset.X)
    print("\nInverse scaled X: ", X_unscaled[0][:10])
    print("Original unscaled X: ", train_dataset_unscaled.X[0][:10])
    print("Equal: ", np.array_equal(X_unscaled, train_dataset_unscaled.X))
    # tiny numeric differences are expected
    print("All close: ", np.allclose(X_unscaled, train_dataset_unscaled.X))

    # 5. validate condition vectors
    import collections
    print(train_dataset.class_names, train_dataset.condition_vector_names)
    print(train_dataset.static_condition_vectors, train_dataset.static_condition_levels)
    levels, levels_without_port = collections.defaultdict(list), collections.defaultdict(list)
    for label, level in train_dataset.static_condition_levels.items():
        levels[tuple(level)].append(label)
        levels_without_port[tuple(level[1:])].append(label)
    print("Duplicates:", not len(levels) == len(train_dataset.static_condition_levels))
    # 'DoS GoldenEye' 'DoS Hulk' have duplicate representations, unfortunately.
    print(levels)
    print(levels_without_port)
