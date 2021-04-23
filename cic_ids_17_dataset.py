"""
TODO:
    1. what columns to keep?
        - actually, it is not useful to keep cols like "Fwd/Bwd Packet Length Mean/Std"
          since we have "Total Fwd/Bwd Packets" and "Total Length of Fwd/Bwd Packets".
          so, we can derive them anyway. GAN might not get them accurately
    2. how to preprocess "categorical" columns (e.g., Port numbers)

"""

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

    # TODO: find representation for ports
    return df


def generate_train_test_split(data_folder_path, write_path="./data/cic-ids-2017_splits",
                              test_size=0.05, seed=0, stratify=False, scale=False, keep_benign=False):
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

    """
    labels = ['Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',  'DoS slowloris',
              'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack \x96 Brute Force',
              'Web Attack \x96 Sql Injection', 'Web Attack \x96 XSS', 'zBENIGN']

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


class CIC17Dataset(data.Dataset):

    def __init__(self, file_path):
        dataset = torch.load(file_path)
        self.X = dataset["features"]
        self.y = dataset["labels"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


if __name__ == '__main__':

    # --------------------------------- Data generation  ---------------------------------
    # generate_train_test_split("./data/cic-ids-2017/TrafficLabelling", stratify=True, scale=False)
    # generate_train_test_split("./data/cic-ids-2017/TrafficLabelling", stratify=True, scale=True)
    # generate_train_test_split("./data/cic-ids-2017/TrafficLabelling",
    #                           write_path="./data/cic-ids-2017_splits_with_benign",
    #                           stratify=True, scale=False, keep_benign=True)

    # --------------------------------- Sanity checks  ---------------------------------
    train_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/train_dataset_scaled.pt")
    test_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/test_dataset_scaled.pt")

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
    class_names = torch.load("./data/cic-ids-2017_splits/seed_0/class_names.pt")
    label_encoder = joblib.load("./data/cic-ids-2017_splits/seed_0/label_encoder.gz")
    print("\nClasses: ", class_names)
    print("Transformed labels: ", *zip(train_dataset.y[:10],
                                       label_encoder.inverse_transform(train_dataset.y[:10])))

    # 4. inverse transform scaling checks
    scaler = joblib.load("./data/cic-ids-2017_splits/seed_0/min_max_scaler.gz")
    train_dataset_unscaled = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/train_dataset.pt")
    X_unscaled = scaler.inverse_transform(train_dataset.X)
    print("\nInverse scaled X: ", X_unscaled[0][:10])
    print("Original unscaled X: ", train_dataset_unscaled.X[0][:10])
    print("Equal: ", np.array_equal(X_unscaled, train_dataset_unscaled.X))
    # tiny numeric differences are expected
    print("All close: ", np.allclose(X_unscaled, train_dataset_unscaled.X))
