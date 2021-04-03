import pandas as pd
import numpy as np
import torch

from pathlib import Path
from torch.utils import data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_preprocess_dataset(data_folder_path):
    """
    Loads and preprocesses the CIC-IDS 2017 dataset:
        - concatenates all csv files
        - fixes column names
        - removes all `BENIGN` flows
        - handles missing values in `Flow Bytes/s`
        - drop ["Flow ID", "Source IP", "Destination IP", "Protocol", "Timestamp"]

    Args:
        data_folder_path: String. Folder path where .csv files reside.

    Returns: pandas.DataFrame

    """
    # load dataset
    p = Path(data_folder_path).glob('**/*')
    files = [x for x in p if x.is_file()]

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
    cols = [col.strip() for col in df.columns]
    df.columns = cols
    df = df[df.Label != "BENIGN"]
    df["Flow Bytes/s"] = df["Flow Bytes/s"].fillna(value=0.0)
    drop_cols = ["Flow ID", "Source IP", "Destination IP", "Protocol", "Timestamp"]
    df = df.drop(drop_cols, axis=1)
    return df


def generate_train_test_split(data_folder_path, write_path="./data/cic-ids-2017_splits",
                              test_size=0.1, seed=0, stratify=False):
    """
    Generate a train-test-split of the CIC-IDS dataset:
        - loads and preprocess dataset
        - encodes the `Label` col
        - split df into train and test set given `test_size` and `seed`.
        - saves the generate split + class array to `write_path`

    Args:
        data_folder_path: String. Folder path where .csv files reside.
        write_path: String. Folder path to save files to.
        test_size: Float. Proportion of test samples
        stratify: Bool. Whether to preserve the original distribution of labels in train/test.
        seed: Int. For reproducibility.

    Returns:

    """
    write_path = Path(write_path) / f"seed_{seed}"
    if not write_path.exists():
        write_path.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing data...")
    df = load_and_preprocess_dataset(data_folder_path)

    # encode target col
    label_encoder = LabelEncoder()
    df.Label = label_encoder.fit_transform(df.Label)
    df.Label.value_counts(), label_encoder.classes_, label_encoder.inverse_transform([1, 2, 3])

    # split train_test
    print("Generating split...")
    X = df.drop("Label", axis=1).values
    y = df.Label.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=seed
    )

    print(f"Saving split to {write_path}...")
    torch.save(X_train, write_path / "X_train.pt")
    torch.save(y_train, write_path / "y_train.pt")
    torch.save(X_test, write_path / "X_test.pt")
    torch.save(y_test, write_path / "y_test.pt")
    torch.save(label_encoder.classes_, write_path / "classes.pt")


class CIC17Dataset(data.Dataset):

    def __init__(self, features_path, labels_path):
        self.X = torch.load(features_path)
        self.y = torch.load(labels_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


if __name__ == '__main__':
    from collections import Counter

    # generate_train_test_split("./data/cic-ids-2017/TrafficLabelling", stratify=True)

    train_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/X_train.pt",
                                 "./data/cic-ids-2017_splits/seed_0/y_train.pt")

    test_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/X_test.pt",
                                "./data/cic-ids-2017_splits/seed_0/y_test.pt")

    print(len(train_dataset)) # 501881
    print(len(test_dataset)) # 55765

    # sanity check for stratify
    train_label_counts = Counter(train_dataset.y)
    test_label_counts = Counter(test_dataset.y)
    print({label: round(count / len(train_dataset), 5) for label, count in train_label_counts.most_common()})
    print({label: round(count / len(test_dataset), 5) for label, count in test_label_counts.most_common()})
    # {3: 0.41437, 9: 0.285, 1: 0.22958, 2: 0.01846, 6: 0.01423, 10: 0.01057, 5: 0.01039, 4: 0.00986, 0: 0.00352, 11: 0.0027, 13: 0.00117, 8: 7e-05, 12: 4e-05, 7: 2e-05}
    # {3: 0.41436, 9: 0.285, 1: 0.22959, 2: 0.01845, 6: 0.01424, 10: 0.01058, 5: 0.0104, 4: 0.00986, 0: 0.00353, 11: 0.00271, 13: 0.00117, 8: 5e-05, 12: 4e-05, 7: 2e-05}

