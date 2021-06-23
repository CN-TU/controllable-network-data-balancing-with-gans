import scipy
import random
import torch
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

def get_label_names():
    return ['Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris',
            'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack \x96 Brute Force',
            'Web Attack \x96 Sql Injection', 'Web Attack \x96 XSS', 'zBENIGN']


def labels_to_labelidx(labels):
    if isinstance(labels, str):
        labels = [labels]
    label_names = get_label_names()
    return [label_names.index(label) for label in labels]


def get_condition_vector_names():
    # as defined by Fares
    # For Flags it does not make sense to compute quantiles actually (they are 0, 1) just add them to the vector
    condition_vector_features = ['Destination Port', 'Flow Duration', 'Total Backward Packets', 'Total Fwd Packets',
                                 'Packet Length Mean', 'Flow Bytes/s', 'Flow IAT Min', 'Flow IAT Max', 'PSH Flag Count',
                                 'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count']
    return condition_vector_features


def run_significance_tests(real_features, real_labels, generated_features, generated_labels,
                           column_names, class_names, num_labels=14, alpha=0.05, test="ks"):
    """
    Conducts a significance test between each real and generated feature column per class,
    assuming a significance level of 5%.
    The siginificance test determines if the distributions of the given real and generated column are different.
    The GAN should be able to generate "realistic" attacks. Therefore, it is successful if the significance test
    does not detect a difference in the distributions.

    Available tests are the Kolmogorov-Smirnov, the two sample ttest, and the two-sample wilcoxon ranksum test.

    Args:
        real_features: Numpy array.
        real_labels: Numpy array.
        generated_features: Numpy array.
        generated_labels: Numpy array.
        column_names: List.
        class_names: List.
        num_labels: Int.
        test: Str. The test to conduct. One of ["ks", "ttest", "ranksums"].

    Returns: A dictionary containing the number of "real" feature columns for each label.

    """
    if test not in ["ks", "ttest", "ranksums"]:
        raise ValueError("Valid tests are 'ks', 'ttest', 'ranksums'")
    stats = []
    for label in range(num_labels):
        real_by_class = real_features[np.where(real_labels == label)]
        generated_by_class = generated_features[np.where(generated_labels == label)]
        is_different = []
        for col in range(len(column_names) - 1):
            real_feature, generated_feature = real_by_class[:, col], generated_by_class[:, col]
            if real_feature.size == 0 or generated_feature.size == 0:
                is_different.append(None)
            else:
                if test == "ks":
                    res = scipy.stats.ks_2samp(real_feature, generated_feature)
                elif test == "ttest":
                    res = scipy.stats.ttest_ind(real_feature, generated_feature)
                elif test == "ranksums":
                    res = scipy.stats.ranksums(real_feature, generated_feature)
                is_different.append(res[1] < alpha)
        stats.append(is_different)

    df_is_different = pd.DataFrame(stats, columns=column_names[:-1], index=class_names[:-1])
    return (df_is_different == False).sum(axis=1).to_dict()


def compute_euclidean_distance_by_class(class_means, generated_features, generated_labels,
                                        column_names, class_names, num_labels=14):
    """
    Args:
        class_means: Numpy array.
        generated_features: Numpy array.
        generated_labels: Numpy array.
        column_names: List.
        class_names: List.
        num_labels: Int.

    Returns: A dictionary containing the mean distance across all columns for each label.
    """
    flows = pd.DataFrame(np.append(generated_features, generated_labels.reshape(-1, 1), axis=1),
                         columns=column_names)
    distances = []
    for label in range(num_labels):
        generated_by_class = flows[flows.Label == label].drop("Label", axis=1)
        mean_by_class = class_means.loc[label]
        distance_by_feature = np.linalg.norm(generated_by_class - mean_by_class, axis=0)
        distances.append(distance_by_feature)

    distances = pd.DataFrame(distances, columns=column_names[:-1], index=range(0, num_labels))
    distances.index = class_names[:-1]
    return distances.mean(axis=1).to_dict()


def set_seeds(seed=None):
    if seed is not None:
        print(f"Setting seed to {seed}.")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


def load_static_condition_levels(path="./data/cic-ids-2017_splits/seed_0/static_condition_levels.pt"):
    return torch.load(path)


def load_dynamic_condition_levels(path="./data/cic-ids-2017_splits/seed_0/dynamic_condition_level_dict.pt"):
    return torch.load(path)


def convert_port_to_binary(port_num, width=16):
    """
    Converts port from integer to binary representation.
    The max port number is an unsigned 16-bit integer, 65535:
        https://stackoverflow.com/questions/113224/what-is-the-largest-tcp-ip-network-port-number-allowable-for-ipv4

    Args:
        port_num: Int.
        width: Int. Number of bits.

    Returns: List.

    """
    bits = np.binary_repr(int(port_num), width=width)
    return [int(val) for val in bits]


def convert_single_level_to_condition(feature, val):
    """
    Utility function to convert single level to its representation used in the condition vector.
    Different features require different processing, hence have to hardcode this.

    Args:
        feature: Str.
        val: Int.

    Returns: List.
    """
    if feature == "Destination Port":
        return convert_port_to_binary(val)
    elif feature in ['PSH Flag Count', 'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count']:
        return [val]
    else:
        vals = [0, 0, 0]
        vals[val] = 1
        return vals


def convert_levels_to_condition_vectors(level_dicts, attack_types=None):
    """
    Utility function to convert the given levels to the (binary) condition vector.
    Different features must be treated differently, e.g., port converted to 16bit vector.

    Expects a list of dictionaries containing feature-level key-value pairs.

    For simplicity, if a feature is left out in the first dictionary and attack_type is given,
    the dynamic condition levels are loaded and the feature is set to the value of a random dynamic condition vector
    for the given class.

    All features:
        ['Destination Port', 'Flow Duration', 'Total Backward Packets', 'Total Fwd Packets',
         'Packet Length Mean', 'Flow Bytes/s', 'Flow IAT Min', 'Flow IAT Max', 'PSH Flag Count',
         'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count']

    Args:
         level_dicts: List. Contains dicts containing feature-level key-value pairs.
         attack_types: List of strings or string.
    Returns: List. The condition vector.

    """
    if isinstance(attack_types, str):
        attack_types = [attack_types] * len(level_dicts)
    if isinstance(level_dicts, dict):
        level_dicts = [level_dicts]

    condition_vectors = []
    all_features = get_condition_vector_names()
    if set(level_dicts[0].keys()) != set(all_features):
        if attack_types is not None:
            # watch out, the attack type keys in default levels are 0-13, not the string names.
            # convert attack_type to its int representation.
            label_names = get_label_names()
            default_levels = load_dynamic_condition_levels()
        else:
            raise ValueError("If not all feature are specified in level_dict, please define"
                             "attack_type to set missing features to their default values.")

    for level_dict, attack_type in zip(level_dicts, attack_types):
        condition_vector = []
        if len(level_dict) != len(all_features):
            # select a random default level
            # default_levels, label_names already exists
            all_levels = default_levels[label_names.index(attack_type)]
            # sample from the default levels of that class.
            default_level = all_levels[np.random.choice(all_levels.shape[0])]
        for i, feature in enumerate(all_features):
            if feature not in level_dict:
                # default_level already exists
                val = int(default_level[i])
            else:
                val = level_dict[feature]
            condition_vector.append(convert_single_level_to_condition(feature, val))
        # flatten condition_vector
        condition_vectors.append([val for rep in condition_vector for val in rep])
    return condition_vectors


def generate_from_levels(gan, levels, attack_types, scaler=None, label_encoder=None,
                         column_names=None):
    """
    Only used within evaluation.ipynb

    Args:
        gan: The trained GAN.
        levels: List.
        attack_types: List.
        scaler: Sklearn scaler.
        label_encoder: Sklearn label encoder.
        column_names: None or np.array.

    Returns: pd.DataFrame

    """
    if not column_names:
        column_names = torch.load("./data/cic-ids-2017_splits/seed_0/column_names.pt")
    condition_vectors = convert_levels_to_condition_vectors(levels, attack_types)
    labels = labels_to_labelidx(attack_types)

    # generate
    flows, _, condition_vectors = gan.generate(
        labels=labels,
        condition_vectors=condition_vectors,
        scaler=scaler,
        label_encoder=label_encoder
    )

    # make df
    df_results = pd.DataFrame(levels).add_suffix("_level")
    df_results["Attack_Type"] = attack_types
    df_flows = pd.DataFrame(flows, columns=column_names[:-1])
    df_results = pd.concat([df_results, df_flows], axis=1)
    return df_results


def load_tf_event_files(paths, architectures=None):
    if isinstance(paths, str):
        paths = [paths]
    # if architectures is not None and not isinstance(architectures, (list, tuple, np.ndarray)):
    if architectures is not None and isinstance(architectures, str):
        architectures = [architectures]
    assert len(paths) == len(architectures)

    metrics = []
    for path, architecture in tqdm(zip(paths, architectures), total=len(paths)):
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        scalar_tags = ea.Tags()['scalars']

        for tag in scalar_tags:
            events_list = ea.Scalars(tag)
            for event in events_list:
                metrics.append({
                    "File_path": path,
                    "Architecture": architecture,
                    "Metric": tag,
                    "wall_time": event[0],
                    "step": event[1],
                    "value": event[2]
                })
    return pd.DataFrame(metrics)


if __name__ == "__main__":
    levels_dict = [{"Destination Port": 80}] * 10
    vec = convert_levels_to_condition_vectors(levels_dict, attack_types="DDoS")
    print(vec)
