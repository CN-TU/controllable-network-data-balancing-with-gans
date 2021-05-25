import io
import scipy
import PIL.Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scikitplot.metrics import plot_confusion_matrix
from torchvision.transforms import ToTensor


def make_distplot(real, fake, feature_name):
    """
    Images cannot be written to TensorBoard directly.
    Easiest way is to write them to IOBuffer, then convert to PyTorch tensor.

    Creates a distribution plot of the given real/fake values in one plot.

    """
    sns.displot({"Real": real, "Fake": fake}, kind="kde", common_norm=False, fill=True, height=5, aspect=1.5)
    plt.title(feature_name)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close("all")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image


def make_confusion_matrix(y_true, y_pred):
    """
    Images cannot be written to TensorBoard directly.
    Easiest way is to write them to IOBuffer, then convert to PyTorch tensor.

    Creates a confusion matrix based on the given true and predicted labels.
    """
    plot_confusion_matrix(y_true, y_pred, figsize=(12, 10), x_tick_rotation=90)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close("all")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image


def run_significance_tests(real_features, real_labels, generated_features, generated_labels,
                           column_names, class_names, num_labels=14):
    """
    Conducts the Kolmogorov-Smirnov for between each real and generated feature column per class,
    assuming a significance level of 5%.
    The siginificance test determines if the distributions of the given real and generated column are different.
    The GAN should be able to generate "realistic" attacks. Therefore, it is successful if the significance test
    does not detect a difference in the distributions.

    Args:
        real_features: Numpy array.
        real_labels: Numpy array.
        generated_features: Numpy array.
        generated_labels: Numpy array.
        column_names: List.
        class_names: List.
        num_labels: Int.

    Returns: A dictionary containing the number of "real" feature columns for each label.

    """
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
                ks_test = scipy.stats.ks_2samp(real_feature, generated_feature)
                is_different.append(ks_test[1] < 0.05)

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


class Logger:

    def __init__(self, summary_writer):
        self.summary_writer = summary_writer

    @staticmethod
    def log_to_commandline(stats, epoch, step, total_steps):
        stats_str = " | ".join([f"{k}: {v:.5f}" for k, v in stats.items()])
        print(f"\nEpoch {epoch} | Batch {step}/{total_steps} |  {stats_str}")

    def log_to_tensorboard(self, stats, epoch, step, steps_per_epoch):
        global_step = epoch * steps_per_epoch + step
        for k, v in stats.items():
            self.summary_writer.add_scalar(k, v, global_step=global_step)

    def add_all_custom_scalars(self):
        layout = {
            "GAN losses": {
                "combined": ["Multiline", ["GAN_losses/G_loss", "GAN_losses/D_loss"]],
                "per_epoch": ["Multiline", ["GAN_losses_epoch/G_loss", "GAN_losses_epoch/D_loss"]]
            },
            "# of real features per class": {
                "Attack type": ["Multiline", ['N_real_features']]
            },
            "Classifier metrics": {
                "Weighted average metrics": ["Multiline", ['Classifier/weighted']],
                "Macro average metrics": ["Multiline", ['Classifier/macro']],
                "Accuracy": ["Multiline", ['Classifier/accuracy']]
            }
        }
        self.summary_writer.add_custom_scalars(layout)


if __name__ == "__main__":
    import torch
    from networks import Generator, Discriminator
    from experiment import CGANExperiment

    # make model
    G, D = Generator(79, 14), Discriminator(79, 14)
    exp = CGANExperiment(G, D, None, None)
    exp.load_model("./models/cgan/04-05-2021_14h52m/model-150.pt", load_optimizer=False)
    label_distribution = {0: 0.01, 1: 0.23, 2: 0.02, 3: 0.38, 4: 0.01, 5: 0.01, 6: 0.015,
                          7: 0.01, 8: 0.01, 9: 0.265, 10: 0.01, 11: 0.01, 12: 0.01, 13: 0.01}
    label_weights = list(label_distribution.values())

    column_names = torch.load("./data/cic-ids-2017_splits/seed_0/column_names.pt")
    class_names = torch.load("./data/cic-ids-2017_splits/seed_0/class_names.pt")
    class_means = torch.load("./data/cic-ids-2017_splits/seed_0/class_means_scaled.pt")

    flows, labels = exp.generate(1024, label_weights)

    distances = compute_euclidean_distance_by_class(class_means, flows, labels, column_names, class_names)
    print(distances)
