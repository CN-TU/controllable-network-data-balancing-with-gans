import io
import scipy
import PIL.Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from scikitplot.metrics import plot_confusion_matrix
from torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter


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


class Logger:

    def __init__(self, log_dir=None, use_wandb=False, wandb_config=None):
        if log_dir:
            # make wandb before calling TensorBoard
            if use_wandb:
                self.setup_wandb(log_dir, wandb_config)
            self.summary_writer = SummaryWriter(log_dir)

    @staticmethod
    def log_to_commandline(stats, epoch, step, total_steps):
        stats_str = " | ".join([f"{k}: {v:.5f}" for k, v in stats.items()])
        print(f"\nEpoch {epoch} | Batch {step}/{total_steps} |  {stats_str}")

    def log_to_tensorboard(self, stats, epoch, step, steps_per_epoch):
        global_step = epoch * steps_per_epoch + step
        for k, v in stats.items():
            self.summary_writer.add_scalar(k, v, global_step=global_step)

    def add_distplot(self, real, fake, col, step):
        dist_plot = make_distplot(real, fake, col)
        self.summary_writer.add_image("Distributions/" + col, dist_plot, step)

    def add_confusion_matric(self, labels, preds, step):
        confusion_matrix = make_confusion_matrix(labels, preds)
        self.summary_writer.add_image("classifier_confusion_matrix", confusion_matrix, step)

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

    def setup_wandb(self, log_dir, wandb_config):
        wandb.login()
        # tracks everything that TensorBoard tracks
        # writes to same dir as TesnorBoard
        wandb.init(project="interdisciplinary_project", name=str(log_dir),
                   dir=log_dir, sync_tensorboard=True, config=wandb_config)

    def watch_wandb(self, G, D):
        wandb.watch(G, log="all")
        wandb.watch(D, log="all")

    def update_wandb_config(self, config):
        wandb.config.update(config)

    def wandb_dummy_log(self):
        wandb.log({"test": 1})


if __name__ == "__main__":
    import torch
    from networks import Generator, Discriminator
    from gans import CGAN

    # make model
    G, D = Generator(79, 14), Discriminator(79, 14)
    exp = CGAN(G, D, None, None)
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
