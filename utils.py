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


class CustomLogger:

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
                "per_epoch": ["Multiline", ["GAN_epoch_losses/G_loss", "GAN_epoch_losses/D_loss"]]
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