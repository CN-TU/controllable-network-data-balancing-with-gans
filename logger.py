import io
import PIL.Image
import seaborn as sns
import matplotlib.pyplot as plt
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


class Logger:

    def __init__(self, log_dir=None, use_wandb=False, wandb_config=None):
        """
        Looger class for experiments.

        Args:
            log_dir: Str. Log dir to write TensorBoard/Wandb logs to.
            use_wandb: Bool. Indicates whether Weights&Biases (wandb) should be used.
                wandb is nice, as it allows for easy debugging via gradient/param tracking of the models.
                It also integrates nicely with TensorBoard.
            wandb_config: Dict.
        """
        if log_dir:
            # make wandb before calling TensorBoard
            if use_wandb:
                self.setup_wandb(str(log_dir), wandb_config)
            self.summary_writer = SummaryWriter(str(log_dir))

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
        wandb.tensorboard.patch(root_logdir=log_dir)
        wandb.init(project="interdisciplinary_project", name=log_dir,
                   dir=log_dir, config=wandb_config)

    def watch_wandb(self, G, D):
        wandb.watch(G, log="all")
        wandb.watch(D, log="all")

    def update_wandb_config(self, config):
        wandb.config.update(config)

    def wandb_dummy_log(self):
        wandb.log({"test": 1})