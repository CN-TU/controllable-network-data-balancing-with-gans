"""
GAN implementations adjusted from
- https://github.com/eriklindernoren/PyTorch-GAN

"""
import torch
import numpy as np

from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report

import utils
from utils import CustomLogger


class BaseExperiment:

    def __init__(self, G, D, G_optimizer, D_optimizer, criterion=None, model_save_dir=None, log_dir=None, device=None):
        """

        Args:
            G: torch.nn.Module. The generator part of the GAN.
            D: torch.nn.Module. The discriminator part of the GAN.
            G_optimizer: torch Optimizer. Generator optimizer.
            D_optimizer: torch Optimizer. Discriminator optimizer.
            criterion: torch criterion. Loss function to optimizer for.
            model_save_dir: String. Directory path to save trained model to.
            log_dir: String. Directory path to log TensorBoard logs to.
            device: torch.device. Either cpu or gpu device.
        """
        self.G, self.D = G, D
        self.G_optimizer, self.D_optimizer = G_optimizer, D_optimizer
        self.criterion = criterion
        self.latent_dim, self.num_labels = self.G.latent_dim, self.G.num_labels
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu") \
            if not device else device
        self.G.to(self.device)
        self.D.to(self.device)
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        time = datetime.now().strftime("%d-%m-%Y_%Hh%Mm")
        if self.model_save_dir:
            self.model_save_dir = Path(model_save_dir) / time
            Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
            print("Saving models to: ", self.model_save_dir)
        if self.log_dir:
            self.log_dir = Path(log_dir) / time
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self.summary_writer = SummaryWriter(self.log_dir)
            print("Writing logs to: ", self.log_dir)
            # # dummy inputs to save graph (cannot save BOTH generator and discriminator unfortunately)
            # noise = torch.zeros((1, self.G.latent_dim))
            # labels = torch.zeros(1, dtype=torch.long)
            # features = torch.zeros((1, self.D.num_features))
            # self.summary_writer.add_graph(self.G, [noise, labels])
            # self.summary_writer.add_graph(self.D, [features, labels])
            self.logger = CustomLogger(self.summary_writer)

    def train_epoch(self, train_loader, epoch, log_freq=50, log_tensorboard_freq=1, label_weights=None):
        """
        Runs a single training epoch.

        Args:
            train_loader: torch.utils.data.DataLoader. Iterable over dataset.
            epoch: Int. Current epoch, important for TensorBoard logging.
            log_freq: Int. Determines the logging frequency for commandline outputs.
            log_tensorboard_freq: Int. Determines the logging frequency of TensorBoard logs.
            label_weights: None or List. Weights used for random generation of fake labels.

        """
        raise NotImplementedError()

    def evaluate(self, test_loader, col_to_idx, cols_to_plot, step,
                 num_samples=1024, label_weights=None, classifier=None,
                 scaler=None, label_encoder=None, run_significance_tests=False):
        """
        Evaluates the generated features:
            - Compares generated feature distributions with actual distributions of the given test set.
            - If classifier, scaler, label_encoder are given, computes the label prediction
              of classifier and generates a confusion matrix.
            - Plots are written to TensorBoard.

        Args:
            test_loader: PyTorch Dataset. Contains the actual flows, i.e.,
                the real feature distributions to compare against
            col_to_idx: Dict. Map column names to index.
            cols_to_plot: List. Names of feature columns to plot.
            step: Int. Current step/epoch.
            num_samples: Int. Number of fake samples to generate.
            label_weights: None or List. Weights used for random generation of fake labels.
            classifier: sklearn model. Use for predicting the classes of the generated samples.
            scaler: sklearn model. The classifier is trained on unscaled flows. However, the GAN (typically) is
                    trained on scaled flows. Therefore, we have to unscale them using the original scaler.
            label_encoder: sklearn model. Original label encoder used to create the dataset. Predicted numeric labels
                           can be transformed to clear-name labels.
            run_significance_tests: Bool. Indicates whether significance test on feature columns should be run or not.

        """
        generated_features, labels = self.generate(num_samples, label_weights)
        generated_features = generated_features.cpu()
        labels = labels.cpu()

        for col in cols_to_plot:
            idx = col_to_idx[col]
            real = test_loader.X[:, idx]
            fake = generated_features[:, idx]
            dist_plot = utils.make_distplot(real, fake.cpu(), col)
            self.summary_writer.add_image("Distributions/" + col, dist_plot, step)

        if classifier and label_encoder:
            label_preds = classifier.predict(generated_features if not scaler
                                             else scaler.inverse_transform(generated_features))
            labels_transformed = label_encoder.inverse_transform(labels)
            label_preds = label_encoder.inverse_transform(label_preds)
            report = classification_report(labels_transformed, label_preds, output_dict=True)
            macro_avg, weighted_avg = report["macro avg"], report["weighted avg"]
            metrics = {
                "accuracy": report["accuracy"], "macro_precision": macro_avg["precision"],
                "macro_recall": macro_avg["recall"], "macro_f1": macro_avg["f1-score"],
                "weighted_precision": weighted_avg["precision"], "weighted_recall": weighted_avg["recall"],
                "weighted_f1": weighted_avg["f1-score"],
            }
            metrics = {"Classifier/" + k: v for k, v in metrics.items()}
            self.logger.log_to_tensorboard(metrics, step, 0, 1)
            print(classification_report(labels_transformed, label_preds))
            confusion_matrix = utils.make_confusion_matrix(labels_transformed, label_preds)
            self.summary_writer.add_image("classifier_confusion_matrix", confusion_matrix, step)

        if run_significance_tests:
            n_real_features_by_class = utils.run_significance_tests(
                test_loader.X, test_loader.y,
                generated_features, labels,
                list(col_to_idx.keys()),
                label_encoder.classes_
            )
            n_real_features_by_class = {"N_real_features/" + k: v for k, v in n_real_features_by_class.items()}
            self.logger.log_to_tensorboard(n_real_features_by_class, step, 0, 1)

    def generate(self, num_samples=1024, label_weights=None):
        """
        Generates the given number of fake flows.

        Args:
            num_samples: Int. Number of samples to generate
            label_weights: None or List. Weights used for random generation of fake labels.

        Returns: torch.Tensor of generated samples, torch.Tensor of generated labels

        """
        with torch.no_grad():
            noise, noise_labels = self.make_noise(num_samples, label_weights)
            generated_features = self.G(noise, noise_labels)
        return generated_features, noise_labels

    def make_noise(self, num_samples, label_weights=None):
        noise = torch.FloatTensor(np.random.normal(0, 1, (num_samples, self.latent_dim))).to(self.device)
        noise_labels = torch.LongTensor(np.random.choice(np.arange(0, self.num_labels),
                                                         num_samples, p=label_weights)).to(self.device)
        return noise, noise_labels

    def save_model(self, epoch):
        torch.save({
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'optim_G_state_dict': self.G_optimizer.state_dict(),
            'optim_D_state_dict': self.D_optimizer.state_dict(),
        }, self.model_save_dir / f"model-{epoch}.pt")

    def load_model(self, model_dict_path, load_optimizer=True):
        checkpoint = torch.load(model_dict_path, map_location=self.device)
        self.G.load_state_dict(checkpoint["G_state_dict"])
        print("Loaded G weights.")
        self.D.load_state_dict(checkpoint["D_state_dict"])
        print("Loaded D weights.")
        if load_optimizer:
            self.G_optimizer.load_state_dict(checkpoint["optim_G_state_dict"])
            for state in self.G_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            print("Loaded G state_dict.")
            self.D_optimizer.load_state_dict(checkpoint["optim_D_state_dict"])
            for state in self.D_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            print("Loaded D state_dict.")


class CGANExperiment(BaseExperiment):

    def __init__(self, G, D, G_optimizer, D_optimizer, criterion=None, model_save_dir=None, log_dir=None, device=None):
        """
        Implement the conditional GAN architecture.

        Args:
            G: torch.nn.Module. The generator part of the GAN.
            D: torch.nn.Module. The discriminator part of the GAN.
            G_optimizer: torch Optimizer. Generator optimizer.
            D_optimizer: torch Optimizer. Discriminator optimizer.
            criterion: torch criterion. Loss function to optimizer for.
            model_save_dir: String. Directory path to save trained model to.
            log_dir: String. Directory path to log TensorBoard logs to.
            device: torch.device. Either cpu or gpu device.
        """
        super().__init__(G, D, G_optimizer, D_optimizer, criterion, model_save_dir, log_dir, device)

    def train_epoch(self, train_loader, epoch, log_freq=50, log_tensorboard_freq=1, label_weights=None):
        """
        Runs a single training epoch.

        Args:
            train_loader: torch.utils.data.DataLoader. Iterable over dataset.
            epoch: Int. Current epoch, important for TensorBoard logging.
            log_freq: Int. Determines the logging frequency for commandline outputs.
            log_tensorboard_freq: Int. Determines the logging frequency of TensorBoard logs.
            label_weights: None or List. Weights used for random generation of fake labels.

        """
        total_steps = len(train_loader)
        running_stats = {"G_loss": 0, "D_loss": 0, "D_loss_fake": 0, "D_loss_real": 0}
        with tqdm(total=total_steps, desc="Train loop") as pbar:
            for step, (features, labels) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)

                # ground truths
                batch_size = features.shape[0]
                real = torch.FloatTensor(batch_size, 1).fill_(1.0).to(self.device)
                fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(self.device)

                # train generator
                generated_features, noise_labels, G_stats = self.fit_generator(batch_size, real,
                                                                               label_weights=label_weights)

                # train discriminator
                D_stats = self.fit_discriminator(features, labels, generated_features,
                                                 noise_labels, real, fake)

                # logging
                stats = {**G_stats, **D_stats}
                running_stats = {k: v + stats[k] for k, v in running_stats.items()}
                if step % log_freq == 0:
                    self.logger.log_to_commandline(stats, epoch, step, total_steps)

                if self.log_dir and step % log_tensorboard_freq == 0:
                    stats = {"GAN_losses/" + k: v for k, v in stats.items()}
                    self.logger.log_to_tensorboard(stats, epoch, step, total_steps)
                pbar.update(1)

        # logs after epoch
        if self.log_dir:
            stats_epoch = {"GAN_losses_epoch/" + k: v / total_steps for k, v in running_stats.items()}
            self.logger.log_to_tensorboard(stats_epoch, epoch, 0, 1)

    def fit_generator(self, batch_size, real, label_weights=None):
        self.G_optimizer.zero_grad()
        noise, noise_labels = self.make_noise(batch_size, label_weights)
        generated_features = self.G(noise, noise_labels)
        validity = self.D(generated_features, noise_labels)
        G_loss = self.criterion(validity, real)
        G_loss.backward()
        self.G_optimizer.step()
        return generated_features, noise_labels, {"G_loss": G_loss.item()}

    def fit_discriminator(self, features, labels, generated_features,
                          noise_labels, real, fake):
        self.D_optimizer.zero_grad()
        validity_real = self.D(features.float(), labels)
        validity_fake = self.D(generated_features.detach(), noise_labels)
        D_loss_real = self.criterion(validity_real, real)
        D_loss_fake = self.criterion(validity_fake, fake)
        D_loss = (D_loss_real + D_loss_fake) / 2
        D_loss.backward()
        self.D_optimizer.step()
        return {"D_loss": D_loss.item(), "D_loss_real": D_loss_real.item(), "D_loss_fake": D_loss_fake.item()}


class CWGANExperiment(BaseExperiment):

    def __init__(self, G, D, G_optimizer, D_optimizer, use_gradient_penalty=False,
                 criterion=None, model_save_dir=None, log_dir=None, device=None):
        """
        Implement the conditional Wasserstein-GAN architecture.

        Args:
            G: torch.nn.Module. The generator part of the GAN.
            D: torch.nn.Module. The discriminator part of the GAN.
            G_optimizer: torch Optimizer. Generator optimizer.
            D_optimizer: torch Optimizer. Discriminator optimizer.
            criterion: torch criterion. Loss function to optimizer for.
            model_save_dir: String. Directory path to save trained model to.
            log_dir: String. Directory path to log TensorBoard logs to.
            device: torch.device. Either cpu or gpu device.
        """
        super().__init__(G, D, G_optimizer, D_optimizer, criterion, model_save_dir, log_dir, device)
        self.use_gradient_penalty = use_gradient_penalty
        self.lambda_gp = 10

    def train_epoch(self, train_loader, epoch, log_freq=50, log_tensorboard_freq=1,
                    G_train_freq=5, label_weights=None):
        """
        Runs a single training epoch.

        Args:
            train_loader: torch.utils.data.DataLoader. Iterable over dataset.
            epoch: Int. Current epoch, important for TensorBoard logging.
            log_freq: Int. Determines the logging frequency for commandline outputs.
            log_tensorboard_freq: Int. Determines the logging frequency of TensorBoard logs.
            G_train_freq:
            label_weights: None or List. Weights used for random generation of fake labels.

        """
        total_steps = len(train_loader)
        running_stats = {"G_loss": 0, "D_loss": 0, "D_loss_fake": 0, "D_loss_real": 0}
        with tqdm(total=total_steps, desc="Train loop") as pbar:
            for step, (features, labels) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)

                # ground truths
                batch_size = features.shape[0]

                noise, noise_labels = self.make_noise(batch_size, label_weights)
                generated_features = self.G(noise, noise_labels)

                # train discriminator
                D_stats = self.fit_discriminator(features, labels, generated_features, noise_labels)

                # train generator
                if step % G_train_freq == 0:
                    generated_features, noise_labels, G_stats = self.fit_generator(noise, noise_labels)

                # logging
                stats = {**G_stats, **D_stats}
                running_stats = {k: v + stats[k] for k, v in running_stats.items()}
                if step % log_freq == 0:
                    self.logger.log_to_commandline(stats, epoch, step, total_steps)

                if self.log_dir and step % log_tensorboard_freq == 0:
                    stats = {"GAN_losses/" + k: v for k, v in stats.items()}
                    self.logger.log_to_tensorboard(stats, epoch, step, total_steps)
                pbar.update(1)

        # logs after epoch
        if self.log_dir:
            stats_epoch = {"GAN_losses_epoch/" + k: v / total_steps for k, v in running_stats.items()}
            self.logger.log_to_tensorboard(stats_epoch, epoch, 0, 1)

    def fit_generator(self, noise, noise_labels):
        self.G_optimizer.zero_grad()
        generated_features = self.G(noise, noise_labels)
        G_loss = -torch.mean(self.D(generated_features, noise_labels))
        G_loss.backward()
        self.G_optimizer.step()
        return generated_features, noise_labels, {"G_loss": G_loss.item()}

    def fit_discriminator(self, features, labels, generated_features, noise_labels):
        self.D_optimizer.zero_grad()
        validity_real = self.D(features.float(), labels)
        validity_fake = self.D(generated_features if self.use_gradient_penalty else generated_features.detach(),
                               noise_labels)
        D_loss_real = -torch.mean(validity_real)
        D_loss_fake = torch.mean(validity_fake)
        if self.use_gradient_penalty:
            gradient_penalty = self.compute_gradient_penalty(features.data, labels, generated_features.data)
            D_loss = D_loss_real + D_loss_fake + self.lambda_gp * gradient_penalty
        else:
            D_loss = D_loss_real + D_loss_fake
        D_loss.backward()
        self.D_optimizer.step()
        # clip weights
        if self.use_gradient_penalty:
            for p in self.D.parameters():
                p.data.clamp_(-0.01, 0.01)
        return {"D_loss": D_loss.item(), "D_loss_real": D_loss_real.item(), "D_loss_fake": D_loss_fake.item()}

    def compute_gradient_penalty(self, features, labels, generated_features):
        alpha = torch.FloatTensor(np.random.random((features.size(0), 1))).to(self.device)
        interpolates = (alpha * features.float() + ((1 - alpha) * generated_features)).requires_grad_(True)
        d_interpolates = self.D(interpolates, labels)
        fake = torch.FloatTensor(features.shape[0], 1).fill_(1.0).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty.to(self.device)
