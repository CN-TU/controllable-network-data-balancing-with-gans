"""
GAN implementations adjusted from
- https://github.com/eriklindernoren/PyTorch-GAN

All supported architectures:
    1. Conditional GAN (CGAN)
    2. Auxiliary Classifier GAN (ACGAN)
    3. Conditional Wasserstein GAN (WGAN)
    4. Conditional Wasserstein GAN with gradient penalty (WGAN-GP)
    5. Auxiliary Classifier Wasserstein GAN (ACWGAN)

"""
import torch
import numpy as np

from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report

import utils
from logger import Logger


class BaseGAN:

    def __init__(self, G, D, G_optimizer, D_optimizer, use_wandb=False,
                 use_static_condition_vectors=False, use_dynamic_condition_vectors=False,
                 model_save_dir=None, log_dir=None, device=None, condition_vector_dict=None):
        """

        Args:
            G: torch.nn.Module. The generator part of the GAN.
            D: torch.nn.Module. The discriminator part of the GAN.
            G_optimizer: torch Optimizer. Generator optimizer.
            D_optimizer: torch Optimizer. Discriminator optimizer.
            model_save_dir: String. Directory path to save trained model to.
            log_dir: String. Directory path to log TensorBoard logs to.
            device: torch.device. Either cpu or gpu device.
            use_wandb: Bool. Indicates whether Weights & Biases model tracking should be used.
            use_static_condition_vectors: Bool. Indicates whether static condition vectors should be used.
            use_dynamic_condition_vectors: Boo. Indicates whether dynamic condition vectors should be used.
            condition_vector_dict: Dict. Used for constructing static condition vectors.
        """
        assert not (use_static_condition_vectors and use_dynamic_condition_vectors)
        if use_static_condition_vectors:
            assert condition_vector_dict is not None
        self.G, self.D = G, D
        self.G_optimizer, self.D_optimizer = G_optimizer, D_optimizer
        self.latent_dim, self.num_labels = self.G.latent_dim, self.G.num_labels
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu") \
            if not device else device
        self.G.to(self.device)
        self.D.to(self.device)
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.condition_vector_dict = condition_vector_dict
        self.use_static_condition_vectors = use_static_condition_vectors
        self.use_dynamic_condition_vectors = use_dynamic_condition_vectors
        time = datetime.now().strftime("%d-%m-%Y_%Hh%Mm")
        if self.model_save_dir:
            self.model_save_dir = Path(model_save_dir) / time
            Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
            print("Saving models to: ", self.model_save_dir)
        if self.log_dir:
            self.log_dir = Path(log_dir) / time
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            print("Writing logs to: ", self.log_dir)
            self.logger = Logger(self.log_dir, self.use_wandb, {"architecture": self.__class__.__name__})
            if self.use_wandb:
                self.logger.watch_wandb(self.G, self.D)

    def train_epoch(self, train_loader, epoch, log_freq=50, log_tensorboard_freq=1, G_train_freq=1, label_weights=None):
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
            for step, (features, labels, condition_vectors) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                if self.use_static_condition_vectors or self.use_dynamic_condition_vectors:
                    condition_vectors = condition_vectors.float().to(self.device)
                else:
                    condition_vectors = None

                # update params
                stats = self._train_epoch(features, labels, step, G_train_freq, label_weights, condition_vectors)

                # logging
                running_stats = {k: v + stats[k] for k, v in running_stats.items() if k in stats}
                if step % log_freq == 0:
                    self.logger.log_to_commandline(stats, epoch, step, total_steps)

                if self.log_dir and step % log_tensorboard_freq == 0:
                    stats = {"GAN_losses/" + k: v for k, v in stats.items()}
                    self.logger.log_to_tensorboard(stats, epoch, step, total_steps)
                pbar.update(1)
            if self.use_wandb:
                # to enable gradient logging, .log() has to be called once after backwards() pass.
                self.logger.wandb_dummy_log()

        # logs after epoch
        if self.log_dir:
            stats_epoch = {"GAN_losses_epoch/" + k: v / total_steps for k, v in running_stats.items()}
            self.logger.log_to_tensorboard(stats_epoch, epoch, 0, 1)

    def _train_epoch(self, features, labels, step, G_train_freq=1, label_weights=None, condition_vectors=None):
        raise NotImplementedError()

    def evaluate(self, dataset, cols_to_plot, step,
                 num_samples=1024, label_weights=None, classifier=None,
                 scaler=None, label_encoder=None, class_means=None,
                 significance_test=None, compute_euclidean_distances=False):
        """
        Evaluates the generated features:
            - Compares generated feature distributions with actual distributions of the given test set.
            - If classifier, scaler, label_encoder are given, computes the label prediction
              of classifier and generates a confusion matrix.
            - Plots are written to TensorBoard.

        Args:
            dataset: PyTorch Dataset. Contains the actual flows, i.e.,
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
            significance_test: Str or None. If given, indicates what significance test should be run.
            compute_euclidean_distances: Bool. Indicates whether euclidean distances between generated features and mean
                    flow of entire dataset should be computed per class.
        """
        self.set_mode("eval")
        col_to_idx = {col: i for i, col in enumerate(dataset.column_names)}
        generated_features, labels, _ = self.generate(num_samples, label_weights=label_weights)
        generated_features = generated_features.cpu()
        labels = labels.cpu()

        for col in cols_to_plot:
            idx = col_to_idx[col]
            real = dataset.X[:, idx]
            fake = generated_features[:, idx]
            self.logger.add_distplot(real, fake.cpu(), col, step)

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
            self.logger.add_confusion_matric(labels_transformed, label_preds, step)

        if significance_test:
            n_real_features_by_class = utils.run_significance_tests(
                dataset.X, dataset.y,
                generated_features, labels,
                list(col_to_idx.keys()),
                label_encoder.classes_,
                test=significance_test
            )
            n_real_features_mean = sum(n_real_features_by_class.values()) / len(n_real_features_by_class)
            n_real_features_by_class["mean_n_real_features"] = n_real_features_mean
            n_real_features_by_class = {"N_real_features/" + k: v for k, v in n_real_features_by_class.items()}
            self.logger.log_to_tensorboard(n_real_features_by_class, step, 0, 1)

        if compute_euclidean_distances and class_means is not None:
            distance_by_class = utils.compute_euclidean_distance_by_class(
                class_means,
                generated_features,
                labels,
                list(col_to_idx.keys()),
                label_encoder.classes_
            )
            distance_by_class = {"Distance_measures/" + k: v for k, v in distance_by_class.items()}
            self.logger.log_to_tensorboard(distance_by_class, step, 0, 1)

    def generate(self, num_samples=1024, label_weights=None, scaler=None, label_encoder=None,
                 condition_vectors=None, labels=None):
        """
        Generates the given number of fake flows.

        Args:
            num_samples: Int. Number of samples to generate
            label_weights: None or List. Weights used for random generation of fake labels.
            scaler: sklearn model.
            label_encoder: sklearn model.
            condition_vectors: None or List. Contains predefined condition vectors.
            labels: None or List. Contains predefined labels.

        Returns: torch.Tensor of generated samples, torch.Tensor of generated labels,
                 (optionally) torch.Tensor of condition vectors

        """
        self.set_mode("eval")
        with torch.no_grad():
            if condition_vectors and labels:
                if not isinstance(condition_vectors, torch.Tensor):
                    condition_vectors = torch.Tensor(condition_vectors).to(self.device)
                noise = self.make_noise(len(condition_vectors))
            else:
                noise, labels, condition_vectors = self.make_noise_and_labels(num_samples, label_weights)
            if self.use_static_condition_vectors or self.use_dynamic_condition_vectors:
                generated_features = self.G(noise, condition_vectors)
            else:
                generated_features = self.G(noise, labels)

        if scaler:
            generated_features = scaler.inverse_transform(generated_features)
        if label_encoder:
            labels = label_encoder.inverse_transform(labels)

        return generated_features, labels, condition_vectors

    def make_noise_and_labels(self, num_samples, label_weights=None):
        noise = self.make_noise(num_samples)
        # make raw labels np.array, to avoid converting back for condition vector
        raw_labels = self.make_labels(num_samples, label_weights)
        noise_labels = torch.LongTensor(raw_labels).to(self.device)
        condition_vectors = None
        if self.use_static_condition_vectors:
            condition_vectors = self.make_static_condition_vectors(raw_labels)
        elif self.use_dynamic_condition_vectors:
            condition_vectors = self.make_dynamic_condition_vectors(raw_labels)
        return noise, noise_labels, condition_vectors

    def make_noise(self, num_samples):
        return torch.FloatTensor(np.random.normal(0, 1, (num_samples, self.latent_dim))).to(self.device)

    def make_labels(self, num_samples, label_weights):
        return np.random.choice(np.arange(0, self.num_labels), num_samples, p=label_weights)

    def make_static_condition_vectors(self, labels):
        condition_vectors = []
        for label in labels:
            condition_vectors.append(self.condition_vector_dict[label])
        return torch.Tensor(condition_vectors).to(self.device)

    def make_dynamic_condition_vectors(self, labels):
        condition_vectors = []

        # #  Old approach: randomly construct the condition vectors (does not work well)
        # condition_vector_features = utils.get_condition_vector_names()
        # for label in labels:
        #     vector = []
        #     for feature in condition_vector_features:
        #         if feature == "Destination Port":
        #             # vector.append(self.condition_vector_dict[label][0])
        #             continue
        #         elif "Flag" in feature:
        #             # either 0 or 1
        #             vector.append(np.random.randint(0, 2))
        #         else:
        #             # random one hot vector with three levels (low/mid/high)
        #             vector += np.eye(3)[np.random.choice(3)].tolist()
        #     condition_vectors.append(vector)

        # New approach: use real condition vectors instead of randomly constructed ones
        # TODO: think about adding the label one-hot vector
        for label in labels:
            vectors = self.condition_vector_dict[label]
            selected = vectors[np.random.choice(vectors.shape[0])]
            condition_vectors.append(selected)
        return torch.Tensor(condition_vectors).to(self.device)

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

    def set_mode(self, mode="train"):
        for key in dir(self):
            module = getattr(self, key)
            if isinstance(module, torch.nn.Module):
                if mode == "train":
                    module.train()
                elif mode == "eval":
                    module.eval()
                else:
                    raise ValueError("Invalid mode; allowed are ['train', 'eval'].")
