"""
GAN implementations adjusted from
- https://github.com/eriklindernoren/PyTorch-GAN

Contains the following architectures:
    1. Conditional GAN (CGAN)
    2. Auxiliary Classifier GAN (ACGAN)
    3. Conditional Wasserstein GAN (WGAN)
    4. Conditional Wasserstein GAN with gradient penalty (WGAN-GP)
    5. Auxiliary Classifier Wasserstein GAN (ACWGAN)

"""

# TODO:
#   - validate condition vector approach
#   - adjust other GAN architectures (only CGAN right now. )

import torch
import numpy as np

from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report

import utils


class BaseGAN:

    def __init__(self, G, D, G_optimizer, D_optimizer, use_wandb=False, model_save_dir=None, log_dir=None, device=None):
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
        """
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
        time = datetime.now().strftime("%d-%m-%Y_%Hh%Mm")
        if self.model_save_dir:
            self.model_save_dir = Path(model_save_dir) / time
            Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
            print("Saving models to: ", self.model_save_dir)
        if self.log_dir:
            self.log_dir = Path(log_dir) / time
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            print("Writing logs to: ", self.log_dir)
            self.logger = utils.Logger(self.log_dir, self.use_wandb, {"architecture": self.__class__.__name__})
            if self.use_wandb:
                self.logger.watch_wandb(self.G, self.D)

    def train_epoch(self, train_loader, epoch, log_freq=50, log_tensorboard_freq=1, G_train_freq=1,
                    label_weights=None, condition_vector_dict=None):
        """
        Runs a single training epoch.

        Args:
            train_loader: torch.utils.data.DataLoader. Iterable over dataset.
            epoch: Int. Current epoch, important for TensorBoard logging.
            log_freq: Int. Determines the logging frequency for commandline outputs.
            log_tensorboard_freq: Int. Determines the logging frequency of TensorBoard logs.
            label_weights: None or List. Weights used for random generation of fake labels.
            condition_vector_dict: Dict. If given, the custom condition vectors are used instead of the label condition.

        """
        total_steps = len(train_loader)
        running_stats = {"G_loss": 0, "D_loss": 0, "D_loss_fake": 0, "D_loss_real": 0}
        with tqdm(total=total_steps, desc="Train loop") as pbar:
            for step, (features, labels) in enumerate(train_loader):
                features = features.to(self.device)
                if condition_vector_dict:
                    # TODO: converting to numpy is costly, better add condition vector to dataset at creation
                    condition_vectors = self.make_condition_vectors(labels.numpy(), condition_vector_dict)
                labels = labels.to(self.device)

                # update params
                stats = self._train_epoch(features, labels, step, G_train_freq, label_weights,
                                          condition_vectors if condition_vector_dict else None,
                                          condition_vector_dict)

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

    def _train_epoch(self, features, labels, step, G_train_freq=1,
                     label_weights=None, condition_vectors=None, condition_vector_dict=None):
        raise NotImplementedError()

    def evaluate(self, dataset, cols_to_plot, step,
                 num_samples=1024, label_weights=None, classifier=None,
                 scaler=None, label_encoder=None, class_means=None,
                 significance_test=None, compute_euclidean_distances=False,
                 condition_vector_dict=None):
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
            condition_vector_dict: Dict. If given, the custom condition vectors are used instead of the label condition.
        """
        self.set_mode("eval")
        col_to_idx = {col: i for i, col in enumerate(dataset.column_names)}
        generated_features, labels, _ = self.generate(num_samples, label_weights=label_weights,
                                                      condition_vector_dict=condition_vector_dict)
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

        if compute_euclidean_distances and not class_means is None:
            distance_by_class = utils.compute_euclidean_distance_by_class(
                class_means,
                generated_features,
                labels,
                list(col_to_idx.keys()),
                label_encoder.classes_
            )
            distance_by_class = {"Distance_measures/" + k: v for k, v in distance_by_class.items()}
            self.logger.log_to_tensorboard(distance_by_class, step, 0, 1)

    def generate(self, num_samples=1024, label_weights=None,
                 scaler=None, label_encoder=None, condition_vector_dict=None):
        """
        Generates the given number of fake flows.

        Args:
            num_samples: Int. Number of samples to generate
            label_weights: None or List. Weights used for random generation of fake labels.
            scaler: sklearn model.
            label_encoder: sklearn model.
            condition_vector_dict: Dict. If given, the custom condition vectors are used instead of the label condition.

        Returns: torch.Tensor of generated samples, torch.Tensor of generated labels,
                 (optionally) torch.Tensor of condition vectors

        """
        self.set_mode("eval")
        with torch.no_grad():
            noise, labels, condition_vectors = self.make_noise_and_labels(num_samples, label_weights,
                                                                          condition_vector_dict)
            generated_features = self.G(noise, labels if condition_vectors is None else condition_vectors)

        if scaler:
            generated_features = scaler.inverse_transform(generated_features)
        if label_encoder:
            labels = label_encoder.inverse_transform(labels)
        if condition_vector_dict:
            return generated_features, labels, condition_vectors
        return generated_features, labels, None

    def make_noise_and_labels(self, num_samples, label_weights=None, condition_vector_dict=None):
        noise = self.make_noise(num_samples)
        # make raw labels np.array, to avoid converting back for condition vector
        raw_labels = self.make_labels(num_samples, label_weights)
        noise_labels = torch.LongTensor(raw_labels).to(self.device)
        condition_vectors = None
        if condition_vector_dict:
            condition_vectors = self.make_condition_vectors(raw_labels, condition_vector_dict)
        return noise, noise_labels, condition_vectors

    def make_noise(self, num_samples):
        return torch.FloatTensor(np.random.normal(0, 1, (num_samples, self.latent_dim))).to(self.device)

    def make_labels(self, num_samples, label_weights):
        return np.random.choice(np.arange(0, self.num_labels), num_samples, p=label_weights)

    def make_condition_vectors(self, labels, condition_vector_dict):
        condition_vectors = []
        for label in labels:
            # condition_vectors.append(condition_vector_dict[label])
            condition_vectors.append(condition_vector_dict[label][1:])
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


class CGAN(BaseGAN):

    def __init__(self, G, D, G_optimizer, D_optimizer, use_wandb=False, model_save_dir=None, log_dir=None, device=None):
        """
        Implements the conditional GAN (cGAN) architecture.
        Paper:
            https://arxiv.org/pdf/1411.1784.pdf

        Args:
            G: torch.nn.Module. The generator part of the GAN.
            D: torch.nn.Module. The discriminator part of the GAN.
            G_optimizer: torch Optimizer. Generator optimizer.
            D_optimizer: torch Optimizer. Discriminator optimizer.
            model_save_dir: String. Directory path to save trained model to.
            log_dir: String. Directory path to log TensorBoard logs to.
            device: torch.device. Either cpu or gpu device.
            use_wandb: Bool. Indicates whether Weights & Biases model tracking should be used.

        """
        super().__init__(G, D, G_optimizer, D_optimizer, use_wandb, model_save_dir, log_dir, device)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

    def _train_epoch(self, features, labels, step, G_train_freq=1,
                     label_weights=None, condition_vectors=None, condition_vector_dict=None):
        """
        Fits GAN. Is called by train_epoch() method.
        Args:
            features: torch.Tensor.
            labels: torch.Tensor.
            label_weights: List.

        Returns: a dictionary, stats

        """
        self.set_mode("train")
        batch_size = features.shape[0]
        real = torch.FloatTensor(batch_size, 1).fill_(1.0).to(self.device)
        fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(self.device)
        noise, noise_labels, noise_condition_vectors = self.make_noise_and_labels(
            batch_size,
            label_weights,
            condition_vector_dict
        )
        generated_features, G_stats = self.fit_generator(
            noise,
            noise_labels if noise_condition_vectors is None else noise_condition_vectors,
            real
        )
        D_stats = self.fit_discriminator(
            features,
            labels if condition_vectors is None else condition_vectors,
            generated_features,
            noise_labels if noise_condition_vectors is None else noise_condition_vectors,
            real,
            fake
        )
        return {**G_stats, **D_stats}

    def fit_generator(self, noise, noise_labels, real):
        self.G_optimizer.zero_grad(set_to_none=True)
        generated_features = self.G(noise, noise_labels)
        validity = self.D(generated_features, noise_labels)
        G_loss = self.criterion(validity, real)
        G_loss.backward()
        self.G_optimizer.step()
        return generated_features, {"G_loss": G_loss.item()}

    def fit_discriminator(self, features, labels, generated_features,
                          noise_labels, real, fake):
        self.D_optimizer.zero_grad(set_to_none=True)
        validity_real = self.D(features.float(), labels)
        validity_fake = self.D(generated_features.detach(), noise_labels)
        D_loss_real = self.criterion(validity_real, real)
        D_loss_fake = self.criterion(validity_fake, fake)
        D_loss = (D_loss_real + D_loss_fake) / 2
        D_loss.backward()
        self.D_optimizer.step()
        return {"D_loss": D_loss.item(), "D_loss_real": D_loss_real.item(), "D_loss_fake": D_loss_fake.item()}


class CWGAN(BaseGAN):

    def __init__(self, G, D, G_optimizer, D_optimizer, clip_val=0.1, lambda_gp=10, lambda_auxiliary=1,
                 use_gradient_penalty=False, use_auxiliary_classifier=False, use_wandb=False,
                 model_save_dir=None, log_dir=None, device=None):
        """
        Implements the conditional Wasserstein-GAN (WGAN),
        conditional WGAN with gradient penalty (WGAN-GP),
        and Auxiliary Classifier WGAN (ACWGAN).
        Papers:
            https://arxiv.org/pdf/1701.07875.pdf
            https://arxiv.org/pdf/1704.00028.pdf

        Args:
            G: torch.nn.Module. The generator part of the GAN.
            D: torch.nn.Module. The discriminator part of the GAN.
            G_optimizer: torch Optimizer. Generator optimizer.
            D_optimizer: torch Optimizer. Discriminator optimizer.
            clip_val: Float.
            lambda_gp: Int.
            use_gradient_penalty: Bool. If true, WGAN-GP is used.
            use_auxiliary_classifier: Bool. If true, ACWGAN is used.
            model_save_dir: String. Directory path to save trained model to.
            log_dir: String. Directory path to log TensorBoard logs to.
            device: torch.device. Either cpu or gpu device.
            use_wandb: Bool. Indicates whether Weights & Biases model tracking should be used.

        """
        super().__init__(G, D, G_optimizer, D_optimizer, use_wandb, model_save_dir, log_dir, device)
        self.use_gradient_penalty = use_gradient_penalty
        self.clip_val = clip_val
        self.lambda_gp = lambda_gp
        self.use_auxiliary_classifier = use_auxiliary_classifier
        self.auxiliary_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.lambda_auxiliary = lambda_auxiliary

    def _train_epoch(self, features, labels, step, G_train_freq=5,
                     label_weights=None, condition_vectors=None, condition_vector_dict=None):
        self.set_mode("train")
        batch_size = features.shape[0]
        noise, noise_labels, _ = self.make_noise_and_labels(batch_size, label_weights)
        generated_features = self.G(noise, noise_labels)

        # train discriminator
        D_stats = self.fit_discriminator(features, labels, generated_features, noise_labels)

        # train generator
        G_stats = dict()
        if step % G_train_freq == 0:
            noise, noise_labels, _ = self.make_noise_and_labels(batch_size, label_weights)
            generated_features, noise_labels, G_stats = self.fit_generator(noise, noise_labels)
            # generated_features, noise_labels, G_stats = self.fit_generator(generated_features, noise_labels)

        return {**G_stats, **D_stats}

    def fit_generator(self, noise, noise_labels):
        self.G_optimizer.zero_grad(set_to_none=True)
        generated_features = self.G(noise, noise_labels)
        G_loss, G_loss_aux, metrics = self.compute_generator_loss(generated_features, noise_labels)
        G_loss.backward(retain_graph=self.use_auxiliary_classifier)
        if self.use_auxiliary_classifier:
            G_loss_aux.backward()
        self.G_optimizer.step()
        return generated_features, noise_labels, metrics

    def fit_discriminator(self, features, labels, generated_features, noise_labels):
        self.D_optimizer.zero_grad(set_to_none=True)
        D_loss, D_loss_aux, metrics = self.compute_discriminator_loss(features, labels,
                                                                      generated_features, noise_labels)
        D_loss.backward(retain_graph=self.use_auxiliary_classifier)
        if self.use_auxiliary_classifier:
            D_loss_aux.backward()
        self.D_optimizer.step()
        # clip weights
        if not self.use_gradient_penalty:
            for p in self.D.parameters():
                p.data.clamp_(-self.clip_val, self.clip_val)
        return metrics

    def compute_generator_loss(self, generated_features, noise_labels):
        if self.use_auxiliary_classifier:
            validity, class_preds = self.D(generated_features)
            G_loss = -torch.mean(validity)
            G_loss_aux = self.auxiliary_loss(class_preds, noise_labels) * self.lambda_auxiliary
            metrics = {"G_loss": G_loss.item(),
                       "G_loss_aux": G_loss_aux.item()}
            return G_loss, G_loss_aux, metrics
        else:
            validity = self.D(generated_features, noise_labels)
            G_loss = -torch.mean(validity)
            metrics = {"G_loss": G_loss.item()}
            return G_loss, None, metrics

    def compute_discriminator_loss(self, features, labels, generated_features, noise_labels):
        if self.use_auxiliary_classifier:
            validity_real, class_preds_real = self.D(features.float())
            validity_fake, class_preds_fake = self.D(generated_features.detach())
            D_loss_real = torch.mean(validity_real)
            D_loss_fake = torch.mean(validity_fake)
            D_loss = -D_loss_real + D_loss_fake
            D_loss_real_aux = self.auxiliary_loss(class_preds_real, labels)
            D_loss_fake_aux = self.auxiliary_loss(class_preds_fake, noise_labels)
            D_loss_aux = (D_loss_real_aux + D_loss_fake_aux) * self.lambda_auxiliary
            pred = np.concatenate([class_preds_real.detach().cpu().numpy(), class_preds_fake.detach().cpu().numpy()], axis=0)
            gt = np.concatenate([labels.detach().cpu().numpy(), noise_labels.detach().cpu().numpy()], axis=0)
            D_acc = np.mean(np.argmax(pred, axis=1) == gt)
        else:
            validity_real = self.D(features.float(), labels)
            validity_fake = self.D(generated_features if self.use_gradient_penalty else generated_features.detach(),
                                   noise_labels if self.use_gradient_penalty else noise_labels.detach())
            # validity_fake = self.D(generated_features, noise_labels)

            D_loss_real = torch.mean(validity_real)
            D_loss_fake = torch.mean(validity_fake)
            if self.use_gradient_penalty:
                gradient_penalty = self.compute_gradient_penalty(features, labels, generated_features)
                # D_loss = -D_loss_real + D_loss_fake + self.lambda_gp * gradient_penalty
                D_loss = -(D_loss_real - D_loss_fake)
                gradient_penalty *= self.lambda_gp
                gradient_penalty.backward(retain_graph=True)
            else:
                D_loss = -D_loss_real + D_loss_fake

        metrics = {"D_loss": D_loss.item(), "D_loss_real": D_loss_real.item(), "D_loss_fake": D_loss_fake.item()}
        if self.use_auxiliary_classifier:
            metrics.update({"D_loss_aux": D_loss_aux.item(), "D_acc": D_acc,})
            return D_loss, D_loss_aux, metrics
        return D_loss, None, metrics

    def compute_gradient_penalty(self, features, labels, generated_features):
        alpha = torch.rand(features.size(0), 1, device=self.device)
        interpolates = alpha * features + ((1 - alpha) * generated_features)
        disc_interpolates = self.D(interpolates.float(), labels)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient_penalty = ((
                                    gradients.view(-1, features.size(1)).norm(2, dim=1) - 1
                            ) ** 2).mean()
        return gradient_penalty


class ACGAN(BaseGAN):

    def __init__(self, G, D, G_optimizer, D_optimizer, lambda_auxiliary=1.0,
                 use_wandb=False, model_save_dir=None, log_dir=None, device=None):
        """
        Implements the Auxiliary classifier GAN (AC-GAN)
        Paper:
            https://arxiv.org/pdf/1610.09585.pdf

        Args:
            G: torch.nn.Module. The generator part of the GAN.
            D: torch.nn.Module. The discriminator part of the GAN.
            G_optimizer: torch Optimizer. Generator optimizer.
            D_optimizer: torch Optimizer. Discriminator optimizer.
            criterion: torch criterion. Loss function to optimizer for.
            model_save_dir: String. Directory path to save trained model to.
            log_dir: String. Directory path to log TensorBoard logs to.
            device: torch.device. Either cpu or gpu device.
            use_wandb: Bool. Indicates whether Weights & Biases model tracking should be used.

        """
        super().__init__(G, D, G_optimizer, D_optimizer, use_wandb, model_save_dir, log_dir, device)
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.auxiliary_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.lambda_auxiliary = lambda_auxiliary

    def _train_epoch(self, features, labels, step, G_train_freq=1,
                     label_weights=None, condition_vectors=None, condition_vector_dict=None):
        self.set_mode("train")
        batch_size = features.shape[0]
        real = torch.FloatTensor(batch_size, 1).fill_(1.0).to(self.device)
        fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(self.device)
        noise, noise_labels, _ = self.make_noise_and_labels(batch_size, label_weights)
        generated_features, G_stats = self.fit_generator(noise, noise_labels, real)
        D_stats = self.fit_discriminator(features, labels, generated_features,
                                         noise_labels, real, fake)
        return {**G_stats, **D_stats}

    def fit_generator(self, noise, noise_labels, real):
        self.G_optimizer.zero_grad(set_to_none=True)
        generated_features = self.G(noise, noise_labels)
        validity, class_preds = self.D(generated_features)
        G_loss_adv = self.adversarial_loss(validity, real)
        G_loss_aux = self.auxiliary_loss(class_preds, noise_labels) * self.lambda_auxiliary
        G_loss = G_loss_adv + G_loss_aux
        G_loss.backward()
        self.G_optimizer.step()
        metrics = {"G_loss": G_loss.item(), "G_loss_adv": G_loss_adv.item(), "G_loss_aux": G_loss_aux.item()}
        return generated_features, metrics

    def fit_discriminator(self, features, labels, generated_features,
                          noise_labels, real, fake):
        self.D_optimizer.zero_grad(set_to_none=True)
        validity_real, class_preds_real = self.D(features.float())
        validity_fake, class_preds_fake = self.D(generated_features.detach())

        D_loss_real = (self.adversarial_loss(validity_real, real) +
                       self.lambda_auxiliary * self.auxiliary_loss(class_preds_real, labels))
        D_loss_fake = (self.adversarial_loss(validity_fake, fake) +
                       self.lambda_auxiliary * self.auxiliary_loss(class_preds_fake, noise_labels))
        D_loss = (D_loss_real + D_loss_fake)

        # Calculate discriminator accuracy
        pred = np.concatenate([class_preds_real.detach().cpu().numpy(), class_preds_fake.detach().cpu().numpy()],
                              axis=0)
        gt = np.concatenate([labels.detach().cpu().numpy(), noise_labels.detach().cpu().numpy()], axis=0)
        D_acc = np.mean(np.argmax(pred, axis=1) == gt)

        D_loss.backward()
        self.D_optimizer.step()

        return {"D_loss": D_loss.item(),
                "D_loss_real": D_loss_real.item(),
                "D_loss_fake": D_loss_fake.item(),
                "D_acc": D_acc}
