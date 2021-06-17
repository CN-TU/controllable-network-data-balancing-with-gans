import torch
from .base_gan import BaseGAN


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
