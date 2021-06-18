import torch
import numpy as np
from .base_gan import BaseGAN


class ACGAN(BaseGAN):

    def __init__(self, G, D, G_optimizer, D_optimizer, lambda_auxiliary=1.0,
                 use_wandb=False, use_static_condition_vectors=False, use_dynamic_condition_vectors=False,
                 model_save_dir=None, log_dir=None, device=None, condition_vector_dict=None):
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
        super().__init__(G, D, G_optimizer, D_optimizer,
                         use_wandb, use_static_condition_vectors, use_dynamic_condition_vectors,
                         model_save_dir, log_dir, device, condition_vector_dict)
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.auxiliary_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.lambda_auxiliary = lambda_auxiliary

    def _train_epoch(self, features, labels, step, G_train_freq=1, label_weights=None, condition_vectors=None):
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
