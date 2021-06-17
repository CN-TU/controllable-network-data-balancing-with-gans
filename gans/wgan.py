import torch
import numpy as np
from .base_gan import BaseGAN


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
        noise, noise_labels, noise_condition_vectors = self.make_noise_and_labels(
            batch_size,
            label_weights,
            condition_vector_dict
        )
        generated_features = self.G(noise, noise_labels if noise_condition_vectors is None else noise_condition_vectors)

        # train discriminator
        D_stats = self.fit_discriminator(
            features,
            labels if condition_vectors is None else condition_vectors,
            generated_features,
            noise_labels if noise_condition_vectors is None else noise_condition_vectors,
        )

        # train generator
        G_stats = dict()
        if step % G_train_freq == 0:
            noise, noise_labels, noise_condition_vectors = self.make_noise_and_labels(batch_size, label_weights,
                                                                                      condition_vector_dict)
            generated_features, noise_labels, G_stats = self.fit_generator(
                noise,
                noise_labels if noise_condition_vectors is None else noise_condition_vectors
            )

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
            pred = np.concatenate([class_preds_real.detach().cpu().numpy(), class_preds_fake.detach().cpu().numpy()],
                                  axis=0)
            gt = np.concatenate([labels.detach().cpu().numpy(), noise_labels.detach().cpu().numpy()], axis=0)
            D_acc = np.mean(np.argmax(pred, axis=1) == gt)
        else:
            validity_real = self.D(features.float(), labels)
            validity_fake = self.D(generated_features if self.use_gradient_penalty else generated_features.detach(),
                                   noise_labels)
                                   # noise_labels if self.use_gradient_penalty else noise_labels.detach())

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
            metrics.update({"D_loss_aux": D_loss_aux.item(), "D_acc": D_acc, })
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
