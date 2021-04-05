"""
Generator/discriminator adjusted from:
 https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py

"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter


class Generator(nn.Module):
    def __init__(self, num_features, num_labels, latent_dim=100):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(num_labels, num_labels)

        def block(in_dim, out_dim, normalize=True):
            layers = [nn.Linear(in_dim, out_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + num_labels, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, num_features),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_embedding(labels)), -1)
        out = self.model(gen_input)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_features, num_labels):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.label_embedding = nn.Embedding(num_labels, num_labels)

        self.model = nn.Sequential(
            nn.Linear(num_features + num_labels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            # nn.Sigmoid()
        )

    def forward(self, features, labels):
        d_in = torch.cat((features, self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class Experiment:

    def __init__(self, G, D, G_optimizer, D_optimizer, criterion, model_save_dir=None, log_dir=None, device=None):
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
            # # dummy inputs to save graph (cannot save BOTH generator and discriminator unfortunately )
            # noise = torch.zeros((1, self.G.latent_dim))
            # labels = torch.zeros(1, dtype=torch.long)
            # features = torch.zeros((1, self.D.num_features))
            # self.summary_writer.add_graph(self.G, [noise, labels])
            # self.summary_writer.add_graph(self.D, [features, labels])

    def train_epoch(self, train_loader, epoch, log_freq=50, log_tensorboard_freq=1):
        total_steps = len(train_loader)
        running_stats = {"G_loss": 0, "D_loss": 0, "D_loss_fake": 0, "D_loss_real": 0}
        with tqdm(total=total_steps, desc="Train loop") as pbar:
            for step, (features, labels) in enumerate(train_loader):
                # ground truths
                batch_size = features.shape[0]
                real = torch.FloatTensor(batch_size, 1).fill_(1.0)
                fake = torch.FloatTensor(batch_size, 1).fill_(0.0)

                # train generator
                generated_features, noise_labels, G_stats = self.fit_generator(real, batch_size)

                # train discriminator
                D_stats = self.fit_discriminator(features, labels, generated_features,
                                                 noise_labels, real, fake)

                # logging
                stats = {**G_stats, **D_stats}
                running_stats = {k: v + stats[k] for k, v in running_stats.items()}
                if step % log_freq == 0:
                    self.log_to_output(stats, epoch, step, total_steps)

                if self.log_dir and step % log_tensorboard_freq == 0:
                    self.log_to_tensorboard(stats, epoch, step, total_steps)
                pbar.update(1)

        # logs after epoch
        if self.log_dir:
            stats_epoch = {k + "_epoch": v / total_steps for k, v in running_stats.items()}
            self.log_to_tensorboard(stats_epoch, epoch, 0, 1)

    def fit_generator(self, real, batch_size):
        self.G_optimizer.zero_grad()
        noise = torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim)))
        noise_labels = torch.LongTensor(np.random.randint(0, self.num_labels, batch_size))
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

    def log_to_output(self, stats, epoch, step, total_steps):
        stats_str = " | ".join([f"{k}: {v:.5f}" for k, v in stats.items()])
        print(f"\nEpoch {epoch} | Batch {step}/{total_steps} |  {stats_str}")

    def log_to_tensorboard(self, stats, epoch, step, steps_per_epoch):
        global_step = epoch * steps_per_epoch + step
        for k, v in stats.items():
            self.summary_writer.add_scalar(k, v, global_step=global_step)

    def save_model(self, epoch):
        torch.save({
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'optim_G_state_dict': self.G_optimizer.state_dict(),
            'optim_D_state_dict': self.D_optimizer.state_dict(),
        }, self.model_save_dir / f"model-{epoch}.pt")

    def evaluate(self):
        pass
