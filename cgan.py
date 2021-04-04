"""
Implementation adjusted from:
 https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py

"""
import torch
import torch.nn as nn
from pathlib import Path
from tensorboardX import SummaryWriter


class Generator(nn.Module):
    def __init__(self, num_features, num_labels, latent_dim=100):
        super().__init__()
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


class CGAN(nn.Module):

    def __init__(self, G, D, lr=0.001, log_dir=None, device=None):
        super().__init__()
        self.G = G
        self.D = D
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu") \
            if not device else device
        self.log_dir = log_dir
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self.summary_writer = SummaryWriter(self.log_dir)

    def train_epoch(self, data_loader):
        """
        takes dataloaders
            - runs one epoch training
            - should be called within a loop (probably easiest to do)

        then train_cgan.py can just load the datasets, instantiate the CGAN and implement the
        outside training loop + handle all the input params (argparse)
        """
        pass

    def fit_generator(self):
        pass

    def fit_discriminator(self):
        pass

    def evaluate(self):
        pass

    def log_to_tensorboard(self):
        pass
