"""
Generator/discriminator adjusted from:
 https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py

"""
import torch
import torch.nn as nn


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
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + num_labels, 256, normalize=False),
            *block(256, 256),
            nn.Linear(256, num_features),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_embedding(labels)), -1)
        out = self.model(gen_input)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_features, num_labels, use_class_head=False, use_label_condition=True):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.use_class_head = use_class_head
        self.use_label_condition = use_label_condition
        if self.use_label_condition:
            self.label_embedding = nn.Embedding(num_labels, num_labels)

        input_size = self.num_features
        if self.use_label_condition:
            input_size += self.num_labels

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )
        self.output_head = nn.Linear(256, 1)
        if self.use_class_head:
            self.class_head = nn.Linear(256, num_labels)

    def forward(self, features, labels=None):
        if labels is not None and self.use_label_condition:
            x = torch.cat((features, self.label_embedding(labels)), -1)
        else:
            x = features
        x = self.net(x)
        validity = self.output_head(x)
        if self.use_class_head:
            classes = self.class_head(x)
            return validity, classes
        return validity


class GeneratorBig(nn.Module):
    def __init__(self, num_features, num_labels, latent_dim=100):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(num_labels, num_labels)

        def block(in_dim, out_dim, normalize=True):
            layers = [nn.Linear(in_dim, out_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + num_labels, 512, normalize=False),
            *block(512, 256),
            *block(256, 256),
            nn.Linear(256, num_features),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_embedding(labels)), -1)
        out = self.model(gen_input)
        return out


class DiscriminatorBig(nn.Module):
    def __init__(self, num_features, num_labels, use_class_head=False, use_label_condition=True):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.use_class_head = use_class_head
        self.use_label_condition = use_label_condition
        if self.use_label_condition:
            self.label_embedding = nn.Embedding(num_labels, num_labels)

        input_size = self.num_features
        if self.use_label_condition:
            input_size += self.num_labels

        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )
        self.output_head = nn.Linear(256, 1)
        if self.use_class_head:
            self.class_head = nn.Linear(256, num_labels)

    def forward(self, features, labels=None):
        if labels is not None and self.use_label_condition:
            x = torch.cat((features, self.label_embedding(labels)), -1)
        else:
            x = features
        x = self.net(x)
        validity = self.output_head(x)
        if self.use_class_head:
            classes = self.class_head(x)
            return validity, classes
        return validity



