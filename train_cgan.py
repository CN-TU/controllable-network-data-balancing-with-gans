"""
Train loop + logging
    - best way to evaluate?
    - can implement some evaluation script, that checks how far
        away the generated features are from ground truth
    - not clear to me how feed the flows into SNORT.

TODO:
    - scale features + inverse-transform
        --> "IDSGAN: Generative Adversarial Networks for Attack Generation against Intrusion Detection"
        --> they do Min-Max scaling
    - class distribution in batches:
        - currently not proportional to the actual class distribution
        - problematic since highly skewed distribution
"""
import argparse
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from cgan import Generator, Discriminator
from cic_ids_17_dataset import CIC17Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--num_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--num_features", type=int, default=79, help="features")
    parser.add_argument("--num_labels", type=int, default=14, help="number of classes for dataset")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    args = parser.parse_args()
    print(f"Args: {args}")

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    print("Loading dataset...")
    train_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/X_train.pt",
                                 "./data/cic-ids-2017_splits/seed_0/y_train.pt")
    test_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/X_test.pt",
                                "./data/cic-ids-2017_splits/seed_0/y_test.pt")
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size)

    print("Making GAN...")
    G = Generator(args.num_features, args.num_labels, latent_dim=args.latent_dim).to(device)
    D = Discriminator(args.num_features, args.num_labels).to(device)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    print("Starting train loop...")
    for epoch in range(args.n_epochs):
        with tqdm(total=len(train_loader), desc="Train loop") as pbar:
            for i, (features, labels) in enumerate(train_loader):

                # make ground truths
                batch_size = features.shape[0]
                real = torch.FloatTensor(batch_size, 1).fill_(1.0)
                fake = torch.FloatTensor(batch_size, 1).fill_(0.0)

                # train generator
                G_optimizer.zero_grad()
                noise = torch.FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim)))
                noise_labels = torch.LongTensor(np.random.randint(0, args.num_labels, batch_size))
                generated_features = G(noise, noise_labels)
                validity = D(generated_features, noise_labels)
                G_loss = criterion(validity, real)
                G_loss.backward()
                G_optimizer.step()

                # train discriminator
                D_optimizer.zero_grad()
                validity_real = D(features.float(), labels)
                validity_fake = D(generated_features.detach(), noise_labels)
                D_loss_real = criterion(validity_real, real)
                D_loss_fake = criterion(validity_fake, fake)
                D_loss = (D_loss_real + D_loss_fake) / 2
                D_loss.backward()
                D_optimizer.step()

                if i % 1 == 0:
                    print(f"\nEpoch {epoch}/{args.n_epochs} | Batch {i}/{len(train_loader)} |  "
                          f"D_loss: {D_loss.item():.5f}  | G_loss: {G_loss.item():.5f}")

                # pbar.update(1)