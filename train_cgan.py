"""
Train loop + logging
    - best way to evaluate?
    - can implement some evaluation script, that checks how far
        away the generated features are from ground truth
    - not clear to me how feed the flows into SNORT.

TODO:
    - class distribution in batches:
        - currently not proportional to the actual class distribution
        - problematic since highly skewed distribution
"""
import argparse
import torch
from torch.utils import data

from cgan import Generator, Discriminator, Experiment
from cic_ids_17_dataset import CIC17Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_cpu", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--num_features", type=int, default=79)
    parser.add_argument("--num_labels", type=int, default=14)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--log_freq", type=int, default=100, help="Write logs to commandline every n steps.")
    parser.add_argument("--log_tensorboard_freq", type=int, default=100,
                        help="Write logs to tensorboard every n steps.")
    parser.add_argument("--save_freq", type=int, default=1, help="Save model every n epochs.")
    parser.add_argument("--log_dir", type=str, default="./tensorboard/cgan", help="Tensorboard log dir.")
    parser.add_argument("--model_save_dir", type=str, default="./models/cgan")
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
    exp = Experiment(G, D, G_optimizer, D_optimizer, criterion,
                     model_save_dir=args.model_save_dir, log_dir=args.log_dir)
    print("Generator:\n", G)
    print("Discriminator:\n", D)

    print("Starting train loop...")
    for epoch in range(args.n_epochs):
        exp.train_epoch(train_loader, epoch, log_freq=args.log_freq, log_tensorboard_freq=args.log_tensorboard_freq)
        if epoch % args.save_freq == 0:
            exp.save_model(epoch)
