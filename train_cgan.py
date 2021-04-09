"""
TODO:
    1. class distribution in batches:
        - currently not proportional to the actual class distribution
        - problematic since highly skewed distribution
    2. evaluation, how do we assess quality of the generated flows?
        - not yet clear how feed the flows into SNORT.
        - compute mean/std for each feature by class
          generate n-samples for each class, compute mean/std
          compare generated distributions against actual distributions
        - also could plot feature distributions for real/fake flows right in TensorBoard (e.g., every nth step)
          similar to: https://www.youtube.com/watch?v=ZFmnchOJseM
          matplotlib plots can be added to TB as images: https://stackoverflow.com/questions/38543850/how-to-display-custom-images-in-tensorboard-e-g-matplotlib-plots

"""
import argparse
import torch
from torch.utils import data

from experiment import CGANExperiment, CWGANExperiment
from cgan import Generator, Discriminator
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
    parser.add_argument("--log_tensorboard_freq", type=int,
                        default=100, help="Write logs to TensorBoard every n steps.")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluate model every n epochs.")
    parser.add_argument("--save_freq", type=int, default=1, help="Save model every n epochs.")
    parser.add_argument("--use_wgan", action="store_true", help="Indicates if the WGAN architecture should be used.")
    parser.add_argument("--use_gp", action="store_true", help="Indicates if gradient should be used in WGAN.")
    parser.add_argument("--log_dir", type=str, default="./tensorboard", help="TensorBoard log dir.")
    parser.add_argument("--model_save_dir", type=str, default="./models")
    args = parser.parse_args()
    print(f"Args: {args}")

    log_dir = args.log_dir + "/cwgan" if args.use_wgan else args.log_dir + "/cgan"
    model_save_dir = args.model_save_dir + "/cwgan" if args.use_wgan else args.model_save_dir + "/cgan"
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    print("Loading dataset...")
    train_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/X_train_scaled.pt",
                                 "./data/cic-ids-2017_splits/seed_0/y_train_scaled.pt")
    test_dataset = CIC17Dataset("./data/cic-ids-2017_splits/seed_0/X_test_scaled.pt",
                                "./data/cic-ids-2017_splits/seed_0/y_test_scaled.pt")
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size)
    column_names = torch.load("./data/cic-ids-2017_splits/seed_0/column_names.pt")
    cols_to_plot = ["Source Port", "Destination Port", "Flow Duration", "Flow Packets/s", "Fwd Packets/s",
                    "Bwd Packets/s", "Packet Length Mean", "Average Packet Size", "Idle Mean"]
    col_to_idx = {col: i for i, col in enumerate(column_names)}
    label_distribution = {0: 0.01, 1: 0.23, 2: 0.02, 3: 0.38, 4: 0.01, 5: 0.01, 6: 0.015,
                          7: 0.01, 8: 0.01, 9: 0.265, 10: 0.01, 11: 0.01, 12: 0.01, 13: 0.01}
    label_weights = list(label_distribution.values())

    print("Making GAN...")
    G = Generator(args.num_features, args.num_labels, latent_dim=args.latent_dim).to(device)
    D = Discriminator(args.num_features, args.num_labels).to(device)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr)

    if args.use_wgan:
        exp = CWGANExperiment(G, D, G_optimizer, D_optimizer, criterion=None,
                              use_gradient_penalty=args.use_gp, model_save_dir=model_save_dir, log_dir=log_dir)
    else:
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.BCEWithLogitsLoss()
        exp = CGANExperiment(G, D, G_optimizer, D_optimizer, criterion,
                             model_save_dir=model_save_dir, log_dir=log_dir)
    print("Generator:\n", G)
    print("Discriminator:\n", D)

    print("Starting train loop...")
    for epoch in range(args.n_epochs):

        if epoch % args.eval_freq == 0:
            exp.evaluate(test_dataset, col_to_idx, cols_to_plot,
                         epoch, label_weights=label_weights)

        exp.train_epoch(train_loader, epoch, log_freq=args.log_freq,
                        log_tensorboard_freq=args.log_tensorboard_freq,
                        label_weights=label_weights)

        if epoch % args.save_freq == 0:
            exp.save_model(epoch)
