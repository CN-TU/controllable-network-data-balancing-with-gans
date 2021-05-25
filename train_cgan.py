"""
Notes:
    1. class distribution in batches:
        - now proportional to the actual class distribution
    2. Evaluation, how do we assess quality of the generated flows?
        - we cannot feed flows into SNORT --> therefore we use external ML model to classify (random forest)
        - visual inspection of feature distributions in TensorBoard, similar to: https://www.youtube.com/watch?v=ZFmnchOJseM
        - euclidean distances between mean real flows per class and generated ones
        - statistical tests
"""
import json
import argparse
import joblib
import torch
from torch.utils import data

from gans import CGAN, CWGAN
from networks import Generator, Discriminator
from cic_ids_17_dataset import CIC17Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_cpu", type=int, default=-1)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--num_features", type=int, default=79)
    parser.add_argument("--num_labels", type=int, default=14)
    parser.add_argument("--G_train_freq", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--clip_val", type=float, default=0.1, help="Gradient clipping. Only used in WGAN.")
    parser.add_argument("--log_freq", type=int, default=100, help="Write logs to commandline every n steps.")
    parser.add_argument("--log_tensorboard_freq", type=int,
                        default=100, help="Write logs to TensorBoard every n steps.")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluate model every n epochs.")
    parser.add_argument("--save_freq", type=int, default=1, help="Save model every n epochs.")
    parser.add_argument("--use_wgan", action="store_true", help="Indicates if the WGAN architecture should be used.")
    parser.add_argument("--use_gp", action="store_true", help="Indicates if gradient should be used in WGAN.")
    parser.add_argument("--use_label_weights", action="store_true",
                        help="Indicates if label weights should be used in generation procedure.")
    parser.add_argument("--run_significance_tests", action="store_true")
    parser.add_argument("--compute_euclidean_distances", action="store_true")
    parser.add_argument("--log_dir", type=str, default="./tensorboard", help="TensorBoard log dir.")
    parser.add_argument("--model_save_dir", type=str, default="./models")
    parser.add_argument("--data_path", type=str, default="./data/cic-ids-2017_splits/seed_0/")
    parser.add_argument("--classifier_path", type=str, default="./models/classifier/20-05-2021_12h01m/classifier.gz")
    args = parser.parse_args()
    print(f"Args: {args}")

    log_dir = args.log_dir + "/cwgan" if args.use_wgan else args.log_dir + "/cgan"
    model_save_dir = args.model_save_dir + "/cwgan" if args.use_wgan else args.model_save_dir + "/cgan"
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    print("Loading dataset...")
    train_dataset = CIC17Dataset(args.data_path + "train_dataset_scaled.pt", is_scaled=True)
    test_dataset = CIC17Dataset(args.data_path + "test_dataset_scaled.pt", is_scaled=True)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size)

    classifier = joblib.load(args.classifier_path)
    cols_to_plot = ["Source Port", "Destination Port", "Flow Duration", "Flow Packets/s", "Fwd Packets/s",
                    "Bwd Packets/s", "Packet Length Mean", "Average Packet Size", "Idle Mean"]
    col_to_idx = {col: i for i, col in enumerate(train_dataset.column_names)}
    label_distribution = {0: 0.01, 1: 0.23, 2: 0.02, 3: 0.38, 4: 0.01, 5: 0.01, 6: 0.015,
                          7: 0.01, 8: 0.01, 9: 0.265, 10: 0.01, 11: 0.01, 12: 0.01, 13: 0.01}
    label_weights = list(label_distribution.values())

    print("Making GAN...")
    G = Generator(args.num_features, args.num_labels, latent_dim=args.latent_dim).to(device)
    D = Discriminator(args.num_features, args.num_labels).to(device)
    if args.use_gp:
        G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    elif args.use_wgan:
        G_optimizer = torch.optim.RMSprop(G.parameters(), lr=args.lr)
        D_optimizer = torch.optim.RMSprop(D.parameters(), lr=args.lr)
    else:
        G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr)
        D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr)

    if args.use_wgan:
        exp = CWGAN(G, D, G_optimizer, D_optimizer, criterion=None, use_gradient_penalty=args.use_gp,
                    model_save_dir=model_save_dir, log_dir=log_dir, clip_val=args.clip_val)
    else:
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.BCEWithLogitsLoss()
        exp = CGAN(G, D, G_optimizer, D_optimizer, criterion, model_save_dir=model_save_dir, log_dir=log_dir)

    print("Generator:\n", G)
    print("Discriminator:\n", D)
    print("G optimizer:\n", G_optimizer)
    print("D optimizer:\n", D_optimizer)

    print("Saving config params to ", log_dir)
    all_params = {"args": vars(args), "log_dir": log_dir, "model_save_dir": model_save_dir,
                  "G": str(G), "D": str(D), "G_optimizer": str(G), "D_optimizer": str(D)}
    with open(exp.log_dir / "all_params.json", "w") as f:
        json.dump(all_params, f)

    print("Starting train loop...")
    for epoch in range(args.n_epochs):

        if epoch % args.eval_freq == 0:
            exp.evaluate(test_dataset, col_to_idx, cols_to_plot, epoch,
                         label_weights=label_weights, classifier=classifier, label_encoder=test_dataset.label_encoder,
                         scaler=test_dataset.scaler, class_means=test_dataset.class_means,
                         run_significance_tests=args.run_significance_tests,
                         compute_euclidean_distances=args.compute_euclidean_distances)

        exp.train_epoch(train_loader, epoch, log_freq=args.log_freq,
                        log_tensorboard_freq=args.log_tensorboard_freq,
                        label_weights=label_weights if args.use_label_weights else None,
                        G_train_freq=args.G_train_freq)

        if epoch % args.save_freq == 0:
            exp.save_model(epoch)

    exp.logger.add_all_custom_scalars()
