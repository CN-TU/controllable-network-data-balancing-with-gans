import argparse
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scikitplot.metrics import plot_confusion_matrix

from cic_ids_17_dataset import CIC17Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./models/classifier")
    args = parser.parse_args()
    print(f"Args: {args}")

    np.random.seed(args.seed)
    time = datetime.now().strftime("%d-%m-%Y_%Hh%Mm")
    save_dir = Path(args.save_dir) / time
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    train_dataset = CIC17Dataset("./data/cic-ids-2017_splits_with_benign/seed_0/X_train.pt",
                                 "./data/cic-ids-2017_splits_with_benign/seed_0/y_train.pt")
    test_dataset = CIC17Dataset("./data/cic-ids-2017_splits_with_benign/seed_0/X_test.pt",
                                "./data/cic-ids-2017_splits_with_benign/seed_0/y_test.pt")
    label_encoder = joblib.load("./data/cic-ids-2017_splits_with_benign/seed_0/label_encoder.gz")
    column_names = torch.load("./data/cic-ids-2017_splits_with_benign/seed_0/column_names.pt")
    idx_to_col = {i: col for i, col in enumerate(column_names)}

    label_distribution = {0: 0.01, 1: 0.23, 2: 0.02, 3: 0.38, 4: 0.01, 5: 0.01, 6: 0.015,
                          7: 0.01, 8: 0.01, 9: 0.265, 10: 0.01, 11: 0.01, 12: 0.01, 13: 0.01}

    print("\nMaking Classifier...")
    classifier = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed,
                                        n_jobs=-1, verbose=2, class_weight=None)

    print("\nStarting training...")
    classifier.fit(train_dataset.X, train_dataset.y)

    print(f"\nSaving classifier to {save_dir}")
    joblib.dump(classifier, save_dir / "classifier.gz")

    print("\nEvaluating...")

    # print(\n"--------------- Train statistics ---------------")
    # y_preds_train = classifier.predict(train_dataset.X)
    # y_preds_train = label_encoder.inverse_transform(y_preds_train)
    # y_true_train = label_encoder.inverse_transform(train_dataset.y)
    # print(classification_report(y_true_train, y_preds_train))
    # plot_confusion_matrix(y_true_train, y_preds_train, figsize=(12, 10), x_tick_rotation=90)
    # plt.show()

    print("\n--------------- Test statistics ---------------")
    y_preds_test = classifier.predict(test_dataset.X)
    y_preds_test = label_encoder.inverse_transform(y_preds_test)
    y_true_test = label_encoder.inverse_transform(test_dataset.y)
    print(classification_report(y_true_test, y_preds_test))
    report = classification_report(y_true_test, y_preds_test, output_dict=True)
    df_report = pd.DataFrame.from_dict(report).to_csv(save_dir / "classification_report.csv")
    # make confusion matrix
    fig, ax = plt.subplots()
    plot_confusion_matrix(y_true_test, y_preds_test, figsize=(12, 10), x_tick_rotation=90, ax=ax)
    fig.savefig(save_dir / "confusion_matrix.png")
    plt.show()
    # make normalized confusion matrix
    fig, ax = plt.subplots()
    plot_confusion_matrix(y_true_test, y_preds_test, figsize=(12, 10), x_tick_rotation=90, normalize=True, ax=ax)
    fig.savefig(save_dir / "confusion_matrix_normalized.png")
    plt.show()


    print("\n--------------- Feature ranking ---------------")
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    ranking = []
    for i in range(len(indices)):
        ranking.append([i + 1, idx_to_col[indices[i]], round(importances[indices[i]], 4)])
    ranking = pd.DataFrame(ranking, columns=["Rank", "Feature", "Importance score"]).set_index("Rank")
    ranking.to_csv(save_dir / "feature_ranking.csv")
    print(ranking)
