import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from scikitplot.metrics import plot_confusion_matrix

from cic_ids_17_dataset import CIC17Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./models/classifier")
    parser.add_argument("--data_path", type=str,
                        default="./data/cic-ids-2017_splits_with_benign/seed_0/")
    parser.add_argument("--remove_benign", action="store_true")
    parser.add_argument("--cross_validate", action="store_true")
    args = parser.parse_args()
    print(f"Args: {args}")

    np.random.seed(args.seed)
    time = datetime.now().strftime("%d-%m-%Y_%Hh%Mm")
    save_dir = Path(args.save_dir) / time
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    train_dataset = CIC17Dataset(args.data_path + "train_dataset.pt")
    test_dataset = CIC17Dataset(args.data_path + "test_dataset.pt")
    idx_to_col = {i: col for i, col in enumerate(train_dataset.column_names)}

    if args.remove_benign:
        benign_label = int(np.where(train_dataset.class_names == "zBENIGN")[0])
        idx_malicious_train = np.where(train_dataset.y != benign_label)[0]
        train_dataset.X = train_dataset.X[idx_malicious_train]
        train_dataset.y = train_dataset.y[idx_malicious_train]

        idx_malicious_test = np.where(test_dataset.y != benign_label)[0]
        test_dataset.X = test_dataset.X[idx_malicious_test]
        test_dataset.y = test_dataset.y[idx_malicious_test]

    # label_distribution = {0: 0.01, 1: 0.23, 2: 0.02, 3: 0.38, 4: 0.01, 5: 0.01, 6: 0.015,
    #                       7: 0.01, 8: 0.01, 9: 0.265, 10: 0.01, 11: 0.01, 12: 0.01, 13: 0.01}

    print("\nMaking Classifier...")
    classifier = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed,
                                        n_jobs=-1, verbose=1, class_weight=None)
    if args.cross_validate:
        print("\nCross validating...")
        scores = cross_validate(classifier, train_dataset.X, train_dataset.y,
                                scoring=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"], cv=5)
        print("\n--------------- Cross validation statistics ---------------")
        print(f"Mean accuracy: {scores['test_accuracy'].mean()} |"
              f" Mean balanced-accuracy: {scores['test_balanced_accuracy'].mean()} |"
              f" Mean f1-macro: {scores['test_f1_macro'].mean()} |"
              f" Mean f1-weighted: {scores['test_f1_weighted'].mean()}")

    print("\nStarting training...")
    classifier.fit(train_dataset.X, train_dataset.y)

    print(f"\nSaving classifier to {save_dir}")
    joblib.dump(classifier, save_dir / "classifier.gz")

    print("\n--------------- Test statistics ---------------")
    y_preds_test = classifier.predict(test_dataset.X)
    y_preds_test = train_dataset.label_encoder.inverse_transform(y_preds_test)
    y_true_test = train_dataset.label_encoder.inverse_transform(test_dataset.y)
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
