from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.utils.helpers import save_json


def evaluate_classifier(model, X_train, y_train, X_test, y_test) -> dict:
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    train_preds = (train_probs >= 0.5).astype(int)
    test_preds = (test_probs >= 0.5).astype(int)

    results = {
        "train_roc_auc": float(roc_auc_score(y_train, train_probs)),
        "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
        "train_pr_auc": float(average_precision_score(y_train, train_probs)),
        "test_pr_auc": float(average_precision_score(y_test, test_probs)),
        "test_confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        "test_classification_report": classification_report(
            y_test, test_preds, output_dict=True
        ),
    }
    return results


def save_metrics(results: dict, output_path: Path) -> None:
    save_json(results, output_path)


def plot_roc_curve(model, X_test, y_test, output_path: Path, title: str) -> None:
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pr_curve(model, X_test, y_test, output_path: Path, title: str) -> None:
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()