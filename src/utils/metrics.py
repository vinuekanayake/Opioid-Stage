import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def compute_classification_metrics(label_ids, predictions):
    """Compute accuracy and F1 for classification."""
    return {
        "accuracy": accuracy_score(label_ids, predictions),
        "f1_macro": f1_score(label_ids, predictions, average="macro"),
    }

def print_classification_report(name, y_true, y_pred, label_names):
    """Print detailed classification report."""
    print(f"\n=== Classification Report: {name} ===")
    print(classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        digits=4
    ))
