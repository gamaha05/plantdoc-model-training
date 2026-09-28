"""
Diagnostic script: run this BEFORE applying any class-imbalance fixes,
against the currently trained model weights, to establish a baseline.
Compare its output to the numbers you get after retraining with fixes.
"""

import json
from collections import Counter

import torch
from datasets import load_dataset

from src import config
from src.data.loader import load_data
from src.evaluate import evaluate_detailed
from src.model import build_model


def show_class_distribution():
    """How imbalanced is the raw dataset, independent of any model."""
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    class_names = dataset["train"].features["label"].names
    counts = Counter(dataset["train"]["label"])

    print("\n=== Train set class distribution ===")
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1])
    for label_idx, count in sorted_counts:
        print(f"{class_names[label_idx]:<45} {count:>6}")

    smallest = sorted_counts[0]
    largest = sorted_counts[-1]
    print(f"\nSmallest class: {class_names[smallest[0]]} ({smallest[1]} examples)")
    print(f"Largest class:  {class_names[largest[0]]} ({largest[1]} examples)")
    print(f"Imbalance ratio: {largest[1] / smallest[1]:.1f}x")


def show_model_performance():
    """Per-class performance of the currently saved model on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config.CLASS_NAMES_OUTPUT_PATH) as f:
        class_names = json.load(f)

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(config.MODEL_OUTPUT_PATH, weights_only=True, map_location=device))

    _, _, test_loader, _ = load_data()

    macro_f1, report = evaluate_detailed(model, test_loader, device, class_names)

    print(f"\n=== Model performance on test set ===")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)


if __name__ == "__main__":
    show_class_distribution()
    show_model_performance()