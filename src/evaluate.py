import torch
from sklearn.metrics import classification_report, f1_score


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Returns accuracy as a percentage."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total


@torch.no_grad()
def evaluate_detailed(model, loader, device, class_names):
    """
    Per-class breakdown via classification_report, plus macro-F1 — the
    metric that actually reflects performance on rare classes, unlike
    plain accuracy which gets dominated by whichever classes have the
    most examples.

    Returns: (macro_f1, report_str)
    """
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    return macro_f1, report