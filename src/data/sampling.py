from collections import Counter

import torch


def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)
