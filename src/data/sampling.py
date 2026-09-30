from collections import Counter

import torch


<<<<<<< HEAD
def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)
=======
def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    """
    Weights inversely proportional to class frequency — rare classes get
    a bigger weight in the loss, so mistakes on them cost more during
    training, offsetting the fact that common classes dominate by sheer
    example count.
    """
    counts = Counter(dataset["label"])
    total = len(dataset)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)
>>>>>>> 3b08dc33b0be16d9f8592df89979cb411a9ad9d7
