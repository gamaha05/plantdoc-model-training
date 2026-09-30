from src.data.sampling import compute_class_weights


def test_rare_class_gets_higher_weight():
    labels = [0, 0, 0] + [1] * 30

    weights = compute_class_weights(labels, num_classes=2)

    assert weights[0] > weights[1]


def test_balanced_classes_get_equal_weight():
    labels = [0, 0, 1, 1]

    weights = compute_class_weights(labels, num_classes=2)

    assert weights[0] == weights[1]