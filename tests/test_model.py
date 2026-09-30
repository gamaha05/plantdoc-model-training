from src.model import build_model


def test_build_model_output_layer_matches_num_classes():
    model = build_model(num_classes=5)
    assert model.fc.out_features == 5


def test_build_model_only_final_layer_is_trainable():
    model = build_model(num_classes=5)

    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]

    assert all(name.startswith("fc.") for name in trainable_params)
    assert len(trainable_params) > 0