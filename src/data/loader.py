import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from src import config


def load_data():
    """
    Downloads the dataset, splits off a validation set from train, and
    applies transforms (with light augmentation on train only — the
    dataset's lab photos are unusually clean, so augmentation helps the
    model generalize instead of overfitting to that cleanliness).

    Returns: (train_loader, val_loader, test_loader, class_names)
    """
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    class_names = dataset["train"].features["label"].names

    train_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    split = dataset["train"].train_test_split(test_size=config.VAL_SPLIT, seed=config.RANDOM_SEED)
    train_data, val_data = split["train"], split["test"]
    test_data = dataset["test"]

    def apply_train_transforms(examples):
        examples["pixel_values"] = [train_transforms(img.convert("RGB")) for img in examples["image"]]
        return examples

    def apply_eval_transforms(examples):
        examples["pixel_values"] = [eval_transforms(img.convert("RGB")) for img in examples["image"]]
        return examples

    train_data.set_transform(apply_train_transforms)
    val_data.set_transform(apply_eval_transforms)
    test_data.set_transform(apply_eval_transforms)

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return pixel_values, labels

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, class_names