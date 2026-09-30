import copy
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src import config
from src.data.loader import load_data
from src.evaluate import evaluate
from src.model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = load_data()
    print(f"Classes: {len(class_names)}")

    model = build_model(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=config.LEARNING_RATE)

    best_val_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(config.MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch + 1}/{config.MAX_EPOCHS}] - Train Loss: {train_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.PATIENCE:
            print(f"No improvement for {config.PATIENCE} epochs — stopping early at epoch {epoch + 1}.")
            break

    # Returning the weights from the best epoch instead of last epoch because
    # the last one could be worse if the model start relearning
    model.load_state_dict(best_model_state)

    test_accuracy = evaluate(model, test_loader, device)
    print(f"Best Val Accuracy: {best_val_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    os.makedirs(os.path.dirname(config.MODEL_OUTPUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_OUTPUT_PATH)
    with open(config.CLASS_NAMES_OUTPUT_PATH, "w") as f:
        json.dump(class_names, f)

    print(f"Saved model to {config.MODEL_OUTPUT_PATH}")
    print(f"Saved class names to {config.CLASS_NAMES_OUTPUT_PATH}")

    try:
        from google.colab import files
        files.download(config.MODEL_OUTPUT_PATH)
        files.download(config.CLASS_NAMES_OUTPUT_PATH)
    except ImportError:
        pass


if __name__ == "__main__":
    main()