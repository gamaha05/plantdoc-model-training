# PlantDoc — Model Training

Training pipeline for a ResNet18-based plant disease classifier — the model that powers [plantdoc-bot](https://github.com/gamaha05/plantdoc-bot), a Telegram bot that diagnoses plant diseases from a photo.

## Overview

The model classifies leaf photos into 38 crop/disease categories using transfer learning on a frozen, ImageNet-pretrained ResNet18. This repo covers only the training pipeline and evaluation — inference and the bot itself live in a separate repository (see above).

## Dataset

[PlantVillage-derived dataset](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset) via HuggingFace `datasets`:

- 43,456 training images / 10,849 test images
- 38 classes (14 crops, healthy + disease variants)
- **Significantly imbalanced**: class sizes range from 122 to 4,406 examples (36.1x ratio between smallest and largest class)

## Architecture

- **Backbone**: ResNet18, ImageNet-pretrained weights, fully frozen
- **Head**: single `Linear` layer replacing the final FC layer, sized for 38 classes — only this layer is trained
- **Augmentation**: random horizontal flip + rotation on train split only (the dataset's lab photos are unusually clean, so augmentation helps generalization instead of overfitting to that cleanliness)
- **Early stopping**: training stops when validation accuracy doesn't improve for 3 consecutive epochs, and the best-epoch weights (not the last epoch's) are what get saved

## Results

| Metric                   | Value           |
| ------------------------ | --------------- |
| Best validation accuracy | 95.4% (epoch 8) |
| Test accuracy            | 95%             |
| Test macro-F1            | 0.937           |

**Why macro-F1, not just accuracy:** given the 36x class imbalance, accuracy alone can be misleading — a model can score high overall while quietly failing on rare classes. Macro-F1 averages F1 per class equally, regardless of class size, so it surfaces that weakness if it exists.

### Per-class analysis

Full breakdown via `src/diagnose.py`, which runs `classification_report` and a confusion analysis on the trained model. The weakest classes:

| Class                         | F1                 | Likely cause                                                                                                                                                |
| ----------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Tomato___Early_blight`       | 0.75               | Visual overlap with `Late_blight` and `Septoria_leaf_spot` — not a rare class (800 train examples), so this looks like class _similarity_, not class _size_ |
| `Corn___Cercospora_leaf_spot` | 0.76               | Visually similar lesion pattern to `Corn___Northern_Leaf_Blight`                                                                                            |
| `Potato___healthy`            | 0.78 (recall 0.67) | Smallest class in the dataset (122 train examples) — a textbook case of insufficient data for that class                                                    |

This distinction matters: the macro-F1/accuracy gap here (93.7% vs 95%) is real but modest — the model doesn't collapse on rare classes wholesale. The weakest classes split into two different problems (data scarcity vs. visual ambiguity between diseases), which call for different fixes rather than one blanket solution.

## Limitations & Future Work

- **Class imbalance mitigation** (weighted loss / weighted sampling) was diagnosed and prototyped but not fully validated against a before/after comparison — a natural next step.
- **Visually similar disease pairs** (e.g. tomato blight variants, corn leaf diseases) likely need more training data or a different modeling approach than reweighting alone.
- No hyperparameter search was performed — learning rate and epoch budget were chosen pragmatically, not tuned.

## Project Structure

```
src/
├── config.py           # hyperparameters and paths
├── model.py             # build_model() — ResNet18 + custom head
├── train.py             # training loop with early stopping
├── evaluate.py           # accuracy + macro-F1/classification_report
├── diagnose.py           # standalone script: class distribution + per-class performance
└── data/
    ├── loader.py          # dataset loading, splits, transforms
    └── sampling.py        # class weight computation (prototyped, not currently wired into training)
tests/
├── test_model.py
└── test_sampling.py
```

## Setup

```bash
git clone https://github.com/gamaha05/plantdoc-model-training.git
cd plantdoc-model-training
pip install -r requirements.txt
```

## Usage

Train:

```bash
python -m src.train
```

Diagnose class imbalance / per-class performance on an already-trained model:

```bash
python -m src.diagnose
```

## Tests

```bash
pytest -v
```

## Model Weights

Trained weights (`plant_disease_model.pth`) and class names (`class_names.json`) are published as [GitHub Releases](../../releases) rather than committed to the repo.
