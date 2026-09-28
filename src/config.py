IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VAL_SPLIT = 0.15
RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 3
LEARNING_RATE = 0.001

DATASET_NAME = "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
DATASET_CONFIG = "default"

MODEL_OUTPUT_PATH = "outputs/plant_disease_model.pth"
CLASS_NAMES_OUTPUT_PATH = "outputs/class_names.json"