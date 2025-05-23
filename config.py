import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Data/train"
VAL_DIR = "Data/val"
BATCH_SIZE = 1/.,xnbvc
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 2
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_P = "genp.pth.tar"
CHECKPOINT_GEN_M = "genm.pth.tar"
CHECKPOINT_CRITIC_P = "criticp.pth.tar"
CHECKPOINT_CRITIC_M = "criticm.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)