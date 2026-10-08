"""
config.py — single source of truth for the Casting Defect Detection DL-MLOps pipeline.

A CNN transfer-learning (ResNet18) visual-inspection pipeline that classifies submersible
pump impeller castings as OK vs Defective, with MLflow tracking/registry, FastAPI/Docker
serving, GitHub Actions CI, statistical + embedding drift detection, retraining and
rollback. Everything tunable lives here so the rest of the codebase stays declarative.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (relative to this file → the folder is fully portable)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"                 # casting_data/{train,test}/{ok_front,def_front}
ARTIFACT_DIR = BASE_DIR / "artifacts"        # models, reports, logs, plots, metadata
SPLIT_DIR = ARTIFACT_DIR / "splits"          # versioned dataset snapshots (file lists)
MLRUNS_DIR = BASE_DIR / "mlruns"             # MLflow local tracking store
for _d in (DATA_DIR, ARTIFACT_DIR, SPLIT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Classes  (binary visual inspection; positive class = DEFECT)
# --------------------------------------------------------------------------- #
# Explicit mapping so the positive (defect) class is index 1 regardless of the
# alphabetical ImageFolder default — recall on defects is what matters for QC.
CLASS_TO_IDX = {"ok_front": 0, "def_front": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
CLASSES = ["ok_front", "def_front"]
POSITIVE_CLASS = "def_front"                  # the class we most care about catching
POSITIVE_IDX = CLASS_TO_IDX[POSITIVE_CLASS]   # 1
NUM_CLASSES = len(CLASSES)

# --------------------------------------------------------------------------- #
# Image preprocessing  (ImageNet-pretrained backbone expects 3×224×224, normalised)
# --------------------------------------------------------------------------- #
IMG_SIZE = 224
# Source images are grayscale → replicated to 3 channels for the pretrained backbone.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VAL_SPLIT = 0.15                              # carved from the train folder, stratified
RANDOM_SEED = 42

# Augmentation (train only) — mimic real inspection-line variation
AUG = {
    "rotation_degrees": 15,
    "translate": 0.05,
    "brightness": 0.2,
    "contrast": 0.2,
    "hflip_p": 0.5,
}

# --------------------------------------------------------------------------- #
# Model / training
# --------------------------------------------------------------------------- #
BACKBONE = "resnet18"                         # torchvision pretrained
FREEZE_BACKBONE = True                        # transfer learning: train the head only
EMBEDDING_DIM = 512                           # resnet18 penultimate feature size
EPOCHS = int(os.getenv("EPOCHS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
# Cap training images for a fast CPU run (stratified subsample). Casting defect detection
# saturates well above 95% with a few thousand images. Set MAX_TRAIN_IMAGES=0 for the full
# ~6,600-image train set. Validation and TEST always use their full splits (honest metrics).
MAX_TRAIN_IMAGES = int(os.getenv("MAX_TRAIN_IMAGES", "2500"))
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))   # 0 = safe/deterministic on CPU/macOS
DEVICE = os.getenv("DEVICE", "cpu")
EARLY_STOP_PATIENCE = 3                        # epochs without val-F1 improvement

# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{MLRUNS_DIR}")
MLFLOW_EXPERIMENT = "casting_defect_detection"
REGISTERED_MODEL = "casting_defect_classifier"
PRODUCTION_ALIAS = "production"               # MLflow registry alias for the live model

# Saved artefacts (also logged to MLflow)
MODEL_PATH = ARTIFACT_DIR / "model.pt"             # current production weights
MODEL_META_PATH = ARTIFACT_DIR / "model_meta.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
PREDICTIONS_LOG = ARTIFACT_DIR / "predictions.log"
REFERENCE_EMBED = ARTIFACT_DIR / "reference_embeddings.npz"   # for embedding drift

# --------------------------------------------------------------------------- #
# Monitoring & drift detection
# --------------------------------------------------------------------------- #
# Statistical drift: per-image summary features (brightness, contrast, edge density,
# sharpness) compared reference vs current with Evidently + PSI.
DRIFT_FEATURES = ["brightness", "contrast", "edge_density", "sharpness", "mean_intensity"]
PSI_THRESHOLD = 0.20                          # >0.2 = significant feature drift
EMBEDDING_DRIFT_THRESHOLD = 0.10              # PSI on the embedding-distance distribution
CONFIDENCE_DROP_THRESHOLD = 0.10              # mean-confidence drop vs reference → alert
DRIFT_SHARE_THRESHOLD = 0.40                  # share of drifted features that triggers retrain

# Simulated "production" corruption to demonstrate drift (camera/lighting degradation).
DRIFT_SIM = {"brightness": 0.6, "blur_radius": 1.5, "noise_std": 12, "rotate": 8}

# --------------------------------------------------------------------------- #
# Retraining / rollback governance
# --------------------------------------------------------------------------- #
RETRAIN_MIN_F1_GAIN = 0.0     # new model must be >= production F1 (minus epsilon) to promote
PROMOTE_EPSILON = 0.01        # tolerance band for promotion decision
ROLLBACK_ON_REGRESSION = True
