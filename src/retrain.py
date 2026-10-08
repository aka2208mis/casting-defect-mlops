"""retrain.py — Stage 4: drift-triggered retraining, version compare, rollback.

Implement: read drift_summary.json; if retraining is recommended, measure the production
model on a drifted batch, train a drift-augmented candidate, compare, then PROMOTE the
candidate to @production only if it improves (within PROMOTE_EPSILON) else ROLL BACK.
Record retraining_decision.json + manage MLflow registry versions.   Run: python -m src.retrain
"""
from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep, evaluate
from src.model import build_model, trainable_parameters, load_model, save_model
from src.monitoring import corrupt
from src.train import _subsample, set_seed, class_weights


def main() -> int:
    drift = json.loads((config.ARTIFACT_DIR / "drift_summary.json").read_text())
    if not drift.get("retrain_recommended"):
        # TODO 4: record a 'no_retrain' decision and return.
        raise NotImplementedError
    # TODO 4: evaluate current production on a fully-drifted eval batch (corrupt_frac=1.0).
    # TODO 4: train a drift-augmented candidate (corrupt_frac~0.4, few epochs, capped).
    # TODO 4: compare candidate vs production F1 on the drifted batch.
    # TODO 4: promote candidate→@production iff cand_f1 >= prod_f1 + PROMOTE_EPSILON, else
    #         rollback (keep incumbent). Register the candidate version; move/keep the alias.
    # TODO 4: write retraining_decision.json (scores, action, version history).
    raise NotImplementedError("Implement retraining + version compare + rollback")


if __name__ == "__main__":
    raise SystemExit(main())
