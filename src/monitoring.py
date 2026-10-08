"""monitoring.py — Stage 4: statistical + embedding drift + confidence monitoring.

Implement the three drift signals against the clean reference baseline:
  1. statistical drift  — Evidently DataDriftPreset + PSI on image features
  2. embedding drift    — PSI on ResNet-embedding distance-to-centroid distribution
  3. confidence         — mean predicted confidence reference vs current
Use a corrupted copy of clean images as the simulated "current" production batch.
Outputs drift_report.html + drift_summary.json.   Run: python -m src.monitoring

Embedding drift (TODO 4) — see the conceptual walkthrough in
Operations_Monitoring_and_Evidence.ipynb (Stage 4.3):
  1. feature extraction  — penultimate 512-dim ResNet embedding (model.EmbeddingExtractor)
  2. embedding generation — embeddings for reference + current batches
  3. feature-space compare — reduce each to distance-to-reference-centroid (one distribution per batch)
  4. drift calculation   — PSI between the two distance distributions (> ~0.10 => drifted)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep
from src.model import load_model, EmbeddingExtractor


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions (quantile bins).

    Bin edges are the quantiles of the reference, so each bin holds about the same
    share of reference data. PSI = sum((cur% - ref%) * ln(cur% / ref%)).
    Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant drift.
    """
    import numpy as np

    reference = np.asarray(reference, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        raise ValueError("psi() needs non-empty reference and current samples.")

    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:  # constant reference: compare "same value" vs "different"
        same = np.isclose(current, reference[0]).mean()
        ref_share, cur_share = np.array([1.0, 0.0]), np.array([same, 1.0 - same])
    else:
        edges[0], edges[-1] = -np.inf, np.inf  # catch values outside the reference range
        ref_share = np.histogram(reference, bins=edges)[0] / reference.size
        cur_share = np.histogram(current, bins=edges)[0] / current.size

    eps = 1e-6  # avoids log(0) for empty bins
    ref_share = np.clip(ref_share, eps, None)
    cur_share = np.clip(cur_share, eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def corrupt(img: Image.Image) -> Image.Image:
    # TODO 4: simulate camera/lighting drift (brightness/blur/rotate/noise via config.DRIFT_SIM).
    raise NotImplementedError


def run() -> dict:
    # TODO 4: build reference (clean) + current (corrupted) batches → features, embeddings,
    #         mean confidence. Run Evidently DataDriftPreset + PSI; embedding PSI on
    #         distance-to-centroid; confidence drop. Write drift_summary.json + drift_report.html
    #         and set retrain_recommended from the configured thresholds.
    raise NotImplementedError


if __name__ == "__main__":
    run()
