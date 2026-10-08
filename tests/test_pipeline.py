"""Unit tests for pipeline building blocks (no dataset required → CI-safe)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep
from src.monitoring import psi


def test_psi_zero_for_identical():
    x = np.random.RandomState(0).normal(size=2000)
    assert psi(x, x) < 1e-6


def test_psi_positive_for_shifted():
    rs = np.random.RandomState(0)
    ref = rs.normal(0, 1, 2000)
    cur = rs.normal(2, 1, 2000)          # clear shift
    assert psi(ref, cur) > 0.2


def test_image_features_keys():
    img = Image.new("L", (300, 300), color=100)
    f = data_prep.image_features(img)
    assert set(f) == set(config.DRIFT_FEATURES)
    assert all(isinstance(v, float) for v in f.values())


def test_transforms_output_shape():
    tf = data_prep.get_transforms(train=False)
    x = tf(Image.new("L", (300, 300), color=128))
    assert tuple(x.shape) == (3, config.IMG_SIZE, config.IMG_SIZE)


def test_class_mapping_defect_positive():
    assert config.CLASS_TO_IDX["def_front"] == 1
    assert config.POSITIVE_IDX == 1
