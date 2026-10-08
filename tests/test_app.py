"""API tests — robust whether or not a trained model/dataset is present (CI-safe)."""
import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
import app as app_module
import config

client = TestClient(app_module.app)


def _fake_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (300, 300), color=128).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["classes"] == config.CLASS_TO_IDX
    assert body["positive_class"] == "def_front"


def test_predict_contract():
    r = client.post("/predict", files={"file": ("x.jpg", _fake_image_bytes(), "image/jpeg")})
    if app_module._state["model"] is None:
        assert r.status_code == 503            # no model yet → graceful 503
    else:
        assert r.status_code == 200
        b = r.json()
        assert b["label"] in config.CLASSES
        assert 0.0 <= b["prob_defect"] <= 1.0
        assert isinstance(b["is_defective"], bool)


def test_predict_rejects_non_image():
    if app_module._state["model"] is None:
        return                                  # 503 path covered above
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
