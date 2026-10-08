"""Stage 3.1: imbalance-aware test-set evaluation and failure analysis."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

import config
from src import data_prep
from src.model import build_model


class ManifestDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(label)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    device = next(model.parameters()).device
    y_true, y_pred, y_prob_defect = [], [], []

    for images, labels in loader:
        logits = model(images.to(device))
        probabilities = torch.softmax(logits, dim=1)

        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().astype(int).tolist())
        y_prob_defect.extend(
            probabilities[:, config.POSITIVE_IDX].cpu().numpy().astype(float).tolist()
        )

    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob_defect)


def compute_metrics(y_true, y_pred, y_prob_defect):
    positive = config.POSITIVE_IDX
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    try:
        roc_auc = float(roc_auc_score(y_true == positive, y_prob_defect))
    except ValueError:
        roc_auc = None

    is_defect = y_true == positive
    predicted_defect = y_pred == positive

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_defect": float(
            precision_score(y_true, y_pred, pos_label=positive, zero_division=0)
        ),
        "recall_defect": float(
            recall_score(y_true, y_pred, pos_label=positive, zero_division=0)
        ),
        "f1_defect": float(f1_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        "roc_auc": roc_auc,
        # Accuracy of a model that calls every casting defective.
        "baseline_accuracy_all_defect": float(is_defect.mean()) if len(y_true) else None,
        "true_positives": int(np.sum(is_defect & predicted_defect)),
        "false_negatives": int(np.sum(is_defect & ~predicted_defect)),
        "false_positives": int(np.sum(~is_defect & predicted_defect)),
        "true_negatives": int(np.sum(~is_defect & ~predicted_defect)),
        "positive_class": config.POSITIVE_CLASS,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(config.NUM_CLASSES))
        ).tolist(),
        "confusion_matrix_labels": [
            config.IDX_TO_CLASS[i] for i in range(config.NUM_CLASSES)
        ],
        "selection_metric": "validation defect F1; test set used for final evaluation",
    }


def plot_eval(y_true, y_pred, y_prob_defect, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
        display_labels=[config.IDX_TO_CLASS[i] for i in range(config.NUM_CLASSES)],
        cmap="Blues",
        values_format="d",
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title("Test confusion matrix")

    if len(np.unique(y_true)) == 2:
        RocCurveDisplay.from_predictions(
            (np.asarray(y_true) == config.POSITIVE_IDX).astype(int),
            y_prob_defect,
            name="Defect",
            ax=axes[1],
        )
        axes[1].plot([0, 1], [0, 1], "--", color="gray")
        axes[1].set_title("Test ROC curve")
    else:
        axes[1].text(0.5, 0.5, "ROC unavailable: test set has one class",
                     ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def failure_cases(records, y_true, y_pred, y_prob_defect, limit=20):
    """Misclassified samples, most confident mistakes first (limit=None: all)."""
    errors = []
    for i in np.flatnonzero(np.asarray(y_true) != np.asarray(y_pred)):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])
        errors.append({
            "path": str(records[i][0]),
            "true_label": config.IDX_TO_CLASS[true_label],
            "predicted_label": config.IDX_TO_CLASS[pred_label],
            "prob_defect": float(y_prob_defect[i]),
            "error_type": (
                "false_negative" if true_label == config.POSITIVE_IDX else "false_positive"
            ),
        })

    errors.sort(key=lambda row: abs(row["prob_defect"] - 0.5), reverse=True)
    return errors if limit is None else errors[:limit]


def evaluate_test_set(version="v1"):
    root = data_prep.find_data_root()
    records = data_prep.load_split(version, "test", root)

    if not records:
        raise ValueError("The test manifest is empty.")
    if not Path(config.MODEL_PATH).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {config.MODEL_PATH}")

    device_name = config.DEVICE
    if device_name != "cpu" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = build_model(pretrained=False).to(device)
    state = torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)

    loader = DataLoader(
        ManifestDataset(records, data_prep.get_transforms(train=False)),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    y_true, y_pred, y_prob_defect = predict(model, loader)
    metrics = compute_metrics(y_true, y_pred, y_prob_defect)
    metrics["dataset_version"] = version
    metrics["test_images"] = len(records)

    # Link the results to the training run that produced the checkpoint.
    meta_path = Path(config.MODEL_META_PATH)
    if meta_path.is_file():
        training_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics["mlflow_run_id"] = training_meta.get("mlflow_run_id")

    cases = failure_cases(records, y_true, y_pred, y_prob_defect, limit=None)
    metrics["misclassified"] = len(cases)

    Path(config.ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.METRICS_PATH).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_eval(y_true, y_pred, y_prob_defect, Path(config.ARTIFACT_DIR) / "model_eval.png")

    failure_path = Path(config.ARTIFACT_DIR) / "failure_cases.json"
    failure_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    return metrics, cases


if __name__ == "__main__":
    metrics, cases = evaluate_test_set()
    print(json.dumps(metrics, indent=2))
    print(
        f"Misclassified: {len(cases)} "
        f"(false negatives: {metrics['false_negatives']}, "
        f"false positives: {metrics['false_positives']})"
    )
    print("Metrics:", config.METRICS_PATH)
    print("Plot:", Path(config.ARTIFACT_DIR) / "model_eval.png")
