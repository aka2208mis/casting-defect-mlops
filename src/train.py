"""Stages 2.3, 2.4 and 3.2: train the head, track it in MLflow, register it."""
import copy
import json
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from src import data_prep
from src.model import (
    EmbeddingExtractor,
    build_model,
    parameter_counts,
    save_model,
    trainable_parameters,
)

PRODUCTION_ALIAS = "production"


class ManifestDataset(Dataset):
    """Turn (image_path, label) records into (tensor, label) pairs."""

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


# Name used by an earlier version of this module.
ImageManifestDataset = ManifestDataset


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int = config.RANDOM_SEED):
    """Seed Python, NumPy and PyTorch (template signature); return the seed."""
    seed_everything(seed)
    return seed


def _label_of(item):
    """Label from a (path, label) record, a dict with "label", or a bare int."""
    if isinstance(item, dict):
        return int(item["label"])
    if isinstance(item, (tuple, list)):
        return int(item[1])
    return int(item)


def class_weights(items, num_classes=None) -> torch.Tensor:
    """Inverse-frequency weights w_c = N / (K * n_c) for CrossEntropyLoss."""
    labels = np.asarray([_label_of(item) for item in items], dtype=int)
    num_classes = num_classes or config.NUM_CLASSES
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"Training labels are missing a class: {counts.tolist()}")
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def get_device():
    name = config.DEVICE
    if name != "cpu" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over loader; trains the head when an optimizer is given."""
    training = optimizer is not None

    # The frozen backbone stays in eval mode so BatchNorm statistics stay fixed.
    model.eval()
    model.fc.train(training)

    total_loss = 0.0
    actual, predicted = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(labels)
        actual.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(1).detach().cpu().tolist())

    positive = config.POSITIVE_IDX
    return {
        "loss": total_loss / len(loader.dataset),
        "f1": f1_score(actual, predicted, pos_label=positive, zero_division=0),
        "recall": recall_score(actual, predicted, pos_label=positive, zero_division=0),
        "precision": precision_score(
            actual, predicted, pos_label=positive, zero_division=0
        ),
    }


def train(version="v1"):
    """Train the head; early-stop on validation defect F1; log everything to MLflow."""
    seed = set_seed()

    root = data_prep.find_data_root()
    train_records = data_prep.load_split(version, "train", root)
    val_records = data_prep.load_split(version, "val", root)

    if not train_records or not val_records:
        raise ValueError("Train and validation manifests must not be empty.")

    for split_name, records in (("train", train_records), ("val", val_records)):
        missing = [str(path) for path, _ in records if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{split_name} has missing images; examples: {missing[:3]}"
            )

    train_loader = DataLoader(
        ManifestDataset(train_records, data_prep.get_transforms(train=True)),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        ManifestDataset(val_records, data_prep.get_transforms(train=False)),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # 2.3.1: class weights w_c = N / (K * n_c), so each class contributes equally.
    class_counts = np.bincount(
        [label for _, label in train_records], minlength=config.NUM_CLASSES
    )
    weights = class_weights(train_records)

    device = get_device()
    model = build_model(pretrained=True).to(device)
    counts = parameter_counts(model)

    # 2.3.1: Adam over the trainable (head) parameters only.
    optimizer = torch.optim.Adam(
        trainable_parameters(model),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    Path(config.ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)

    # 2.4.1: experiment, parameters, per-epoch metrics.
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    history = []
    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False

    with mlflow.start_run(run_name=f"resnet18_{version}") as run:
        mlflow.set_tags({
            "dataset_version": version,
            "model_family": "resnet18_frozen_backbone",
        })
        mlflow.log_params({
            "dataset_version": version,
            "seed": seed,
            "backbone": config.BACKBONE,
            "pretrained_weights": "IMAGENET1K_V1",
            "frozen_backbone": True,
            "total_params": counts["total"],
            "trainable_params": counts["trainable"],
            "optimizer": "Adam",
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "batch_size": config.BATCH_SIZE,
            "epochs_requested": config.EPOCHS,
            "early_stop_metric": "val_f1",
            "early_stop_patience": config.EARLY_STOP_PATIENCE,
            "loss": "CrossEntropyLoss (class-weighted)",
            "class_weights": json.dumps([round(float(w), 4) for w in weights]),
            "train_class_counts": json.dumps(class_counts.tolist()),
            "img_size": config.IMG_SIZE,
            "train_images": len(train_records),
            "validation_images": len(val_records),
        })

        for epoch in range(1, config.EPOCHS + 1):
            train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_metrics = run_epoch(model, val_loader, loss_fn, device)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_precision_defect": val_metrics["precision"],
                "val_recall_defect": val_metrics["recall"],
            }
            history.append(row)
            mlflow.log_metrics(
                {key: value for key, value in row.items() if key != "epoch"},
                step=epoch,
            )

            print(
                f"Epoch {epoch}/{config.EPOCHS} — "
                f"train loss: {row['train_loss']:.4f}, "
                f"val loss: {row['val_loss']:.4f}, "
                f"val F1: {row['val_f1']:.4f}, "
                f"defect recall: {row['val_recall_defect']:.4f}"
            )

            # 2.3.2: keep the best validation-F1 weights; stop when F1 stalls.
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.EARLY_STOP_PATIENCE:
                    stopped_early = epoch < config.EPOCHS
                    if stopped_early:
                        print(
                            f"Early stopping at epoch {epoch}: validation F1 has "
                            f"not improved for {stale_epochs} epochs."
                        )
                    break

        if best_state is None:
            raise RuntimeError("Training did not produce a best checkpoint.")

        model.load_state_dict(best_state)
        save_model(model, config.MODEL_PATH)

        best_row = history[best_epoch - 1]
        mlflow.log_metrics({
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
            "best_val_precision_defect": best_row["val_precision_defect"],
            "best_val_recall_defect": best_row["val_recall_defect"],
            "epochs_completed": len(history),
        })

        metadata = {
            "dataset_version": version,
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "epochs_requested": config.EPOCHS,
            "stopped_early": stopped_early,
            "best_val_f1": float(best_f1),
            "best_val_precision_defect": float(best_row["val_precision_defect"]),
            "best_val_recall_defect": float(best_row["val_recall_defect"]),
            "class_weights": [float(w) for w in weights],
            "history": history,
            "class_to_idx": config.CLASS_TO_IDX,
            "positive_class": config.POSITIVE_CLASS,
            "mlflow_run_id": run.info.run_id,
        }

        mlflow.log_dict({"history": history}, "history.json")
        mlflow.log_artifact(str(config.MODEL_PATH), artifact_path="checkpoint")

        # 2.4.2: log the best model with a signature and an input example.
        model_for_logging = copy.deepcopy(model).cpu().eval()
        input_example = np.zeros(
            (1, 3, config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32
        )
        with torch.no_grad():
            example_output = model_for_logging(torch.from_numpy(input_example))
        signature = infer_signature(input_example, example_output.numpy())

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model_for_logging,
            name="casting_defect_model",
            signature=signature,
            input_example=input_example,
        )

        metadata["mlflow_model_uri"] = model_info.model_uri
        metadata["mlflow_model_id"] = getattr(model_info, "model_id", None)
        Path(config.MODEL_META_PATH).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(config.MODEL_META_PATH), artifact_path="metadata")

    print("Best epoch:", best_epoch)
    print(f"Best validation F1: {best_f1:.4f}")
    print("Checkpoint:", config.MODEL_PATH)
    print("Metadata:", config.MODEL_META_PATH)
    print("MLflow run:", run.info.run_id)
    return metadata


def register_and_promote(run_id=None, alias=PRODUCTION_ALIAS):
    """Register a run's logged model (3.2.1) and point the alias at it (3.2.2).

    Defaults to the run in model_meta.json: the same weights saved to
    config.MODEL_PATH and served by app.py, so registry and API always agree.
    Re-running is idempotent: an already-registered run reuses its version.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    model_name = config.REGISTERED_MODEL
    meta_path = Path(config.MODEL_META_PATH)

    if not meta_path.is_file():
        raise FileNotFoundError(f"Training metadata not found: {meta_path}. Train first.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    run_id = run_id or metadata.get("mlflow_run_id")
    if not run_id:
        raise ValueError("No MLflow run id in model_meta.json; train first or pass run_id.")

    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {config.MLFLOW_EXPERIMENT}")

    # MLflow 3: models are LoggedModel objects linked to their source run.
    logged_models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        output_format="list",
    )
    run_models = [m for m in logged_models if getattr(m, "source_run_id", None) == run_id]
    if not run_models:
        raise RuntimeError(
            f"No logged model found for run {run_id}. "
            "Re-run training with the current src/train.py."
        )
    logged_model = max(run_models, key=lambda m: getattr(m, "creation_timestamp", 0) or 0)

    existing = client.search_model_versions(f"name='{model_name}'")
    same_run = [v for v in existing if getattr(v, "run_id", None) == run_id]

    if same_run:
        version = max(same_run, key=lambda v: int(v.version))
        created_new_version = False
    else:
        version = mlflow.register_model(
            model_uri=f"models:/{logged_model.model_id}",
            name=model_name,
            await_registration_for=300,
        )
        created_new_version = True
    version_number = str(version.version)

    try:
        previous_version = str(client.get_model_version_by_alias(model_name, alias).version)
    except MlflowException:
        previous_version = None

    client.set_registered_model_alias(model_name, alias, version_number)

    # Record why this version was promoted.
    tags = {
        "dataset_version": metadata.get("dataset_version"),
        "selection_metric": "best validation defect F1 (early stopping)",
        "best_epoch": metadata.get("best_epoch"),
        "best_val_f1": metadata.get("best_val_f1"),
        "best_val_recall_defect": metadata.get("best_val_recall_defect"),
    }
    metrics_path = Path(config.METRICS_PATH)
    if metrics_path.is_file():
        test_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if test_metrics.get("mlflow_run_id") == run_id:
            for key in ("recall_defect", "precision_defect", "f1_defect", "roc_auc"):
                tags[f"test_{key}"] = test_metrics.get(key)

    for key, value in tags.items():
        if value is not None:
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            client.set_model_version_tag(model_name, version_number, key, text)

    description = (
        f"ResNet18 (frozen ImageNet backbone, 2-class head) trained on dataset "
        f"{metadata.get('dataset_version')}; selected at epoch "
        f"{metadata.get('best_epoch')} by validation defect F1 "
        f"{metadata.get('best_val_f1', 0):.4f}."
    )
    try:
        client.update_model_version(model_name, version_number, description=description)
    except Exception as error:
        # MLflow 3 file store cannot save versions that carry model metrics.
        print(f"Note: version description stored as a tag ({type(error).__name__}).")
        client.set_model_version_tag(model_name, version_number, "description", description)

    production = client.get_model_version_by_alias(model_name, alias)

    metadata.update({
        "mlflow_run_id": run_id,
        "registered_model": model_name,
        "registered_version": str(production.version),
        "model_version": str(production.version),
        "production_alias": alias,
    })
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Version history as evidence (rollback = move the alias to an older version).
    all_versions = client.search_model_versions(f"name='{model_name}'")
    history = [
        {
            "name": v.name,
            "version": str(v.version),
            "run_id": getattr(v, "run_id", None),
            "status": getattr(v, "status", None),
            "created_ms": getattr(v, "creation_timestamp", None),
            "aliases": list(getattr(v, "aliases", []) or []),
            "tags": dict(getattr(v, "tags", {}) or {}),
        }
        for v in sorted(all_versions, key=lambda v: int(v.version))
    ]
    history_path = Path(config.ARTIFACT_DIR) / "model_registry_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "name": model_name,
        "version": str(production.version),
        "alias": alias,
        "run_id": run_id,
        "created_new_version": created_new_version,
        "previous_version": previous_version,
        "history_path": str(history_path),
    }


def _image_stats(path):
    """data_prep.image_features for one image (accepts a path or a PIL image)."""
    try:
        return data_prep.image_features(path)
    except (TypeError, AttributeError):
        with Image.open(path) as image:
            return data_prep.image_features(image.convert("L"))


def save_reference_baseline(net, ref_items) -> dict:
    """Stage 4 reference: features + embeddings for clean reference images.

    net: the trained model. ref_items: (path, label) records of clean images
    (main() passes a seeded validation sample). Writes, in config.ARTIFACT_DIR:
    - reference_features.csv: data_prep.image_features per image + path, label
    - reference_embeddings.npz: embeddings (N x 512), labels, paths (row-aligned)
    """
    import datetime

    import pandas as pd

    ref_items = sorted(ref_items, key=lambda item: str(item[0]))
    if not ref_items:
        raise ValueError("ref_items is empty.")

    output_dir = Path(config.ARTIFACT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = data_prep.find_data_root()

    def relative(path):
        try:
            return str(Path(path).relative_to(root))
        except ValueError:
            return str(path)

    paths = [relative(path) for path, _ in ref_items]
    labels = np.array([_label_of(item) for item in ref_items], dtype=np.int64)

    features = pd.DataFrame([
        {"path": p, "label": int(l), **_image_stats(path)}
        for p, l, (path, _) in zip(paths, labels, ref_items)
    ])

    device = next(net.parameters()).device
    extractor = EmbeddingExtractor(net).to(device).eval()
    loader = DataLoader(
        ManifestDataset(ref_items, data_prep.get_transforms(train=False)),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    chunks = []
    with torch.no_grad():
        for images, _ in loader:
            chunks.append(extractor(images.to(device)).cpu().numpy())
    embeddings = np.concatenate(chunks).astype(np.float32)

    features_path = output_dir / "reference_features.csv"
    embeddings_path = output_dir / "reference_embeddings.npz"
    features.to_csv(features_path, index=False)
    np.savez(embeddings_path, embeddings=embeddings, labels=labels, paths=np.array(paths))

    summary = {
        "n_images": len(ref_items),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "features_path": str(features_path),
        "embeddings_path": str(embeddings_path),
        "embedding_dim": int(embeddings.shape[1]),
        "feature_means": features.drop(columns=["path", "label"]).mean().round(4).to_dict(),
    }
    (output_dir / "reference_baseline.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    """Full workflow: train + MLflow tracking -> registry @production -> Stage 4 reference.

    Data quality checks and the versioned v1 split are produced in
    Data_Preparation.ipynb; here we only confirm the split exists, because
    re-scanning all 7,348 images on Google Drive would add many minutes.
    """
    version = "v1"
    split_dir = Path(config.SPLIT_DIR) / version
    if not (split_dir / "metadata.json").is_file():
        raise FileNotFoundError(
            f"Split {version} not found in {split_dir}. Run Data_Preparation.ipynb first."
        )

    metadata = train(version)

    registry = register_and_promote()
    print(
        f"Registered {registry['name']} version {registry['version']} "
        f"as @{registry['alias']}"
    )

    # Stage 4 reference: a seeded sample of clean validation images.
    root = data_prep.find_data_root()
    val_records = data_prep.load_split(version, "val", root)
    sample = random.Random(metadata["seed"]).sample(val_records, min(300, len(val_records)))

    from src.model import load_model
    net = load_model(device=get_device())
    reference = save_reference_baseline(net, sample)
    print(
        f"Reference baseline: {reference['n_images']} validation images -> "
        f"{reference['features_path']}, {reference['embeddings_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())