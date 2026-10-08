"""data_prep.py — Stage 1: discovery, quality validation, versioned splits, transforms. 

Implement: locate the casting folders; run data-quality checks (missing/corrupt/duplicate/
dimension/class-distribution/consistency); build reproducible stratified train/val/test
splits with a versioned snapshot + metadata.json; define preprocessing + augmentation
transforms; and per-image feature extraction used by drift monitoring.
"""
from __future__ import annotations

import hashlib, json, os, random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
















def find_data_root(base=None):
    """Locate the casting train/test root inside data/."""
    from pathlib import Path
    import config

    base = Path(base) if base is not None else Path(config.BASE_DIR) / "data"

    if not base.is_dir():
        raise FileNotFoundError(f"Data folder not found: {base}")

    candidates = {base}
    candidates.update(
        folder.parent for folder in base.rglob("train") if folder.is_dir()
    )

    valid_roots = [
        root
        for root in sorted(candidates)
        if all(
            (root / split / class_name).is_dir()
            for split in ("train", "test")
            for class_name in config.CLASS_TO_IDX
        )
    ]

    if not valid_roots:
        raise FileNotFoundError(
            f"No complete dataset found under {base}. Expected train and test "
            "folders, each containing ok_front and def_front."
        )

    if len(valid_roots) > 1:
        raise ValueError(
            f"Multiple dataset roots found: {valid_roots}. "
            "Pass the intended root as the base argument."
        )

    return valid_roots[0]


def list_images(root, split=None):
    """Return sorted image paths from train, test, or both."""
    from pathlib import Path
    import config

    root = Path(root)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if split not in (None, "train", "test"):
        raise ValueError("split must be 'train', 'test', or None.")

    splits = ("train", "test") if split is None else (split,)
    images = []

    for split_name in splits:
        for class_name in config.CLASS_TO_IDX:
            folder = root / split_name / class_name

            if not folder.is_dir():
                raise FileNotFoundError(f"Missing folder: {folder}")

            images.extend(
                path
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in extensions
            )

    return sorted(images)

def validate_quality(root, expected_size=(300, 300)):
    """
    Check image readability, dimensions, exact duplicates and class balance.

    passed=True means every BLOCKING check passed:
    - All required class folders contain readable images.
    - No missing/unreadable files.
    - No unexpected image dimensions.
    - No identical image stored under two different labels.

    Exact duplicates are REVIEW checks, not blocking: they are reported here
    and removed by build_splits() in Stage 1.3 (keeping the test copy).

    expected_size=None reports dimensions without enforcing a specific size.
    No images are changed or deleted.
    """
    from pathlib import Path
    from collections import Counter, defaultdict
    import hashlib
    from PIL import Image
    import config

    root = Path(root)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    missing_folders = []
    missing_files = []
    corrupt = []
    unexpected_dimensions = []
    dimensions = Counter()
    hash_groups = defaultdict(list)
    distribution = {}
    total_images = 0

    for split in ("train", "test"):
        distribution[split] = {}

        for class_name in config.CLASS_TO_IDX:
            folder = root / split / class_name
            stats = {"found": 0, "readable": 0}
            distribution[split][class_name] = stats

            if not folder.is_dir():
                missing_folders.append(str(folder))
                continue

            paths = sorted(
                p for p in folder.rglob("*")
                if p.suffix.lower() in extensions
                and (p.is_file() or p.is_symlink())
            )

            stats["found"] = len(paths)
            total_images += len(paths)

            for path in paths:
                relative_path = path.relative_to(root).as_posix()

                if not path.exists():
                    missing_files.append(relative_path)
                    continue

                try:
                    # Verify image integrity.
                    with Image.open(path) as image:
                        image.verify()

                    # Reopen: verify() does not decode all image pixels.
                    with Image.open(path) as image:
                        image.load()
                        width, height = image.size

                    if width <= 0 or height <= 0:
                        raise ValueError("Invalid image dimensions")

                    stats["readable"] += 1
                    dimensions[f"{width}x{height}"] += 1

                    if (
                        expected_size is not None
                        and (width, height) != tuple(expected_size)
                    ):
                        unexpected_dimensions.append({
                            "path": relative_path,
                            "actual": [width, height],
                            "expected": list(expected_size),
                        })

                    # Hash file bytes to detect exact duplicate files.
                    digest = hashlib.sha256()
                    with path.open("rb") as image_file:
                        for chunk in iter(
                            lambda: image_file.read(1024 * 1024), b""
                        ):
                            digest.update(chunk)

                    hash_groups[digest.hexdigest()].append({
                        "path": relative_path,
                        "split": split,
                        "class": class_name,
                    })

                except FileNotFoundError:
                    missing_files.append(relative_path)

                except Exception as error:
                    corrupt.append({
                        "path": relative_path,
                        "error": str(error),
                    })

    duplicates = [
        {"sha256": digest, "files": files}
        for digest, files in hash_groups.items()
        if len(files) > 1
    ]

    cross_split_duplicates = [
        group for group in duplicates
        if len({file["split"] for file in group["files"]}) > 1
    ]

    label_conflicts = [
        group for group in duplicates
        if len({file["class"] for file in group["files"]}) > 1
    ]

    # Count extra copies, not the first image in each duplicate group.
    duplicate_count = sum(
        len(group["files"]) - 1 for group in duplicates
    )

    empty_classes = []

    for split, class_stats in distribution.items():
        split_total = sum(s["found"] for s in class_stats.values())

        for class_name, stats in class_stats.items():
            stats["percentage"] = (
                round(100 * stats["found"] / split_total, 2)
                if split_total else 0.0
            )

            if stats["readable"] == 0:
                empty_classes.append(f"{split}/{class_name}")

    # Blocking: the data cannot be used safely until these pass.
    blocking_checks = {
        "images_found": total_images > 0,
        "required_folders_present": not missing_folders,
        "no_missing_files": not missing_files,
        "no_corrupt_files": not corrupt,
        "dimensions_ok": not unexpected_dimensions,
        "all_classes_readable": not empty_classes,
        "no_label_conflicts": not label_conflicts,
    }

    # Review: reported issues that build_splits() resolves by deduplication.
    review_checks = {
        "no_exact_duplicates": not duplicates,
        "no_cross_split_duplicates": not cross_split_duplicates,
    }

    checks = {**blocking_checks, **review_checks}

    warnings = []

    if total_images != 7348:
        warnings.append(
            f"Found {total_images} image files; the template expects "
            "approximately 7348. Verify whether this is a sample."
        )

    if duplicates:
        warnings.append(
            f"{duplicate_count} exact duplicate copies found "
            f"({len(cross_split_duplicates)} groups span train and test). "
            "build_splits() keeps one copy per image (the test copy) "
            "so no image can appear in two final splits."
        )

    return {
        "root": str(root),
        "total_images": total_images,
        "expected_size": (
            list(expected_size) if expected_size is not None else None
        ),
        "missing_folders": missing_folders,
        "missing_files": missing_files,
        "corrupt": corrupt,
        "dimensions": dict(dimensions),
        "unexpected_dimensions": unexpected_dimensions,
        "distribution": distribution,
        "duplicate_group_count": len(duplicates),
        "duplicate_count": duplicate_count,
        "duplicates": duplicates,
        "cross_split_duplicates": cross_split_duplicates,
        "label_conflicts": label_conflicts,
        "empty_classes": empty_classes,
        "checks": checks,
        "blocking_checks": blocking_checks,
        "review_checks": review_checks,
        "passed": all(blocking_checks.values()),
        "fully_clean": all(checks.values()),
        "warnings": warnings,
    }


def load_split(version, name, root):
    """Load saved JSON records as (image_path, label) pairs."""
    import json
    from pathlib import Path
    import config

    if name not in ("train", "val", "test"):
        raise ValueError("name must be 'train', 'val', or 'test'.")

    manifest_path = Path(config.SPLIT_DIR) / version / f"{name}.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}. "
            f"Run build_splits(root, '{version}') first."
        )

    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(root)

    return [(root / record["path"], int(record["label"])) for record in records]

def build_splits(root, version="v1"):
    from pathlib import Path
    from collections import defaultdict, Counter
    from datetime import datetime, timezone
    import hashlib
    import json
    import re

    from PIL import Image
    from sklearn.model_selection import train_test_split
    import config

    root = Path(root).resolve()
    seed = getattr(config, "SEED", 42)  # brief requires seed 42
    val_fraction = config.VAL_SPLIT

    if not re.fullmatch(r"[A-Za-z0-9_-]+", version):
        raise ValueError("Use a version such as v1 or v2.")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    groups = defaultdict(list)
    inventory = []

    # Read and hash all images again so the snapshot matches current data.
    for split in ("train", "test"):
        for class_name, label in config.CLASS_TO_IDX.items():
            folder = root / split / class_name

            if not folder.is_dir():
                raise FileNotFoundError(f"Missing folder: {folder}")

            paths = sorted(
                p for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in extensions
            )

            if not paths:
                raise ValueError(f"No images in {folder}")

            for path in paths:
                # Stop if an unreadable image is encountered.
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()

                hasher = hashlib.sha256()
                with path.open("rb") as file:
                    for chunk in iter(lambda: file.read(1024 * 1024), b""):
                        hasher.update(chunk)

                record = {
                    "path": path.relative_to(root).as_posix(),
                    "label": int(label),
                    "class": class_name,
                    "source_split": split,
                    "sha256": hasher.hexdigest(),
                }

                inventory.append(record)
                groups[record["sha256"]].append(record)

    # Keep one copy per hash. Prefer the original test copy.
    retained = []
    excluded = []

    for digest in sorted(groups):
        copies = groups[digest]

        if len({r["label"] for r in copies}) > 1:
            raise ValueError(
                "Identical image has conflicting labels: "
                + str([r["path"] for r in copies])
            )

        copies = sorted(
            copies,
            key=lambda r: (
                0 if r["source_split"] == "test" else 1,
                r["path"],
            ),
        )

        retained.append(copies[0])

        for duplicate in copies[1:]:
            excluded.append({
                **duplicate,
                "kept_path": copies[0]["path"],
                "reason": "exact_duplicate",
            })

    train_pool = sorted(
        [r for r in retained if r["source_split"] == "train"],
        key=lambda r: r["path"],
    )

    test = sorted(
        [r for r in retained if r["source_split"] == "test"],
        key=lambda r: r["path"],
    )

    required_labels = set(config.CLASS_TO_IDX.values())

    for name, records in (("training pool", train_pool), ("test", test)):
        if {r["label"] for r in records} != required_labels:
            raise ValueError(f"A class is missing from the {name}.")

    try:
        train, val = train_test_split(
            train_pool,
            test_size=val_fraction,
            random_state=seed,
            stratify=[r["label"] for r in train_pool],
        )
    except ValueError as error:
        raise ValueError(
            "Not enough images per class for the requested "
            "stratified validation split."
        ) from error

    splits = {
        "train": sorted(train, key=lambda r: r["path"]),
        "val": sorted(val, key=lambda r: r["path"]),
        "test": test,
    }

    # Confirm all final splits contain both classes.
    for name, records in splits.items():
        if {r["label"] for r in records} != required_labels:
            raise ValueError(f"A class is missing from the {name} split.")

    # Prove no exact duplicate contents cross final splits.
    hashes = {
        name: {r["sha256"] for r in records}
        for name, records in splits.items()
    }

    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if hashes[first] & hashes[second]:
            raise ValueError(f"Leakage detected between {first} and {second}.")

    fingerprint = hashlib.sha256(
        json.dumps(
            sorted(inventory, key=lambda r: r["path"]),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    metadata = {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "dataset_fingerprint": fingerprint,
        "seed": seed,
        "validation_fraction_of_clean_train": val_fraction,
        "class_to_idx": dict(config.CLASS_TO_IDX),
        "positive_class": config.POSITIVE_CLASS,
        "original_image_count": len(inventory),
        "unique_image_count": len(retained),
        "excluded_duplicate_count": len(excluded),
        "split_policy": (
            "Preserve original test membership after deduplication. "
            "Prefer test copies; stratify remaining train into train/val."
        ),
        "cross_split_hash_overlap": 0,
        "split_info": {
            name: {
                "size": len(records),
                "class_counts": {
                    class_name: sum(
                        r["class"] == class_name for r in records
                    )
                    for class_name in config.CLASS_TO_IDX
                },
            }
            for name, records in splits.items()
        },
    }

    output_dir = Path(config.SPLIT_DIR) / version
    payloads = {
        **{f"{name}.json": records for name, records in splits.items()},
        "excluded_duplicates.json": excluded,
    }

    # Preserve existing versions; reuse only if data and split results match.
    if output_dir.exists():
        metadata_path = output_dir / "metadata.json"

        if not metadata_path.is_file():
            raise ValueError("Incomplete existing version. Use a new version.")

        previous = json.loads(metadata_path.read_text())

        same_settings = all(
            previous.get(key) == metadata[key]
            for key in (
                "dataset_fingerprint",
                "seed",
                "validation_fraction_of_clean_train",
                "class_to_idx",
                "positive_class",
                "split_policy",
            )
        )

        same_files = all(
            (output_dir / name).is_file()
            and json.loads((output_dir / name).read_text()) == content
            for name, content in payloads.items()
        )

        if not (same_settings and same_files):
            raise ValueError(
                f"{version} already contains different data or splits. "
                "Choose a new version, such as v2."
            )

        return previous

    output_dir.mkdir(parents=True)

    for name, content in payloads.items():
        (output_dir / name).write_text(
            json.dumps(content, indent=2), encoding="utf-8"
        )

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return metadata

def image_features(image):
    """Calculate descriptive features from a PIL image or image path."""
    from pathlib import Path
    import numpy as np
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32)
    else:
        gray = np.asarray(image.convert("L"), dtype=np.float32)

    if min(gray.shape) < 3:
        raise ValueError("Image must be at least 3 × 3 pixels.")

    # Gradient magnitude: strength of intensity changes.
    gradient_y, gradient_x = np.gradient(gray)
    gradient = np.hypot(gradient_x, gradient_y)

    # Discrete Laplacian, excluding image borders.
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )

    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "edge_density": float((gradient > 20).mean()),
        "sharpness": float(laplacian.var()),
        "dark_ratio": float((gray < 50).mean()),
        # Compatibility with the supplied monitoring configuration.
        "mean_intensity": float(gray.mean()),
    }

def get_transforms(train=False):
    """Convert a PIL image into a normalized 3 × 224 × 224 tensor."""
    import config
    from torchvision import transforms as T

    operations = [
        # Replicate grayscale information across three channels.
        T.Grayscale(num_output_channels=3),
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    ]

    if train:
        operations.extend([
            T.RandomHorizontalFlip(p=config.AUG["hflip_p"]),

            # Random rotation and translation in one operation.
            T.RandomAffine(
                degrees=config.AUG["rotation_degrees"],
                translate=(
                    config.AUG["translate"],
                    config.AUG["translate"],
                ),
                interpolation=T.InterpolationMode.BILINEAR,
                fill=128,
            ),

            T.ColorJitter(
                brightness=config.AUG["brightness"],
                contrast=config.AUG["contrast"],
            ),
        ])

    operations.extend([
        # Convert pixels from 0–255 to a float tensor in 0–1.
        T.ToTensor(),

        # Normalize each channel using ImageNet statistics.
        T.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
        ),
    ])

    return T.Compose(operations)


# ---- image_features restricted to config.DRIFT_FEATURES ----
_all_image_features = image_features


def image_features(image):
    """Drift features for one image: exactly config.DRIFT_FEATURES, as floats."""
    import config

    features = _all_image_features(image)
    missing = [name for name in config.DRIFT_FEATURES if name not in features]
    if missing:
        raise KeyError(f"image_features is missing drift features: {missing}")
    return {name: float(features[name]) for name in config.DRIFT_FEATURES}
