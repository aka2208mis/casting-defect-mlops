"""Stage 2.2: ImageNet-pretrained ResNet18 with a frozen backbone and 2-class head."""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

import config


def build_model(pretrained=True):
    """Build ResNet18 with a frozen backbone and a trainable 2-class head."""
    if config.NUM_CLASSES != 2:
        raise ValueError("This stage expects exactly two classes.")

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # 2.2.1: freeze every pretrained parameter.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 2.2.2: replace the 1000-class ImageNet head; new layers train by default.
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


def trainable_parameters(model):
    """Return trainable parameters for the optimizer."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def parameter_counts(model):
    """Return total, trainable, and frozen parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def save_model(model, path=None):
    """Save the model's weights (state_dict) to path (default: config.MODEL_PATH)."""
    path = Path(path or config.MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_model(checkpoint=None, device=None):
    """Load a saved checkpoint without downloading pretrained weights."""
    checkpoint_path = Path(checkpoint or config.MODEL_PATH)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if device is None:
        device = config.DEVICE
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"

    device = torch.device(device)
    model = build_model(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class EmbeddingExtractor(nn.Module):
    """Extract 512-d feature vectors from a ResNet before its classifier."""

    def __init__(self, model):
        super().__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-1])

    def forward(self, images):
        return self.backbone(images).flatten(start_dim=1)
