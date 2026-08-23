"""Frozen image and feature contract for FactoryMind Visual Quality v1."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2


VISUAL_CATEGORY = "zipper"
VISUAL_INPUT_SIZE = (256, 256)
VISUAL_MIN_DIMENSION = 32
VISUAL_PATCH_GRID = (16, 16)
VISUAL_PATCH_COUNT = 256
VISUAL_EMBEDDING_DIM = 384
VISUAL_FEATURE_LAYERS = ("layer2", "layer3")
VISUAL_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
VISUAL_IMAGE_FORMATS = ("JPEG", "PNG")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_PREPROCESS = v2.Compose([
    v2.ToImage(),
    v2.Resize(VISUAL_INPUT_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _validate_dimensions(image: Image.Image) -> None:
    if image.width < VISUAL_MIN_DIMENSION or image.height < VISUAL_MIN_DIMENSION:
        raise ValueError("Image width and height must each be at least 32 pixels.")


def normalize_color(image: Image.Image) -> Image.Image:
    """Return deterministic RGB pixels without modifying the supplied PIL image."""
    _validate_dimensions(image)
    if image.mode == "RGB":
        return image.copy()
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")
    try:
        return image.convert("RGB")
    except (ValueError, OSError) as exc:
        raise ValueError(f"Image color mode {image.mode!r} cannot be converted safely to RGB.") from exc


def _decode_payload(payload: bytes, expected_format: str | None) -> Image.Image:
    if not payload:
        raise ValueError("Image input is empty.")
    try:
        with Image.open(BytesIO(payload)) as probe:
            probe.verify()
        with Image.open(BytesIO(payload)) as decoded:
            actual_format = decoded.format
            if actual_format not in VISUAL_IMAGE_FORMATS:
                raise ValueError("Only JPEG and PNG images are accepted.")
            if expected_format is not None and actual_format != expected_format:
                raise ValueError("Image extension does not match the decoded image format.")
            return normalize_color(decoded)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Image is corrupt or cannot be decoded.") from exc


def load_visual_image(
    source: str | Path | bytes | bytearray | BinaryIO | Image.Image,
    *,
    filename: str | None = None,
) -> Image.Image:
    """Validate one supported input and return a detached RGB PIL image."""
    if isinstance(source, Image.Image):
        return normalize_color(source)
    expected_format = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix not in VISUAL_IMAGE_EXTENSIONS:
            raise ValueError("Only JPEG and PNG images are accepted.")
        if not path.is_file():
            raise ValueError("Image path does not identify a readable file.")
        expected_format = "PNG" if suffix == ".png" else "JPEG"
        return _decode_payload(path.read_bytes(), expected_format)
    if filename is not None:
        suffix = Path(filename).suffix.lower()
        if suffix not in VISUAL_IMAGE_EXTENSIONS:
            raise ValueError("Only JPEG and PNG images are accepted.")
        expected_format = "PNG" if suffix == ".png" else "JPEG"
    elif not isinstance(source, Image.Image):
        raise ValueError("A .jpg, .jpeg, or .png filename is required for byte or file-like input.")
    if isinstance(source, (bytes, bytearray)):
        return _decode_payload(bytes(source), expected_format)
    if hasattr(source, "read"):
        payload = source.read()
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("File-like image input must return bytes.")
        return _decode_payload(bytes(payload), expected_format)
    raise TypeError("Image input must be a path, bytes, binary file-like object, or PIL Image.")


def preprocess_visual_image(image: Image.Image) -> torch.Tensor:
    rgb = normalize_color(image)
    tensor = _PREPROCESS(rgb)
    if tensor.shape != (3, *VISUAL_INPUT_SIZE) or not torch.isfinite(tensor).all():
        raise RuntimeError("Frozen visual preprocessing produced an invalid tensor.")
    return tensor


class VisualFeatureExtractor(nn.Module):
    """Frozen ResNet18 layer2/layer3 extractor matching Notebooks 16–17."""

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.freeze()

    def freeze(self) -> "VisualFeatureExtractor":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.stem(inputs)
        layer2 = self.layer2(stem)
        layer3 = self.layer3(layer2)
        return layer2, layer3


def build_visual_extractor(
    *, state_dict: dict[str, torch.Tensor] | None = None, pretrained: bool = False,
    device: str | torch.device = "cpu",
) -> VisualFeatureExtractor:
    if state_dict is not None and pretrained:
        raise ValueError("Supply serialized state or request pretrained weights, not both.")
    model = VisualFeatureExtractor(pretrained=pretrained)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model.to(torch.device(device)).freeze()


def extract_patch_embeddings(
    model: VisualFeatureExtractor, image_batch: torch.Tensor, *, device: str | torch.device,
) -> np.ndarray:
    """Return L2-normalized (batch, 256, 384) patches with exact notebook alignment."""
    if image_batch.ndim != 4 or image_batch.shape[1:] != (3, *VISUAL_INPUT_SIZE):
        raise ValueError("Image batch must have shape (batch, 3, 256, 256).")
    model.freeze()
    with torch.inference_mode():
        layer2, layer3 = model(image_batch.to(device))
        if layer2.shape[1:] != (128, 32, 32) or layer3.shape[1:] != (256, 16, 16):
            raise RuntimeError("ResNet18 intermediate feature shapes violate the frozen contract.")
        layer3_aligned = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        combined = torch.cat([
            F.adaptive_avg_pool2d(layer2, VISUAL_PATCH_GRID),
            F.adaptive_avg_pool2d(layer3_aligned, VISUAL_PATCH_GRID),
        ], dim=1)
        patches = combined.permute(0, 2, 3, 1).reshape(
            combined.shape[0], VISUAL_PATCH_COUNT, VISUAL_EMBEDDING_DIM
        )
        patches = F.normalize(patches, dim=-1)
        result = patches.cpu().numpy().astype(np.float32)
    if result.shape[1:] != (VISUAL_PATCH_COUNT, VISUAL_EMBEDDING_DIM):
        raise RuntimeError("Patch embedding shape violates the frozen contract.")
    if not np.isfinite(result).all():
        raise RuntimeError("Patch embeddings contain non-finite values.")
    return result
