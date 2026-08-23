from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.visual_features import (
    build_visual_extractor, extract_patch_embeddings, load_visual_image,
    normalize_color, preprocess_visual_image,
)


def encoded_image(mode="RGB", size=(40, 36), fmt="PNG", color=None):
    if color is None: color = 128 if mode == "L" else ((10, 20, 30, 0) if mode == "RGBA" else (10, 20, 30))
    image = Image.new(mode, size, color)
    output = BytesIO(); image.save(output, format=fmt); return output.getvalue()


@pytest.mark.parametrize(("fmt", "filename"), [("PNG", "part.png"), ("JPEG", "part.jpg")])
def test_valid_png_and_jpeg(fmt, filename):
    image = load_visual_image(encoded_image(fmt=fmt), filename=filename)
    assert image.mode == "RGB" and image.size == (40, 36)


def test_grayscale_and_rgba_conversion_are_deterministic():
    grayscale = load_visual_image(encoded_image("L"), filename="part.png")
    rgba = load_visual_image(encoded_image("RGBA"), filename="part.png")
    assert grayscale.mode == "RGB" and grayscale.getpixel((0, 0)) == (128, 128, 128)
    assert rgba.getpixel((0, 0)) == (255, 255, 255)
    source = Image.new("RGBA", (32, 32), (100, 50, 25, 128))
    assert np.array_equal(np.asarray(normalize_color(source)), np.asarray(normalize_color(source)))


def test_minimum_dimensions_and_invalid_inputs():
    assert load_visual_image(encoded_image(size=(32, 32)), filename="ok.png").size == (32, 32)
    with pytest.raises(ValueError, match="at least 32"):
        load_visual_image(encoded_image(size=(31, 32)), filename="small.png")
    with pytest.raises(ValueError, match="corrupt"):
        load_visual_image(b"not an image", filename="bad.png")
    with pytest.raises(ValueError, match="Only JPEG and PNG"):
        load_visual_image(encoded_image(), filename="part.gif")
    with pytest.raises(ValueError, match="filename is required"):
        load_visual_image(encoded_image())


def test_preprocessing_is_deterministic():
    image = Image.new("RGB", (53, 41), (10, 20, 30))
    first = preprocess_visual_image(image); second = preprocess_visual_image(image)
    assert first.shape == (3, 256, 256) and torch.equal(first, second)


def test_feature_contract_from_serialized_artifact():
    import joblib
    root = Path(__file__).resolve().parents[1]
    bundle = joblib.load(root / "models" / "factorymind_visual_quality_model_v1.joblib")
    model = build_visual_extractor(state_dict=bundle.backbone_state_dict, device="cpu")
    assert not model.training and sum(p.requires_grad for p in model.parameters()) == 0
    tensor = preprocess_visual_image(Image.new("RGB", (40, 40), (100, 120, 140)))[None]
    first = extract_patch_embeddings(model, tensor, device="cpu")
    second = extract_patch_embeddings(model, tensor, device="cpu")
    assert first.shape == (1, 256, 384) and np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.allclose(np.linalg.norm(first, axis=-1), 1.0, atol=1e-5)
