"""Build the frozen self-contained FactoryMind Visual Quality v1 artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

from .visual_features import (
    IMAGENET_MEAN, IMAGENET_STD, VISUAL_CATEGORY, VISUAL_EMBEDDING_DIM,
    VISUAL_FEATURE_LAYERS, VISUAL_IMAGE_FORMATS, VISUAL_INPUT_SIZE,
    VISUAL_MIN_DIMENSION, VISUAL_PATCH_COUNT, VISUAL_PATCH_GRID,
    build_visual_extractor, extract_patch_embeddings, normalize_color,
    preprocess_visual_image,
)
from .visual_pipeline import (
    VISUAL_DISCLAIMER, VISUAL_MODEL_VERSION, VISUAL_THRESHOLD_QUANTILE,
    VISUAL_WARNING, VisualModelBundle, create_visual_runtime,
    score_patch_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad" / VISUAL_CATEGORY
MODEL_PATH = PROJECT_ROOT / "models" / "factorymind_visual_quality_model_v1.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "factorymind_visual_quality_model_v1.metadata.json"
SEED = 42
PROJECTION_DIM = 16
CORESET_RATIO = 0.05
CORESET_COUNT = 2458
CHUNK_SIZE = 8192
NOTEBOOK17_THRESHOLD = 0.454687


class _PathDataset(Dataset):
    def __init__(self, paths: list[Path]): self.paths = paths
    def __len__(self) -> int: return len(self.paths)
    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            tensor = preprocess_visual_image(normalize_color(image))
        return tensor, str(self.paths[index])


def validate_dataset(root: Path = DATA_ROOT) -> dict[str, list[Path]]:
    train = sorted((root / "train" / "good").glob("*.png"))
    test_good = sorted((root / "test" / "good").glob("*.png"))
    defect_dirs = sorted(p for p in (root / "test").iterdir() if p.is_dir() and p.name != "good")
    anomalies = [p for directory in defect_dirs for p in sorted(directory.glob("*.png"))]
    masks = sorted((root / "ground_truth").glob("*/*_mask.png"))
    if (len(train), len(test_good), len(anomalies), len(masks), len(defect_dirs)) != (240, 32, 119, 119, 7):
        raise ValueError("MVTec zipper data does not match the frozen dataset contract.")
    return {"train": train, "test_good": test_good, "anomalies": anomalies, "masks": masks, "defect_dirs": defect_dirs}


def frozen_split(train_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    reference, development = train_test_split(
        sorted(train_paths), test_size=0.20, random_state=SEED, shuffle=True
    )
    reference, development = sorted(reference), sorted(development)
    if len(reference) != 192 or len(development) != 48:
        raise RuntimeError("Frozen visual development split was not reproduced.")
    return reference, development


def extract_paths(model, paths: list[Path], device: torch.device, batch_size: int = 12):
    loader = DataLoader(_PathDataset(paths), batch_size=batch_size, shuffle=False, num_workers=0)
    batches, ordered = [], []
    with torch.inference_mode():
        for images, names in loader:
            batches.append(extract_patch_embeddings(model, images, device=device))
            ordered.extend(Path(name) for name in names)
    return np.concatenate(batches), ordered


def projected_kcenter_indices(
    memory: np.ndarray, *, seed: int = SEED, projection_dim: int = PROJECTION_DIM,
    retention_ratio: float = CORESET_RATIO, chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    values = np.asarray(memory, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Coreset input must be a finite two-dimensional array.")
    retained = int(np.ceil(len(values) * retention_ratio))
    projected = GaussianRandomProjection(
        n_components=projection_dim, random_state=seed
    ).fit_transform(values).astype(np.float32)
    selected = np.empty(retained, dtype=np.int64)
    selected[0] = int(np.argmax(np.sum((projected - projected.mean(axis=0)) ** 2, axis=1)))
    minimum_distances = np.full(len(projected), np.inf, dtype=np.float32)
    for step in range(retained):
        center = projected[selected[step]]
        for start in range(0, len(projected), chunk_size):
            stop = min(start + chunk_size, len(projected))
            difference = projected[start:stop] - center
            squared = np.einsum("ij,ij->i", difference, difference)
            minimum_distances[start:stop] = np.minimum(minimum_distances[start:stop], squared)
        minimum_distances[selected[: step + 1]] = -1
        if step + 1 < retained:
            selected[step + 1] = int(np.argmax(minimum_distances))
    return selected


def build_metadata(bundle: VisualModelBundle, defect_subtypes: list[str]) -> dict:
    return {
        "model_name": "FactoryMind Visual Quality Inspection",
        "model_version": VISUAL_MODEL_VERSION, "model_family": "PatchCore-style nearest-neighbor anomaly detection",
        "dataset": "MVTec AD", "dataset_category": VISUAL_CATEGORY,
        "dataset_license_note": "MVTec AD CC BY-NC-SA; noncommercial constraints must be reviewed before use.",
        "training_protocol": "normal-only", "reference_image_count": 192, "development_normal_count": 48,
        "test_good_count": 32, "test_anomaly_count": 119, "defect_subtypes": defect_subtypes,
        "accepted_image_formats": list(VISUAL_IMAGE_FORMATS), "minimum_source_dimensions": [VISUAL_MIN_DIMENSION] * 2,
        "input_size": list(VISUAL_INPUT_SIZE), "input_color_mode": "RGB",
        "rgba_handling": "alpha composite over opaque white, then RGB", "resize_method": "bilinear",
        "antialias": True, "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "backbone": "ResNet18", "backbone_weights": "IMAGENET1K_V1", "backbone_frozen": True,
        "feature_layers": list(VISUAL_FEATURE_LAYERS), "feature_map_grid": list(VISUAL_PATCH_GRID),
        "patch_embedding_dim": VISUAL_EMBEDDING_DIM, "patches_per_image": VISUAL_PATCH_COUNT,
        "full_reference_patch_count": 49_152,
        "coreset_method": "seeded Gaussian projection plus greedy farthest-point k-center",
        "coreset_projection_dim": PROJECTION_DIM, "coreset_seed": SEED,
        "coreset_retention_ratio": CORESET_RATIO, "coreset_patch_count": CORESET_COUNT,
        "nearest_neighbor_algorithm": "brute", "nearest_neighbor_metric": "euclidean",
        "patch_score_method": "nearest coreset Euclidean distance",
        "image_score_method": "maximum of 256 patch scores",
        "threshold_method": "97.5th percentile of 48 held-out development-normal image scores",
        "threshold_quantile": bundle.threshold_quantile, "threshold_raw_score": bundle.threshold,
        "display_scale_low": bundle.display_low, "display_scale_high": bundle.display_high,
        "notebook_16_image_level_metrics": {"roc_auc": .948004, "average_precision": .984192, "precision": .933884, "recall": .949580, "f1": .941667, "tn": 24, "fp": 8, "fn": 6, "tp": 113},
        "notebook_16_pixel_level_metrics": {"roc_auc": .971613, "average_precision": .447934, "aupro": None},
        "per_subtype_metrics": {"split_teeth_detection_rate": 1.0, "broken_teeth_detection_rate": .8947368421},
        "false_positive_false_negative_tradeoff": "PatchCore FP=8/FN=6 versus baseline FP=3/FN=22; greater sensitivity with more manual reinspection.",
        "known_limitations": ["One public research category.", "No external factory validation.", "Development-stage threshold.", "No calibrated probability.", "Pixel AP regressed versus the simple baseline."],
        "output_interpretation": "Distance-based visual anomaly score and development decision; not probability or certification.",
        "warning": VISUAL_WARNING, "disclaimer": VISUAL_DISCLAIMER,
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__, "numpy_version": np.__version__,
        "pillow_version": Image.__version__, "sklearn_version": sklearn.__version__,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def train_and_save(*, device: str | torch.device | None = None) -> tuple[Path, Path, dict]:
    np.random.seed(SEED); torch.manual_seed(SEED)
    resolved = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    data = validate_dataset(); reference, development = frozen_split(data["train"])
    model = build_visual_extractor(pretrained=True, device=resolved)
    reference_embeddings, order = extract_paths(model, reference, resolved)
    if order != reference or reference_embeddings.shape != (192, 256, 384):
        raise RuntimeError("Reference extraction violates the frozen contract.")
    full_memory = np.ascontiguousarray(reference_embeddings.reshape(-1, VISUAL_EMBEDDING_DIM))
    selected = projected_kcenter_indices(full_memory)
    if len(selected) != CORESET_COUNT:
        raise RuntimeError("Frozen coreset count was not reproduced.")
    coreset = np.ascontiguousarray(full_memory[selected], dtype=np.float32)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    provisional = VisualModelBundle(state, coreset, 1.0, 0.0, 1.0, {})
    runtime = create_visual_runtime(provisional, device=resolved)
    development_embeddings, development_order = extract_paths(runtime.extractor, development, resolved)
    if development_order != development:
        raise RuntimeError("Development extraction order changed.")
    scores, maps = [], []
    for patches in development_embeddings:
        _, raw_map, score = score_patch_embeddings(runtime, patches)
        scores.append(score); maps.append(raw_map)
    threshold = float(np.quantile(scores, VISUAL_THRESHOLD_QUANTILE))
    display_low = float(np.quantile(np.stack(maps), .05))
    if abs(threshold - NOTEBOOK17_THRESHOLD) >= 5e-6:
        raise RuntimeError(f"Notebook 17 threshold parity failed: {threshold:.9f}")
    config = {"seed": SEED, "input_size": VISUAL_INPUT_SIZE, "feature_layers": VISUAL_FEATURE_LAYERS,
              "patch_grid": VISUAL_PATCH_GRID, "embedding_dim": VISUAL_EMBEDDING_DIM,
              "projection_dim": PROJECTION_DIM, "coreset_ratio": CORESET_RATIO,
              "image_score": "max_patch_distance", "alignment": "Notebook 16/17 exact"}
    bundle = VisualModelBundle(state, coreset, threshold, display_low, threshold, config)
    metadata = build_metadata(bundle, [p.name for p in data["defect_dirs"]])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return MODEL_PATH, METADATA_PATH, metadata


def main() -> None:
    started = perf_counter(); model_path, metadata_path, metadata = train_and_save()
    print(f"threshold={metadata['threshold_raw_score']:.9f}")
    print(f"display_low={metadata['display_scale_low']:.9f}")
    print(f"model={model_path.relative_to(PROJECT_ROOT)} ({model_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"metadata={metadata_path.relative_to(PROJECT_ROOT)}")
    print(f"build_seconds={perf_counter()-started:.3f}")


if __name__ == "__main__": main()
