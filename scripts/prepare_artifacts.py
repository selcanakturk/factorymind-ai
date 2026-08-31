"""Verify release artifacts and explicitly build the restricted visual bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "models" / "artifact_manifest.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Artifact manifest must contain a non-empty artifacts list.")
    return manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_artifacts(root: Path, manifest: dict) -> list[str]:
    """Return actionable problems without changing the artifact directory."""
    problems: list[str] = []
    for item in manifest["artifacts"]:
        path = root / "models" / item["filename"]
        if not path.is_file():
            problems.append(f"missing:{item['filename']}")
            continue
        expected = item.get("sha256")
        if expected and sha256(path) != expected:
            problems.append(f"checksum:{item['filename']}")
    return problems


def prepare_artifacts(
    root: Path = PROJECT_ROOT,
    *,
    build_missing: bool = False,
    visual_builder: Callable[..., object] | None = None,
) -> list[str]:
    manifest = load_manifest(root / "models" / "artifact_manifest.json")
    problems = inspect_artifacts(root, manifest)
    missing_visual = [
        problem for problem in problems
        if problem.startswith("missing:factorymind_visual_quality_")
    ]
    other_problems = [problem for problem in problems if problem not in missing_visual]

    if other_problems:
        for problem in other_problems:
            kind, filename = problem.split(":", 1)
            if kind == "missing":
                print(f"ERROR: versioned runtime artifact is missing: models/{filename}")
            else:
                print(f"ERROR: checksum mismatch for versioned artifact: models/{filename}")
        return other_problems + missing_visual

    if missing_visual:
        dataset = root / "data" / "raw" / "mvtec_ad" / "zipper"
        if not dataset.is_dir():
            print(
                "ERROR: visual artifacts require the official MVTec AD zipper dataset at "
                "data/raw/mvtec_ad/zipper. Obtain it under the MVTec AD license; this "
                "script does not download or modify the dataset."
            )
            return missing_visual
        try:
            from src.visual_train import validate_dataset

            validate_dataset(dataset)
        except (OSError, ValueError) as exc:
            print(
                "ERROR: MVTec AD zipper dataset validation failed at "
                f"data/raw/mvtec_ad/zipper: {exc}"
            )
            return missing_visual
        if not build_missing:
            print(
                "ERROR: visual artifacts are absent. Re-run with --build-missing to "
                "explicitly start the deterministic CPU build (src.visual_train)."
            )
            return missing_visual
        print("Building missing MVTec-derived visual artifacts on CPU; this may take time.")
        if visual_builder is None:
            from src.visual_train import train_and_save

            visual_builder = train_and_save
        visual_builder(device="cpu")

    remaining = inspect_artifacts(root, manifest)
    if remaining:
        return remaining

    # Preserve the backend's strict, authoritative artifact compatibility checks.
    if root == PROJECT_ROOT:
        from backend.app.core.model_loader import load_model_resources

        load_model_resources()
    print("All required FactoryMind runtime artifacts are present and valid.")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-missing",
        action="store_true",
        help="Explicitly build missing local-build artifacts from user-supplied datasets.",
    )
    args = parser.parse_args()
    return 1 if prepare_artifacts(build_missing=args.build_missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
