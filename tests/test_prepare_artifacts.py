import hashlib
import json
from pathlib import Path

from scripts.prepare_artifacts import inspect_artifacts, prepare_artifacts


def write_fixture(tmp_path: Path, *, include_visual: bool = True) -> tuple[Path, dict]:
    models = tmp_path / "models"
    models.mkdir()
    entries = []
    for filename, distribution in [
        ("tabular.joblib", "git"),
        ("factorymind_visual_quality_model_v1.joblib", "local_build"),
        ("factorymind_visual_quality_model_v1.metadata.json", "local_build"),
    ]:
        payload = filename.encode()
        if distribution == "git" or include_visual:
            (models / filename).write_bytes(payload)
        entries.append(
            {
                "filename": filename,
                "distribution": distribution,
                "sha256": hashlib.sha256(payload).hexdigest()
                if distribution == "git"
                else None,
            }
        )
    manifest = {"manifest_version": "test", "artifacts": entries}
    (models / "artifact_manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def test_all_fixture_artifacts_present(tmp_path):
    root, manifest = write_fixture(tmp_path)
    assert inspect_artifacts(root, manifest) == []
    assert prepare_artifacts(root) == []


def test_files_only_validation_succeeds_without_runtime_loading(tmp_path):
    root, manifest = write_fixture(tmp_path)
    assert inspect_artifacts(root, manifest) == []
    assert prepare_artifacts(root, files_only=True) == []


def test_missing_versioned_artifact_is_reported_without_changes(tmp_path):
    root, manifest = write_fixture(tmp_path)
    target = root / "models" / "tabular.joblib"
    target.unlink()
    before = sorted(path.relative_to(root) for path in root.rglob("*"))
    assert inspect_artifacts(root, manifest) == ["missing:tabular.joblib"]
    assert prepare_artifacts(root) == ["missing:tabular.joblib"]
    assert sorted(path.relative_to(root) for path in root.rglob("*")) == before


def test_invalid_checksum_is_reported(tmp_path):
    root, _ = write_fixture(tmp_path)
    (root / "models" / "tabular.joblib").write_bytes(b"changed")
    assert prepare_artifacts(root) == ["checksum:tabular.joblib"]


def test_missing_visual_dataset_returns_nonempty_without_building(tmp_path):
    root, _ = write_fixture(tmp_path, include_visual=False)
    called = False

    def builder(**_kwargs):
        nonlocal called
        called = True

    problems = prepare_artifacts(root, build_missing=True, visual_builder=builder)
    assert len(problems) == 2
    assert not called
    assert not (root / "data").exists()
