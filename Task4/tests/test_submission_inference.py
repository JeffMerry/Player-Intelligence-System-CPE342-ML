"""Portable inference tests do not use Train, downloaded weights, or training."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Task4 import predict as submission


@pytest.fixture
def package(tmp_path, monkeypatch):
    bundle = tmp_path / "relocated_bundle"
    (bundle / "checkpoints").mkdir(parents=True)
    (bundle / "config").mkdir()
    (bundle / "examples/images").mkdir(parents=True)
    weights = bundle / "checkpoints/model.pt"
    weights.write_bytes(b"identity only: mock loader must not train or download")
    config = {
        "schema_version": 1, "class_names": list(submission.pipeline.CLASS_NAMES),
        "members": [{"name": "resnet18", "weight": 1., "tta": True,
                     "checkpoint": "checkpoints/model.pt",
                     "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}],
        "image_size": [224, 448], "data_identity": {"training": "recorded provenance only"},
        "full_train_refit": True, "development_outer_metrics_before_refit": {"macro_f1": .9},
        "kaggle_score": None,
    }
    (bundle / "config/ensemble.json").write_text(json.dumps(config))
    rows = pd.DataFrame({"id": ["ANS00001", "ANS00002"], "file_name": ["one.jpg", "two.jpg"],
                         "label": ["ignored", "ignored"]})
    rows.to_csv(bundle / "examples/input.csv", index=False)
    pd.DataFrame({"id": rows.id.iloc[::-1].to_numpy()}).to_csv(bundle / "examples/sample.csv", index=False)
    for name in rows.file_name:
        Image.new("RGB", (23, 12)).save(bundle / "examples/images" / name)
    files = {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in bundle.rglob("*") if p.is_file()}
    for name in ("predict.py", "dinov2_pipeline.py"):
        (bundle / name).write_text("test placeholder", encoding="utf-8")
        files[name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
    (bundle / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": files}))

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must not train or prepare Train partitions")
    monkeypatch.setattr(submission.pipeline, "train_member", forbidden)
    monkeypatch.setattr(submission.pipeline, "prepare_partitions", forbidden)
    monkeypatch.setattr(submission.pipeline, "run_development", forbidden)

    def load(path, context, device):
        assert context.identity == config["data_identity"]
        return nn.Identity(), submission.pipeline.PipelineConfig(), 1
    monkeypatch.setattr(submission.pipeline, "load_model", load)

    def predict(model, frame, *args, **kwargs):
        assert frame.columns.tolist() == ["id", "file_name"]
        assert kwargs["with_labels"] is False
        return np.array([[.9, .025, .025, .025, .025], [.025, .9, .025, .025, .025]], dtype=np.float32)
    monkeypatch.setattr(submission.pipeline, "predict", predict)
    return bundle


def test_portable_inference_without_train_and_sample_reordering(package):
    result = submission.predict_bundle(
        package, package / "examples/input.csv", package / "examples/images",
        package / "outputs/predictions.csv", sample_csv=package / "examples/sample.csv", device="cpu")
    assert result.predictions.id.tolist() == ["ANS00002", "ANS00001"]
    assert result.predictions.task4.tolist() == [1, 0]
    assert result.metadata["validation_score_of_refit"] is None
    assert result.metadata["csv_sha256"] == hashlib.sha256(result.output_path.read_bytes()).hexdigest()
    assert not (package / "data").exists()


def test_tampered_package_stops_before_prediction(package):
    (package / "checkpoints/model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        submission.verify_bundle(package)


def test_inference_does_not_require_training_assets(package):
    manifest_path = package / "checksums.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["training_assets/split_manifest.csv"] = "archived but deliberately absent"
    manifest_path.write_text(json.dumps(manifest))
    submission.verify_bundle(package, full=False)
    with pytest.raises(ValueError):
        submission.verify_bundle(package, full=True)


@pytest.mark.parametrize("relative", ["../model.pt", "/absolute/model.pt", "C:\\secret.pt", ""])
def test_unsafe_package_paths_rejected(package, relative):
    with pytest.raises(ValueError):
        submission.contained_path(package, relative)


def test_archived_results_cannot_be_overwritten(package):
    with pytest.raises(ValueError, match="read-only"):
        submission.predict_bundle(package, package / "examples/input.csv", package / "examples/images",
                                  package / "results/task4.csv", device="cpu")


def test_duplicate_ids_and_missing_images_rejected(package):
    csv = package / "duplicates.csv"
    pd.DataFrame({"id": ["same", "same"], "file_name": ["one.jpg", "two.jpg"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="unique"):
        submission.read_inputs(csv, package / "examples/images")
    (package / "examples/images/one.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        submission.read_inputs(package / "examples/input.csv", package / "examples/images")
