"""Portable, offline inference for the submitted Task 4 ensemble. Never trains."""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

if __package__:
    from . import dinov2_pipeline as pipeline
else:
    import dinov2_pipeline as pipeline


def contained_path(root, relative):
    """Resolve package-controlled paths without allowing escape from the bundle."""
    value = str(relative).replace("\\", "/")
    parts = PurePosixPath(value)
    root = Path(root).resolve()
    if not value or parts.is_absolute() or ".." in parts.parts or ":" in value:
        raise ValueError(f"Expected a relative path inside the package: {relative}")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes package: {relative}")
    return path


def find_bundle(start=None):
    start = Path(start or Path.cwd()).resolve()
    roots = [start, *start.parents]
    candidates = [candidate for p in roots for candidate in (p, p / "Task4", p / "Task4/dist/Task4_submission", p / "dist/Task4_submission")]
    for root in candidates:
        if (root / "config/ensemble.json").is_file() and (root / "checkpoints").is_dir():
            return root
    raise FileNotFoundError("Cannot find Task4 or Task4_submission; set BUNDLE_DIR or --bundle explicitly")


def verify_bundle(bundle, full=True):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "checksums.json").read_text(encoding="utf-8"))
    if manifest.get("algorithm") != "sha256" or not manifest.get("files"):
        raise ValueError("Invalid checksums.json")
    files = manifest["files"]
    if not full:
        essential = {"predict.py", "dinov2_pipeline.py", "config/ensemble.json"}
        config = json.loads((bundle / "config/ensemble.json").read_text(encoding="utf-8"))
        essential.update(member["checkpoint"] for member in config["members"])
        if essential - files.keys():
            raise ValueError("Required inference files missing from checksums.json")
        files = {name: files[name] for name in essential}
    for relative, expected in files.items():
        path = contained_path(bundle, relative)
        if not path.is_file():
            raise ValueError(f"Package file missing: {relative}. Restore artifacts from the submission ZIP; see ARTIFACTS.md.")
        if pipeline.file_sha256(path) != expected:
            raise ValueError(f"Package checksum mismatch: {relative}")
    return manifest


def load_ensemble(bundle):
    bundle = Path(bundle).resolve()
    config = json.loads((bundle / "config/ensemble.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("class_names") != list(pipeline.CLASS_NAMES):
        raise ValueError("Unsupported ensemble schema or class mapping")
    members = config.get("members", [])
    if not members or len({m["name"] for m in members}) != len(members):
        raise ValueError("Ensemble members must be nonempty and unique")
    if any(not np.isfinite(m["weight"]) or m["weight"] < 0 for m in members):
        raise ValueError("Invalid ensemble weight")
    if not np.isclose(sum(m["weight"] for m in members), 1):
        raise ValueError("Ensemble weights must sum to one")
    for member in members:
        if member["name"] not in pipeline.MODEL_SPECS or not isinstance(member["tta"], bool):
            raise ValueError("Unknown model or invalid TTA flag")
        path = contained_path(bundle, member["checkpoint"])
        if not path.is_file() or pipeline.file_sha256(path) != member["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {member['name']}")
    return config


def read_inputs(csv_path, image_root, sample_path=None):
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if not {"id", "file_name"} <= set(frame) or frame.empty:
        raise ValueError("Input CSV must contain nonempty id,file_name rows")
    frame = frame[["id", "file_name"]].copy()  # never read/use labels for inference
    frame["file_name"] = frame["file_name"].str.replace("\\", "/", regex=False)
    if not frame.id.is_unique or frame.id.eq("").any():
        raise ValueError("Input IDs must be unique and nonempty")
    root = Path(image_root).resolve()
    for filename in frame.file_name:
        path = contained_path(root, filename)
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {filename}")
    if sample_path is None:
        ids = frame.id.to_numpy(dtype=str)
    else:
        sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        if "id" not in sample or not sample.id.is_unique or sample.id.eq("").any():
            raise ValueError("Invalid sample IDs")
        ids = sample.id.to_numpy(dtype=str)
        if len(ids) != len(frame):
            raise ValueError("Input and sample must contain exactly the same IDs")
    order = pd.Index(frame.id).get_indexer(ids)
    if (order < 0).any():
        raise ValueError("Input and sample IDs differ")
    return frame, ids, order


@dataclass
class PredictionResult:
    output_path: Path
    predictions: pd.DataFrame
    metadata: dict


def predict_bundle(bundle, input_csv, image_root, output, *, sample_csv=None,
                   device="auto", batch_size=None, verify=True, overwrite=False, cpu_threads=4):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output == Path(input_csv).resolve() or (sample_csv and output == Path(sample_csv).resolve()):
        raise ValueError("Output cannot overwrite an input CSV")
    if output.is_relative_to(bundle / "results"):
        raise ValueError("Archived results are read-only; write predictions under outputs/ instead")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    if verify:
        verify_bundle(bundle, full=False)
    config = load_ensemble(bundle)
    frame, ids, order = read_inputs(input_csv, image_root, sample_csv)
    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: auto, cpu, cuda")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    # Identity is recorded training provenance, not a dependency on raw Train files.
    context = SimpleNamespace(identity=config["data_identity"])
    total = np.zeros((len(frame), 5), dtype=np.float64)
    for member in config["members"]:
        path = contained_path(bundle, member["checkpoint"])
        model, cfg, recorded_batch = pipeline.load_model(path, context, selected_device)
        cfg = replace(cfg, num_workers=0)
        if list(cfg.image_size) != config["image_size"]:
            raise ValueError("Checkpoint input size does not match ensemble config")
        batch = batch_size or recorded_batch
        print(f"Predicting {member['name']}: {len(frame)} images, device={selected_device}, flip_tta={member['tta']}", flush=True)
        probabilities = pipeline.predict(model, frame, image_root, cfg, batch, selected_device, with_labels=False)
        if member["tta"]:
            flipped = pipeline.predict(model, frame, image_root, cfg, batch, selected_device, flip=True, with_labels=False)
            probabilities = (probabilities + flipped) / 2
        total += member["weight"] * probabilities
        del model
        gc.collect()
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()
    pipeline.validate_probabilities(total, len(frame))
    probabilities = total[order].astype(np.float32)
    predictions = pd.DataFrame({"id": ids, "task4": probabilities.argmax(1).astype(np.int64)})
    encoded = predictions.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if output.exists() and not overwrite and output.read_bytes() != encoded:
        raise FileExistsError("A different output exists; choose another path or --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    pd.testing.assert_frame_equal(pd.read_csv(output, dtype={"id": str}), predictions)
    metadata = {
        "rows": len(predictions), "device": str(selected_device),
        "input_csv_sha256": pipeline.file_sha256(input_csv),
        "csv_sha256": pipeline.file_sha256(output), "members": config["members"],
        "training_provenance": config["data_identity"], "full_train_refit": config["full_train_refit"],
        "development_outer_metrics_before_refit": config["development_outer_metrics_before_refit"],
        "validation_score_of_refit": None, "kaggle_score": config.get("kaggle_score"),
    }
    pipeline.write_json(output.with_suffix(".metadata.json"), metadata)
    return PredictionResult(output, predictions, metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = predict_bundle(args.bundle, args.input_csv, args.image_root, args.output,
                            sample_csv=args.sample_csv, device=args.device, batch_size=args.batch_size,
                            cpu_threads=args.cpu_threads, overwrite=args.overwrite)
    print("Output:", result.output_path)
    print("Rows:", result.metadata["rows"])
    print("SHA256:", result.metadata["csv_sha256"])


if __name__ == "__main__":
    main()
