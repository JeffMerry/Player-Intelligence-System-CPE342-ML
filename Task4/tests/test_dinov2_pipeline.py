"""Offline checks: real random-weight forward passes; no training or downloads."""
import gc
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Task4 import dinov2_pipeline as p


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not download or train")
    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(p, "train_member", forbidden)
    torch.set_num_threads(2)


@pytest.mark.parametrize("name", ["resnet18", "convnext_tiny", "swin_t", "dinov2_small", "dinov2_base"])
def test_real_architecture_forward_and_optimizer_coverage(name):
    cfg = p.PipelineConfig()
    model = p.build_model(name, cfg, pretrained=False).eval()
    with torch.inference_mode():
        result = model(torch.zeros(1, 3, 224, 448))
    assert result.shape == (1, 5)
    assert torch.isfinite(result).all()
    groups = p.optimizer_groups(model, p.MODEL_SPECS[name], cfg)
    parameters = [parameter for group in groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert {id(parameter) for parameter in parameters} == {id(parameter) for parameter in model.parameters()}
    if name.startswith("dinov2_"):
        assert model.backbone.num_prefix_tokens == 5
        assert model.head.in_features == model.backbone.embed_dim * 10
        backbone_lrs = [g["lr"] for g in groups if g["backbone"]]
        assert min(backbone_lrs) < max(backbone_lrs)
    del model, result
    gc.collect()


def test_large_is_opt_in():
    with pytest.raises(ValueError, match="opt-in"):
        p.PipelineConfig(model_names=("resnet18", "dinov2_large")).validate()
    with pytest.raises(ValueError, match="allow_large"):
        p.build_model("dinov2_large", p.PipelineConfig(), pretrained=False)


@pytest.fixture
def data(tmp_path):
    raw = tmp_path / "data"
    (raw / "train").mkdir(parents=True)
    (raw / "test").mkdir()
    rows = []
    for i in range(120):
        name = f"train_{i}.jpg"
        Image.new("RGB", (23, 12), (i, 0, 0)).save(raw / "train" / name)
        rows.append(dict(id=f"PLY{i:05d}", file_name=name, label=str(i % 5)))
    train = pd.DataFrame(rows)
    test = pd.DataFrame([dict(id=f"ANS{i:05d}", file_name=f"test_{i}.jpg", label="DO_NOT_USE") for i in range(10)])
    for name in test.file_name:
        Image.new("RGB", (23, 12)).save(raw / "test" / name)
    train.to_csv(raw / "train.csv", index=False)
    test.to_csv(raw / "test.csv", index=False)
    manifest = train.copy()
    manifest["duplicate_group"] = [str(i // 2) for i in range(len(train))]
    manifest["subset"] = ["validation" if (i // 10) % 5 == 0 else "fit" for i in range(len(train))]
    split = tmp_path / "split.csv"
    manifest.to_csv(split, index=False)
    meta = tmp_path / "split_metadata.json"
    meta.write_text(json.dumps({
        "split_manifest_sha256": p.file_sha256(split),
        "train_csv_sha256": p.file_sha256(raw / "train.csv"),
        "fit_images": 90, "validation_images": 30,
    }))
    return p.prepare_partitions(raw, split, meta, p.PipelineConfig())


def test_three_way_group_separation_and_test_labels_discarded(data):
    assert "label" not in data.test
    frames = [data.fit, data.selection, data.outer]
    assert sum(map(len, frames)) == len(data.train)
    for left, right in __import__("itertools").combinations(frames, 2):
        assert set(left.duplicate_group).isdisjoint(right.duplicate_group)


def test_transforms_are_deterministic_for_validation(data):
    cfg = p.PipelineConfig()
    dataset = p.ImageDataset(data.outer, data.data_dir / "train", p.image_transform(cfg))
    a, y = dataset[0]
    b, z = dataset[0]
    assert torch.equal(a, b) and y == z
    assert a.shape == (3, 224, 448)


def test_bootstrap_pairs_identical_methods_by_group():
    labels = np.tile(np.arange(5), 4)
    probabilities = np.eye(5)[labels] * .9 + .02
    result = p.paired_group_bootstrap(probabilities, probabilities, labels,
                                      np.repeat(np.arange(10), 2), samples=100)
    assert result["p_gain_nonpositive"] == 1
    assert result["gain_ci95"] == [0, 0]


def test_selection_uses_equal_weights_and_flip_threshold():
    labels = np.tile(np.arange(5), 4)
    good = np.eye(5)[labels] * .9 + .02
    bad = np.roll(good, 1, axis=1)
    selected, candidates = p.choose_inner_recipe({
        "resnet18": {"plain": good, "flip": bad},
        "dinov2_small": {"plain": good, "flip": good},
    }, labels, p.PipelineConfig())
    assert selected["macro_f1"] == 1
    assert len(selected["members"]) == 1
    assert all(not m["tta"] for c in candidates for m in c["members"])
    for candidate in candidates:
        assert np.isclose(sum(m["weight"] for m in candidate["members"]), 1)


def test_export_aligns_sample_ids_and_refuses_failed_gate(data, tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    p.write_json(run / "experiment.json", {"data_identity": data.identity})
    weights = run / "fake.pt"
    weights.write_bytes(b"test checkpoint identity; no model loaded")
    recipe = {"passes": False, "development_checkpoints": {"resnet18": str(weights)},
              "selected": {"members": [{"name": "resnet18", "weight": 1., "tta": False}]},
              "outer_candidate": {"macro_f1": .5}}
    p.write_json(run / "selection.json", recipe)
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"id": data.test.id.iloc[::-1].to_numpy()}).to_csv(sample, index=False)
    with pytest.raises(ValueError, match="gate"):
        p.export_test(run, data, sample, tmp_path / "outputs", device="cpu")
    with pytest.raises(ValueError, match="gate"):
        p.refit_selected(run, data, device="cpu")
    recipe["passes"] = True
    p.write_json(run / "selection.json", recipe)
    monkeypatch.setattr(p, "load_model", lambda *args: (nn_identity(), p.PipelineConfig(), 2))
    probability = np.eye(5)[np.arange(10) % 5] * .9 + .02
    monkeypatch.setattr(p, "predict", lambda *args, **kwargs: probability.copy())
    output = p.export_test(run, data, sample, tmp_path / "outputs", device="cpu")
    result = pd.read_csv(output)
    assert result.id.tolist() == data.test.id.iloc[::-1].tolist()
    assert result.task4.tolist() == (np.arange(10) % 5)[::-1].tolist()
    info = json.loads(output.with_name("prediction_metadata.json").read_text())
    assert info["kaggle_score"] is None and info["validation_score_of_refit"] is None
    assert info["csv_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def nn_identity():
    return torch.nn.Identity()


def test_checkpoint_roundtrip_without_pretrained_download(data, tmp_path):
    cfg = p.PipelineConfig(model_names=("resnet18",))
    model = p.build_model("resnet18", cfg, pretrained=False).eval()
    sample = torch.zeros(1, 3, 224, 448)
    with torch.inference_mode():
        expected = model(sample)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "model_name": "resnet18",
                "spec": asdict(p.MODEL_SPECS["resnet18"]), "config": asdict(cfg),
                "class_names": list(p.CLASS_NAMES), "data_identity": data.identity, "micro_batch": 1}, path)
    restored, _, _ = p.load_model(path, data, torch.device("cpu"))
    with torch.inference_mode():
        torch.testing.assert_close(expected, restored(sample))


def test_notebook_default_flags_do_not_train_refit_or_export(data, tmp_path, monkeypatch):
    import nbformat
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/training.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    audit_end = next(i for i, c in enumerate(notebook.cells) if c.id == "3eb91fcc")
    monkeypatch.setattr(p, "prepare_partitions", lambda *args: data)

    def forbidden(*args, **kwargs):
        raise AssertionError("Default notebook must not train, refit or export")

    for operation in ("run_development", "refit_selected", "export_test", "build_model"):
        monkeypatch.setattr(p, operation, forbidden)
    import matplotlib
    matplotlib.use("Agg")
    scope = {"PROJECT_ROOT": Path(__file__).resolve().parents[2],
             "RESULTS_DIR": tmp_path / "results", "BUNDLE_DIR": None, "Path": Path, "pd": pd, "json": json,
             "display": lambda *args: None, "show_recorded_outputs": lambda *args: None}
    for i, cell in enumerate(notebook.cells[audit_end + 1:], start=audit_end + 1):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"notebook_cell_{i}", "exec"), scope)
    assert scope["advanced_run_dir"] is None
    assert scope["selection_report"] is None
    assert not scope["RUN_DEVELOPMENT"] and not scope["RUN_FULL_REFIT"]
    assert not scope["RUN_TEST_INFERENCE"] and not scope["RUN_MODEL_SMOKE"]
