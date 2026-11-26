"""Restore only approved artifacts; source files and trained weights are not changed by tests."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from Task4.predict import find_bundle
from Task4.restore_artifacts import restore


def test_bundle_discovery_from_project_notebooks_and_standalone(tmp_path):
    repo = tmp_path / "repo"
    task = repo / "Task4"
    (task / "config").mkdir(parents=True)
    (task / "checkpoints").mkdir()
    (task / "config/ensemble.json").write_text("{}")
    fallback = repo / "dist/Task4_submission"
    (fallback / "config").mkdir(parents=True)
    (fallback / "checkpoints").mkdir()
    (fallback / "config/ensemble.json").write_text("{}")
    assert find_bundle(repo) == task
    assert find_bundle(task / "notebooks") == task
    assert find_bundle(fallback / "notebooks") == fallback


@pytest.mark.parametrize("location", ["root", "notebooks", "standalone"])
def test_training_bootstrap_uses_correct_project_and_bundle(tmp_path, monkeypatch, location):
    from Task4.scripts.prepare_task4_submission import BOOTSTRAP
    repo = tmp_path / "repo"
    task = repo / "Task4"
    (task / "config").mkdir(parents=True)
    (task / "notebooks").mkdir()
    (task / "config/ensemble.json").write_text("{}")
    (task / "dinov2_pipeline.py").write_text("# discovery marker")
    flat = repo / "dist/Task4_submission"
    (flat / "config").mkdir(parents=True)
    (flat / "notebooks").mkdir()
    (flat / "config/ensemble.json").write_text("{}")
    (flat / "dinov2_pipeline.py").write_text("# discovery marker")
    start = {"root": repo, "notebooks": task / "notebooks", "standalone": flat / "notebooks"}[location]
    monkeypatch.chdir(start)
    scope = {}
    exec(BOOTSTRAP, scope)
    assert scope["PROJECT_ROOT"] == (flat if location == "standalone" else repo)
    assert scope["BUNDLE_DIR"] == (flat if location == "standalone" else task)


def artifact_fixture(tmp_path, *, invalid_path=None):
    task = tmp_path / "Task4"
    (task / "config").mkdir(parents=True)
    (task / "predict.py").write_text("source must survive")
    archive = tmp_path / "Task4_submission.zip"
    content = b"fake checkpoint for extraction only"
    relative = invalid_path or "checkpoints/dinov2_base.pt"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("Task4_submission/" + relative, content)
        zipped.writestr("Task4_submission/predict.py", "do not extract this")
    manifest = {"schema_version": 1,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": {relative: hashlib.sha256(content).hexdigest()}}
    (task / "config/artifacts.json").write_text(json.dumps(manifest))
    return task, archive, content


def test_restore_is_idempotent_and_preserves_source(tmp_path):
    task, archive, content = artifact_fixture(tmp_path)
    assert restore(archive, task) == {"restored": 1, "already_present": 0}
    assert (task / "checkpoints/dinov2_base.pt").read_bytes() == content
    assert (task / "predict.py").read_text() == "source must survive"
    assert restore(archive, task) == {"restored": 0, "already_present": 1}


def test_restore_preserves_conflicting_local_artifact(tmp_path):
    task, archive, _ = artifact_fixture(tmp_path)
    (task / "checkpoints").mkdir()
    path = task / "checkpoints/dinov2_base.pt"
    path.write_bytes(b"local")
    with pytest.raises(FileExistsError):
        restore(archive, task)
    assert path.read_bytes() == b"local"


@pytest.mark.parametrize("relative", ["predict.py", "checkpoints/../../escape.pt"])
def test_restore_rejects_source_or_escaping_paths(tmp_path, relative):
    task, archive, _ = artifact_fixture(tmp_path, invalid_path=relative)
    with pytest.raises(ValueError):
        restore(archive, task)


def test_restore_rejects_wrong_archive_hash(tmp_path):
    task, archive, _ = artifact_fixture(tmp_path)
    with archive.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="ZIP SHA256"):
        restore(archive, task)
