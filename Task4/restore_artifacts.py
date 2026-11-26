"""Restore only large/data artifacts from the supplied ZIP; never overwrite source code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root, relative):
    value = str(relative).replace("\\", "/")
    parts = PurePosixPath(value)
    root = Path(root).resolve()
    if not value or parts.is_absolute() or ".." in parts.parts or ":" in value:
        raise ValueError("Artifact path must stay inside Task4")
    destination = (root / value).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Artifact path escapes Task4")
    return destination


def restore(archive, task_dir=None):
    task_dir = Path(task_dir or Path(__file__).resolve().parent).resolve()
    archive = Path(archive).resolve()
    manifest = json.loads((task_dir / "config/artifacts.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported artifact manifest")
    if file_sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("ZIP SHA256 differs from config/artifacts.json")
    restored = skipped = 0
    with ZipFile(archive) as zipped:
        # Validate every target and archive entry before extracting anything.
        planned = []
        for relative, expected in manifest["files"].items():
            allowed = (relative.startswith(("checkpoints/", "training_assets/", "examples/images/"))
                       or (relative.startswith("examples/") and relative.endswith(".csv"))
                       or relative == "results/task4.csv"
                       or (relative.startswith("results/") and relative.endswith(".npz")))
            if not allowed or relative.endswith("README.md"):
                raise ValueError(f"Not a restorable artifact: {relative}")
            destination = contained_path(task_dir, relative)
            member = zipped.getinfo("Task4_submission/" + relative)
            if member.is_dir() or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Directories/symlinks cannot be restored as files")
            if destination.exists():
                if file_sha256(destination) != expected:
                    raise FileExistsError(f"Different local artifact exists; preserving it: {relative}")
                skipped += 1
            else:
                planned.append((relative, expected, destination, member))
        for relative, expected, destination, member in planned:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".restore.tmp")
            digest = hashlib.sha256()
            with zipped.open(member) as source, temporary.open("wb") as output:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Extracted artifact SHA256 mismatch: {relative}")
            temporary.replace(destination)
            restored += 1
            print("Restored:", relative, flush=True)
    return {"restored": restored, "already_present": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(restore(args.archive))


if __name__ == "__main__":
    main()
