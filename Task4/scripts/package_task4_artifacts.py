"""Package only Git-excluded Task 4 artifacts. Never trains or downloads weights."""
from pathlib import Path
import argparse
import json
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Task4.restore_artifacts import contained_path, file_sha256


def package(output):
    task = ROOT / "Task4"
    manifest_path = task / "config/artifacts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    for relative, expected in files.items():
        path = contained_path(task, relative)
        if file_sha256(path) != expected:
            raise ValueError(f"Source artifact differs: {relative}")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Preserving existing archive: {output}")
    temporary = output.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=3) as zipped:
        for relative in sorted(files):
            zipped.write(task / relative, "Task4_submission/" + relative)
    with ZipFile(temporary) as zipped:
        assert set(zipped.namelist()) == {"Task4_submission/" + p for p in files}
        assert zipped.testzip() is None
    temporary.replace(output)
    digest = file_sha256(output)
    manifest.update(archive_name=output.name, archive_sha256=digest, download_url=None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".zip.sha256").write_text(digest + "  " + output.name + "\n", encoding="utf-8")
    print(f"Created: {output}\nFiles: {len(files)}\nSHA256: {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Task4/dist/Task4_artifacts.zip")
    package(parser.parse_args().output)
