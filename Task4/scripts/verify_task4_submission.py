"""Verify relocated CPU inference and notebook review without training or model downloads."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]


def verify(bundle):
    bundle = Path(bundle).resolve()
    work = ROOT / "Task4/.ipynb_checkpoints" / ("Task4_relocated_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    work.parent.mkdir(exist_ok=True)
    assert work.resolve().is_relative_to((ROOT / "Task4/.ipynb_checkpoints").resolve())
    shutil.copytree(bundle, work, ignore=shutil.ignore_patterns("outputs", "dist", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"))
    print("Relocated copy:", work, flush=True)
    check_script = work / "offline_verification.py"
    check_script.write_text('''import json,os,sys
from pathlib import Path
import pandas as pd
import torch
import huggingface_hub
import predict

bundle=Path.cwd().resolve()
source=Path(sys.argv[1]).resolve()
def forbidden(*args,**kwargs):
    raise AssertionError("No training or pretrained downloads allowed during verification")
torch.hub.download_url_to_file=forbidden
huggingface_hub.hf_hub_download=forbidden
predict.pipeline.train_member=forbidden
predict.pipeline.prepare_partitions=forbidden
predict.pipeline.run_development=forbidden

def guard(event,args):
    if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(source) and not path.is_relative_to(bundle):
            raise AssertionError("Unexpected original-project dependency: "+str(path))
        if path.is_relative_to(bundle/"training_assets"):
            raise AssertionError("Inference must not read Train split/reference artifacts")
    if event=="socket.connect":
        raise AssertionError("Network use forbidden during offline inference")
sys.addaudithook(guard)
result=predict.predict_bundle(bundle,bundle/"examples/input.csv",bundle/"examples/images",
                             bundle/"outputs/verified.csv",sample_csv=bundle/"examples/sample.csv",device="cpu")
expected=pd.read_csv(bundle/"examples/expected_predictions.csv",dtype={"id":str})
pd.testing.assert_frame_equal(result.predictions,expected)
print("Relocated CPU predictions match the five archived rows",flush=True)
''', encoding="utf-8")
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, "-X", "utf8", str(check_script), str(ROOT)],
                            cwd=work, env=environment, text=True, encoding="utf-8",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError("Relocated offline inference failed")

    guard_cell = nbformat.v4.new_code_cell('''import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
import huggingface_hub
import dinov2_pipeline as guarded_pipeline
def forbidden(*args,**kwargs):
    raise AssertionError("Verification must not train, read Train partitions, or download weights")
for operation in ("train_member","prepare_partitions","run_development","refit_selected"):
    setattr(guarded_pipeline,operation,forbidden)
torch.hub.download_url_to_file=forbidden
huggingface_hub.hf_hub_download=forbidden
''')
    for name in ("inference", "training"):
        print("Running notebook in relocated copy:", name, flush=True)
        notebook = nbformat.read(work / "notebooks" / f"{name}.ipynb", as_version=4)
        notebook.cells.insert(0, guard_cell)
        client = NotebookClient(notebook, timeout=600, kernel_name="python3",
                                resources={"metadata": {"path": str(work)}})
        client.execute()
        if name == "inference":
            notebook.cells.pop(0)
            # Save actual inference outputs; do not replace the preserved training outputs.
            nbformat.write(notebook, bundle / "notebooks/inference.ipynb")
            nbformat.write(notebook, ROOT / "Task4/notebooks/inference.ipynb")
    record = {
        "relocated_cpu_inference": "passed", "example_rows": 5,
        "predictions_match_archived_csv": True, "original_project_access_blocked": True,
        "train_split_access_blocked_for_cli": True, "pretrained_downloads_blocked": True,
        "training_calls_blocked": True, "inference_notebook_run_all": "passed",
        "training_notebook_review_run_all": "passed", "new_training_performed": False,
        "environment_note": "Existing Python environment; CPU execution tested, fresh CPU-wheel installation not tested.",
    }
    (bundle / "results/verification.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    from prepare_task4_submission import refresh_checksums
    refresh_checksums(bundle)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "Task4")
    verify(parser.parse_args().bundle)
