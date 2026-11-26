"""Assemble a portable teacher submission from completed Task 4 artifacts. Never trains."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import io
import json
import shutil
import sys
import textwrap
import zipfile
from datetime import datetime
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Task4 import dinov2_pipeline as pipeline

RUN_ID = "dinov2_ensemble_20261125_012806_431894"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.write_json(path, value)


def copy_file(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def replace_assignments(source, replacements):
    lines = source.splitlines(keepends=True)
    edits = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    edits.append((node.lineno - 1, node.end_lineno, replacements[target.id] + "\n"))
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    return "".join(lines)


BOOTSTRAP = '''from pathlib import Path
import sys
import json
import hashlib
from importlib.metadata import version
import numpy as np
import pandas as pd
from IPython.display import display

# False = เปิดดู outputs เดิมและสรุปจาก artifacts โดยไม่อ่านภาพ Train ทั้งชุด
RUN_DATA_PREPARATION = False
cwd = Path.cwd().resolve()
roots = [cwd, *cwd.parents]
PROJECT_ROOT = None
for p in roots:
    if (p / "Task4/dinov2_pipeline.py").is_file():
        PROJECT_ROOT = p
        break
    if (p / "dinov2_pipeline.py").is_file() and (p / "config/ensemble.json").is_file():
        PROJECT_ROOT = p.parent if p.name == "Task4" else p
        break
if PROJECT_ROOT is None:
    raise FileNotFoundError("เปิด notebook จากโฟลเดอร์โปรเจกต์หรือ Task4_submission")
candidates = [candidate for p in roots for candidate in (p, p / "Task4", p / "Task4/dist/Task4_submission", p / "dist/Task4_submission")]
BUNDLE_DIR = next((p for p in candidates if (p / "config/ensemble.json").is_file()), None)
RESULTS_DIR = (BUNDLE_DIR if BUNDLE_DIR else PROJECT_ROOT) / "results"
DATA_DIR = PROJECT_ROOT / "data/raw/Task4"
if RUN_DATA_PREPARATION:
    for name in ["train.csv", "test.csv", "train", "test"]:
        assert (DATA_DIR / name).exists(), f"เตรียมข้อมูลตาม README ก่อน: {name}"
print("Project:", PROJECT_ROOT)
print("Data audit enabled:", RUN_DATA_PREPARATION)
print("ผลที่เห็นใน cells ด้านล่างเป็นผลบันทึกจากรอบเดิม ไม่ใช่การฝึกใหม่")

archive_path = RESULTS_DIR / "notebook_outputs.json"
recorded_outputs = json.loads(archive_path.read_text(encoding="utf-8")) if archive_path.is_file() else {}
def show_recorded_outputs(cell_id):
    for output in recorded_outputs.get(cell_id, []):
        if output["output_type"] == "stream":
            print(output["text"], end="", file=sys.stderr if output.get("name") == "stderr" else sys.stdout)
        elif output["output_type"] in {"display_data", "execute_result"}:
            display(output["data"], raw=True, metadata=output.get("metadata", {}))
        elif output["output_type"] == "error":
            print("Historical output:", output["ename"], output["evalue"])
'''


def build_training_notebook(original, bundle):
    notebook = copy.deepcopy(original)
    summary_start = next((i for i, c in enumerate(notebook.cells)
                          if c.cell_type == "markdown" and c.source.startswith("## สรุปจาก artifacts ของรอบที่ส่ง")), len(notebook.cells))
    notebook.cells = notebook.cells[:summary_start]
    audit_end = next(i for i, c in enumerate(notebook.cells) if c.id == "3eb91fcc")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if cell.id == "71573146":
            cell.source = BOOTSTRAP
        elif index <= audit_end:
            body = cell.source
            if body.startswith("if RUN_DATA_PREPARATION:\n"):
                body = textwrap.dedent(body.split("\nelse:\n", 1)[0].split("\n", 1)[1])
            cell.source = ("if RUN_DATA_PREPARATION:\n" + textwrap.indent(body, "    ")
                           + f'\nelse:\n    show_recorded_outputs("{cell.id}")\n')
        elif "RUN_DEVELOPMENT =" in cell.source:
            cell.source = replace_assignments(cell.source, {
                "RUN_DEVELOPMENT": "RUN_DEVELOPMENT = False",
                "RUN_FULL_REFIT": "RUN_FULL_REFIT = False",
                "RUN_TEST_INFERENCE": "RUN_TEST_INFERENCE = False",
                "RUN_MODEL_SMOKE": "RUN_MODEL_SMOKE = False",
                "USE_REFIT_FOR_TEST": "USE_REFIT_FOR_TEST = False",
                "EXISTING_RUN_DIR": "EXISTING_RUN_DIR = None",
                "SPLIT_PATH": 'SPLIT_PATH = (BUNDLE_DIR / "training_assets/split_manifest.csv" if BUNDLE_DIR and (BUNDLE_DIR / "training_assets/split_manifest.csv").is_file() else PROJECT_ROOT / "reports/experiments/task4/split_manifest.csv")',
                "SAMPLE_PATH": 'SAMPLE_PATH = (BUNDLE_DIR / "training_assets/sample_submission.csv" if BUNDLE_DIR and (BUNDLE_DIR / "training_assets/sample_submission.csv").is_file() else PROJECT_ROOT / "data/raw/sample_submission.csv")',
                "REFERENCE_RUN": 'REFERENCE_RUN = (BUNDLE_DIR / "training_assets/reference_baseline" if BUNDLE_DIR and (BUNDLE_DIR / "training_assets/reference_baseline").is_dir() else OUTPUT_ROOT / "resnet18_20261124_155809_039967")',
            })
            old_import = '''if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Task4 import dinov2_pipeline as advanced'''
            new_import = '''if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if (PROJECT_ROOT / "Task4/dinov2_pipeline.py").is_file():
    from Task4 import dinov2_pipeline as advanced
else:
    import dinov2_pipeline as advanced'''
            if old_import in cell.source:
                cell.source = cell.source.replace(old_import, new_import)
        elif cell.source.startswith("partitions = advanced.prepare_partitions"):
            cell.source = ('partitions = None\n'
                           'if RUN_DEVELOPMENT or RUN_FULL_REFIT or RUN_TEST_INFERENCE or RUN_MODEL_SMOKE:\n'
                           + textwrap.indent(cell.source, "    ")
                           + '\nelse:\n    print("Review mode: ไม่โหลด Train; ดูสรุปผลจาก artifacts ท้าย notebook")\n')
        elif cell.source.startswith("advanced_run_dir ="):
            if "    if partitions is not None:" not in cell.source:
                cell.source = cell.source.replace(
                    "    advanced.assert_run_identity(advanced_run_dir, partitions)",
                    "    if partitions is not None:\n        advanced.assert_run_identity(advanced_run_dir, partitions)")
            if "show_recorded_outputs(" not in cell.source:
                cell.source += f'\nif not RUN_DEVELOPMENT and advanced_run_dir is None:\n    show_recorded_outputs("{cell.id}")\n'
        elif cell.source.startswith("if RUN_FULL_REFIT:") and "show_recorded_outputs(" not in cell.source:
            cell.source += f'\nif not RUN_FULL_REFIT:\n    show_recorded_outputs("{cell.id}")\n'
        elif cell.source.startswith("if RUN_TEST_INFERENCE:") and "show_recorded_outputs(" not in cell.source:
            cell.source += f'\nif not RUN_TEST_INFERENCE:\n    show_recorded_outputs("{cell.id}")\n'

    notebook.cells[0].source = """# Task 4 — Training และผลการทดลองที่บันทึกไว้

Notebook นี้อธิบาย EDA → group split → inner selection → development training → full-train refit
**Outputs เดิมถูกเก็บไว้ครบ** เปิดอ่านรูป กราฟ ตาราง และ logs ได้โดยไม่ฝึกใหม่
สำหรับลองทำนายจากโมเดลสำเร็จแล้ว ให้เปิด `inference.ipynb`
"""
    notebook.cells[1].source = """## โหมดเริ่มต้น: เปิดดูผลโดยไม่ฝึก

`RUN_DATA_PREPARATION`, `RUN_DEVELOPMENT`, `RUN_FULL_REFIT`, `RUN_TEST_INFERENCE` และ `RUN_MODEL_SMOKE` เริ่มเป็น False
Run All จะข้ามขั้นอ่านภาพ/ฝึก และอ่านสรุปจาก results ที่จัดให้ ไม่เริ่มฝึกซ้ำ
รูป/outputs ในส่วนเดิมเป็นประวัติจากเครื่องผู้จัดทำ อาจแสดง path หรือ execution counts เดิม
ตัวแปรใน kernel ไม่ถูกบันทึก ต้องรันส่วนตั้งค่าใหม่ก่อนใช้โค้ดต่อ

ผลจริง: development outer Macro F1 **0.9882345069508871**; inner selection **0.9887623315174263**
สูตรสุดท้าย: DINOv2-Base 50% + ConvNeXt-Tiny 50% โดย ConvNeXt ใช้ plain+flip TTA
ทั้งสองตัว refit ครบ 8 epochs บน Train 31,546 ภาพแล้ว ผล Test 25,889 แถวอยู่ใน `results/task4.csv`
คะแนนหลัง refit แบบอิสระและคะแนน Kaggle ยังไม่ทราบ
"""
    notebook.cells[2].source = """### Environment และการทำซ้ำ

ดู README สำหรับติดตั้ง `requirements.txt` ไฟล์เดียว และคำสั่งเลือก PyTorch CPU/CUDA ใน virtual environment ใหม่
การเปิดอ่าน outputs เดิมไม่ต้องฝึก ส่วนการทำซ้ำต้องเตรียม CSV และภาพทั้งหมดตาม `data/README.md`
เมื่อจะฝึกจริงเปิด RUN_DEVELOPMENT; เปิด RUN_DATA_PREPARATION เฉพาะเมื่อจะตรวจภาพและสร้าง split ใหม่
มี split เดิมใน training_assets ให้ใช้เพื่อเทียบผลบนแถวเดียวกัน การฝึกซ้ำใช้เวลาหลายชั่วโมงและต้องดาวน์โหลด pretrained weights ครั้งแรก
"""
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace(
                "- แม่แบบนี้ยังไม่ได้ฝึก ต้องรายงานคะแนนจากผลการทดลองจริงของแต่ละรอบ",
                "- รุ่นนี้ฝึกและ refit แล้ว: outer development F1 0.988235 เป็นของ development models ไม่ใช่คะแนนหลัง refitหรือ Kaggle")
    notebook.metadata["task4_modern_pipeline"] = {
        "development_completed": True, "full_refit_completed": True,
        "test_export_completed": True, "outputs_preserved": True,
        "notebook_retrained_during_packaging": False,
    }

    summary_source = '''summary_path = RESULTS_DIR / "validation_summary.json"
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    display(pd.DataFrame(summary["scores"]))
    display(pd.DataFrame(summary["members"]))
    print("Refit epochs:", summary["refit_epochs"])
    print("Refit Train images:", summary["refit_rows"])
    print("Final CSV rows:", summary["test_rows"])
    print("คะแนน refit/Kaggle ยังไม่ทราบ; ไม่มีการฝึกซ้ำใน cell นี้")
else:
    print("ดาวน์โหลดชุด Task4_submission สำหรับรันอ่าน artifacts; outputs เดิมยังเปิดดูได้")
'''
    summary_cell = nbformat.v4.new_code_cell(summary_source, id="task4-saved-summary")
    summary = json.loads((bundle / "results/validation_summary.json").read_text(encoding="utf-8"))
    table = pd.DataFrame(summary["scores"])
    summary_cell.outputs = [nbformat.v4.new_output("display_data", data={
        "text/html": table.to_html(index=False), "text/plain": table.to_string(index=False)})]

    plot_source = '''import matplotlib.pyplot as plt
history_dir = RESULTS_DIR / "history"
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for phase, ax in zip(["development", "full_refit"], axes):
    for path in sorted(history_dir.glob(phase + "_*.csv")):
        history = pd.read_csv(path)
        ax.plot(history["epoch"], history["train_loss"], label=path.stem.removeprefix(phase + "_"))
    ax.set_title(phase + " — recorded training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.grid(alpha=0.3)
    if ax.lines:
        ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
'''
    plot_cell = nbformat.v4.new_code_cell(plot_source, id="task4-saved-history")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for phase, ax in zip(("development", "full_refit"), axes):
        for path in sorted((bundle / "results/history").glob(phase + "_*.csv")):
            history = pd.read_csv(path)
            ax.plot(history.epoch, history.train_loss, label=path.stem.removeprefix(phase + "_"))
        ax.set(title=phase + " — recorded training loss", xlabel="Epoch", ylabel="Train loss")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    plot_cell.outputs = [nbformat.v4.new_output("display_data", data={
        "image/png": base64.b64encode(buffer.getvalue()).decode(),
        "text/plain": "Training histories read from saved CSV files; no retraining."})]
    notebook.cells += [nbformat.v4.new_markdown_cell(
        "## สรุปจาก artifacts ของรอบที่ส่ง\nตารางและกราฟส่วนนี้สร้างจากไฟล์ผลจริงระหว่างจัดชุดส่ง ไม่ได้ฝึกใหม่"),
        summary_cell, plot_cell]

    original_by_id = {cell.id: cell for cell in original.cells if not cell.id.startswith("task4-saved-")}
    for cell in notebook.cells:
        if cell.id in original_by_id and cell.cell_type == "code":
            assert cell.outputs == original_by_id[cell.id].outputs
            assert cell.execution_count == original_by_id[cell.id].execution_count
    return notebook


def inference_notebook():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    n = nbformat.v4.new_notebook()
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    n.cells = [md("""# Task 4 — ลองทำนายด้วยโมเดลที่ฝึกเสร็จแล้ว

Notebook นี้ **ไม่มี training loop** และไม่ต้องมีข้อมูล Train/split/pretrained cache
ใช้ checkpoints หลัง refit ของ DINOv2-Base 50% + ConvNeXt-Tiny 50% (plain+flip สำหรับ ConvNeXt)
Outputs ที่บันทึกไว้ด้านล่างเป็นตัวอย่างจากชุดส่ง เปิดดูได้ทันที หากรันใหม่จะทำเฉพาะ inference
ติดตั้ง requirements ตาม README และเลือก kernel ก่อนรัน; CPU ใช้ได้แต่ช้ากว่า GPU
"""), code('''from pathlib import Path
import sys
import json
import pandas as pd
from IPython.display import display

cwd = Path.cwd().resolve()
roots = [cwd, *cwd.parents]
candidates = [candidate for p in roots for candidate in (p, p / "Task4", p / "Task4/dist/Task4_submission", p / "dist/Task4_submission")]
BUNDLE_DIR = next((p for p in candidates if (p / "config/ensemble.json").is_file()), None)
if BUNDLE_DIR is None:
    raise FileNotFoundError("เปิดจากชุด Task4_submission หรือกำหนด BUNDLE_DIR ตามตำแหน่งชุดส่ง")
sys.path.insert(0, str(BUNDLE_DIR))
import predict as submission_predict
print("Bundle:", BUNDLE_DIR)
'''), md("## 1. ตั้งค่า input\nค่าเริ่มต้นใช้ตัวอย่าง Test ห้าภาพ ไม่ใช่ labeled validation จึงไม่คำนวณ accuracy จากตัวอย่างนี้"),
        code('''INPUT_CSV = BUNDLE_DIR / "examples/input.csv"
IMAGE_ROOT = BUNDLE_DIR / "examples/images"
SAMPLE_CSV = BUNDLE_DIR / "examples/sample.csv"
OUTPUT_CSV = BUNDLE_DIR / "outputs/example_predictions.csv"
DEVICE = "cpu"  # เปลี่ยนเป็น "auto" หรือ "cuda" เมื่อต้องการใช้ NVIDIA GPU
'''), md("## 2. ตรวจไฟล์และดูสูตร ensemble\nไม่มีการดาวน์โหลด weights; หากไฟล์หายหรือ checksum ไม่ตรงจะหยุดก่อนทำนาย"),
        code('''manifest = submission_predict.verify_bundle(BUNDLE_DIR)
ensemble = submission_predict.load_ensemble(BUNDLE_DIR)
display(pd.DataFrame(ensemble["members"])[["name", "weight", "tta", "checkpoint"]])
print("Package files verified:", len(manifest["files"]))
'''), md("## 3. ทำนายและตรวจ CSV"),
        code('''result = submission_predict.predict_bundle(
    BUNDLE_DIR, INPUT_CSV, IMAGE_ROOT, OUTPUT_CSV,
    sample_csv=SAMPLE_CSV, device=DEVICE,
)
display(result.predictions)
print("Rows:", result.metadata["rows"])
print("SHA256:", result.metadata["csv_sha256"])
expected = pd.read_csv(BUNDLE_DIR / "examples/expected_predictions.csv", dtype={"id": str})
print("ตรงกับ predictions รอบที่บันทึกไว้:", result.predictions.equals(expected))
print("การตรงกันนี้ตรวจ reproducibility ไม่ใช่ accuracy เพราะไม่มี Test labels")
'''), md("## 4. ดูภาพกับคำทำนาย"), code('''import matplotlib.pyplot as plt
from PIL import Image
input_rows = pd.read_csv(INPUT_CSV, dtype=str).set_index("id")
preview = result.predictions.head(5)
fig, axes = plt.subplots(1, len(preview), figsize=(16, 3), squeeze=False)
for ax, row in zip(axes.flat, preview.itertuples(index=False)):
    with Image.open(IMAGE_ROOT / input_rows.loc[row.id, "file_name"]) as image:
        ax.imshow(image)
    ax.set_title(f"{row.id}\\nPred: {ensemble['class_names'][row.task4]}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()
'''), md("""## ทำนายข้อมูลชุดอื่นหรือ Test เต็มชุด

เปลี่ยน INPUT_CSV/IMAGE_ROOT/SAMPLE_CSV ให้เป็นข้อมูลที่ต้องทำนาย และเลือก OUTPUT_CSV ใต้ outputs/ แล้วรันใหม่
CSV input ต้องมี `id,file_name` (คอลัมน์อื่นรวมถึง label จะไม่ถูกใช้)
ต้องมี IDs ตรงกับ sample หากระบุ sample; ไม่เขียนทับ CSV ผลจริงใน results/

ตัวอย่าง CLI จาก root ของชุดส่ง:
```text
python predict.py --input-csv examples/input.csv --image-root examples/images --sample-csv examples/sample.csv --output outputs/example_predictions.csv --device cpu
```

ผลเต็ม 25,889 แถวที่บันทึกจากเครื่องผู้จัดทำอยู่ใน results/task4.csv
Development outer Macro F1 0.988235 เป็นของโมเดลก่อน refit ไม่ใช่คะแนนของภาพตัวอย่างนี้หรือคะแนน Kaggle
""")]
    return n


def refresh_checksums(bundle):
    bundle = Path(bundle)
    if bundle.resolve() == (ROOT / "Task4").resolve():
        # .gitattributes keeps these sources as LF on clone. Never normalize raw artifacts.
        for path in bundle.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".txt", ".json", ".csv", ".ipynb"}:
                continue
            parts = path.relative_to(bundle).parts
            if any(part in {".ipynb_checkpoints", "__pycache__", "outputs", "dist", ".pytest_cache", ".venv", "venv"} for part in parts):
                continue
            if parts[0] in {"checkpoints", "training_assets"} and path.name != "README.md":
                continue
            if parts[0] == "examples" and path.name != "README.md":
                continue
            if parts[0] == "results" and path.name == "task4.csv":
                continue
            text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
    files = {}
    for path in sorted(Path(bundle).rglob("*")):
        relative = path.relative_to(bundle)
        if not path.is_file() or relative.as_posix() == "checksums.json":
            continue
        if any(part in {"outputs", "__pycache__", ".ipynb_checkpoints", "dist", ".pytest_cache", ".venv", "venv"} for part in relative.parts):
            continue
        files[relative.as_posix()] = pipeline.file_sha256(path)
    write_json(Path(bundle) / "checksums.json", {"algorithm": "sha256", "files": files})


def finalize(bundle):
    bundle = Path(bundle).resolve()
    refresh_checksums(bundle)
    manifest = json.loads((bundle / "checksums.json").read_text())
    archive = bundle.with_suffix(".zip")
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for relative in [*manifest["files"], "checksums.json"]:
            z.write(bundle / relative, arcname=bundle.name + "/" + relative)
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    archive.with_suffix(".zip.sha256").write_text(pipeline.file_sha256(archive) + "  " + archive.name + "\n")
    print("ZIP:", archive, archive.stat().st_size, "bytes")


PACKAGE_README = r"""# Task 4 — ชุดส่งสำหรับตรวจโมเดล

สูตรสุดท้าย: **DINOv2-Base 50% + ConvNeXt-Tiny 50%** โดย ConvNeXt ใช้ plain+flip TTA
ทั้งสองโมเดล refit 8 epochs บน Train 31,546 ภาพแล้ว ไม่ต้องฝึกใหม่เพื่อทดสอบ

## เริ่มทดสอบเร็ว (ไม่ต้องมี Train และไม่ต้องใช้อินเทอร์เน็ตหลังติดตั้งแพ็กเกจ)

1. แตก ZIP และเปิด Terminal ที่ Task4_submission
2. สร้าง virtual environment ด้วย Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

requirements.txt รวม dependencies ในไฟล์เดียว และไม่ล็อก suffix CPU/CUDA
หากต้องการเลือก backend ใน environment ใหม่ ให้ติดตั้ง torch/torchvision จาก official index ก่อน:
CUDA 12.8: `python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128`
CPU: `python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu`
แล้วติดตั้ง requirements.txt ตามปกติ ไม่ได้เลือก GPU backend ให้อัตโนมัติ
บน Linux ใช้ `.venv/bin/python` แทน `.venv\Scripts\python.exe`; macOS ยังไม่ได้ทดสอบ

3. เลือก kernel ของ environment นี้ แล้วเปิด **notebooks/inference.ipynb** และ Run All
หรือรัน CLI:

```text
python predict.py --input-csv examples/input.csv --image-root examples/images --sample-csv examples/sample.csv --output outputs/example_predictions.csv --device cpu
```

หากไม่ได้ activate environment ให้ใช้ path ของ Python ใน .venv แทนคำว่า python
ค่า device เปลี่ยนเป็น auto/cuda ได้; CPU ทำนายได้แต่ข้อมูลเต็มชุดอาจใช้เวลานาน
ไม่มีการดาวน์โหลด pretrained weights ระหว่าง inference โหลดจาก checkpoints ที่แนบเท่านั้น

## Notebook สองไฟล์

- **inference.ipynb**: โหลดโมเดลสำเร็จแล้ว ทำนายและตรวจ CSV มีตัวอย่าง outputs บันทึกไว้
- **training.ipynb**: เก็บ EDA/รูป/ตาราง/logs จากการทดลองเดิม พร้อมสรุปจาก artifacts เปิดอ่านได้โดยไม่รันใหม่
  ค่าเริ่มต้นข้าม data audit และ training/refit/export; ทำซ้ำการฝึกได้เมื่อเตรียมข้อมูลตาม data/README.md และเปิด flags ที่ต้องการ
  Outputs/paths ที่แสดงในประวัติอาจเป็นของเครื่องผู้จัดทำ ไม่ใช่ paths ที่ใช้รันจริงในชุดส่ง

## รูปแบบ input และ output

CSV input ต้องมี id,file_name และภาพอยู่ใต้ image-root เช่น `ANS00001,sjqyfwc.jpg`
ทุก ID ต้องไม่ซ้ำ ถ้าระบุ sample ต้องมี IDs ชุดเดียวกัน โค้ดเรียงผลตาม sample
Output มีเฉพาะ id,task4 โดย 0=7-11 Knights, 1=FiveN, 2=Free Fried, 3=PUBE, 4=Roblock
ไม่ใช้ label ของ input ในการทำนาย ไม่รวมผล Tasks 1/2/3/5 และไม่ส่ง Kaggle อัตโนมัติ
ไฟล์ที่มีอยู่ต่างจากผลใหม่จะไม่ถูกเขียนทับ เว้นแต่ระบุ --overwrite; results/ เป็นผล archive ที่ห้ามเขียนทับ

## ผลที่ส่งและข้อจำกัด

- Inner selection Macro F1: **0.9887623315174263**
- Outer development Macro F1 ก่อน refit: **0.9882345069508871**
- ResNet18 reference: **0.9631383592615694**
- ไม่มี independent validation score ของโมเดลหลัง refit; คะแนน Kaggle ยังไม่ทราบ
- Outer เดิมมีประวัติการใช้เลือกโมเดล จึงไม่ใช่ independent final test
- ตัวอย่างห้าภาพเป็น Test ไม่มี ground-truth labels การทำนายตรงกับ CSV เดิมไม่ใช่การวัด accuracy
- Different devices/versions may produce small numerical differences; exact bitwise equality is not guaranteed

ผล Test เต็ม 25,889 แถวอยู่ใน results/task4.csv พร้อม metadata/checksums
Training histories และ outer probabilities เป็น artifacts จริง ไม่ได้ฝึกใหม่ระหว่างจัดชุดส่ง
สูตร Ensemble และคะแนนที่รายงานมาจากผลการทดลองที่บันทึกไว้ในโปรเจกต์นี้

## ไฟล์สำคัญ

- config/ensemble.json: น้ำหนัก, TTA, relative checkpoint paths, SHA256 และ training provenance
- checkpoints/: เฉพาะสอง final checkpoints รวมประมาณ 458 MB
- results/: CSV เต็มชุด, คะแนน, probabilities และ training logs
- training_assets/: split, metadata และ reference probabilities สำหรับทำซ้ำ ไม่รวมภาพ Train ทั้งชุด
- checksums.json: SHA256 ของไฟล์ที่ส่ง; outputs/ ที่เกิดเมื่อทดลองใหม่ไม่รวมใน archive

การทำนายใช้ checkpoints และ config ที่จัดเตรียมไว้ ไม่ต้องมี pretrained cache, .git หรือ .venv จากเครื่องผู้จัดทำ
"""


def assemble(run, prediction_dir, output):
    run, prediction_dir, output = Path(run).resolve(), Path(prediction_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output exists; choose a new directory (no overwrite)")
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    refit = json.loads((run / "refit.json").read_text(encoding="utf-8"))
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    prediction_meta = json.loads((prediction_dir / "prediction_metadata.json").read_text(encoding="utf-8"))
    assert selection["passes"] and prediction_meta["full_train_refit"]
    assert prediction_meta["members"] == selection["selected"]["members"]
    assert pipeline.file_sha256(prediction_dir / "task4.csv") == prediction_meta["csv_sha256"]
    assert pipeline.file_sha256(ROOT / "Task4/dinov2_pipeline.py") == experiment["code_sha256"]
    required_bytes = sum(Path(path).stat().st_size for path in refit["checkpoints"].values())
    if shutil.disk_usage(ROOT).free < required_bytes * 3 + 200 * 2**20:
        raise OSError("Need space for bundle, relocated verification copy and ZIP")
    output.mkdir(parents=True)
    for name in ("predict.py", "dinov2_pipeline.py", "requirements.txt"):
        copy_file(ROOT / "Task4" / name, output / name)
    members = []
    for member in selection["selected"]["members"]:
        name = member["name"]
        path = Path(refit["checkpoints"][name])
        digest = pipeline.file_sha256(path)
        assert digest == prediction_meta["checkpoint_sha256"][name]
        relative = f"checkpoints/{name}.pt"
        copy_file(path, output / relative)
        assert pipeline.file_sha256(output / relative) == digest
        members.append({**member, "checkpoint": relative, "sha256": digest})
    config = {"schema_version": 1, "source_run": run.name, "class_names": list(pipeline.CLASS_NAMES),
              "members": members, "image_size": experiment["config"]["image_size"],
              "data_identity": experiment["data_identity"], "full_train_refit": True,
              "development_outer_metrics_before_refit": selection["outer_candidate"],
              "validation_score_of_refit": None, "kaggle_score": None}
    write_json(output / "config/ensemble.json", config)
    for name in ("task4.csv", "prediction_metadata.json"):
        copy_file(prediction_dir / name, output / "results" / name)
    copy_file(run / "outer_validation.npz", output / "results/outer_validation.npz")
    for phase in ("development", "full_refit"):
        for path in sorted((run / phase).glob("*/history.csv")):
            copy_file(path, output / "results/history" / f"{phase}_{path.parent.name}.csv")
    for name in ("split_manifest.csv", "split_metadata.json"):
        copy_file(ROOT / "reports/experiments/task4" / name, output / "training_assets" / name)
    copy_file(run / "partitions.csv", output / "training_assets/partitions.csv")
    copy_file(ROOT / "data/raw/sample_submission.csv", output / "training_assets/sample_submission.csv")
    reference = Path(experiment["reference_dir"])
    for name in ("run_info.json", "best_validation.npz"):
        copy_file(reference / name, output / "training_assets/reference_baseline" / name)
    write_json(output / "training_assets/experiment.json", {**experiment, "reference_dir": "reference_baseline"})
    write_json(output / "results/selection_summary.json", {
        key: selection[key] for key in ("selected", "inner_candidates", "outer_reference", "outer_candidate",
                                       "outer_gain", "bootstrap", "passes", "data_identity", "outer_note")})
    summary = {"scores": [
        {"scope": "inner selection (before refit)", **{key: selection["selected"][key] for key in ("macro_f1", "val_loss")}},
        {"scope": "reused outer development (before refit)", **selection["outer_candidate"]},
        {"scope": "outer ResNet18 reference", **selection["outer_reference"]}],
        "members": selection["selected"]["members"], "refit_epochs": refit["epochs"],
        "refit_rows": refit["rows"], "test_rows": prediction_meta["rows"],
        "validation_score_of_refit": None, "kaggle_score": None}
    write_json(output / "results/validation_summary.json", summary)

    test = pd.read_csv(ROOT / "data/raw/Task4/test.csv", dtype=str, keep_default_na=False).head(5)
    (output / "examples/images").mkdir(parents=True)
    test[["id", "file_name"]].to_csv(output / "examples/input.csv", index=False)
    test[["id"]].to_csv(output / "examples/sample.csv", index=False)
    full_predictions = pd.read_csv(prediction_dir / "task4.csv", dtype={"id": str})
    full_predictions.set_index("id").loc[test.id].reset_index().to_csv(output / "examples/expected_predictions.csv", index=False)
    for name in test.file_name:
        copy_file(ROOT / "data/raw/Task4/test" / name, output / "examples/images" / name)
    (output / "README.md").write_text(PACKAGE_README, encoding="utf-8")
    (output / "data").mkdir()
    (output / "data/README.md").write_text("""# ข้อมูลสำหรับฝึกซ้ำ

Inference ใช้เฉพาะภาพที่ต้องทำนาย ไม่ต้องมี Train
หากจะทำซ้ำ training ให้เตรียมชุดข้อมูลรายวิชารุ่นเดียวกับที่ระบุ SHA256 ใน config/ensemble.json ใต้ root ของชุดส่ง:

data/raw/Task4/train.csv (id,file_name,label)
data/raw/Task4/test.csv (id,file_name,label)
data/raw/Task4/train/<file_name>
data/raw/Task4/test/<file_name>

ภาพทั้งชุด 230×120 RGB JPEG; Train 31,546 / Test 25,889
ใช้ file_name จับคู่ภาพ ไม่ใช่ {id}.jpg
มี split เดิมใน training_assets ไม่ต้องสร้างใหม่เพื่อทำซ้ำการเปรียบเทียบเดิม
ไม่รวมภาพ Train/Test ทั้งชุดใน ZIP นี้เพื่อลดขนาด ให้ใช้ข้อมูลรายวิชาที่อาจารย์มีหรือไฟล์ที่ทีมส่งตามช่องทางที่กำหนด
""", encoding="utf-8")

    original_path = ROOT / "Task4/notebooks/training.ipynb"
    original_bytes = original_path.read_bytes()
    original = nbformat.reads(original_bytes.decode("utf-8"), as_version=4)
    write_json(output / "results/notebook_outputs.json", {
        c.id: c.get("outputs", []) for c in original.cells
        if c.cell_type == "code" and not c.id.startswith("task4-saved-")})
    training = build_training_notebook(original, output)
    inference = inference_notebook()
    transformer = TransformerManager()
    for notebook in (training, inference):
        nbformat.validate(notebook)
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(transformer.transform_cell(cell.source), filename=f"cell_{i}")
    backup = original_path.parent / ".ipynb_checkpoints" / (
        "task4-before-teacher-package-" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".ipynb")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(original_bytes)
    (ROOT / "Task4/notebooks").mkdir(exist_ok=True)
    (output / "notebooks").mkdir()
    for destination in (ROOT / "Task4/notebooks/training.ipynb", output / "notebooks/training.ipynb", original_path):
        nbformat.write(training, destination)
    for destination in (ROOT / "Task4/notebooks/inference.ipynb", output / "notebooks/inference.ipynb"):
        nbformat.write(inference, destination)
    refresh_checksums(output)
    print("Bundle:", output)
    print("Original outputs preserved; legacy backup:", backup)
    print("Checkpoints copied; no training or model inference during assembly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "models/Task4" / RUN_ID)
    parser.add_argument("--predictions", type=Path, default=ROOT / "submissions" / (RUN_ID + "_refit"))
    parser.add_argument("--output", type=Path, default=ROOT / "Task4/dist/Task4_submission")
    parser.add_argument("--finalize", action="store_true", help="Refresh checksums and create ZIP of an assembled bundle")
    args = parser.parse_args()
    if args.finalize:
        finalize(args.output)
    else:
        assemble(args.run, args.predictions, args.output)


if __name__ == "__main__":
    main()
