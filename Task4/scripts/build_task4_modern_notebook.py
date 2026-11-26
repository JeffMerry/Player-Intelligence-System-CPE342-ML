"""Build the multi-architecture notebook; keep its data audit and back up legacy ML cells."""
import ast
import copy
from datetime import datetime
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "Task4/notebooks/training.ipynb"


def build():
    original = PATH.read_bytes()
    notebook = nbformat.reads(original.decode("utf-8"), as_version=4)
    end = next(i for i, c in enumerate(notebook.cells) if c.id == "3eb91fcc")
    audit = copy.deepcopy(notebook.cells[:end + 1])
    audit[0].source = """# Task 4 — DINOv2 และ ensemble หลายสถาปัตยกรรม

จำแนก screenshot 5 เกมด้วย Macro F1 ใช้ Train เท่านั้นในการฝึกและเลือกสูตร
ใช้ Transfer Learning และ Ensemble โดยฝึกและประเมินด้วยข้อมูลและ split ของโปรเจกต์นี้
"""
    audit[1].source = """## วิธีรันและสถานะผล

ตรวจข้อมูล/ภาพซ้ำ → บันทึก split → inner fit/selection → ฝึกหลายสถาปัตยกรรม → เลือก TTA/ensemble → outer gate → optional full refit → Test export

**ยังไม่ได้ฝึก pipeline DINOv2 ใหม่นี้** รูปและตารางในส่วนตรวจข้อมูลเป็นผลเดิม ส่วนโมเดลใหม่ยังไม่มีคะแนน
ค่า `RUN_DEVELOPMENT`, `RUN_FULL_REFIT`, `RUN_TEST_INFERENCE` เริ่มเป็น `False` จึงไม่เริ่มฝึกหรือทำนาย Test เมื่อ Run All
Notebook ก่อนแก้ถูกสำรองใน `Task4/.ipynb_checkpoints/` และดูรุ่นเดิมจาก Git commit `c371779` ได้
"""
    audit[2].source = """### เตรียม environment ก่อนรัน

ใช้ Python 3.12 ติดตั้งจาก project root: `python -m pip install -r Task4/requirements.txt`
หากใช้ notebook: `%pip install -r <path จริงของ Task4/requirements.txt>` แล้ว Restart Kernel ก่อนเริ่ม
requirements อ้าง Windows/CUDA เดิม บน Kaggle ให้ตรวจ torch/torchvision ที่มีอยู่ก่อนติดตั้ง CUDA build เพิ่ม
"""
    for cell in audit:
        if cell.id in {"8671bcad", "8c628a5a"}:
            cell.cell_type = "markdown"
            cell.source = "คำสั่งติดตั้งเดิมย้ายไปรวมใน `Task4/requirements.txt`"
            cell.pop("outputs", None)
            cell.pop("execution_count", None)

    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    modern = [md("""## 7. สิ่งที่เปลี่ยนจาก ResNet18 + frozen/MLP

| ส่วน | รุ่นใหม่ |
|---|---|
| โมเดล | ResNet18, ConvNeXt-T, Swin-T, DINOv2-S/B; DINOv2-L optional |
| DINO head | CLS + spatial pooling 2×4 + global pooling → LayerNorm → Linear 5 classes |
| Weights | กำหนด revision และ SHA256 ของ DINOv2 เพื่อให้ทำซ้ำได้ |
| ภาพ | Letterbox สูง 224 × กว้าง 448, BICUBIC, ImageNet normalization |
| Augmentation | Flip, affine, color jitter, grayscale, random erasing ตาม STRONG recipe |
| Fine-tune | 8 epochs, AdamW, AMP, warmup 1 epoch, cosine LR, DINO layer-wise LR decay |
| VRAM | Micro batch ตามโมเดล + accumulation เป้าหมาย 32, gradient checkpointing, OOM retry |
| Ensemble | Single/top-k/DINO-only equal-weight candidates; เลือกจาก inner selection |

เลือกสมาชิก Ensemble จากผลบน Inner Selection โดยไม่กำหนดว่าต้องใช้ทุกโมเดลร่วมกัน
DINOv2-L ต้องเปิดเอง เพราะเครื่องที่ใช้มี VRAM 6 GB ไม่รับประกันว่าโมเดลใหญ่จะพอดีหรือคะแนนสูงขึ้น
"""), code('''import sys
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Task4 import dinov2_pipeline as advanced

RUN_DEVELOPMENT = False
RUN_FULL_REFIT = False
RUN_TEST_INFERENCE = False
RUN_MODEL_SMOKE = False  # forward ด้วย random weights ไม่ดาวน์โหลด pretrained
USE_REFIT_FOR_TEST = False
EXISTING_RUN_DIR = None  # ใส่ path รอบใหม่ที่สำเร็จแล้วเพื่อเปิดดู/ทำนายโดยไม่ฝึกซ้ำ

CFG = advanced.PipelineConfig(
    model_names=("resnet18", "convnext_tiny", "swin_t", "dinov2_small", "dinov2_base"),
    image_size=(224, 448), epochs=8, effective_batch=32,
    num_workers=0, gradient_checkpointing=True,
    allow_large=False,  # เพิ่ม "dinov2_large" ใน model_names และตั้ง True เพื่อทดลองโมเดลขนาด Large
)
CFG.validate()

# บน Kaggle ตั้ง paths ให้ตรง Input จริง และเขียน outputs ใต้ /kaggle/working
DATA_DIR = PROJECT_ROOT / "data/raw/Task4"
SPLIT_PATH = PROJECT_ROOT / "reports/experiments/task4/split_manifest.csv"
SPLIT_METADATA_PATH = SPLIT_PATH.with_name("split_metadata.json")
SAMPLE_PATH = PROJECT_ROOT / "data/raw/sample_submission.csv"
OUTPUT_ROOT = PROJECT_ROOT / "models/Task4"
SUBMISSION_ROOT = PROJECT_ROOT / "submissions"

# ใช้ ResNet18 ที่บันทึกไว้เป็น reference; ถ้าไม่มีให้ฝึก reference ใหม่
REFERENCE_RUN = OUTPUT_ROOT / "resnet18_20261124_155809_039967"
if not REFERENCE_RUN.is_dir():
    REFERENCE_RUN = None

software = {}
for package in ("torch", "torchvision", "timm", "huggingface-hub", "safetensors"):
    try:
        software[package] = version(package)
    except PackageNotFoundError:
        software[package] = "ยังไม่ติดตั้ง: ดู Task4/requirements.txt"
display(pd.Series(software, name="version").to_frame())
display(pd.DataFrame([
    {"name": name, **asdict(advanced.MODEL_SPECS[name])}
    for name in CFG.model_names
]))
print("Reference:", REFERENCE_RUN)
print("Training enabled:", RUN_DEVELOPMENT)
'''), md("""## 8. Inner selection และ provenance

Outer คือ Validation 6,310 ภาพเดิม ส่วน Fit 25,236 ภาพเดิมแยก inner fit/selection แบบรักษากลุ่มอีกครั้ง
ทุกโมเดลใหม่ฝึกเฉพาะ inner fit; ใช้ selection เลือก TTA/สมาชิก แล้วตรวจ outer เฉพาะสูตรที่เลือก
Outer เดิมถูกดูและใช้เลือกโมเดลแล้ว จึงยังเป็น development evidence ไม่ใช่ independent test ใหม่
"""), code('''partitions = advanced.prepare_partitions(
    DATA_DIR, SPLIT_PATH, SPLIT_METADATA_PATH, CFG
)
display(pd.DataFrame([
    {"partition": name, "images": len(frame), "groups": frame.duplicate_group.nunique()}
    for name, frame in [("inner fit", partitions.fit),
                        ("inner selection", partitions.selection),
                        ("reused outer", partitions.outer)]
]))
print("Dataset identity:", partitions.identity)
'''), md("""## 9. Optional model smoke — ไม่ฝึก

สร้าง random-weight models และทดสอบ forward ทีละตัว ไม่ดาวน์โหลด pretrained
ผ่านขั้นนี้ยืนยันเฉพาะรูปทรง ไม่ยืนยัน VRAM ตอน backward หรือ weights ที่ยังไม่ได้ดาวน์โหลด
"""), code('''if RUN_MODEL_SMOKE:
    import gc
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in CFG.model_names:
        if name == "dinov2_large":
            print("ข้าม Large ใน smoke ปกติ; ทดสอบแยกบนเครื่องทรัพยากรเพียงพอ")
            continue
        smoke_model = advanced.build_model(name, CFG, pretrained=False).to(device).eval()
        with torch.inference_mode():
            output = smoke_model(torch.zeros(1, 3, *CFG.image_size, device=device))
        assert output.shape == (1, 5) and torch.isfinite(output).all()
        print(name, tuple(output.shape))
        del smoke_model, output
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
else:
    print("ยังไม่รัน model smoke test")
'''), md("""## 10. Development training และเลือกสูตร

เปิด `RUN_DEVELOPMENT=True` เมื่อต้องการฝึกจริง ครั้งแรกดาวน์โหลด weights และตรวจ DINO SHA256
ใช้ plain+flip TTA เมื่อ inner Macro F1 gain ≥ 0.001; เลือก single/top-k/DINO-only equal-weight สูตรด้วย inner เท่านั้น
Outer gate: gain ≥ 0.002 และ paired group bootstrap 2,000 draws ให้สัดส่วน gain ≤ 0 ต่ำกว่า 0.10
เป็น descriptive evidence บน reused holdout ไม่รับประกัน Test; ถ้าไม่ผ่านไม่สร้าง candidate
"""), code('''advanced_run_dir = Path(EXISTING_RUN_DIR) if EXISTING_RUN_DIR else None
selection_report = None
if RUN_DEVELOPMENT:
    advanced_run_dir, selection_report = advanced.run_development(
        partitions, CFG, OUTPUT_ROOT, reference_dir=REFERENCE_RUN
    )
elif advanced_run_dir is not None:
    advanced.assert_run_identity(advanced_run_dir, partitions)
    selection_report = json.loads(
        (advanced_run_dir / "selection.json").read_text(encoding="utf-8")
    )
else:
    print("ยังไม่เริ่ม development training; ผลโมเดลใหม่ยังไม่มี")

if selection_report is not None:
    display(pd.DataFrame(selection_report["inner_candidates"]))
    print("Selected members:", selection_report["selected"]["members"])
    print("Outer reference:", selection_report["outer_reference"])
    print("Outer candidate:", selection_report["outer_candidate"])
    print("Outer gain:", selection_report["outer_gain"])
    print("Gate passed:", selection_report["passes"])
    print("Run:", advanced_run_dir)
'''), md("""## 11. Optional full-train refit

เปิด `RUN_FULL_REFIT=True` หลังผ่าน gate จะเริ่ม selected backbones ใหม่แล้วฝึก Train ทั้งหมด 8 epochs เดิม
ไม่ใช้ outer early stopping และไม่อ้าง outer score เดิมเป็นคะแนน refit
หากไม่ refit ใช้ development checkpoints เดียวกับที่ประเมินสำหรับ Test ด้วย `USE_REFIT_FOR_TEST=False`
"""), code('''if RUN_FULL_REFIT:
    if advanced_run_dir is None:
        raise RuntimeError("ยังไม่มี development run; ตั้ง EXISTING_RUN_DIR หรือฝึกก่อน")
    refit_paths = advanced.refit_selected(advanced_run_dir, partitions)
    print("Refit checkpoints:", refit_paths)
else:
    print("ยังไม่เริ่ม full-train refit")
'''), md("""## 12. Test inference และ candidate export

เปิด `RUN_TEST_INFERENCE=True` หลังเลือกสูตรและผ่าน gate; ถ้าใช้ refit เปิด `USE_REFIT_FOR_TEST=True` ด้วย
จับคู่ sample ด้วย ID ส่งออก `id,task4` คลาส 0–4 พร้อม probabilities และ provenance ในโฟลเดอร์ใหม่
ไม่เขียนทับ CSV เดิม ไม่รวม Tasks อื่นหรือส่ง Kaggle อัตโนมัติ
"""), code('''if RUN_TEST_INFERENCE:
    if advanced_run_dir is None:
        raise RuntimeError("ยังไม่มี development run")
    candidate_csv = advanced.export_test(
        advanced_run_dir, partitions, SAMPLE_PATH, SUBMISSION_ROOT,
        use_refit=USE_REFIT_FOR_TEST,
    )
    print("Candidate:", candidate_csv)
    display(pd.read_csv(candidate_csv, dtype={"id": str}).head())
else:
    print("ยังไม่ทำนาย Test หรือสร้าง candidate ใหม่")
'''), md("""## ข้อแตกต่างและข้อจำกัด

- ใช้ภาพและ split ของโปรเจกต์นี้ โดยรักษากลุ่มภาพคล้ายไม่ให้ข้ามชุด
- DINOv2-L เป็นตัวเลือกเพิ่มเติมและไม่ได้เปิดใช้โดยค่าเริ่มต้น
- Micro batch + accumulation ไม่เท่ากับ batch ใหญ่ทุกอย่าง โดยเฉพาะ BatchNorm ของ ResNet18
- ใช้ GPU เดียวร่วมกับ gradient accumulation
- Flip/strong augmentation ต้องวัดใหม่ ไม่ใช้ Test ออกแบบ training policy หรือทำ pseudo-labeling
- Outer เดิมเคยถูกใช้แล้ว แม้รอบนี้เลือกสูตรจาก inner ก็ยังไม่ใช่ independent final estimate
- Teacher hints ที่ทำ: augmentation/ViT/Swin/ensemble; imbalance แสดงใน EDA แต่ยังไม่เพิ่ม class weights; VGG และการฝึก group K-fold เลื่อนไว้เพื่อจำกัดงบ
- แม่แบบนี้ยังไม่ได้ฝึก ต้องรายงานคะแนนจากผลการทดลองจริงของแต่ละรอบ

เอกสารโมเดล: [DINOv2 model card](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m), [timm features](https://huggingface.co/docs/timm/en/feature_extraction)
""")]
    notebook.cells = audit + modern
    notebook.metadata["task4_modern_pipeline"] = {
        "executed": False,
        "note": "Multi-architecture training template; model outputs require a completed experiment.",
    }
    nbformat.validate(notebook)
    transformer = TransformerManager()
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(transformer.transform_cell(cell.source), filename=f"cell_{i}")
    backup = PATH.parent / ".ipynb_checkpoints" / (
        "task4-before-dinov2-" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".ipynb")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(original)
    assert PATH.read_bytes() == original, "Notebook changed during build"
    nbformat.write(notebook, PATH)
    print("Updated:", PATH)
    print("Legacy backup:", backup)
    print("Cells:", len(notebook.cells), "(new model cells have no outputs)")


if __name__ == "__main__":
    build()
