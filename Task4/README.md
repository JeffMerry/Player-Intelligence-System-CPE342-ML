# Task 4 — Game Title Detection

Notebook แยกสำหรับส่งตรวจ:
- `notebooks/training.ipynb`: EDA, training และ outputs ที่บันทึกจากการทดลองจริง
- `notebooks/inference.ipynb`: ทำนายด้วยสอง final checkpoints ไม่ต้องมี Train หรือฝึกซ้ำ
- `task4-game-classification.ipynb`: ไฟล์นำทางไปยัง notebook สองไฟล์ข้างต้น

Implementation: `dinov2_pipeline.py`; portable inference CLI: `predict.py`

ใช้ Transfer Learning และ Ensemble หลายสถาปัตยกรรม โดยฝึกและประเมินด้วยข้อมูลและ split ของโปรเจกต์นี้
Development และ full-train refit เสร็จแล้ว สูตรที่เลือกคือ DINOv2-Base 50% + ConvNeXt-Tiny 50%
ConvNeXt ใช้ plain+flip TTA; outer development Macro F1 ก่อน refit = 0.9882345069508871


## เริ่มใช้งาน

1. ใช้ Python 3.12 ติดตั้งจาก project root:
   `python -m pip install -r Task4/requirements.txt`
2. เปิด notebook เลือก kernel ที่ติดตั้ง dependencies แล้ว; หากเพิ่งติดตั้งให้ Restart Kernel
3. หลัง clone ให้กู้ artifacts ตาม [ARTIFACTS.md](ARTIFACTS.md) ก่อนทำนาย:
   ดาวน์โหลด **Task4_artifacts.zip** จาก [โฟลเดอร์ Google Drive](https://drive.google.com/drive/folders/1J7OTKaxK3lW-8vFKFq-Dkxw8OGuoY9or?usp=sharing) แล้ววางใน `Task4/dist/`
   `python Task4/restore_artifacts.py --archive Task4/dist/Task4_artifacts.zip`
   จากนั้นเปิด `Task4/notebooks/inference.ipynb` หรือรันจาก project root:
   `python Task4/predict.py --input-csv Task4/examples/input.csv --image-root Task4/examples/images --sample-csv Task4/examples/sample.csv --output Task4/outputs/example_predictions.csv --device cpu`
   Dependencies ทั้งหมดรวมใน `Task4/requirements.txt` ไฟล์เดียว รวมเครื่องมือทดสอบ
   ไฟล์นี้ไม่ล็อก suffix CPU/CUDA จึงใช้กับ torch 2.9.1+cu128 ที่ติดตั้งอยู่แล้วได้
   ไม่ได้เลือก GPU backend ให้อัตโนมัติ หากเริ่ม environment ใหม่และต้องการเลือก backend ชัดเจน ดูคำสั่งด้านล่าง
4. ตรวจ config และ paths ก่อนเปิด `RUN_DEVELOPMENT=True`
5. เปิด full refit และ test inference แยกกันเมื่อ candidate ผ่าน gate

Training/refit/test และ data-preparation flags เริ่มเป็น `False` จึงไม่เริ่มฝึกหรืออ่านภาพ Train ทั้งชุดเมื่อ Run All
Outputs เดิมยังอยู่ครบ และสรุปท้าย notebook อ่านจาก artifacts ในชุดส่ง
เมื่อ Run All ในโหมดเปิดดู จะนำ outputs เดิมมาแสดงซ้ำจาก `results/notebook_outputs.json`
จึงยังเห็นกราฟ/ตารางเดิมโดยไม่ต้องคำนวณภาพหรือฝึกใหม่
หากต้องทำซ้ำ EDA เปิด `RUN_DATA_PREPARATION=True` หลังเตรียมข้อมูลตาม README

## โมเดลและวิธีฝึก

- ค่าเริ่มต้น: ResNet18, ConvNeXt-T, Swin-T, DINOv2 ViT-S/14 และ ViT-B/14 แบบมี register tokens
- DINOv2 ViT-L/14 optional: เพิ่ม `dinov2_large` ใน `model_names` และเปิด `allow_large=True`
- DINO head: CLS + 2×4 grid + global pooling → LayerNorm → Linear 5 classes
- DINO weights กำหนด revision/SHA256 เพื่อให้ใช้น้ำหนักเริ่มต้นชุดเดิม; ไม่ใช้ `trust_remote_code`
- Fine-tune ทั้ง backbone 8 epochs ด้วย strong augmentation, AMP, warmup, cosine LR, DINO layer-wise LR decay
- Letterbox สูง 224 × กว้าง 448; micro batch ตามโมเดลและ gradient accumulation เป้าหมาย 32
- Gradient checkpointing, OOM restart ลด micro batch และ time budget ต่อโมเดล
- ใช้ GPU เดียวและ gradient accumulation การเลือก T4×2 ใน Kaggle ไม่ทำให้ใช้สองตัวอัตโนมัติ

## Validation และ ensemble

เก็บ outer validation เดิม แล้วแบ่ง inner fit/selection ภายใน Fit โดยรักษา duplicate groups
เลือก plain/flip TTA และ single/top-k/DINO-only equal-weight candidates จาก inner selection เท่านั้น
จากนั้นตรวจสูตรที่เลือกบน outer เทียบ baseline เดิมที่มี split provenance ตรงกัน
ถ้าไม่มี baseline เดิม ใช้ ResNet18 ที่ฝึกใหม่บน inner fit เป็น reference

Gate ก่อนฝึก: outer Macro F1 gain ≥ 0.002 และสัดส่วน bootstrap gain ≤ 0 ต่ำกว่า 0.10
ใช้ paired bootstrap แบบ resample ทั้ง duplicate group 2,000 draws ถ้าไม่ผ่านไม่ refit/export candidate
Outer เดิมเคยใช้เลือกโมเดลแล้ว จึงไม่ใช่ independent test ใหม่

Optional refit ใช้ Train ทั้งหมดด้วย epoch budget เดิม ไม่อ้างคะแนน outer เป็นคะแนน refit
หากไม่ refit ใช้ development checkpoints เดียวกับที่ประเมินสำหรับ Test

## ข้อมูลและผลลัพธ์

- Data: `data/raw/Task4/`; CSV `id,file_name,label`; ภาพใน `train/`, `test/`
- Split เดิมในโปรเจกต์: `reports/experiments/task4/`; ชุดสำหรับ clone/ส่งตรวจอยู่ใน `Task4/training_assets/` หลัง restore
- Runs: `models/Task4/dinov2_ensemble_<timestamp>/`
- Candidate: `submissions/dinov2_ensemble_<timestamp>_<scope>/task4.csv` คอลัมน์ `id,task4`
- Mapping: `0=7-11 Knights, 1=FiveN, 2=Free Fried, 3=PUBE, 4=Roblock`

เก็บ checkpoints, probabilities, config, software versions, hashes และ recipe แยกรอบ
ข้อมูล/โมเดล/ผลทำนายถูก Git ignore ต้องเตรียมในเครื่องเอง
กู้ checkpoints, examples, full CSV และ training assets จาก ZIP ได้โดยไม่ต้องฝึกใหม่
`split_manifest.csv` มี IDs/labels ไม่ได้ commit; ใช้ไฟล์เดิมจาก ZIP เพื่อทำซ้ำการทดลองบน split เดียวกัน

โครงสร้าง Task4 เป็นโครงสร้างหลักสำหรับ merge เข้า main:
`predict.py`, `dinov2_pipeline.py`, `config/`, `notebooks/`, `examples/`, `results/`, `checkpoints/`, `training_assets/`
Git เก็บ source/notebooks พร้อม outputs/config/summary/history logs และ README ของโฟลเดอร์ artifacts
เครื่องมือจัด ZIP และตรวจชุดส่งอยู่ใน `Task4/scripts/` ส่วนกฎ ignore และรูปแบบบรรทัดอยู่ใน `Task4/.gitignore` และ `Task4/.gitattributes`
จึงสามารถ commit เฉพาะ `Task4/` ได้โดยไม่ต้องแก้ไฟล์ตั้งค่าระดับ project root
Weights, examples ที่เป็นข้อมูล, full prediction CSV และ raw split แยกจาก Git ตาม .gitignore
คำสั่ง restore ไม่เขียนโค้ดจาก ZIP ทับ source จึงใช้ ZIP เป็นแหล่ง artifacts ได้แม้ source จะปรับปรุง paths ภายหลัง

บน Kaggle ต้องแนบข้อมูล, split และ source ของโปรเจกต์ แล้วแก้ paths ตาม Input จริง
ตั้ง OUTPUT_ROOT/SUBMISSION_ROOT ใต้ `/kaggle/working`; ไม่สมมติ dataset slug
requirements ระบุเวอร์ชันไลบรารีโดยไม่ล็อก suffix CPU/CUDA ตรวจ torch/torchvision ของ Kaggle ก่อนติดตั้งเพิ่ม


## ตรวจโค้ดโดยไม่ฝึก

ติดตั้ง dependencies ด้วย `python -m pip install -r Task4/requirements.txt`
แล้วรัน `python -m pytest Task4/tests -q`
ทดสอบใช้ random weights สำหรับ forward เท่านั้น บล็อกการดาวน์โหลด pretrained และการฝึก
ครอบคลุมสถาปัตยกรรม S/B, optimizer groups, checkpoint roundtrip, group separation,
ensemble selection, bootstrap, ID alignment, failed gate และ flags ค่าเริ่มต้นของ notebook
DINOv2-L ยังไม่ได้ทดสอบ forward/full-training บน GPU 6 GB

## เลือก PyTorch backend ใน environment ใหม่ (optional)

CUDA 12.8 ตามรอบที่ใช้ฝึก:
`python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128`

CPU:
`python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu`

จากนั้นติดตั้ง `python -m pip install -r Task4/requirements.txt` ตามปกติ
ไม่จำเป็นต้องติดตั้งใหม่ใน environment ปัจจุบันที่ใช้งานได้แล้ว

## จัดชุดส่งอาจารย์

ส่ง GitHub คู่กับ **Task4_artifacts.zip** บน Google Drive:
- GitHub เก็บโค้ด, notebooks พร้อม outputs, config, requirements, คู่มือ, สรุปผล และประวัติการฝึก
- ZIP เก็บเฉพาะ final checkpoints, ภาพและ CSV ตัวอย่าง, ผลเต็ม/ความน่าจะเป็น และข้อมูลแบ่งชุด
- ZIP ไม่มีโค้ดหรือ notebook ซ้ำกับ GitHub และไม่มีภาพ Train/Test ทั้งชุด

สร้าง ZIP ด้วย `python Task4/scripts/package_task4_artifacts.py`
จะได้ `Task4/dist/Task4_artifacts.zip` และ `Task4/dist/Task4_artifacts.zip.sha256` พร้อมอัปเดต `config/artifacts.json`
สคริปต์ตรวจ hash ของไฟล์ต้นทางและ ZIP ไม่ฝึกหรือดาวน์โหลด weights และไม่เขียนทับ ZIP เดิม
จากนั้นอัปโหลด ZIP ไป Google Drive ตั้งสิทธิ์ให้อาจารย์ดาวน์โหลดได้ และเพิ่มลิงก์ใน README นี้, ARTIFACTS.md และ `config/artifacts.json`
ผู้รับ clone GitHub → ดาวน์โหลด ZIP → restore → ติดตั้ง requirements → เปิด inference.ipynb

ตรวจสำเนาที่เปลี่ยนตำแหน่งด้วย `python Task4/scripts/verify_task4_submission.py --bundle Task4`
คำสั่งนี้ทำนายตัวอย่างจริงบน CPU และรัน notebooks โดยบล็อก training/downloads; ไม่ได้ติดตั้ง CPU environment ใหม่
สคริปต์ตรวจจะบันทึกผลใหม่ใน inference notebook จึงควรบันทึกการแก้ไขของตนเองก่อนรัน
ชุดส่งมีเฉพาะสอง final checkpoints, config relative paths, examples, ผล CSV, logs และ split สำหรับทำซ้ำ
ไม่รวม .venv, caches, โมเดลทดลองทุกตัว หรือภาพ Train/Test ทั้งชุด
ดูรายละเอียดการกู้ไฟล์ใน [ARTIFACTS.md](ARTIFACTS.md)
ลิงก์ที่ให้เป็นโฟลเดอร์ Google Drive ให้เปิดและเลือกดาวน์โหลด `Task4_artifacts.zip` ด้วยตนเอง สคริปต์ restore ไม่ดาวน์โหลดไฟล์ให้อัตโนมัติ
