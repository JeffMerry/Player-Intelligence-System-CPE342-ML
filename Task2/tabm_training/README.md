# TabM training code (Kaggle GPU)

โค้ดชุดนี้ใช้ฝึก TabM ของ Task 2 บน Kaggle GPU (T4) เก็บไว้เป็นหลักฐานว่า checkpoint ใน `../checkpoints/` ได้มาอย่างไร notebook ไม่ได้เรียกใช้ไฟล์เหล่านี้

| ไฟล์ | รายละเอียด |
|---|---|
| `task2_tabm_gpu.py` | ฝึกแต่ละ stage: pilot, inner/outer 5 folds (OOF), `dev_inner` → `development` (เลือก 16 epochs ด้วย early stopping แล้วฝึกบน development) และ `final` (ฝึกบน train ทั้งหมด 16 epochs) |
| `task2_tabm_common.py` | preprocessing (median + missing indicator + StandardScaler + one-hot จาก state ที่ fit บนข้อมูลฝึก), Macro F1 และ blend |
| `PROTOCOL.md` | กติกาที่ตรึงไว้ก่อนรัน: architecture, optimizer, การแบ่งข้อมูล และเกณฑ์ตัดสิน |

ค่าที่ใช้: `d_block=256`, `n_blocks=2`, `k=16`, dropout 0.1, AdamW lr 0.002, weight decay 0.0003, batch 512, สูงสุด 80 epochs, patience 12, seed 20261600

สคริปต์ต้องการ packet ที่มี `train.csv`, `test.csv`, `schema.json`, `roles.json`, `splits.npz`, `lr_states/` และ wheels ของ `tabm` 0.0.3 / `rtdl_num_embeddings` 0.0.12 (`python task2_tabm_gpu.py --bundle <packet> --output <dir>`) packet นี้ไม่ได้อยู่ใน repo เพราะมีข้อมูล train/test
