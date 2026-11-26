# Task 2 — Player Segment Classification

- Target: `segment` (0 Casual, 1 Competitive Grinder, 2 Social, 3 Whale)
- Metric: **Macro F1** ของ 4 คลาส (`f1_score(..., average='macro')`)
- ข้อมูล: `../data/raw/Task2/` (`train.csv`, `test.csv`, `datacard.txt`) และ `../data/raw/sample_submission.csv`
- โค้ด: [notebooks/task-2-player-segment-classification.ipynb](notebooks/task-2-player-segment-classification.ipynb)
- ผลทำนาย: `../submissions/task2.csv` คอลัมน์ `id,task2`
- ไฟล์ผลระหว่างทาง: `../models/Task2/`
- TabM checkpoints: [checkpoints/](checkpoints/) · โค้ดที่ใช้ฝึก TabM บน Kaggle GPU: [tabm_training/](tabm_training/)

## วิธีรัน

```powershell
pip install -r Task2/requirements.txt
```

เปิด notebook ด้วย kernel ที่ติดตั้ง requirements แล้ว แล้วกด **Restart → Run All** (CPU ประมาณ 3–4 นาที) รันได้ทั้งจาก root หรือจาก `Task2/notebooks/`
notebook ไม่ฝึก TabM ใหม่ แต่โหลด checkpoint ใน `checkpoints/` (ตรวจ SHA256 ก่อน) แล้วทำนายบน CPU ส่วน LR และ XGBoost ฝึกใหม่บน CPU ทุกครั้ง

## สูตรที่ใช้

| ส่วน | ค่า |
|---|---|
| ข้อมูลฝึก | 104,000 แถว แบ่ง development 88,400 / audit 15,600 (stratified, seed 342) |
| Features | feature ดิบ 44 ตัว (34 ตัวเลข, 10 หมวดหมู่), `preferred_game_mode` เป็นหมวดหมู่ ไม่ engineer เพิ่ม |
| Logistic Regression | median + missing indicator + StandardScaler + one-hot (122 คอลัมน์), `C=0.5`, lbfgs |
| TabM | 16-member ensemble MLP, 2 blocks × 256, dropout 0.1, input 122 คอลัมน์ชุดเดียวกับ LR, AdamW lr 0.002, 16 epochs (เลือกด้วย early stopping บน inner split ของ development), ฝึกบน Kaggle GPU T4 |
| XGBoost | features ดิบ + one-hot, `max_depth=3`, `min_child_weight=20`, lr 0.04, 780 rounds (เลือกจาก 4 configs บน inner split ของ development) |
| **โมเดลสุดท้าย** | `argmax((0.65·LR + 0.25·TabM + 0.10·XGB) × decision weights)` น้ำหนักตรึงไว้ก่อนดูผล audit |
| Decision weights | `[1.0000, 1.1146, 1.1753, 1.1241]` (เลือกจากข้อมูลชุดก่อน ไม่จูนใหม่) |

## ผล (คะแนนภายใน ไม่ใช่คะแนน Kaggle)

| โมเดล | Development 5-fold OOF Macro F1 | Audit Macro F1 |
|---|---:|---:|
| LR | 0.751680 | 0.758128 |
| TabM | — | 0.759370 |
| XGBoost | — | 0.751823 |
| LR 75% + TabM 25% | 0.753686 | 0.760453 |
| **LR 65% + TabM 25% + XGB 10%** | **0.754468** | **0.761132** |

OOF ของ blend มาจาก fold models ที่ฝึกบน GPU ระหว่างพัฒนา (notebook รัน OOF เฉพาะ LR) ส่วน audit ถูกดูหลายครั้งระหว่างพัฒนา จึงเป็นการเช็คว่าไม่แย่ลง ไม่ใช่การวัดผลอิสระครั้งแรก

## คะแนน Kaggle

`submissions/task2.csv` จาก notebook นี้ตรงกับคอลัมน์ Task 2 ในไฟล์ที่ส่ง Kaggle และได้คะแนนรวม public **0.82618** ทุกแถว
Kaggle รายงานเป็นคะแนนรวม 5 tasks จึงไม่ทราบคะแนน Task 2 อย่างเดียว

## การจับคู่ ID

`test.csv` ใช้ `player_id` ส่วน sample submission ใช้ `ANS00001`–`ANS25889`
จับคู่ตามลำดับแถว: test แถวที่ i ↔ sample แถวที่ i (วิธีเดียวกับ Task 1 และ Task 3)

## ข้อจำกัด

- Social เป็นคลาสที่ยากที่สุด (audit F1 ของ blend 0.723) และ Whale ยังมี recall ต่ำ (0.713)
- TabM ฝึกบน GPU ถ้าฝึกใหม่ด้วยโค้ดใน `tabm_training/` ผลอาจต่างเล็กน้อยเพราะ GPU ไม่ deterministic ทั้งหมด จึงเก็บ checkpoint ที่ใช้จริงไว้ใน repo
- ใช้เฉพาะข้อมูล train ที่มี label ไม่ใช้ test ฝึกหรือเลือกโมเดล ไม่ใช้ pretrained model, pseudo-label หรือ AutoML
