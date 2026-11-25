# Task 1 — Anti-Cheat Pre-Filter

- Target: `is_cheater` (0 หรือ 1)
- Metric: **binary F2 ของคลาส 1** (ยืนยันจากผลส่ง Kaggle ดูด้านล่าง)
- ข้อมูล: `../data/raw/Task1/` (`train.csv`, `test.csv`, `datacard.txt`) และ `../data/raw/sample_submission.csv`
- โค้ด: [notebooks/task-1-anti-cheatdetection.ipynb](notebooks/task-1-anti-cheatdetection.ipynb)
- ผลทำนาย: `../submissions/task1.csv` คอลัมน์ `id,task1`
- โมเดลและไฟล์ตรวจสอบ: `../models/Task1/`
- สรุปการรัน: `../reports/experiments/task1_run_metadata.json`

## ที่มา

สูตรมาจากงาน Task 1 ใน [i3artuf/CPE342-kaggle-competition](https://github.com/i3artuf/CPE342-kaggle-competition)
(`task-1-anti-cheatdetection.ipynb`, `scripts/experiment_task1_augmentation_small_gain.py`, `scripts/make_task1_threshold_probe.py`)
คอลัมน์ Task 1 นี้อยู่ใน submission ที่ได้ Kaggle public **0.74155** (15 พ.ย. 2026) และทุก submission ต่อมาจนถึง best **0.82618** (17 พ.ย. 2026)

## วิธีรัน

```powershell
pip install -r Task1/requirements.txt
```

เปิดโน้ตบุ๊กแล้ว **Run All** (ใช้ CPU ประมาณ 2 นาที) รันได้ทั้งจาก root หรือจาก `Task1/notebooks/`
โน้ตบุ๊กตรวจ SHA256 ของ `train.csv` ให้ตรงกับรุ่นข้อมูลที่ใช้ตอนได้คะแนน
(`d8dbdcaf8f9d042d2490a0dfd01b730cea3112717131f7f04d573f52958941d4`)

## สูตรที่ใช้

| ส่วน | ค่า |
|---|---|
| ข้อมูลฝึก | 90,816 แถวที่มี label (ตัด 13,184 แถวที่ไม่มี label) |
| Features | feature ดิบ 31 ตัว + สร้างเพิ่ม 21 ตัว (จำนวนค่าว่าง, ratio, interaction) |
| Missingness augmentation | สำเนา train แล้วสุ่มปิดค่า 10% เป็น NaN (seed 42) ต่อท้าย train |
| โมเดล | LightGBM 285 ต้น, lr 0.045, num_leaves 31, min_child_samples 20, reg_lambda 2, colsample/subsample 0.85, seed 42 |
| Threshold | 0.20 |

## เหตุผลที่ใช้ binary F2 และ threshold 0.20

หน้า Overview เขียนว่า "F2 Score (Macro)" แต่เมื่อกลุ่มเปลี่ยนเฉพาะ threshold ของ Task 1 จาก 0.38 (Macro F2 สูงสุด)
เป็น 0.20 (binary F2 สูงสุด) คะแนน public รวมขึ้นจาก 0.72752 เป็น 0.74155 (+0.01403)
ถ้า Kaggle ใช้ Macro F2 คะแนนควรลดลงราว 0.011 จึงสรุปว่า Kaggle คิด F2 ของคลาส 1

## ผล (คะแนนภายใน ไม่ใช่คะแนน Kaggle)

| รายการ | ผล |
|---|---:|
| Development cross-fit binary F2 (5-fold OOF, 72,652 แถว) | 0.8353 |
| Audit binary F2 (18,164 แถว, กันไว้ไม่ใช้เลือกโมเดล) | 0.8394 (bootstrap 95%: 0.8360–0.8432) |
| Audit recall / precision ของคนโกง | 0.9724 / 0.5425 |
| Test ที่ทายว่าโกง | 59.46% |

ทุก candidate ในโน้ตบุ๊ก (raw, engineered, regularized, signal-only, augmentation และ blend)
ได้ cross-fit binary F2 ต่างกันไม่เกิน 0.0013 ซึ่งต่ำกว่าเกณฑ์ 0.002 จึงใช้สูตรที่มีคะแนน Kaggle ยืนยันแล้ว

อัตราทายโกง 59.46% และจำนวนแถวที่เปลี่ยนเมื่อขยับ threshold จาก 0.38 เป็น 0.20 (4,813 แถว)
ตรงกับบันทึกของไฟล์ที่ได้คะแนน จึงยืนยันว่าโน้ตบุ๊กสร้างคอลัมน์ Task 1 ชุดเดียวกัน

## การจับคู่ ID

`test.csv` ใช้ `player_id` ส่วน sample submission ใช้ `ANS00001`–`ANS25889`
จับคู่ตามลำดับแถว: test แถวที่ i ↔ `ANS` แถวที่ i (วิธีเดียวกับไฟล์ที่ส่ง Kaggle แล้วได้คะแนน)
