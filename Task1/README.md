# Task 1 — Anti-Cheat Pre-Filter

- Target: `is_cheater` (0 หรือ 1)
- Metric: **F2 แบบ Macro** (เฉลี่ย F2 ของคลาส 0 และ 1 ตามหน้า Overview ของ Kaggle; PDF เขียนแค่ "F2")
- ข้อมูล: `../data/raw/Task1/` (`train.csv`, `test.csv`, `datacard.txt` จาก `01_binary_classification/`) และ `../data/raw/sample_submission.csv`
- ผลทำนาย: `../submissions/task1.csv` คอลัมน์ `id,task1`

## วิธีรัน (จาก root ของโปรเจกต์)

```powershell
pip install -r Task1/requirements.txt
python Task1/train.py     # 5-fold CV, เลือก threshold, บันทึกโมเดลและ OOF (~4 นาที)
python Task1/predict.py   # ทำนาย test และเขียน submissions/task1.csv
```

จากนั้นเปิด notebook เพื่อดูการวิเคราะห์:

| ไฟล์ | เนื้อหา |
|---|---|
| [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) | label ที่หาย, สัดส่วนคลาส, ค่าว่าง, feature ที่แยกคลาสได้, ช่วงค่าเฉพาะคลาส |
| [notebooks/02_model_analysis.ipynb](notebooks/02_model_analysis.ipynb) | การเลือก threshold, confusion matrix, distribution ของ probability, feature importance (ต้องรัน `train.py` ก่อน) |

## ไฟล์ที่สร้าง

| ไฟล์ | เนื้อหา |
|---|---|
| `models/Task1/ensemble.joblib` | โมเดล 5 fold × (LightGBM + XGBoost) × 3 seeds, รายชื่อ feature, threshold |
| `models/Task1/oof.npz` | out-of-fold probabilities |
| `reports/experiments/task1_cv.json` | ผล CV, พารามิเตอร์, seed, เวอร์ชัน library |
| `reports/experiments/task1_feature_importance.csv` | LightGBM gain |
| `reports/experiments/task1_experiments.md` | บันทึกทุกการทดลองที่ลองและสิ่งที่เรียนรู้ |
| `reports/figures/task1_*.png` | กราฟสำหรับรายงาน (สร้างจาก notebook) |

## ผลสำรวจข้อมูลที่สำคัญ

- train 104,000 แถว แต่ **13,184 แถว (12.7%) ไม่มี label** จึงตัดออก เหลือ 90,816 แถว เป็นคนโกง 34.9%
- 98% ของแถวมีค่าว่างอย่างน้อย 1 คอลัมน์ สัดส่วนค่าว่างใน train กับ test ต่างกันมาก แต่ไม่สัมพันธ์กับ target
  จึงไม่สร้าง missing indicator และปล่อยให้ LightGBM/XGBoost จัดการ NaN เอง
- **6 feature มีช่วงค่าที่พบเฉพาะคลาสเดียว** เช่น `crosshair_placement` < 0.82 เป็นคนโกงทั้งหมด
  ส่วนบัญชีอายุเกิน 1,000 วัน เพื่อนเกิน 269 คน หรือได้รับรายงานเกิน 19 ครั้งเป็นผู้เล่นปกติทั้งหมด
  ครอบคลุมราว 33% ของแถว และเป็น 5 feature ที่สำคัญที่สุดของโมเดล
- ในกลุ่มบัญชีที่ถูกรายงาน คนโกงมี K/D, headshot และ accuracy **ต่ำกว่า**ผู้เล่นปกติเล็กน้อย
  ตรงกับโจทย์ที่บอกว่าบัญชีที่ถูกรายงานจำนวนมากเป็นผู้เล่นเก่งที่ไม่ได้โกง
- Tree model ดีกว่าโมเดลเชิงเส้นและ Naive Bayes มาก (AUC 0.89 เทียบกับ 0.76 และ 0.85)

## วิธีการ

- ใช้ feature ดิบทั้ง 31 ตัว (feature ที่สร้างเพิ่มไม่ช่วย)
- Ensemble: LightGBM และ XGBoost แต่ละตัวเฉลี่ย 3 seeds (42, 7, 2026) แล้วเฉลี่ยสองโมเดล
- 5-fold StratifiedKFold (seed 42) เลือก threshold บน OOF ให้ Macro F2 สูงสุด
- test ใช้ค่าเฉลี่ยของโมเดลทั้ง 5 fold

## ผล 5-fold CV (out-of-fold, Macro F2)

| Model | AUC | Macro F2 | Threshold |
|---|---:|---:|---:|
| LightGBM (3 seeds) | 0.8916 | 0.7793 | 0.380 |
| XGBoost (3 seeds) | 0.8904 | 0.7797 | 0.415 |
| **LGBM + XGB (ที่ใช้ส่ง)** | **0.8916** | **0.7799** | **0.405** |

ที่ threshold 0.405: precision 0.685, recall 0.750, Macro F2 รายfold 0.774–0.783
threshold ที่ให้ Macro F2 ห่างจากค่าสูงสุดไม่เกิน 0.002 อยู่ในช่วง 0.365–0.425 จึงไม่ไวต่อการเลือก threshold
บน test ทายว่าโกง 37.8% ใกล้กับ 38.2% ที่ CV ทายไว้

รายละเอียดการทดลองทั้งหมดอยู่ใน `reports/experiments/task1_experiments.md`

## ข้อควรยืนยัน

- **การจับคู่ ID:** test ใช้ `player_id` แต่ sample ใช้ `ANS00001...` หน้า Data ของ Kaggle เขียนว่า
  test ทุก Task ใช้ช่วง ID เดียวกัน (ANS00001–ANS25889) จึงจับคู่ตามลำดับแถว (test แถวที่ i → `ANS` แถวที่ i)
  ทั้ง PDF และหน้า Kaggle ไม่ได้ระบุตรงๆ ควรยืนยันเมื่อส่งรวมครั้งแรก
- **Macro หรือไม่:** Kaggle ระบุ "F2 Score (Macro)" แต่ PDF เขียนแค่ "F2" ถ้าคะแนน leaderboard ของ Task 1
  ต่ำกว่าที่คาดมาก ให้ลองเปลี่ยนไปเลือก threshold ด้วย F2 ของคลาส 1 (threshold ~0.19)
- **ตัวคั่นไฟล์:** หน้า Data เขียนว่าใช้ `|` แต่ `sample_submission.csv` ที่ดาวน์โหลดมาใช้ `,`
