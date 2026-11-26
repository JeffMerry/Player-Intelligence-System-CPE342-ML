# Task 3 — Player Monthly Spending Prediction

- Target: `spending_30d` (ยอดใช้จ่าย 30 วันถัดไป หน่วยบาท)
- Metric: **NMAE score** = `1 / (1 + sum(|y - ŷ|) / sum(y))` ยิ่งสูงยิ่งดี (สไลด์หน้า 19 และ 25)
- ข้อมูล: `../data/raw/Task3/` (`train.csv`, `test.csv`, `datacard.txt`) และ `../data/raw/sample_submission.csv`
- โค้ด: [notebooks/task-3-monthly-spending-prediction.ipynb](notebooks/task-3-monthly-spending-prediction.ipynb)
- ผลทำนาย: `../submissions/task3.csv` คอลัมน์ `id,task3` (และ `task3_predictions_detail.csv` ที่มี `player_id`)
- ตารางผลระหว่างทางและ run manifest: `../models/Task3/`
- กราฟ: `../reports/figures/Task3/`
- บันทึกการทดลอง: `../reports/experiments/task3_spending.md` และ `../reports/experiments/task3_run_metadata.json`

## วิธีรัน

```powershell
pip install -r Task3/requirements.txt
```

เปิด notebook ด้วย kernel ที่ติดตั้ง requirements แล้ว แล้วกด **Restart → Run All** (CPU ประมาณ 17 นาที รวม MLP) รันได้ทั้งจาก root หรือจาก `Task3/notebooks/`
notebook หา root ของ repo จาก `TEAM_GUIDE.md` และตรวจขนาดข้อมูลด้วย assert ก่อนเริ่ม

ทดสอบแล้วด้วย Python 3.14.10 และ 3.13.3 ได้ผลตรงกันทุกค่า (outputs ที่บันทึกใน notebook รันด้วย 3.13.3) เวอร์ชันไลบรารีระบุไว้ใน `requirements.txt`

## สูตรที่ใช้

| ส่วน | ค่า |
|---|---|
| ข้อมูลฝึก | 104,000 แถว แบ่ง dev 88,400 / locked holdout 15,600 (stratified ตามช่วงยอดเงิน, seed 342) |
| Features | feature ดิบ 32 ตัว + สร้างเพิ่ม 30 ตัว (ยอดซื้อสร้างใหม่, ratio, interaction, recency, seasonality, log) |
| โมเดลที่เทียบ | 17 แบบ: Dummy, Ridge, ElasticNet, RF, ExtraTrees, LightGBM/XGBoost/CatBoost (L1), log-target, Tweedie, two-stage |
| Tuning | Optuna TPE 20 trials บน LightGBM L1 |
| Validation | Stratified 3-fold CV บน dev → OOF predictions |
| Tree ensemble | weighted ensemble: two-stage mixture median 0.51 + LightGBM Optuna 0.39 + LightGBM raw 0.10 |
| Scratch MLP | 186 inputs (62 features + signed log1p + missing indicators), 3×128 ReLU, dropout 0.1, L1 loss บน y/10,000, AdamW, seed 20261116, CPU deterministic; จำนวน epoch เลือกด้วย inner early stopping (dev ได้ 12) |
| **โมเดลสุดท้าย** | **tree ensemble 75% + MLP 25%** (น้ำหนักตรึงไว้ก่อนดูผล ไม่ได้จูน) |
| Final fit | train ทั้งหมด 104,000 แถว จำนวนรอบ/epoch ตรึงจาก development, คำทำนายไม่ต่ำกว่า 0 |

**Two-stage mixture median:** classifier ทำนายความน่าจะเป็น p ที่จะจ่าย และ quantile regressors (0.05–0.70) ทำนายยอดของคนที่จ่าย
ถ้า p ≤ 0.5 ทาย 0 บาท ถ้าไม่ใช่ทาย quantile ที่ระดับ `1 − 0.5/p` ซึ่งเป็น median ของการแจกแจงผสม เหมาะกับ metric แบบ absolute error

## ผล (คะแนนภายใน ไม่ใช่คะแนน Kaggle)

| รายการ | ผล |
|---|---:|
| Dev OOF MAE ของ tree ensemble (3-fold, 88,400 แถว) | 2,860.76 บาท (NMAE 0.785820) |
| **Dev OOF ของ blend 75/25** (ดีขึ้นทุก fold) | **2,839.66 บาท (NMAE 0.787064)** |
| **Locked holdout NMAE** ของ tree ensemble (15,600 แถว, ไม่ใช้เลือกโมเดล) | **0.78903** |
| Locked holdout ของ blend 75/25 (เช็คซ้ำหลังเปิด holdout แล้ว) | NMAE 0.78942, MAE 2,801.63 บาท |
| Locked holdout MAE / RMSE / R² | 2,808.12 บาท / 13,297.09 บาท / 0.8104 |
| Holdout MAE bootstrap 95% | 2,607–3,010 บาท |
| Baseline ทาย median ทุกคน (dev validation) | MAE 10,502 บาท |

## คะแนน Kaggle

`submissions/task3.csv` จาก notebook นี้ตรงกับคอลัมน์ Task 3 ในไฟล์ที่ส่ง Kaggle และได้คะแนนรวม public **0.82618** ทุกแถว (ต่างกันไม่เกิน 1e-8 บาท ซึ่งมาจากการปัดทศนิยม 8 หลักตอนเขียน CSV)
Kaggle รายงานเป็นคะแนนรวม 5 tasks จึงไม่ทราบคะแนน Task 3 อย่างเดียว

## การจับคู่ ID

`test.csv` ใช้ `player_id` ส่วน sample submission ใช้ `ANS00001`–`ANS25889`
จับคู่ตามลำดับแถว: test แถวที่ i ↔ sample แถวที่ i (notebook ไม่ sort และไม่ merge test) วิธีเดียวกับ Task 1

## ข้อจำกัด

- ผู้เล่น top 1% สร้าง error ประมาณ 24% และโมเดลทายกลุ่มนี้ต่ำกว่าจริง ส่วนกลุ่มที่ทายว่าจ่ายแต่จริงไม่จ่ายสร้าง error รวมประมาณ 31%
- ข้อมูลไม่มี timestamp จึงยืนยันไม่ได้ว่า features ย้อนหลังและ `segment` เกิดก่อนช่วง 30 วันที่ทำนายทุกแถว
- ใช้เฉพาะข้อมูล train ที่มี label ไม่ใช้ test ฝึกหรือเลือกโมเดล ไม่ใช้ pseudo-label และไม่ใช้ AutoML
