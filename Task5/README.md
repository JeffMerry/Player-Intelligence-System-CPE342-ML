# Task 5 — Account Security Monitoring

- Target ใน submission: `is_anomaly` (0 หรือ 1)
- วิธีการ: Unsupervised anomaly detection (หากลุ่มบัญชีที่แยกตัวด้วย gap detection)
- Metric: F3
- ข้อมูล: `../data/raw/Task5/test.csv`
- Notebook: [`notebooks/task5_anomaly_detection.ipynb`](notebooks/task5_anomaly_detection.ipynb)
- ผลทำนาย: `../submissions/task5.csv` คอลัมน์ `id,task5`

## วิธีรัน

1. วาง `test.csv` (และ `datacard.txt`) จาก Kaggle ไว้ที่ `data/raw/Task5/`
2. ติดตั้ง dependencies: `pip install -r Task5/requirements.txt`
3. เปิด notebook แล้ว Restart & Run All (ใช้เวลาไม่ถึง 1 นาทีบน CPU)

ถ้ามี `data/raw/sample_submission.csv` notebook จะใช้ id จากไฟล์นั้น ถ้าไม่มีจะสร้าง `ANS00001`… ตามลำดับแถวของ `test.csv`
ถ้าวางไฟล์รวมที่ส่ง Kaggle ไว้ที่ `submissions/final_submission.csv` เซลล์สุดท้ายจะเช็คว่าคอลัมน์ `task5` ตรงกันทุกแถว

## สรุปวิธี

1. สร้างสถิติต่อบัญชีจาก feature 4 ช่วงเวลา: median ของ 4 ช่วง (`_med`) และช่วง 4 − ค่าเฉลี่ยช่วง 1–3 (`_jump`)
2. ในหางซ้าย/ขวาข้างละไม่เกิน 5% หาช่องว่างที่กว้างที่สุด ถ้ากว้าง ≥ 0.2×IQR ถือเป็นกลุ่มที่แยกตัว
3. Flag = union ของทุกกลุ่ม → 19 กลุ่ม, 1,808 บัญชี (6.98%) จัดเป็น 5 ชนิดที่ไม่ซ้อนกัน
4. เทียบกับ Isolation Forest (คำใบ้ในสไลด์) ซึ่งจับได้ดีเฉพาะชนิด sudden change

พารามิเตอร์: `MAX_TAIL=0.05`, `MIN_TAIL=0.002`, `MIN_GAP=0.2`, `MIN_UNIQUE=200` ผลเหมือนเดิมทุกค่า `MIN_GAP ≤ 0.2` และ `MAX_TAIL` 0.03–0.07
ส่วนที่เป็นวิธีหลักไม่มีการสุ่ม (Isolation Forest ที่ใช้เทียบตั้ง `random_state=42`)

## ข้อมูลที่ใช้และกติกา

- Task 5 ได้มาเพียง `test.csv` โดยไม่มี label ตามโจทย์ จึงใช้ feature ของ `test.csv` แบบ unsupervised เท่านั้น
- ไม่ใช้ label, pseudo-label, pretrained model หรือ AutoML และไม่มีพารามิเตอร์ใดตั้งจากคะแนนบน leaderboard
- ไม่มี label จึงวัด F3 ในเครื่องไม่ได้
