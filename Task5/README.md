# Task 5 — Account Security Monitoring

- Target ใน submission: `is_anomaly` (0 หรือ 1)
- วิธีการ: Unsupervised anomaly detection (หากลุ่มบัญชีที่แยกตัวด้วย gap detection)
- Metric: F3 (จาก probe บน leaderboard น่าจะคิดเป็น binary F3 ของคลาส 1 ดูหัวข้อ "ประวัติการส่ง Kaggle")
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
- ไม่ใช้ label, pseudo-label, pretrained model หรือ AutoML
- พารามิเตอร์ของ gap detection เป็นค่าทั่วไปที่ตั้งไว้ก่อน ไม่ได้ค้นหาจากคะแนนบน leaderboard แต่ **การเลือกใช้วิธีนี้เกิดขึ้นหลังจากทดลองส่ง Kaggle หลายรอบ** (ดูหัวข้อถัดไป)
- ไม่มี label จึงวัด F3 ในเครื่องไม่ได้

## ประวัติการส่ง Kaggle

Task 5 ไม่มีข้อมูลที่มี label ให้วัดผลในเครื่อง ทีมจึงใช้คะแนน public (คะแนนรวม 5 tasks) เป็น feedback ระหว่างพัฒนา ตามลำดับนี้

| ช่วง | สิ่งที่ส่ง | ผล |
|---|---|---|
| 12–13 พ.ย. | unsupervised ensemble (Isolation Forest, LOF, PCA reconstruction, One-Class SVM และ autoencoder) แล้ว flag ตามอัตราคงที่ 10%, 20%, 30%, 40%, 50% และ 60% | คะแนนรวมดีสุดที่ 40% (0.71802) |
| 15 พ.ย. | **probe 3 ไฟล์** เปลี่ยนเฉพาะคอลัมน์ task5: ทุกแถวเป็น 0, ทุกแถวเป็น 1 และสุ่ม 50% | คะแนนรวม 0.61305 / 0.70604 / 0.67817 ใช้คำนวณว่า metric น่าจะเป็น binary F3 ของคลาส 1 (ไม่ใช่ Macro F3) และประมาณว่า test มีบัญชีผิดปกติราว 8% |
| 15 พ.ย. | gap detection (notebook นี้) 1,808 บัญชี (6.98%) | คะแนนรวม 0.74155 → **0.80401** |

**ข้อควรรู้:**
- probe ไม่ได้ใช้ label รายแถว แต่คะแนนที่ได้มาจาก label ของ test ทางอ้อม ทีมจึงรู้สัดส่วนบัญชีผิดปกติโดยประมาณ (~8%) ก่อนทำ gap detection
- gap detection ไม่ได้ใช้ตัวเลขจาก probe เป็น input และไม่ได้ตั้งเป้าอัตรา flag ไว้ จุดตัดได้มาจากช่องว่างในข้อมูลเอง แต่อัตราที่ได้ (6.98%) ใกล้กับค่าประมาณจาก probe จึงแยกไม่ได้ทั้งหมดว่าความรู้นี้มีผลต่อการตัดสินใจใช้วิธีนี้หรือไม่
- ถือว่าเป็นการใช้ leaderboard เป็น feedback และอธิบายไว้ในรายงานด้วย
