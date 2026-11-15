# Task 5 — Account Security Monitoring

- Target ใน submission: `is_anomaly` (0 หรือ 1)
- วิธีการ: Unsupervised anomaly detection
- Metric: F3
- ข้อมูล: `../data/raw/Task5/`
- โมเดล: `../models/Task5/`
- ผลทำนาย: `../submissions/task5.csv` คอลัมน์ `id,task5`

## ข้อกำหนดที่ต้องยืนยันก่อน fit

คู่มือระบุว่ามีเพียง `test.csv` แต่มีข้อห้ามฝึกบน test ด้วย
ต้องตรวจคำชี้แจงทางการก่อนเลือกข้อมูลสำหรับ fit โมเดลและ preprocessing
ห้ามถือว่า unsupervised learning ได้รับการยกเว้นโดยอัตโนมัติ

## งานที่ต้องเพิ่ม

1. ตรวจ schema และบันทึกข้อสรุปเรื่องข้อมูลที่อนุญาตให้ fit
2. ทำ EDA ใน `notebooks/` โดยไม่ใช้ hidden labels
3. เพิ่ม `train.py` หรือขั้นตอน fit ที่สอดคล้องกับกติกาที่ยืนยันแล้ว
4. เพิ่ม `predict.py` พร้อมบันทึกหลักเกณฑ์ threshold และการจับคู่ ID
5. บันทึก dependencies, คำสั่งรัน, seed, พารามิเตอร์ และข้อจำกัดการประเมิน

ยังไม่มีโค้ดฝึกหรือทำนาย ไม่สร้าง labels หรือคะแนน validation ขึ้นเอง
