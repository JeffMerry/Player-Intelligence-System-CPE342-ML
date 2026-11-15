# Task 2 — Player Segment Classification

- Target: `segment` (0 Casual, 1 Grinder, 2 Social, 3 Whale)
- Metric: F1 — ต้องยืนยัน averaging จากกติกาทางการ
- ข้อมูล: `../data/raw/Task2/`
- โมเดล: `../models/Task2/`
- ผลทำนาย: `../submissions/task2.csv` คอลัมน์ `id,task2`

## งานที่ต้องเพิ่ม

1. ตรวจ schema, missing values, class distribution และความเสี่ยง leakage
2. ทำ EDA ใน `notebooks/` และกำหนด stratified validation ตามความเหมาะสม
3. เพิ่ม `train.py` และประเมินด้วย metric ที่ยืนยันแล้ว
4. เพิ่ม `predict.py` พร้อมตรวจการจับคู่ ID กับ sample submission
5. บันทึก dependencies, คำสั่งรัน, seed, พารามิเตอร์ และผลทดลอง

ยังไม่มีโค้ดฝึกหรือทำนาย ใช้โมเดลที่ฝึกจากข้อมูลที่อนุญาตเท่านั้น
