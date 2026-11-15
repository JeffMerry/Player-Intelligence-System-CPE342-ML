# Task 3 — Player Monthly Spending Prediction

- Target: `spending_30d` (บาท)
- Metric: Normalized MAE score ตามคู่มือ
- ข้อมูล: `../data/raw/Task3/`
- โมเดล: `../models/Task3/`
- ผลทำนาย: `../submissions/task3.csv` คอลัมน์ `id,task3`

## งานที่ต้องเพิ่ม

1. ตรวจ schema, missing values, zero spending, outliers และความเสี่ยง leakage
2. ทำ EDA ใน `notebooks/` และกำหนด validation split
3. เพิ่ม `train.py` และเปรียบเทียบ baseline ด้วย metric ทางการ
4. เพิ่ม `predict.py` ตรวจ inverse transform, ค่าติดลบ และการจับคู่ ID
5. บันทึก dependencies, คำสั่งรัน, seed, พารามิเตอร์ และผลทดลอง

ยังไม่มีโค้ดฝึกหรือทำนาย ใช้โมเดลที่ฝึกจากข้อมูลที่อนุญาตเท่านั้น
