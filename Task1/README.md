# Task 1 — Anti-Cheat Pre-Filter

- Target: `is_cheater` (0 หรือ 1)
- Metric: F2
- ข้อมูล: `../data/raw/Task1/`
- โมเดล: `../models/Task1/`
- ผลทำนาย: `../submissions/task1.csv` คอลัมน์ `id,task1`

## งานที่ต้องเพิ่ม

1. ตรวจ schema, missing values, class imbalance และความเสี่ยง leakage
2. ทำ EDA ใน `notebooks/` และกำหนด validation split
3. เพิ่ม `train.py` สำหรับ baseline และเลือก threshold จาก validation ด้วย F2
4. เพิ่ม `predict.py` พร้อมตรวจการจับคู่ ID กับ sample submission
5. บันทึก dependencies, คำสั่งรัน, seed, พารามิเตอร์ และผลทดลอง

ยังไม่มีโค้ดฝึกหรือทำนาย ใช้โมเดลที่ฝึกจากข้อมูลที่อนุญาตเท่านั้น
