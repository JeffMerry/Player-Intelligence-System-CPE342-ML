# Task 4 — Game Title Detection

- Target: `game_title`
- Classes: `7-11K`, `fiveN`, `freefried`, `pube`, `roblock`
- Metric: Macro F1
- ข้อมูล: `../data/raw/Task4/`
- โมเดล: `../models/Task4/`
- ผลทำนาย: `../submissions/task4.csv` คอลัมน์ `id,task4`

## งานที่ต้องเพิ่ม

1. ตรวจไฟล์ภาพ ขนาดภาพ labels และภาพซ้ำที่อาจทำให้ validation รั่ว
2. ทำ EDA ใน `notebooks/` และกำหนด validation split
3. เพิ่ม `train.py` พร้อม transforms และ class mapping ที่ชัดเจน
4. เพิ่ม `predict.py` พร้อมตรวจชื่อคลาสและการจับคู่ภาพกับ submission ID
5. บันทึก dependencies, คำสั่งรัน, seed, พารามิเตอร์ และผลทดลอง

ยังไม่มีโค้ดฝึกหรือทำนาย Task นี้อนุญาต pretrained models ตามคู่มือ
