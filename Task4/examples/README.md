# ตัวอย่าง inference

รูปห้าภาพและ input/sample/expected CSV อยู่ใน ZIP artifacts แยกจาก Git
กู้ด้วย `python Task4/restore_artifacts.py --archive <path-to-Task4_submission.zip>` จาก project root
จากนั้นรัน `notebooks/inference.ipynb` หรือคำสั่ง predict ใน README หลักของ Task4

ตัวอย่างมาจาก Test ไม่มี ground-truth labels จึงใช้ตรวจการรันซ้ำ ไม่ใช่คำนวณ accuracy
