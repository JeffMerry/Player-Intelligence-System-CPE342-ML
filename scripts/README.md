# สคริปต์ส่วนกลาง

โฟลเดอร์นี้เตรียมไว้สำหรับสคริปต์ที่จะเพิ่มหลังตรวจข้อมูลจริง:

- `merge_submission.py`: รวม CSV ของ Task1–Task5 ด้วย ID ตาม sample submission ทางการ
- `validate_submission.py`: ตรวจชื่อและลำดับคอลัมน์, จำนวนแถว, ID ซ้ำหรือขาด,
  missing values, ค่าที่ไม่เป็น finite, classes และช่วงค่าที่กติกากำหนด

ผลลัพธ์ต้องมีคอลัมน์ `id,task1,task2,task3,task4,task5`
ใช้จำนวนแถวและลำดับ ID จาก sample submission จริง ห้าม hardcode จากตัวอย่างในคู่มือ
ขณะนี้ยังไม่ได้สร้างสคริปต์ทั้งสองรายการ
