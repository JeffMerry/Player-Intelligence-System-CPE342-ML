# Karena Player Intelligence System — CPE342

โปรเจกต์ Machine Learning สำหรับทีม 5 คน ประกอบด้วย 5 งาน และรวมผลเป็น CSV สำหรับส่ง Kaggle

## โครงสร้างโปรเจกต์

```text
reference/             เอกสารอ้างอิงและข้อกำหนด
data/raw/              ข้อมูลต้นฉบับ แยก Task1–Task5
data/processed/        โฟลเดอร์เสริมสำหรับข้อมูลที่เตรียมแล้ว ปัจจุบันไม่มี สร้างเมื่อจำเป็น
Task1/                 Anti-Cheat Pre-Filter
Task2/                 Player Segment Classification
Task3/                 Player Monthly Spending Prediction
Task4/                 Game Title Detection
Task5/                 Account Security Monitoring
scripts/               สคริปต์ส่วนกลางสำหรับรวมและตรวจ submission
models/                โมเดลที่ฝึกแล้ว แยก Task1–Task5
submissions/           ผลทำนายและ CSV สำหรับส่ง Kaggle
reports/figures/       ภาพและกราฟสำหรับรายงาน
reports/experiments/   บันทึกการทดลอง
```

## เริ่มต้นใช้งาน

อ่าน [คู่มือโปรเจกต์](reference/CPE342_PROJECT_GUIDE.md) และตรวจสอบกติกาทางการก่อนพัฒนา
โครงสร้างพร้อมสำหรับเริ่ม EDA และพัฒนาแต่ละ Task แล้ว ในเครื่องนี้มีข้อมูล Task 1–5
รวมถึง CSV และโฟลเดอร์ภาพ train/test ของ Task 4 ข้อมูลเหล่านี้ไม่ได้ส่งขึ้น Git
สมาชิกแต่ละคนต้องดาวน์โหลดและจัดวางข้อมูลในเครื่องของตนเอง
ยังไม่มี notebook ฝึกโมเดลหรือสคริปต์รวม submission ที่ใช้งานได้

อ่าน [คู่มือโครงสร้างและการแบ่ง branch สำหรับทีม](TEAM_GUIDE.md) ก่อนเริ่มงาน

สร้าง Python environment จากโฟลเดอร์หลัก (ตัวอย่างสำหรับ Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

ไฟล์ requirements เริ่มต้นยังไม่มี dependencies ให้เพิ่มพร้อมระบุเวอร์ชันเมื่อเลือกเครื่องมือที่ใช้จริง
Dependencies เฉพาะงานให้บันทึกใน `TaskN/requirements.txt`

## การทำงานร่วมกัน

1. ผู้ดูแลนำโครงสร้างกลางเข้า `main` ก่อนเริ่มแบ่งงาน
2. สมาชิกอัปเดต `main` แล้วสร้าง branch ตามงาน เช่น `task1/anti-cheat`
3. พัฒนาในโฟลเดอร์ Task ของตนเอง และบันทึกวิธีรันกับผลการทดลอง
4. เปิด Pull Request เข้า `main` และให้เพื่อนตรวจสอบก่อน merge
5. ประสานผู้ดูแลไฟล์กลางเมื่อแก้ requirements หรือสคริปต์รวม submission

ตัวอย่างเริ่มงานหลังโครงสร้างกลางอยู่บน remote แล้ว:

```powershell
git switch main
git pull origin main
git switch -c task1/anti-cheat
```

## ข้อตกลงข้อมูลและผลทำนาย

- เก็บข้อมูลต้นฉบับใน `data/raw/TaskN/` โดยไม่แก้ไขต้นฉบับ
- Fit preprocessing ด้วยข้อมูลฝึกที่กติกาอนุญาตเท่านั้น
- เก็บโมเดลใน `models/TaskN/` และผลทำนายใน `submissions/`
- แต่ละ Task ส่งออก CSV ที่มี `id` และ `taskN` เช่น `id,task1` พร้อมหนึ่งผลทำนายต่อ ID
- รวมผลด้วย ID ตาม `sample_submission.csv` ทางการ พร้อมรักษาลำดับและชื่อคอลัมน์
- ยืนยันการจับคู่ ID ของแต่ละชุดข้อมูลก่อนสร้างผลทำนาย ห้ามสมมติว่าลำดับแถวตรงกัน
- ไม่ commit ข้อมูล โมเดล ผลทำนาย หรือ secrets ซึ่งถูกยกเว้นไว้ใน `.gitignore`
- ใช้ pretrained models ได้เฉพาะ Task 4 และไม่ใช้ AutoML
- Task 5 ต้องยืนยันวิธี fit ข้อมูลกับกติกาทางการก่อน เพราะคู่มือระบุทั้ง test-only และข้อห้ามฝึกบน test

## สถานะเริ่มต้น

ขั้นถัดไปคือทำ EDA และ baseline ใน notebook ของแต่ละ Task แล้วเพิ่มสคริปต์รวมและตรวจ submission
`train.py` และ `predict.py` สามารถแยกเพิ่มภายหลังได้เมื่อจำเป็น ไม่บังคับถ้า notebook รันซ้ำได้ครบ
ต้องยืนยัน metric ของ Task 2, รูปแบบผลทำนาย Task 4, การจับคู่ submission ID
และข้อกำหนดการ fit ข้อมูล Task 5 ก่อนดำเนินงานส่วนที่เกี่ยวข้อง
