# ข้อมูลสำหรับฝึกซ้ำ

เมื่อใช้โครงสร้าง repository ให้เตรียม `data/raw/Task4/` ที่ project root (ข้าง Task1–Task5)
ไม่ใช่ `Task4/data/raw/Task4/` ส่วนชุด ZIP แบบ standalone ใช้ data/raw/Task4 ใต้ root ของชุดส่ง

Inference ใช้เฉพาะภาพที่ต้องทำนาย ไม่ต้องมี Train
หากจะทำซ้ำ training ให้เตรียมชุดข้อมูลรายวิชารุ่นเดียวกับที่ระบุ SHA256 ใน config/ensemble.json ใต้ root ของชุดส่ง:

data/raw/Task4/train.csv (id,file_name,label)
data/raw/Task4/test.csv (id,file_name,label)
data/raw/Task4/train/<file_name>
data/raw/Task4/test/<file_name>

ภาพทั้งชุด 230×120 RGB JPEG; Train 31,546 / Test 25,889
ใช้ file_name จับคู่ภาพ ไม่ใช่ {id}.jpg
มี split เดิมใน training_assets ไม่ต้องสร้างใหม่เพื่อทำซ้ำการเปรียบเทียบเดิม
ไม่รวมภาพ Train/Test ทั้งชุดใน ZIP นี้เพื่อลดขนาด ให้ใช้ข้อมูลรายวิชาที่อาจารย์มีหรือไฟล์ที่ทีมส่งตามช่องทางที่กำหนด
