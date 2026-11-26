# โมเดลและข้อมูลประกอบที่แยกจาก Git

Source code, notebooks พร้อม outputs, config, สรุปผล และ training logs อยู่ใน Git
ไฟล์ขนาดใหญ่และข้อมูลที่มี IDs/labels ทั้งชุดอยู่ใน `Task4_artifacts.zip` แยก ไม่ต้องฝึกใหม่หลัง clone
ZIP ไม่มีโค้ด Python, notebooks, requirements, config หรือ README ซ้ำกับ GitHub จึงต้องใช้ร่วมกับ source ที่ clone มา

**ดาวน์โหลด: [โฟลเดอร์ Google Drive](https://drive.google.com/drive/folders/1J7OTKaxK3lW-8vFKFq-Dkxw8OGuoY9or?usp=sharing)**
เปิดโฟลเดอร์แล้วเลือก `Task4_artifacts.zip` ลิงก์นี้เป็นหน้าโฟลเดอร์ ไม่ใช่ลิงก์ดาวน์โหลด ZIP โดยตรง

## หลัง clone

ดาวน์โหลด `Task4_artifacts.zip` จาก Google Drive แล้ววางใน `Task4/dist/` ของโปรเจกต์ (สร้างโฟลเดอร์ได้)
ตรวจ hash ตาม `config/artifacts.json`
ค่า `download_url` ใน `config/artifacts.json` ชี้ไปยังโฟลเดอร์นี้เพื่อให้ผู้ใช้ดาวน์โหลดเอง สคริปต์ restore ทำงานกับ ZIP ในเครื่องเท่านั้น
ในเครื่องผู้จัดทำมี ZIP ที่ `Task4/dist/Task4_artifacts.zip`

จาก project root รัน:

```powershell
python Task4/restore_artifacts.py --archive Task4/dist/Task4_artifacts.zip
```

หาก ZIP อยู่ตำแหน่งอื่น ให้เปลี่ยนค่า `--archive` คำสั่งนี้ใช้ Python standard library ไม่ต้องติดตั้ง PyTorch ก่อน
จะกู้เฉพาะ artifacts ที่อยู่ใน whitelist และตรวจ SHA256 ทุกไฟล์
ไฟล์ใน ZIP มี prefix `Task4_submission/` เพื่อรองรับ restore โดยสคริปต์จะนำเข้า `Task4/` อัตโนมัติ ไม่ต้องแตกไฟล์เอง
ไม่ดึงโค้ด/notebooks/README/checksums จาก ZIP มาทับ source ที่อยู่ใน Git
ไฟล์ที่มีอยู่และตรง hash จะถูกข้าม ถ้าต่างจะหยุดโดยไม่เขียนทับ

## Artifacts ที่กู้

- `checkpoints/*.pt`: DINOv2-Base และ ConvNeXt-Tiny หลัง full-train refit
- `examples/images/` และ CSV ตัวอย่างห้าภาพ
- `results/task4.csv` และ `results/*.npz`: ผลเต็มชุด/outer probabilities
- `training_assets/`: split, official sample และ reference probabilities สำหรับทำซ้ำ

ไม่มีภาพ Train/Test ทั้งชุดใน ZIP นี้ ถ้าจะฝึกซ้ำให้เตรียมข้อมูลตาม `data/README.md`
ผลเต็มชุดเป็น `id,task4` 25,889 แถว ไม่ใช่ submission รวมทั้งห้า tasks

## สร้าง ZIP ใหม่

จาก project root รัน `python Task4/scripts/package_task4_artifacts.py`
สคริปต์สร้าง ZIP เฉพาะไฟล์ประกอบและไฟล์ `.zip.sha256` พร้อมอัปเดต manifest โดยไม่ฝึกโมเดล
ไม่เขียนทับ ZIP เดิม หากต้องการสร้างอีกชุดใช้ `--output Task4/dist/Task4_artifacts_v2.zip` แล้วปรับชื่อในคู่มือให้ตรงกัน
เมื่ออัปโหลดแล้วให้เติมลิงก์ Google Drive ที่หน้านี้และค่า `download_url` ใน `config/artifacts.json`
