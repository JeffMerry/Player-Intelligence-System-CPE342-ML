# Task 1 — บันทึกการทดลอง

ทุกผลเป็น **5-fold StratifiedKFold (seed 42) out-of-fold** บน 90,816 แถวที่มี label
Metric คือ **Macro F2** (ตาม Kaggle "F2 Score (Macro)") และเลือก threshold บน OOF
ส่วนต่างที่น้อยกว่า ~0.003 ถือว่าอยู่ในระดับ noise เพราะ Macro F2 รายfold ต่างกันประมาณ ±0.003

## ผลการทดลอง

| # | โมเดล / วิธี | AUC | Macro F2 | หมายเหตุ |
|---|---|---:|---:|---|
| 1 | Logistic regression (median impute + scale) | 0.765 | – | baseline เชิงเส้น |
| 2 | Histogram Naive Bayes (64 quantile bins/feature) | 0.846 | 0.738 | สมมติว่า feature อิสระต่อกัน |
| 3 | WoE encoding + logistic regression | 0.848 | 0.739 | โมเดลแบบบวก (scorecard) |
| 4 | LightGBM, num_leaves 3 (เกือบเป็นแบบบวก) | 0.889 | 0.777 | |
| 5 | LightGBM + WoE features | 0.886 | 0.775 | ไม่ช่วย |
| 6 | LightGBM + engineered ratios (reports/day, kd×hs, …) | 0.891 | – | ไม่ช่วย (F2 คลาส 1 เท่าเดิม) |
| 7 | LightGBM random search 25 ชุด (early stopping) | ≤ 0.891 | ≤ 0.779 | ไม่มีชุดไหนดีกว่าค่าที่ใช้ |
| 8 | LightGBM (num_leaves 31, lr 0.02, 1300 trees), 1 seed | 0.891 | 0.779 | |
| 9 | XGBoost (depth 6, lr 0.02, 1000 trees), 1 seed | 0.890 | 0.779 | |
| **10** | **LightGBM + XGBoost, เฉลี่ย 3 seeds ต่อโมเดล (ที่ใช้ส่ง)** | **0.892** | **0.780** | threshold 0.405 |

## สิ่งที่เรียนรู้

- **Metric:** ตอนแรกเลือก threshold ด้วย F2 ของคลาส 1 อย่างเดียว ได้ threshold 0.185 ซึ่งทายว่าโกง 60% ของ test
  เมื่อเปลี่ยนเป็น Macro F2 ตามหน้า Kaggle ได้ threshold ~0.40 ต้องอ่าน metric จากหน้า Kaggle ให้ละเอียดก่อนจูน
- **ค่าว่างเกิดแบบสุ่ม:** อัตราคนโกงแทบเท่ากันไม่ว่าจะมีค่าว่างหรือไม่ จึงไม่ต้องสร้าง missing indicator
  และปล่อยให้ boosting จัดการ NaN เอง
- **ช่วงค่าเฉพาะคลาส:** 6 feature มีช่วงค่าที่พบในคลาสเดียว ครอบคลุม ~33% ของแถว tree model จับขอบเขตเหล่านี้ได้
  ด้วย split เดียว จึงเป็นเหตุผลหนึ่งที่ boosting ดีกว่าโมเดลเชิงเส้นมาก
- **ยังมี interaction:** Naive Bayes ที่ใช้ distribution ของแต่ละ feature ได้ AUC เพียง 0.846 ขณะที่ boosting ได้ 0.891
  แสดงว่าสัญญาณบางส่วนอยู่ใน interaction ระหว่าง feature
- **ชนเพดาน:** ทุกโมเดลกลุ่ม boosting ได้ AUC ราว 0.89 การจูนและการสร้าง feature เพิ่มไม่ช่วยอย่างมีนัยสำคัญ
  การเฉลี่ยหลาย seed ทำให้ผลนิ่งขึ้นเล็กน้อย

## ยังไม่ได้ลอง

- ใช้ 13,184 แถวที่ไม่มี label แบบ semi-supervised (ต้องตรวจกับกฎก่อน กฎห้าม pseudo-label บน test
  แต่แถวเหล่านี้อยู่ใน train)
- CatBoost (ยังไม่ได้ติดตั้ง)
