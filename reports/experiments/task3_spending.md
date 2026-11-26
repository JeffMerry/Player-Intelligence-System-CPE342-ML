# Task 3 — Monthly Spending Prediction: บันทึกการทดลอง

## ข้อมูลการรัน

| รายการ | ค่า |
|---|---|
| ผู้รับผิดชอบ | Kawinpop |
| วันที่รัน | 2026-11-26 |
| Base commit | `68b29dc` (branch `task3/spending`) |
| Notebook | `Task3/notebooks/task-3-monthly-spending-prediction.ipynb` |
| คำสั่งรัน | `pip install -r Task3/requirements.txt` แล้ว Restart → Run All (CPU ประมาณ 13 นาที) |
| Environment | Python 3.14.10; numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, optuna 5.0.0 |
| Seed | 342 (split, models, Optuna TPE, bootstrap) |
| Metadata เต็ม | `reports/experiments/task3_run_metadata.json` |

## เวอร์ชันข้อมูล

ประกาศ TA ระบุว่า Task 3 ไม่มีการเปลี่ยน dataset

| ไฟล์ | SHA256 |
|---|---|
| `train.csv` (104,000 × 34) | `fa62c4250acb525d61feae7f53ef7256f58ee38860cc7a172ccc63016a4e314e` |
| `test.csv` (25,889 × 33) | `12c8ae5c09353b9c644d858f2cc6115a606336a6b3e0fac7a55e5f903300caef` |
| `datacard.txt` | `caba9ef45ae9e6e004cf788ac896b26a2b7333c66573363beb135d65bc7b18c5` |

## Metric

`NMAE score = 1 / (1 + sum(|y − ŷ|) / sum(y))` ยิ่งสูงยิ่งดี เมื่อวัดบนแถวชุดเดียวกัน `sum(y)` คงที่ การลด MAE จึงเท่ากับการเพิ่ม NMAE
ใช้ MAE หน่วยบาทเป็นเกณฑ์เลือกโมเดลและ early stopping ส่วน RMSE กับ R² ใช้ประกอบการอ่านผลเท่านั้น

## Validation

1. กัน **locked holdout 15%** (15,600 แถว) ก่อนเริ่มทุกอย่าง โดย stratify ตามช่วงยอดเงิน 8 ช่วง
2. แบ่ง dev 88,400 แถวเป็น `dev_fit` 70,720 / `dev_validation` 17,680 แถว สำหรับเปรียบเทียบโมเดล, early stopping และ Optuna
3. นำ shortlist เข้า **Stratified 3-fold CV** บน dev ด้วยจำนวนรอบที่ตรึงไว้ (best rounds × 1.1) เพื่อสร้าง OOF predictions
4. เลือกวิธีรวมโมเดลจาก OOF **ก่อน** เปิด holdout แล้ววัด holdout ครั้งเดียว
5. Fit preprocessing (imputer/scaler ของโมเดลเชิงเส้น) เฉพาะใน training partition ส่วน feature engineering คำนวณภายในแถว

## Features

- ดิบ 32 ตัว (ไม่ใช้ `player_id`)
- สร้างเพิ่ม 30 ตัว: `missing_count`, ยอดซื้อสร้างใหม่ (`total_transactions × avg_transaction_value`) และ `history_with_transaction_fallback`, ratio ต่อ transaction/วัน/ชั่วโมงเล่น, interaction ของยอดล่าสุดกับ VIP/premium/event, recency ของการซื้อ, `season_sin/cos`, log1p ของ 5 คอลัมน์ และ `history_discrepancy`

## ผลเปรียบเทียบโมเดล (dev_validation, 17,680 แถว)

| โมเดล | MAE (บาท) | R² |
|---|---:|---:|
| two_stage_mixture_median | 2,865.08 | 0.806 |
| lightgbm_optuna_l1 | 2,872.87 | 0.807 |
| lightgbm_fe_l1 (= quantile50 = upper_clip) | 2,902.80 | 0.806 |
| catboost_fe_l1 | 2,941.94 | 0.805 |
| two_stage_gamma | 2,955.42 | 0.808 |
| xgboost_raw_l1 | 2,956.45 | 0.801 |
| lightgbm_raw_l1 | 2,970.31 | 0.800 |
| lightgbm_fe_log_l1 | 2,995.54 | 0.797 |
| lightgbm_fe_tweedie | 3,006.85 | 0.807 |
| catboost_raw_l1 | 3,225.44 | 0.787 |
| random_forest_raw | 3,244.75 | 0.796 |
| extra_trees_raw | 3,267.43 | 0.794 |
| ridge_raw | 5,029.15 | 0.734 |
| lightgbm_fe_log_l2 | 6,577.67 | 0.513 |
| elasticnet_raw | 9,067.77 | 0.540 |
| dummy_median | 10,502.36 | −0.114 |

- Feature engineering: LightGBM ลด MAE 67.5 บาท, CatBoost ลด 283.5 บาท
- Spender classifier ของ two-stage: AUC 0.787, Brier 0.186
- Optuna best: engineered features, `learning_rate` 0.0244, `num_leaves` 61, `min_child_samples` 73, `colsample_bytree` 0.871, `subsample` 0.760, `reg_alpha` 0.0012, `reg_lambda` 0.0266, `max_bin` 511

## Cross-validation และ ensemble (dev OOF, 88,400 แถว)

| โมเดล | MAE เฉลี่ย 3 folds | SD | น้ำหนัก |
|---|---:|---:|---:|
| two_stage_mixture_median | 2,875.6 | 68.5 | 0.510 |
| lightgbm_optuna_l1 | 2,882.3 | 75.0 | 0.394 |
| lightgbm_fe_l1 | 2,905.4 | 63.5 | 0.000 |
| lightgbm_raw_l1 | 2,943.1 | 75.2 | 0.096 |

| วิธีรวม | OOF MAE (บาท) |
|---|---:|
| **weighted (เลือก)** | **2,860.76** |
| median | 2,867.25 |
| equal_mean | 2,869.35 |
| single | 2,875.59 |

น้ำหนักหาด้วย SLSQP (ไม่ติดลบ รวมกันเป็น 1) บน OOF ชุดเดียวกับที่วัด คะแนน OOF ของ weighted จึงดีเกินจริงเล็กน้อย

จำนวนรอบที่ตรึง: LightGBM Optuna 2,406, LightGBM FE 1,976, LightGBM raw 1,979
two-stage ใช้ classifier 600 trees และ regressors 900 trees คงที่ ค่า 990 ใน metadata เป็นแค่ budget ที่ส่งเข้า wrapper และไม่ถูกใช้

## ผลสุดท้าย

| รายการ | ผล |
|---|---:|
| **Locked holdout NMAE** | **0.789033** |
| Locked holdout MAE | 2,808.12 บาท (bootstrap 95%: 2,607.03–3,009.54) |
| Locked holdout RMSE / R² | 13,297.09 บาท / 0.8104 |
| Dev OOF NMAE ของ ensemble | 0.785820 |
| `submissions/task3.csv` | 25,889 แถว, SHA256 `e841cdbcd2efbb5e38711b63644226779955dcd8e7c4e38c04707a3ff50d81cc` |

ทั้งหมดเป็นคะแนนภายใน ไม่ใช่คะแนน Kaggle เพราะ Kaggle รายงานแค่คะแนนรวม 5 tasks จึงไม่ทราบคะแนน Kaggle ของ Task 3 อย่างเดียว

การรันซ้ำวันที่ 2026-11-26 ด้วย Python 3.14.10 ได้ผลตรงกับรอบเดิม (Python 3.13.3) ทุกค่า ทั้ง holdout, OOF, น้ำหนัก ensemble และตาราง baseline

## Error analysis (locked holdout)

| ช่วงยอดจริง | แถว | MAE (บาท) | สัดส่วน error รวม |
|---|---:|---:|---:|
| 0 | 7,507 | 1,838 | 31.5% |
| (0, 100] | 376 | 47 | 0.0% |
| (100, 500] | 1,435 | 231 | 0.8% |
| (500, 2,000] | 1,895 | 511 | 2.2% |
| (2,000, 10,000] | 2,276 | 659 | 3.4% |
| (10,000, 50,000] | 838 | 4,028 | 7.7% |
| (50,000, 100,000] | 808 | 9,669 | 17.8% |
| > 100,000 | 465 | 34,408 | 36.5% |

- Top 1% (ยอดจริงเกิน 146,143 บาท) สร้าง error 24.2% ของทั้งหมด และมักถูกทายต่ำกว่าจริง
- แถวที่ขาดข้อมูล 4 ช่องขึ้นไปมี MAE 3,880 บาท เทียบกับ 2,596 บาทของแถวที่ข้อมูลครบ
- Permutation importance (two-stage, dev fold): `history_with_transaction_fallback` +7,224 บาท, `vip_spending_interaction` +1,056, `segment` +456

## Scratch MLP blend (notebook ขั้น 14.2)

โมเดลสุดท้ายผสม tree ensemble 75% กับ scratch MLP (L1 loss บนหน่วยบาท) 25% น้ำหนักตรึงไว้ก่อนดูผล

| Fold | MLP epochs | Tree NMAE | Blend NMAE | Gain |
|---|---:|---:|---:|---:|
| 0 | 17 | 0.782430 | 0.784092 | +0.001663 |
| 1 | 23 | 0.790354 | 0.790760 | +0.000406 |
| 2 | 13 | 0.784729 | 0.786376 | +0.001647 |
| รวม | | 0.785820 | 0.787064 | +0.001244 |

Locked holdout (เช็คซ้ำ ไม่ใช่การวัดอิสระ): tree 0.789033 → blend 0.789418 ใช้ 12 epochs ตอนฝึกบน train ทั้งหมด
`submissions/task3.csv` ตรงกับคอลัมน์ Task 3 ของไฟล์ที่ได้ Kaggle public 0.82618 ทุกแถว

การทดลองนอก notebook ที่ทำแล้วแต่ไม่ได้เลือก เพราะผลดีขึ้นน้อยกว่าเกณฑ์ 0.002 NMAE:
- เปลี่ยนการ stop จาก log-MAE เป็น THB-NMAE: +0.00044
- Conditional stacking ด้วย L1: +0.00011
- Calibrate spender probability: −0.00001
- เพิ่ม quantile 0.01/0.025: ≈ 0

## ข้อสังเกตและงานถัดไป

- จุดอ่อนหลักอยู่ที่สองปลาย คือคนที่ทายว่าจ่ายแต่จริงไม่จ่าย และคนจ่ายหนักที่ทายต่ำไป ถ้าจะพัฒนาต่อควรเริ่มที่ classifier "จะจ่ายไหม" (AUC 0.787)
- ข้อมูลไม่มี timestamp จึงยืนยันไม่ได้ว่า features ย้อนหลังและ `segment` เกิดก่อนช่วงที่ทำนาย
- ส่วนต่างระหว่างโมเดลอันดับต้น ๆ เล็กกว่า SD ข้าม fold การเลือกโมเดลจึงมีความไม่แน่นอนอยู่บ้าง
