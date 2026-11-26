# TabM checkpoints

Checkpoint ของ TabM ที่ฝึกบน Kaggle GPU (T4) ด้วยโค้ดใน `../tabm_training/` notebook โหลดไฟล์เหล่านี้แล้วทำนายบน CPU ไม่ต้องฝึกใหม่

| ไฟล์ | ฝึกบน | ใช้ทำนาย | SHA256 |
|---|---|---|---|
| `tabm_development.pt` | development 88,400 แถว, 16 epochs | audit 15,600 แถว | `c87791db28137a98d4abb58643b74d079b83fdf28aa8a3785f395c389569b860` |
| `tabm_final.pt` | train ทั้งหมด 104,000 แถว, 16 epochs | test 25,889 แถว | `6afc4085aed4d6385608d437be3af2fe3b6fa1b9df86f1c78979ab86cfa29ddc` |

แต่ละไฟล์เป็น dict ที่มี key `state` (state_dict ของ `tabm.TabM.make(n_num_features=122, d_out=4, n_blocks=2, d_block=256, dropout=0.1, k=16, arch_type='tabm', start_scaling_init='random-signs')`)
notebook ตรวจ SHA256 ก่อนโหลดทุกครั้ง
