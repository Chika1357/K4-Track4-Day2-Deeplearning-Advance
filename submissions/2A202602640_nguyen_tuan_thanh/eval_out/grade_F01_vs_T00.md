## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 96.43% (mean 3 seed) |
| I2 | Macro-F1 cải thiện so với mốc | 4 | 5 | final 0.9541, mốc 0.9486, Δ=+0.0055, s=0.0027 |
| I3 | Recall hai lớp khó | 4 | 4 | Chinee Apple 90.1% (mốc 88.5%), Snake Weed 94.4% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.1103, sau 0.0089 |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9611, test 0.9541, chênh 0.0070 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 1.9 ms (ngân sách 100 ms), đo đúng cách |

**Tổng các ý đã chấm: 19 / 20** (phần I tối đa 20).

Ngưỡng điểm là TẠM THỜI (xem khối hằng số đầu file eval.py và RUBRIC.md mục I).
