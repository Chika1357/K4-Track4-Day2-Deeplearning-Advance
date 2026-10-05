# Kiểm tra cuối — Nguyen Tuan Thanh — 2A202602640

Kiểm tra ngày 05-10-2026 từ các file của lần chạy RTX 6000 Ada. Không huấn luyện hoặc forward test lại.

- 33 CSV dự đoán khớp toàn bộ Filename và nhãn của fold 0 chính thức.
- Chỉ số từng seed và mean ± std (ddof=1) tính lại bằng `eval.py` gốc; khớp log và sheet `Final` trong sai số 1e-6.
- 15 bản ghi có đủ 12 epoch lịch sử; chọn đúng epoch macro-F1 val cao nhất; đủ 15 ảnh đường cong. Chỉ số summary khớp logit validation đã lưu.
- Có 7 bản ghi mà macro-F1 val sau nạp checkpoint khác history tại epoch tốt nhất; độ lệch tuyệt đối tối đa 0.0005344. Giữ nguyên dữ liệu gốc; báo cáo nêu rõ khác biệt này.
- 13 lượt huấn luyện thực và 2 bản ghi tái sử dụng: T00/seed0 ← B03/seed0; F01/seed0 ← T04/seed0.
- Workbook có đủ 7 sheet bắt buộc và `Setup`; không phát hiện ô lỗi công thức trong giá trị lưu.
- `code/eval.py` khớp bản gốc của repo lớp (bỏ khác biệt kết thúc dòng Windows/Linux).
- Notebook có output đã chạy đủ 8 ô code, không có output lỗi; được giữ nguyên byte từ bản tải về server.
- Đã đọc ảnh sai và ma trận nhầm lẫn; báo cáo phân biệt quan sát với giả thuyết và không tuyên bố đã có Grad-CAM.
- Có link Colab cho bản có output và bản chạy lại. Không cần quyền đọc Drive của sinh viên.
- Gói nộp không có dataset ảnh hoặc checkpoint.

## Chỉ số tính lại

| Chỉ số F01 test | Mean | Std mẫu, 3 seed |
|---|---:|---:|
| Macro-F1 | 0.9540859860 | 0.0027106875 |
| Top-1 | 0.9642619523 | 0.0022148427 |
| ECE | 0.0089385440 | 0.0008658699 |

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


Điểm trên chỉ là phần chất lượng model (I), không phải tổng điểm bài; giảng viên quyết định điểm chính thức.
Độ trễ GPU là số đo lưu từ server, không được đo lại trên CPU lúc kiểm tra này.
Workbook được giữ nguyên từ exporter của lần chạy; không thay đổi giá trị hoặc định dạng khi hoàn thiện tài liệu.
