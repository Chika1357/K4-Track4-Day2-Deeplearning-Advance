# Báo cáo tạm — Lab Day 2: DeepWeeds

**Sinh viên:** Nguyen Tuan Thanh · **MSSV:** 2A202602640

**Trạng thái:** chưa có đủ kết quả thực nghiệm riêng được đưa vào repo. Báo cáo này là bản chuẩn bị;
các ô “Chờ log riêng” phải được cập nhật từ Colab. Số liệu của bài tham khảo được ghi ở phần riêng có nguồn.

## 1. Bài toán và thiết lập dự kiến

Phân loại 9 lớp DeepWeeds; sử dụng fold 0 chia sẵn. Train cập nhật trọng số; validation chọn backbone,
công thức, checkpoint, phương pháp suy luận và nhiệt độ; test chỉ đánh giá sau khi chốt cấu hình.

Cấu hình notebook: 12 epoch, batch 64, AMP, AdamW, LR backbone/head 1e-4/1e-3,
weight decay 0.05 (norm và mọi bias không decay), warmup + cosine.
Checkpoint chọn theo macro-F1 validation; baseline và cấu hình cuối dùng seed 0, 1, 2.
Tên GPU, phiên bản thư viện, số đếm split và kết quả kiểm tra pipeline sẽ cập nhật từ log thật.

## 2. Kết quả backbone của sinh viên — đang chờ

| exp_id | Backbone dự kiến chạy | Macro-F1 val | Top-1 val | Độ trễ |
|---|---|---|---|---|
| B01 | ResNet-18 | Chờ log riêng | Chờ log riêng | Chờ đo |
| B02 | ResNeXt-50 32x4d | Chờ log riêng | Chờ log riêng | Chờ đo |
| B03 | ConvNeXt-Tiny | Chờ log riêng | Chờ log riêng | Chờ đo |
| B04 | DeiT-Tiny | Chờ log riêng | Chờ log riêng | Chờ đo |
| B05 | MobileNetV3-Small | Chờ log riêng | Chờ log riêng | Chờ đo |

Chưa kết luận backbone nào tốt nhất. Lựa chọn dựa trên macro-F1 validation và độ trễ đã đo trong vòng này.

## 3. Huấn luyện, suy luận và chung kết — đang chờ

Ba trục: fine-tune/đóng băng; augmentation cơ bản/CutMix; CE/label smoothing; thêm một kết hợp.
Thử TTA, crop, độ phân giải, gộp xác suất/logit, ensemble, temperature scaling và FP16/AMP.
Đo p50/p95/p99 sau warmup và đồng bộ GPU, ít nhất 50 lần.

| Nhóm kết quả của sinh viên | Trạng thái trong repo |
|---|---|
| Các ablation và kết hợp | Chờ log riêng |
| Độ chính xác, hiệu chuẩn và độ trễ suy luận | Chờ log riêng |
| Baseline test, 3 seed | Chờ dự đoán riêng |
| Chung kết test, 3 seed | Chờ dự đoán riêng |
| Ma trận nhầm lẫn và phân tích ảnh sai | Chờ dự đoán riêng |

Các cấu hình/seed giống hệt được tái sử dụng có ghi `reused_from`, không coi là seed độc lập mới.
`results.xlsx` và các kết luận cuối chỉ tạo từ log, dự đoán và kết quả `eval.py` của lần chạy riêng.

## 4. Kết quả tham khảo có nguồn — không phải kết quả của sinh viên

Nguồn: [bài của Nguyễn Trần Kiên (2A202602571), commit 941d9fb](https://github.com/picuisme/K4-Track4-Day2-Deeplearning-Advance/blob/941d9fb/submissions/2A202602571_nguyen_tran_kien/report.md).
DeepWeeds fold 0, 12 epoch, ConvNeXt-Tiny; mean ± std qua ba seed.

| Kết quả trong bài tham khảo | Macro-F1 test | Top-1 test |
|---|---:|---:|
| REF_T00 — công thức nền, một view | 0.9503 ± 0.0016 | 0.9614 ± 0.0012 |
| REF_F01 — Mixup + label smoothing, toàn ảnh 256 + temperature scaling | 0.9578 ± 0.0035 | 0.9679 ± 0.0023 |

Bảng này cung cấp mốc tham khảo trong lúc chờ kết quả Colab. Backbone, optimizer và thiết kế thí nghiệm riêng
có khác biệt; không thể suy ra bài của Thanh đạt các chỉ số trên.
Xem [bảng tham khảo chi tiết và nguồn](references/friend-results.md).

## 5. Việc cần cập nhật trước khi nộp

1. Đưa log, biểu đồ và dự đoán riêng từ Colab vào thư mục này.
2. Chạy bước xuất kết quả để tạo `results.xlsx` và thay báo cáo tạm bằng báo cáo từ log.
3. Điền link notebook, phần cứng và số đếm dữ liệu thực tế.
4. Kiểm tra mean ± std, chỉ số từng lớp, ma trận nhầm lẫn, ECE và độ trễ.
5. Nếu còn thiếu thí nghiệm ở thời điểm nộp, ghi rõ các phần chưa hoàn thành; giữ nguồn cho các số tham khảo.
