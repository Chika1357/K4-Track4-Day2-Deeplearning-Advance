# Kết quả tham khảo — bài của Nguyễn Trần Kiên

**Đây là kết quả của bài tham khảo, không phải kết quả chạy của Nguyen Tuan Thanh (2A202602640).**
Các bảng dưới chỉ cung cấp mốc đối chiếu trong lúc chờ kết quả Colab riêng. Không đưa các số này vào
sheet kết quả thực nghiệm của sinh viên hoặc dùng để tạo `predictions/`, checkpoint hay đường cong training của sinh viên.

Nguồn cố định: [bài của Nguyễn Trần Kiên, MSSV 2A202602571, commit 941d9fb](https://github.com/picuisme/K4-Track4-Day2-Deeplearning-Advance/tree/941d9fb/submissions/2A202602571_nguyen_tran_kien).
Chi tiết: [report.md](https://github.com/picuisme/K4-Track4-Day2-Deeplearning-Advance/blob/941d9fb/submissions/2A202602571_nguyen_tran_kien/report.md)
và [log chung kết](https://github.com/picuisme/K4-Track4-Day2-Deeplearning-Advance/blob/941d9fb/submissions/2A202602571_nguyen_tran_kien/logs/step4_final.json).

## Thiết lập của bài tham khảo

- DeepWeeds, fold 0 nguyên bản; 12 epoch, batch 64, AMP; chọn checkpoint bằng macro-F1 validation.
- GPU Tesla T4; baseline và chung kết mỗi bên seed 0, 1, 2; std mẫu (`ddof=1`).
- Chung kết dùng `convnext_tiny.fb_in1k`, Mixup + label smoothing, toàn ảnh 256 và temperature scaling khớp trên validation.

## So sánh backbone trên validation — một seed

Mã `REF_*` chỉ thuộc bảng tham khảo, không trùng mã thí nghiệm riêng của sinh viên.

| Mã tham khảo | Backbone | Macro-F1 val | Top-1 val |
|---|---|---:|---:|
| REF_B01 | ResNet-50 | 0.9290 | 0.9434 |
| REF_B02 | ResNeXt-50 32x4d | 0.9377 | 0.9523 |
| REF_B03 | ConvNeXt-Tiny | 0.9554 | 0.9663 |
| REF_B04 | DeiT-Small | 0.9528 | 0.9660 |
| REF_B05 | Swin-Tiny | 0.9582 | 0.9703 |
| REF_B06 | EfficientNet-B0 | 0.8478 | 0.8843 |

## Chung kết trên test — mean ± std qua ba seed

| Mã tham khảo | Cấu hình trong bài nguồn | Macro-F1 test | Top-1 test | ECE test |
|---|---|---:|---:|---:|
| REF_T00 | ConvNeXt-Tiny, công thức nền, center crop 224 | 0.9503 ± 0.0016 | 0.9614 ± 0.0012 | 0.0078 ± 0.0004 |
| REF_F01 | Mixup + label smoothing, toàn ảnh 256, temperature scaling | 0.9578 ± 0.0035 | 0.9679 ± 0.0023 | 0.0061 ± 0.0016 |
| REF_F02 | Cùng mô hình chung kết, center crop 224, temperature scaling | 0.9546 ± 0.0040 | 0.9648 ± 0.0023 | 0.0098 ± 0.0004 |

Trong bài nguồn, macro-F1 chung kết tăng khoảng 0.0075 so với baseline. ECE của REF_F01 trước temperature scaling
là 0.1030 ± 0.0023. Recall test trung bình: Chinee Apple 87.2%, Snake Weed 94.0%.

## Giới hạn khi đối chiếu với bài của Thanh

Bài của Thanh dùng ResNet-18, ResNeXt-50, ConvNeXt-Tiny, DeiT-Tiny và MobileNetV3-Small;
nhóm backbone khác bài nguồn. Optimizer của Thanh loại weight decay cho bias head.
Phần cứng Colab và các lựa chọn cuối của Thanh phải lấy từ log riêng; không mặc định là Tesla T4 hoặc giống bài nguồn.
Do đó, những số ở đây không dự đoán chắc chắn kết quả của Thanh và không thay thế các thí nghiệm còn thiếu.
