# DeepWeeds — Nguyen Tuan Thanh — 2A202602640

Trạng thái hiện tại: **code đã chuẩn bị; chưa có đủ kết quả huấn luyện riêng trong repo**.
Notebook sẽ tạo `results.xlsx`, `report.md`, `curves/`, `predictions/`, `logs/` và `eval_out/` từ lần chạy thật.
Xem [báo cáo tạm](report.md). Kết quả thực nghiệm sẽ được cập nhật từ log Colab.

Notebook Colab: <!-- NOTEBOOK_LINK -->bổ sung link notebook đã lưu trước khi nộp<!-- /NOTEBOOK_LINK -->

## Chạy ngay trên Colab miễn phí

1. Import `code/lab_day2_colab.ipynb` vào Google Colab.
2. Runtime > Change runtime type > **T4 GPU**.
3. Upload thư mục dự án lên Drive. Sửa `PROJECT_DIR` trong notebook nếu tên/đường dẫn thư mục khác
   `MyDrive/K4-Track4-Day2-Deeplearning-Advance`. Notebook đọc trực tiếp code từ Drive, không cần ZIP đầu vào.
4. Google Drive lưu dữ liệu chạy ở `MyDrive/DeepWeeds_2A202602640/epochs12_v1/`.
   Ảnh/cache nằm trên đĩa `/content` cho nhanh, checkpoint/log nằm trên Drive.
5. Đặt `NOTEBOOK_LINK` thành link notebook Colab của mình (Share > General access phù hợp để giảng viên mở).
6. Chạy hết đến ô xuất bài nộp. Tải `submission_2A202602640.zip` từ Drive; file này không chứa checkpoint/dataset.

Profile mặc định `deadline`: 5 backbone, 3 trục với ít nhất 2 giá trị mỗi trục,
một kết hợp, nhiều phương pháp suy luận; baseline và chung kết mỗi bên seed `[0, 1, 2]`.
**12 epoch/model**, trong mức 10–15 đề xuất.
Tốc độ phụ thuộc GPU được cấp và tốc độ tải/lưu Drive; chương trình in thời gian thật và ước lượng sau vòng backbone.
Thời gian có thể vượt hạn 2 giờ trên Colab miễn phí. Không đổi epoch/batch giữa vòng so sánh đã chạy.

## Nếu Colab ngắt

Mở lại notebook, bật GPU, mount Drive và dùng lại **RUN_NAME** cùng cấu hình.
Chạy lại ô tải dữ liệu (ảnh trong `/content` có thể đã mất), sau đó chạy các bước còn lại.
Checkpoint cuối mỗi epoch chứa optimizer, lịch LR, AMP, EMA và lịch sử; seed augmentation được đặt theo epoch.
Chương trình tiếp tục từ epoch hoàn chỉnh gần nhất và không chạy test lần hai.
Log đã có nhưng thiếu checkpoint sẽ báo lỗi rõ; khôi phục checkpoint hoặc dùng RUN_NAME mới.

## Quy tắc thực nghiệm

- DeepWeeds fold 0 nguyên bản; chọn toàn bộ cấu hình bằng validation; không gộp val vào train.
- Chọn checkpoint bằng macro-F1 validation, hòa giữ epoch sớm.
- Cố định ngân sách/seed giữa các backbone; ghi đúng tag trọng số.
- Không weight decay cho norm, bias hoặc position embedding, gồm cả bias head.
- Các lần training có cấu hình và seed giống hệt được tái sử dụng, ghi `reused_from`.
  Chúng không được coi là seed độc lập hoặc lần huấn luyện mới.
- Test được forward một lần/seed/cấu hình đã chốt; `eval_out` có thể tạo lại từ CSV mà không forward test.
- Độ trễ là forward trên GPU, không tính JPEG/resize/chép CPU→GPU; warmup 10, ≥50 lần đo.
- Chạy `selftest.py` và test repo trước huấn luyện. `eval.py` dùng nguyên bản.

## Sản phẩm cuối

`results.xlsx` (7 sheet bắt buộc), `report.md` (bản nháp có số liệu thật), `curves/`, `code/`,
README này với link Colab, `predictions/`. Giữ thêm log và kết quả `eval.py` để truy vết.
Trước khi nộp, đọc lại phần nhận xét ảnh sai trong báo cáo; chương trình đánh dấu các quan sát cần người xem ảnh xác nhận.

## Phiên bản và phần cứng của lần chạy

`timm==1.0.29`; các phiên bản còn lại được ghi thực tế trong log, không giả định GPU được Colab cấp.

<!-- RUN_INFO -->
Chưa chạy training. Thông tin này sẽ được notebook cập nhật tự động.
<!-- /RUN_INFO -->

## Dữ liệu và quy trình đánh giá

Dataset và quy trình đánh giá: tài liệu `README.md`, `GUIDE.md`, `RUBRIC.md` của repo lớp;
nhãn/split DeepWeeds nguyên bản và ảnh Zenodo, MD5 `b7b30f96d466fba86016aa5a26606e0f`.
