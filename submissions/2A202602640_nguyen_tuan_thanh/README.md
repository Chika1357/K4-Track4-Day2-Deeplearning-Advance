# DeepWeeds — Nguyen Tuan Thanh — 2A202602640

**Đã hoàn thành thí nghiệm trên GPU NVIDIA RTX 6000 Ada Generation**, 12 epoch/model, batch 64,
fold 0 nguyên bản. Cấu hình cuối F01 đạt macro-F1 test **0.9541 ± 0.0027**,
top-1 **0.9643 ± 0.0022**, mean ± std mẫu qua 3 seed.

## Notebook và kết quả

- [Notebook có toàn bộ output — mở trên Colab](https://colab.research.google.com/github/Chika1357/K4-Track4-Day2-Deeplearning-Advance/blob/main/submissions/2A202602640_nguyen_tuan_thanh/code/lab_day2_rtx6000_ada_executed.ipynb). Đây là lần chạy trên server; 8/8 ô code đã thực thi.
- [Notebook chạy lại trên Colab](https://colab.research.google.com/github/Chika1357/K4-Track4-Day2-Deeplearning-Advance/blob/main/submissions/2A202602640_nguyen_tuan_thanh/code/lab_day2_colab.ipynb). Chọn GPU và làm theo các ô cấu hình Drive.
- [Notebook có output trong bài nộp](code/lab_day2_rtx6000_ada_executed.ipynb).
- [Báo cáo](report.md), [bảng kết quả](results.xlsx), [kiểm tra cuối](verification.md).

Link Colab mở trực tiếp notebook từ GitHub; không cần quyền truy cập Drive của sinh viên.
Output notebook được giữ nguyên từ lần chạy RTX 6000 Ada. Báo cáo và README được hoàn thiện sau khi chạy.
Kernel cần có môi trường GPU khi thực thi lại; xem output đã lưu không cần thực thi notebook.

## Cấu trúc bài nộp

`results.xlsx` có 7 sheet bắt buộc và sheet `Setup`; `report.md`; `curves/` (15 ảnh);
`code/`; README với link notebook; `predictions/` (33 CSV). Giữ thêm `logs/`, `figures/`
và `eval_out/` để đối chiếu. Dữ liệu ảnh và checkpoint được lưu trên server, không nằm trong gói nộp.

## Chạy trên server

Thư mục gốc là `~/research777`: `code/` chứa toàn bộ repo lớp, `data/` chứa dữ liệu/cache,
`venvs/deepweeds/` chứa virtualenv, `output/` chứa từng lần chạy. Từ thư mục repo:

```bash
source ~/research777/venvs/deepweeds/bin/activate
cd ~/research777/code
python submissions/2A202602640_nguyen_tuan_thanh/code/selftest.py
python submissions/2A202602640_nguyen_tuan_thanh/code/test_workflow.py
python -m unittest discover -s tests
```

Mở `code/lab_day2_rtx6000_ada.ipynb` trong thư mục bài nộp bằng Jupyter,
chọn kernel đã đăng ký với virtualenv, rồi chạy các ô theo thứ tự.
Notebook mặc định dùng `~/research777/output/rtx6000_epochs12_v1/`;
để làm một lần chạy độc lập, đổi thư mục `OUTPUT` ở ô cấu hình. Bản có output giữ nguyên cấu hình của lần đã hoàn thành.
Giữ GPU được cấp bởi `CUDA_VISIBLE_DEVICES` của phiên trường.

Cũng có thể chạy cùng pipeline bằng terminal trong `tmux`:

```bash
python -u submissions/2A202602640_nguyen_tuan_thanh/code/colab_runner.py \
  --work "$HOME/research777/data" \
  --output "$HOME/research777/output/independent_epochs12_v1/2A202602640_nguyen_tuan_thanh" \
  --profile deadline --epochs 12 --batch-size 64 --workers 2 \
  --stage prepare 0 1 2 3 4 5
```

## Chạy lại trên Colab

1. Mở link notebook Colab ở trên, chọn runtime có GPU.
2. Clone/tải toàn bộ repo lớp về Drive tại `MyDrive/K4-Track4-Day2-Deeplearning-Advance`.
   Nếu đường dẫn khác, sửa `PROJECT_DIR` trong notebook.
3. Giữ profile `deadline`, 12 epoch, batch 64 và workers 2; đặt tên lần chạy riêng ở `RUN_NAME`.
4. Điền `NOTEBOOK_LINK`, chạy các ô theo thứ tự đến ô xuất kết quả. Ảnh/cache ở `/content`, checkpoint/log ở Drive.

Một lần chạy trên GPU khác có thể cho sai khác số học và độ trễ khác.
Các con số trong bài nộp này là của RTX 6000 Ada, không phải số đo trên T4.

## Quy tắc thực nghiệm và tiếp tục khi ngắt

- Train cập nhật trọng số; validation chọn backbone/công thức/checkpoint/suy luận/nhiệt độ; test chỉ đánh giá sau khi chốt.
- Giữ nguyên fold 0, không gộp val vào train. Chọn checkpoint theo macro-F1 val, hòa giữ epoch sớm.
- 5 backbone cùng công thức và seed; 3 trục ablation có kiểm soát và 1 kết hợp; mốc và chung kết dùng seed 0, 1, 2.
- Không weight decay cho norm, mọi bias hoặc position embedding.
- Có 15 bản ghi và 15 đường cong: 13 lượt huấn luyện thực; T00/seed0 tái sử dụng B03/seed0,
  F01/seed0 tái sử dụng T04/seed0. Các bản ghi có `reused_from`, không coi là lần train độc lập mới.
- Test forward một lần/seed/cấu hình đã chốt. Tính lại chỉ số từ CSV không forward test lại.
- Latency: warmup 10, 50 lần đo, đồng bộ CUDA, p50/p95/p99; không tính JPEG/resize/chép CPU→GPU.
- Nếu phiên ngắt, giữ nguyên cấu hình và thư mục output; tải lại dữ liệu nếu cần, rồi chạy tiếp.
  Checkpoint mỗi epoch chứa optimizer, scheduler, AMP, EMA và lịch sử. Khi resume, cần giữ cả checkpoint cùng log.
- ZIP bài nộp không chứa checkpoint. Có log nhưng thiếu checkpoint sẽ báo lỗi khi cố tiếp tục training.

## Phiên bản và phần cứng thực tế

<!-- RUN_INFO -->
| Mục | Giá trị |
|---|---|
| GPU/thiết bị huấn luyện | NVIDIA RTX 6000 Ada Generation |
| python | 3.10.12 |
| torch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |
| timm | 1.0.29 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| Seed | quét sàng: 0; mốc T00 và chung kết F01: [0, 1, 2] |
| Epoch / batch | 12 / 64 |
| Số bản ghi / huấn luyện thực | 15 bản ghi; 13 lượt train thực, 2 bản ghi tái sử dụng (tổng 0.63 giờ GPU) |
| Công cụ đếm GMAC | fvcore.FlopCountAnalysis |
<!-- /RUN_INFO -->

## Dữ liệu và đánh giá

DeepWeeds fold 0 nguyên bản; ảnh Zenodo có MD5 `b7b30f96d466fba86016aa5a26606e0f`.
Định nghĩa và quy trình theo `README.md`, `GUIDE.md`, `RUBRIC.md` của repo lớp.
`code/eval.py` trong bài nộp là bản gốc; kiểm tra cuối dùng CSV dự đoán và nhãn fold chính thức.
