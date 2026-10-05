"""bonus.py - các phần làm thêm (RUBRIC mục 2), đều dùng mô hình đã huấn luyện, không huấn luyện lại.

    robustness(lab)      lệch phân phối tự tạo trên VAL (tối, sáng gắt, nhiễu Gauss, mờ, giảm tương phản):
                         macro-F1 và ECE trước/sau temperature scaling với T khớp trên val sạch.  (+2)
    gradcam_errors(lab)  Grad-CAM cho các ảnh test bị đoán sai đã liệt kê ở error_analysis.          (+1)
    export_onnx(lab)     xuất ONNX, kiểm tra đầu ra khớp PyTorch và so độ trễ.                         (+1)
Linear probe DINOv2 đóng băng (+2) là lần chạy X01 trong Bước 1 (cùng hàm train.run, init="frozen").
Không phần nào dùng test để chọn cấu hình: robustness chạy trên val; Grad-CAM chỉ để GIẢI THÍCH lỗi sau Bước 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import train  # noqa: E402
from dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


# --------------------------------------------------------------------------- #
# Lệch phân phối tự tạo
# --------------------------------------------------------------------------- #
def _pixel_op(op):
    """Bọc một phép biến đổi trên ảnh [0, 1] thành phép biến đổi trên tensor đã chuẩn hoá ImageNet."""
    import torch

    def fn(x):
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (op((x * std + mean).clamp(0, 1)).clamp(0, 1) - mean) / std
    return fn


def corruptions(seed: int = 0) -> dict:
    """Tên -> hàm biến đổi batch ảnh đã chuẩn hoá. Nhiễu dùng generator cố định để lặp lại được."""
    import torch
    import torchvision.transforms.functional as TF

    def noise(sigma):
        def op(img):
            g = torch.Generator(device="cpu").manual_seed(seed)
            return img + sigma * torch.randn(img.shape, generator=g).to(img.device, img.dtype)
        return op

    def contrast(c):
        return lambda img: (img - img.mean(dim=(1, 2, 3), keepdim=True)) * c + img.mean(dim=(1, 2, 3), keepdim=True)

    return {
        "sạch (gốc)": lambda x: x,
        "tối (x0,4)": _pixel_op(lambda img: img * 0.4),
        "rất tối (x0,2)": _pixel_op(lambda img: img * 0.2),
        "sáng gắt (x1,8)": _pixel_op(lambda img: img * 1.8),
        "nhiễu Gauss σ=0,05": _pixel_op(noise(0.05)),
        "nhiễu Gauss σ=0,10": _pixel_op(noise(0.10)),
        "mờ Gauss σ=2": _pixel_op(lambda img: TF.gaussian_blur(img, kernel_size=9, sigma=2.0)),
        "tương phản thấp (x0,5)": _pixel_op(contrast(0.5)),
    }


def robustness(lab, seed: int | None = None) -> dict:
    """Đánh giá F01 (1 view) và mốc T00 trên VAL bị làm hỏng; ECE trước/sau temperature scaling (T từ val sạch)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import inference as inf
    from dataset import denormalize
    from experiments import metrics_of

    seed = lab.plan.seeds[0] if seed is None else seed
    dev = lab.device()
    _, val_df, _ = lab.splits()
    loader = lab.eval_loader(val_df)
    cors = corruptions()
    rows = []
    for tag, cfg in (("T00", lab.t00_cfg(seed)), ("F01", lab.final_cfg(seed))):
        model = train.load_checkpoint_model(cfg, dev)
        views = {k: (lambda x, f=f: inf.center_crop(f(x), 224)) for k, f in cors.items()}
        _, y, V = inf.predict_views(model, loader, dev, views)
        T = inf.fit_temperature(V["sạch (gốc)"], y)                    # T khớp trên val SẠCH
        for k in cors:
            before, after = metrics_of(inf.softmax(V[k]), y), metrics_of(inf.apply_temperature(V[k], T), y)
            rows.append({"model": f"{tag} seed {seed}", "corruption": k, "val_macro_f1": before["macro_f1"],
                         "val_top1": before["top1"], "ece_before_ts": before["ece"], "ece_after_ts": after["ece"],
                         "T_from_clean_val": T, "recall_negatives": float(before["recall"][8]),
                         "min_weed_recall": float(before["recall"][:8].min())})
        del model

    x, yb, _ = next(iter(loader))
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.8))
    for a, (k, f) in zip(ax.ravel(), cors.items()):
        a.imshow(denormalize(f(x[:1])[0]).permute(1, 2, 0).numpy())
        a.set_title(k, fontsize=9)
        a.axis("off")
    fig.suptitle(f"Các kiểu lệch phân phối tự tạo trên một ảnh val ({CLASS_NAMES[int(yb[0])]})", fontsize=10)
    fig.tight_layout()
    fig.savefig(lab.paths.figures / "bonus_corruptions.png", dpi=110)
    plt.close(fig)
    out = {"rows": rows, "split": "val (không dùng test)", "inference": "1 view center crop 224"}
    lab.save("bonus_robustness", out)
    return out


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def gradcam(model, x, class_idx: int):
    """Grad-CAM trên đầu ra của `forward_features` (tầng đặc trưng cuối trước pooling).

    Hỗ trợ đặc trưng dạng NCHW (CNN, ConvNeXt), NHWC (Swin của timm) và chuỗi token (ViT/DeiT: bỏ các token tiền tố,
    xếp lại thành lưới). Trả về bản đồ [0, 1] kích thước bằng ảnh vào (H, W), numpy.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.enable_grad():
        feats = model.forward_features(x)
        feats.retain_grad()
        logits = model.forward_head(feats)
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        a, g = feats.detach()[0], feats.grad.detach()[0]
    c = model.num_features
    if a.ndim == 3 and a.shape[0] == c:          # (C, H, W)
        pass
    elif a.ndim == 3 and a.shape[-1] == c:        # (H, W, C)
        a, g = a.permute(2, 0, 1), g.permute(2, 0, 1)
    elif a.ndim == 2:                             # (L, C) token
        n_prefix = getattr(model, "num_prefix_tokens", 0)
        a, g = a[n_prefix:], g[n_prefix:]
        side = int(round(a.shape[0] ** 0.5))
        a, g = a.T.reshape(c, side, side), g.T.reshape(c, side, side)
    else:
        raise ValueError(f"không hiểu dạng đặc trưng {tuple(feats.shape)}")
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((w * a).sum(0))
    cam = cam / (cam.max() + 1e-12)
    cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return cam.cpu().numpy()


def gradcam_errors(lab, seed: int | None = None, max_images: int = 8) -> dict:
    """Grad-CAM (lớp dự đoán và lớp đúng) cho các ảnh test bị đoán sai ở figures/errors_test.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from dataset import build_transforms, denormalize

    seed = lab.plan.seeds[0] if seed is None else seed
    errs = lab.load("step4_errors")
    if not errs or not errs.get("misclassified_shown"):
        print("Chưa có danh sách ảnh sai (chạy lab.error_analysis() sau Bước 4).")
        return {}
    dev = lab.device()
    cfg = lab.final_cfg(seed)
    model = train.load_checkpoint_model(cfg, dev)
    tf = build_transforms(False, 224)
    items = errs["misclassified_shown"][:max_images]
    fig, ax = plt.subplots(3, len(items), figsize=(len(items) * 2.4, 7.6), squeeze=False)
    for j, it in enumerate(items):
        x = tf(Image.open(lab.paths.images_dir / it["Filename"]).convert("RGB"))[None].to(dev)
        img = denormalize(x[0]).permute(1, 2, 0).cpu().numpy()
        ax[0, j].imshow(img)
        ax[0, j].set_title(f"thật: {CLASS_NAMES[it['y_true']]}\nđoán: {CLASS_NAMES[it['y_pred']]}", fontsize=7.5)
        for r, (cls, lab_) in enumerate(((it["y_pred"], "lớp ĐOÁN"), (it["y_true"], "lớp ĐÚNG")), start=1):
            ax[r, j].imshow(img)
            ax[r, j].imshow(gradcam(model, x, cls), cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax[r, j].set_title(f"Grad-CAM {lab_}", fontsize=7.5)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"Grad-CAM trên ảnh test bị đoán sai (F01 seed {seed}, {cfg.backbone})", fontsize=10)
    fig.tight_layout()
    fig.savefig(lab.paths.figures / "bonus_gradcam_errors.png", dpi=110)
    plt.close(fig)
    out = {"backbone": cfg.backbone, "seed": seed, "files": [it["Filename"] for it in items],
           "layer": "đầu ra forward_features (đặc trưng cuối trước pooling)"}
    lab.save("bonus_gradcam", out)
    return out


# --------------------------------------------------------------------------- #
# ONNX
# --------------------------------------------------------------------------- #
def export_onnx(lab, seed: int | None = None) -> dict:
    """Xuất F01 sang ONNX (batch 1, 224), kiểm tra đầu ra khớp PyTorch, so độ trễ ONNX Runtime với PyTorch.

    Cách đo giống benchmark.py (warmup, >= 50 lần, p50/p95/p99). Với ONNX Runtime, mỗi lần đo gồm cả việc chép
    đầu vào numpy vào session (trên GPU là chép CPU->GPU), nên con số ORT-GPU hơi bất lợi so với PyTorch-GPU
    (đầu vào đã nằm sẵn trên GPU). Lỗi cài đặt onnxruntime-gpu trên Kaggle được ghi lại thay vì làm hỏng notebook.
    """
    import torch
    from benchmark import bench, latency_report

    seed = lab.plan.seeds[0] if seed is None else seed
    cfg = lab.final_cfg(seed)
    out: dict = {"backbone": cfg.backbone, "seed": seed, "rows": []}
    try:
        import onnxruntime as ort
        model = train.load_checkpoint_model(cfg, torch.device("cpu"))
        x = torch.randn(1, 3, 224, 224)
        path = lab.paths.ckpt_dir / f"{cfg.exp_id}_seed{seed}.onnx"
        kw = dict(input_names=["image"], output_names=["logits"], opset_version=17)
        try:
            torch.onnx.export(model, x, str(path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(model, x, str(path), **kw)
        out["onnx_size_mb"] = path.stat().st_size / 1e6
        with torch.inference_mode():
            ref = model(x).numpy()
        bw, bi = lab.plan.bench_warmup, lab.plan.bench_iters
        xn = x.numpy()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        for prov in providers:
            try:
                sess = ort.InferenceSession(str(path), providers=[prov])
                if sess.get_providers()[0] != prov:
                    raise RuntimeError(f"ONNX Runtime không khởi tạo được {prov}")
                diff = float(np.abs(sess.run(None, {"image": xn})[0] - ref).max())
                r = bench(lambda: sess.run(None, {"image": xn}), warmup=bw, iters=bi)
                out["rows"].append({"runtime": f"ONNX Runtime {ort.__version__} ({prov})", "batch": 1, "img_size": 224,
                                    "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "max_abs_diff_vs_pytorch": diff})
            except Exception as e:  # noqa: BLE001
                out["rows"].append({"runtime": f"ONNX Runtime ({prov})", "error": f"{type(e).__name__}: {str(e)[:200]}"})
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for d in devices:
            r = latency_report(model, 1, 224, "fp32", d, warmup=bw, iters=bi)
            out["rows"].append({"runtime": f"PyTorch {torch.__version__} FP32 ({r['gpu']})", "batch": 1, "img_size": 224,
                                "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "max_abs_diff_vs_pytorch": 0.0})
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print("Xuất/đo ONNX không thành công:", out["error"])
    lab.save("bonus_onnx", out)
    return out

# Tham khảo implementation: picuisme/K4-Track4-Day2-Deeplearning-Advance @ 941d9fb.
# Bản cho Nguyen Tuan Thanh: sửa optimizer/resume/report/Colab; số liệu phải chạy riêng.
