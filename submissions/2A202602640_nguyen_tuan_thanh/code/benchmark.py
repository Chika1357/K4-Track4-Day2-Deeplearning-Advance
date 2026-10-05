"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo được tuân theo trong file này:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý: chỉ đo forward của model trên tensor đã nằm sẵn trên thiết bị (đầu vào ngẫu nhiên).
    Giải mã JPEG, resize, chuẩn hoá và chép CPU->GPU nằm ngoài con số này.
"""
from __future__ import annotations

import time


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    Trả về {"p50", "p95", "p99", "mean", "min", "n"}.
    """
    import numpy as np

    if iters < 1:
        raise ValueError("iters phải >= 1")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "min": float(t.min()), "n": int(iters)}


def _device_name(device: str) -> str:
    import platform

    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"CPU ({platform.processor() or platform.machine()})"


def _prepare(model, batch_size: int, img_size: int, dtype: str, device: str):
    """Bản sao model ở eval + đầu vào ngẫu nhiên đúng dtype. Trả về (fn forward, sync)."""
    import copy

    import torch

    if dtype not in ("fp32", "amp", "fp16"):
        raise ValueError("dtype phải là fp32 | amp | fp16")
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    if dtype == "fp16" and dev.type != "cuda":
        raise ValueError("fp16 chỉ đo trên GPU")
    m = copy.deepcopy(model).to(dev).float().eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        m, x = m.half(), x.half()

    def fn():
        with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=dtype == "amp" and dev.type == "cuda"):
            m(x)

    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    return fn, sync, dev


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, fused_bn: bool = False, label: str = "") -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast) | "fp16" (model.half()). Model gốc không bị thay đổi (đo trên bản sao).
    `fused_bn` chỉ là nhãn ghi vào kết quả: hãy truyền vào model đã qua inference.fuse_conv_bn.
    Trả về dict ghi thẳng vào sheet `Latency`. Ở batch 1 AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật.
    """
    import torch

    fn, sync, dev = _prepare(model, batch_size, img_size, dtype, device)
    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    return {"config": label, "gpu": _device_name(dev.type), "dtype": dtype, "batch": batch_size,
            "img_size": img_size, "fused_bn": bool(fused_bn), "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "mean": r["mean"], "images_per_s": batch_size / (r["p50"] / 1000.0), "n": r["n"], "warmup": warmup,
            "torch": torch.__version__, "preprocessing_included": False}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view ở batch 1: đo thật K lượt forward liên tiếp cho một ảnh, so với K * p50 của 1 lượt.

    kw: img_size, dtype, device, warmup, iters, label (như latency_report). Trả về dict như latency_report,
    thêm "k_views" và "k_times_single_p50".
    """
    import torch

    img_size = kw.get("img_size", 224)
    dtype, device = kw.get("dtype", "fp32"), kw.get("device", "cuda")
    warmup, iters = kw.get("warmup", 10), kw.get("iters", 100)
    fn, sync, dev = _prepare(model, 1, img_size, dtype, device)
    single = bench(fn, warmup=warmup, iters=iters, sync=sync)

    def k_fn():
        for _ in range(k_views):
            fn()

    r = bench(k_fn, warmup=warmup, iters=iters, sync=sync)
    return {"config": kw.get("label", f"TTA K={k_views}"), "gpu": _device_name(dev.type), "dtype": dtype, "batch": 1,
            "img_size": img_size, "fused_bn": bool(kw.get("fused_bn", False)), "p50": r["p50"], "p95": r["p95"],
            "p99": r["p99"], "mean": r["mean"], "images_per_s": 1.0 / (r["p50"] / 1000.0), "n": r["n"],
            "warmup": warmup, "torch": torch.__version__, "preprocessing_included": False,
            "k_views": k_views, "k_times_single_p50": k_views * single["p50"]}

# Bản cho Nguyen Tuan Thanh: sửa optimizer/resume/report/Colab; số liệu phải chạy riêng.
