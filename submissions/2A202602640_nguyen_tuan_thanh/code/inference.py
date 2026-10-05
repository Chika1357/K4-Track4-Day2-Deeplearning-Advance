"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra (trang 68),
temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val; nhiệt độ T khớp trên VAL rồi
áp dụng sang test (README.md, S2 và S4).

Giao diện (giữ theo starter):
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
Phần thêm:
    predict_views(model, loader, device, views)      -> nhiều view trong MỘT lượt đọc dữ liệu
    VIEW_SETS / build_views(method)                  -> định nghĩa các phương pháp I00-I04

Quy ước: loader đánh giá trả ảnh GỐC 256x256 đã chuẩn hoá (build_transforms(False, None)); mỗi "view" là một
hàm cắt/lật/resize batch đó trên GPU. View mốc I00 = cắt giữa 224, trùng với CenterCrop(224) lúc val.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# View (biến đổi batch trên GPU)
# --------------------------------------------------------------------------- #
def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W): đảo chiều rộng (slide trang 75)."""
    import torch

    return torch.flip(x, dims=[-1])


def center_crop(x, crop: int):
    h, w = x.shape[-2:]
    top, left = (h - crop) // 2, (w - crop) // 2
    return x[..., top:top + crop, left:left + crop]


def views_multicrop(x, crop: int, flips: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`; `flips=True` thêm bản lật ngang của từng crop (10 view).

    Trả về list các batch; phần tử đầu là crop giữa (trùng view mốc).
    """
    h, w = x.shape[-2:]
    crops = [center_crop(x, crop), x[..., :crop, :crop], x[..., :crop, w - crop:],
             x[..., h - crop:, :crop], x[..., h - crop:, w - crop:]]
    if flips:
        crops = crops + [view_hflip(c) for c in crops]
    return crops


def resize(x, size: int):
    import torch.nn.functional as F

    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)


def views_multiscale(x, sizes):
    """Resize TOÀN BỘ ảnh về từng kích thước trong `sizes`, trả về list các batch.

    Giới hạn: model phải nhận được ảnh khác kích thước lúc train. CNN có global pooling (ResNet, ResNeXt,
    ConvNeXt, EfficientNet) thì được; DeiT/Swin của timm cố định 224 (position embedding / cửa sổ) nên báo
    lỗi ở kích thước khác - khi đó chỉ dùng được size 224.
    """
    return [resize(x, s) for s in sizes]


def build_views(method: str, crop: int = 224) -> list:
    """Danh sách hàm view của một phương pháp suy luận (mỗi hàm: batch 256x256 -> batch đưa vào model).

    "center" (I00) | "hflip" (K=2) | "5crop" | "10crop" | "full<size>" (resize toàn ảnh, ví dụ full256, full288)
    | "full<size>+hflip"
    """
    if method == "center":
        return [lambda x: center_crop(x, crop)]
    if method == "hflip":
        return [lambda x: center_crop(x, crop), lambda x: view_hflip(center_crop(x, crop))]
    if method in ("5crop", "10crop"):
        n = 5 if method == "5crop" else 10
        return [(lambda x, i=i: views_multicrop(x, crop, flips=n == 10)[i]) for i in range(n)]
    if method.startswith("full"):
        base, _, extra = method.partition("+")
        size = int(base[4:])
        fns = [lambda x: resize(x, size)]
        if extra == "hflip":
            fns.append(lambda x: view_hflip(resize(x, size)))
        return fns
    raise ValueError(f"không biết phương pháp view {method!r}")


# --------------------------------------------------------------------------- #
# Chạy model
# --------------------------------------------------------------------------- #
def predict_logits(model, loader, device, view=None, amp: bool = False):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Trả về (filenames: list[str], y_true: ndarray, logits: ndarray[N, 9]).
    """
    names, y, out = predict_views(model, loader, device, {"v": view or view_identity}, amp=amp)
    return names, y, out["v"]


def predict_views(model, loader, device, views: dict, amp: bool = False):
    """Như predict_logits nhưng tính nhiều view trong một lượt đọc dữ liệu.

    `views`: {tên: hàm view}. Trả về (filenames, y_true, {tên: logits[N, 9]}).
    View nào model không chạy được (ví dụ ViT ở độ phân giải khác 224) được trả về None.
    """
    import torch

    model.eval()
    names, ys = [], []
    outs = {k: [] for k in views}
    failed = {}
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            for k, fn in views.items():
                if k in failed:
                    continue
                try:
                    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                        logits = model(fn(x))
                    outs[k].append(logits.float().cpu().numpy())
                except (AssertionError, RuntimeError, ValueError) as e:
                    failed[k] = f"{type(e).__name__}: {str(e)[:120]}"
            names.extend(f)
            ys.append(y.numpy())
    result = {k: (None if k in failed else np.concatenate(v).astype(np.float32)) for k, v in outs.items()}
    if failed:
        print("View không chạy được với model này:", failed)
    return names, np.concatenate(ys), result


# --------------------------------------------------------------------------- #
# Gộp dự đoán
# --------------------------------------------------------------------------- #
def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62). Trả về xác suất (N, 9) đã chuẩn hoá.

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    views = [np.asarray(v, dtype=np.float64) for v in logits_per_view]
    if space == "prob":
        p = np.mean([softmax(v) for v in views], axis=0)
        return p / p.sum(axis=1, keepdims=True)
    if space == "logit":
        return softmax(np.mean(views, axis=0))
    raise ValueError(f"space={space!r} không hợp lệ (prob | logit)")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed).

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file (người gọi
    phải kiểm tra thứ tự Filename trước khi gọi).
    """
    probs = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    if len({p.shape for p in probs}) != 1:
        raise ValueError("các mô hình phải dự đoán trên cùng một tập ảnh")
    p = np.mean(probs, axis=0)
    return p / p.sum(axis=1, keepdims=True)


def probs_to_logits(probs):
    """log(p): 'logit' tương đương của một dự đoán đã gộp theo xác suất, để temperature scaling áp dụng được."""
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None))


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #
def nll(logits, labels, T: float = 1.0) -> float:
    p = softmax(np.asarray(logits, dtype=np.float64) / T)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Tối ưu một tham số log T bằng LBFGS (float64); kiểm tra lại bằng lưới thô và lấy nghiệm NLL nhỏ hơn.
    Accuracy không đổi vì chia logit cho số dương không đổi thứ tự lớp. KHÔNG khớp T trên test.
    """
    import torch

    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    t_lbfgs = float(log_t.exp().item())
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
    t_grid = float(grid[int(np.argmin([nll(val_logits, val_labels, t) for t in grid]))])
    candidates = [t for t in (t_lbfgs, t_grid) if np.isfinite(t) and t > 0]
    return min(candidates, key=lambda t: nll(val_logits, val_labels, t))


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    if T <= 0:
        raise ValueError("T phải > 0")
    return softmax(np.asarray(logits, dtype=np.float64) / T)


# --------------------------------------------------------------------------- #
# Gộp BatchNorm vào conv
# --------------------------------------------------------------------------- #
def has_batchnorm(model) -> bool:
    import torch.nn as nn

    return any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


def _fuse_pair(conv, bn):
    """Gộp một cặp (Conv2d, BatchNorm2d) ở chế độ eval:

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Sửa trọng số ngay trên bản sao của conv (giữ nguyên lớp con như Conv2dSame của timm và mọi padding/stride).
    Trả về (conv đã gộp, module thay cho BN): Identity, hoặc phần kích hoạt nếu BN là BatchNormAct2d của timm.
    """
    import copy

    import torch
    import torch.nn as nn

    fused = copy.deepcopy(conv)
    with torch.no_grad():
        gamma = bn.weight if bn.weight is not None else torch.ones_like(bn.running_mean)
        beta = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
        scale = gamma / torch.sqrt(bn.running_var + bn.eps)
        bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias = nn.Parameter(beta + (bias - bn.running_mean) * scale)
    rest = [m for m in (getattr(bn, "drop", None), getattr(bn, "act", None))
            if m is not None and not isinstance(m, nn.Identity)]
    return fused, (nn.Sequential(*rest) if rest else nn.Identity())


def fuse_conv_bn(model, check_input=None, tol: float = 1e-3):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    Làm trên BẢN SAO của model (model gốc không đổi). Với từng module cha, mỗi cặp con liền kề
    (Conv2d, BatchNorm2d) được gộp: conv mới có bias, BN thay bằng Identity.
    Kiểm tra: so đầu ra trước/sau gộp trên `check_input` (mặc định một batch ngẫu nhiên 2x3x224x224);
    sai số lớn nhất được ghi vào `fused.fuse_max_abs_diff` và `fused.fuse_pairs`. Nếu lệch quá `tol` thì báo lỗi
    (nghĩa là có cặp liền kề theo thứ tự khai báo nhưng không liền kề trong forward).
    Kiến trúc không có BatchNorm2d (ViT, Swin, ConvNeXt dùng LayerNorm): không có gì để gộp, trả về bản sao
    với fuse_pairs = 0.
    """
    import copy

    import torch
    import torch.nn as nn

    fused = copy.deepcopy(model).eval()
    pairs = 0
    for parent in list(fused.modules()):
        children = list(parent.named_children())
        for (n1, c1), (n2, c2) in zip(children, children[1:]):
            if (isinstance(c1, nn.Conv2d) and isinstance(c2, nn.BatchNorm2d)
                    and c2.track_running_stats and c1.out_channels == c2.num_features):
                new_conv, new_bn = _fuse_pair(c1, c2)
                setattr(parent, n1, new_conv)
                setattr(parent, n2, new_bn)
                pairs += 1
    fused.fuse_pairs = pairs
    p = next(model.parameters())
    x = check_input if check_input is not None else torch.randn(2, 3, 224, 224, device=p.device, dtype=p.dtype)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        diff = (model(x) - fused(x)).abs().max().item()
    model.train(was_training)
    fused.fuse_max_abs_diff = diff
    if pairs and diff > tol:
        raise RuntimeError(f"gộp BN làm đầu ra lệch {diff:.3e} > {tol}: kiến trúc này không gộp được theo cặp liền kề")
    return fused

# Bản cho Nguyen Tuan Thanh: sửa optimizer/resume/report/Colab; số liệu phải chạy riêng.
