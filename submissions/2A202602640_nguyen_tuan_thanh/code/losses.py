"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện (giữ theo starter):
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
Các kiểm tra tự viết (focal gamma=0 == CE, label smoothing eps=0 == CE, lam của CutMix đúng diện tích
thật...) nằm ở selftest.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`.

      "ce"          : cross-entropy thường
      "ls"          : cross-entropy + label smoothing (kw: smoothing=0.1)
      "focal"       : focal loss (kw: gamma=2.0, alpha=None hoặc vector trọng số lớp)
      "ce_weighted" : cross-entropy có trọng số lớp (kw: weight=tensor độ dài 9, tính từ TRAIN).
                      Lấy trung bình có trọng số như torch.nn.CrossEntropyLoss(weight=...).
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is None:
            raise ValueError("ce_weighted cần weight= (xem class_weights)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(weight, dtype=torch.float32))
    raise ValueError(f"loss kind={kind!r} không hợp lệ (ce | ls | focal | ce_weighted)")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt (không gọi CrossEntropyLoss(label_smoothing=...)); selftest.py kiểm tra kết quả trùng
    với bản của PyTorch và eps = 0 cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải nằm trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)   # phần 1[k == y]
        uniform = -logp.mean(dim=1)                             # phần eps / K trải đều trên K lớp
        return ((1.0 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    gamma = 0 và alpha = None cho đúng cross-entropy (kiểm tra ở selftest.py, sai số < 1e-6).
    alpha: None hoặc vector trọng số theo lớp. Lấy trung bình thường trên batch.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt).clamp(min=0.0) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (không dùng val hay test).

    - beta = 0: tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng" w_c = (1 - beta) / (1 - beta ** n_c)
      (Cui et al., arXiv:1901.05555), chuẩn hoá tổng trọng số về số lớp
    Cả hai cách chuẩn hoá đều cho trung bình trọng số = 1.
    """
    n = torch.as_tensor(counts, dtype=torch.float64)
    if (n <= 0).any():
        raise ValueError("mọi lớp phải có ít nhất 1 ảnh trong train")
    if beta == 0:
        w = 1.0 / n
        w = w / w.mean()
    else:
        if not 0.0 < beta < 1.0:
            raise ValueError("beta phải nằm trong [0, 1)")
        w = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), n))
        w = w / w.sum() * len(n)
    return w.float()


def rand_bbox(h: int, w: int, lam: float):
    """Hộp CutMix: tỉ lệ cạnh sqrt(1 - lam), tâm đều trên ảnh, cắt lại theo biên ảnh."""
    cut = math.sqrt(max(0.0, 1.0 - lam))
    cut_h, cut_w = int(h * cut), int(w * cut)
    cy = int(torch.randint(0, h, (1,)).item())
    cx = int(torch.randint(0, w, (1,)).item())
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn. Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].

    - lam ~ Beta(alpha, alpha)
    - "mixup" : x_mix = lam * x + (1 - lam) * x[perm]
    - "cutmix": dán một hộp chữ nhật của x[perm] vào x; `lam` được TÍNH LẠI theo diện tích thật của hộp
      sau khi bị cắt ở biên ảnh: lam = 1 - (diện tích hộp) / (H * W)  (slide trang 48)
    """
    if alpha <= 0:
        return x, (y, y, 1.0)
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = rand_bbox(h, w, lam)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    else:
        raise ValueError(f"mode={mode!r} không hợp lệ (mixup | cutmix)")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Tương đương dùng nhãn mềm lam * onehot(y_a) + (1 - lam) * onehot(y_b) với mọi loss tuyến tính theo
    nhãn (CE, label smoothing). Accuracy trên batch đã trộn không còn nghĩa bình thường: đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)

# Tham khảo implementation: picuisme/K4-Track4-Day2-Deeplearning-Advance @ 941d9fb.
# Bản cho Nguyen Tuan Thanh: sửa optimizer/resume/report/Colab; số liệu phải chạy riêng.
