"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện (giữ theo starter):
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module (head mới khởi tạo bằng init_head)
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
Phần thêm:
    set_train_mode(model)   -> model.train() nhưng giữ BatchNorm của backbone đóng băng ở eval
    head_parameters(model)  -> danh sách tham số của head phân loại mới
"""
from __future__ import annotations

# Backbone dùng trong bài, GHIM tag trọng số để tái lập được (timm.list_pretrained("resnet50*")).
# Tất cả đều tiền huấn luyện trên ImageNet-1k (không dùng bản in12k/in22k) để so sánh kiến trúc công bằng hơn;
# resnet50 và resnext50 cùng công thức torchvision gốc (tv_in1k).
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50.tv_in1k",
    "resnext50": "resnext50_32x4d.tv_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",      # mạng nhẹ
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",  # bonus: đóng băng + linear probe
}

GMAC_TOOL = "chưa đếm"  # count_gmacs ghi lại công cụ đã dùng vào đây


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", img_size: int | None = None,
                drop_path_rate: float | None = None):
    """Tạo model phân loại 9 lớp bằng timm.

    `init` (trục A): "scratch" (không tải trọng số) | "frozen" (đóng băng backbone, chỉ train head)
                     | "finetune" (tinh chỉnh toàn bộ).
    timm thay head mới (khởi tạo ngẫu nhiên) khi num_classes khác cấu hình gốc.
    Tag trọng số thực sự dùng được ghi vào `model.weights_tag` (ví dụ "resnet50.tv_in1k").
    """
    import timm

    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init={init!r} không hợp lệ")
    name = SUGGESTED_BACKBONES.get(name, name)
    use_pretrained = pretrained and init != "scratch"
    kwargs = dict(pretrained=use_pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate is not None:
        kwargs["drop_path_rate"] = drop_path_rate
    if "dinov2" in name and img_size is not None:
        kwargs["img_size"] = img_size  # DINOv2 mặc định 518; timm nội suy lại position embedding
    model = timm.create_model(name, **kwargs)

    cfg = getattr(model, "pretrained_cfg", {}) or {}
    tag = ".".join(x for x in (cfg.get("architecture"), cfg.get("tag")) if x) or name
    model.weights_tag = tag if use_pretrained else f"{name.split('.')[0]} (từ đầu, không tải trọng số)"
    if use_pretrained:
        from dataset import IMAGENET_MEAN, IMAGENET_STD
        mean, std = tuple(cfg.get("mean", IMAGENET_MEAN)), tuple(cfg.get("std", IMAGENET_STD))
        if (mean, std) != (IMAGENET_MEAN, IMAGENET_STD):
            raise ValueError(f"{tag} cần mean/std {mean}/{std}, khác chuẩn hoá ImageNet đang dùng trong dataset.py")
    init_head(model)
    model.frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def init_head(model, std: float = 0.01) -> None:
    """Khởi tạo lại head mới: trọng số ~ N(0, std^2), bias = 0.

    Khởi tạo mặc định của nn.Linear trên đặc trưng 1280-2048 chiều cho logit lệch nhau khá xa, làm loss ban đầu
    lớn hơn ln 9 rõ rệt (đo ở Bước 0: EfficientNet-B0 khoảng 3,6). Với std nhỏ, logit ban đầu gần 0 nên
    loss CE ban đầu xấp xỉ -ln(1/9) = 2,197 như checklist của slide trang 59. Head vẫn phụ thuộc seed.
    """
    import torch.nn as nn

    head = model.get_classifier()
    for h in (head if isinstance(head, (tuple, list)) else [head]):
        if isinstance(h, nn.Linear):
            nn.init.normal_(h.weight, mean=0.0, std=std)
            if h.bias is not None:
                nn.init.zeros_(h.bias)


def head_parameters(model) -> list:
    """Tham số của head phân loại (model.get_classifier())."""
    head = model.get_classifier()
    heads = head if isinstance(head, (tuple, list)) else [head]
    return [p for h in heads for p in h.parameters()]


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    Backbone đóng băng thì BatchNorm cũng phải ở chế độ eval: nếu không, running_mean/var vẫn bị cập nhật
    theo batch DeepWeeds dù trọng số không đổi, và các tầng phía sau nhận phân phối khác lúc tiền huấn
    luyện. Vòng train gọi `set_train_mode(model)` thay cho `model.train()` để giữ điều này.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in head_parameters(model):
        p.requires_grad = True
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone đóng băng thì đưa mọi BatchNorm về eval."""
    import torch.nn as nn

    model.train()
    if getattr(model, "frozen_backbone", False):
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Nhóm tham số theo slide Day 2, tách thêm bias head để mọi bias đều có decay = 0.

    1. backbone, ndim > 1 (trọng số conv/linear): lr_backbone, có weight decay
    2. backbone, ndim <= 1 (norm, bias) + tham số timm khai báo `no_weight_decay()` (pos_embed, cls_token,
       bảng relative position bias...): lr_backbone, weight_decay = 0
    3. head weight: lr_head, có weight decay; head bias: lr_head, decay = 0
    Tham số requires_grad == False bị bỏ qua; nhóm rỗng không được trả về.
    """
    head_ids = {id(p) for p in head_parameters(model)}
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay, head, head_no_decay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            (head_no_decay if p.ndim <= 1 else head).append(p)
        elif p.ndim <= 1 or name in skip or any(name.endswith(s) for s in ("relative_position_bias_table",)):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"name": "backbone_decay", "params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"name": "backbone_no_decay", "params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"name": "head", "params": head, "lr": lr_head, "weight_decay": weight_decay},
        {"name": "head_no_decay", "params": head_no_decay, "lr": lr_head, "weight_decay": 0.0},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def _count_macs_hooks(model, x) -> float:
    """Tự đếm MAC bằng hook: Conv2d và Linear (bỏ qua phép nhân ma trận trong attention)."""
    import torch
    import torch.nn as nn

    total = [0.0]

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total[0] += out.numel() / out.shape[0] * k

    def linear_hook(m, inp, out):
        total[0] += out.numel() / out.shape[0] * m.in_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return total[0]


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (đếm phép nhân-cộng, không phải FLOPs = 2 x MAC).

    Ưu tiên fvcore.FlopCountAnalysis (đếm conv, linear, matmul/einsum; "flop" của fvcore = 1 phép nhân-cộng).
    Tắt fused attention của timm trong lúc đếm để matmul của attention hiện ra cho fvcore.
    Nếu không có fvcore: tự đếm Conv2d + Linear bằng hook (thiếu matmul của attention, thấp hơn vài %).
    Công cụ đã dùng được ghi vào biến module `GMAC_TOOL`.
    """
    global GMAC_TOOL
    import copy
    import logging
    import warnings

    import torch

    m = copy.deepcopy(model).cpu().float().eval()
    x = torch.zeros(1, 3, img_size, img_size)
    try:
        from fvcore.nn import FlopCountAnalysis

        for mod in m.modules():
            if hasattr(mod, "fused_attn"):
                mod.fused_attn = False
        logging.getLogger("fvcore").setLevel(logging.ERROR)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            fca = FlopCountAnalysis(m, x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            macs = float(fca.total())
        GMAC_TOOL = "fvcore.FlopCountAnalysis"
    except Exception as e:  # không có fvcore, hoặc trace lỗi với kiến trúc này
        macs = _count_macs_hooks(m, x)
        GMAC_TOOL = f"hook Conv2d+Linear (fvcore không dùng được: {type(e).__name__})"
    return macs / 1e9

# Bản cho Nguyen Tuan Thanh: sửa optimizer/resume/report/Colab; số liệu phải chạy riêng.
