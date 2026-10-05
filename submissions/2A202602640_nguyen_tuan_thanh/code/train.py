"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F, X).

MỘT hàm `run(cfg)` dùng chung cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) được tính bằng `eval.compute_metrics` của repo gốc để cùng định nghĩa
với lúc chấm; file dự đoán ghi bằng `eval.save_predictions`.

Mỗi lần chạy ghi vào <out_dir>/<exp_id>/seed<k>/:
    config.json, history.csv, lr_steps.npy, val_logits.npz, summary.json (và test_logits.npz + TEST_DONE.json
    nếu bật save_test_predictions), cùng checkpoint <ckpt_dir>/<exp_id>_seed<k>.pt và ảnh curves/<exp_id>_<mota>.png.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import random
import sys
import time
import typing
from dataclasses import dataclass
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    """Tìm thư mục chứa eval.py của repo gốc (biến môi trường LAB_REPO_DIR, hoặc đi ngược từ code/)."""
    env = os.environ.get("LAB_REPO_DIR")
    if env and (Path(env) / "eval.py").exists():
        return Path(env)
    for p in (CODE_DIR, *CODE_DIR.parents):
        if (p / "eval.py").exists():
            return p
    raise FileNotFoundError("không tìm thấy eval.py của repo gốc; đặt biến môi trường LAB_REPO_DIR")


for _p in (str(CODE_DIR), str(find_repo_root())):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn, dùng trong tên ảnh curves/<exp_id>_<desc>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | vflip
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    mix_prob: float = 0.5             # xác suất trộn một batch khi `mix` bật
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None   # None: không dùng; 0: 1/n_c; >0: class-balanced
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    optimizer: str = "adamw"          # adamw | sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.01        # cosine giảm về min_lr_ratio * LR đỉnh
    grad_clip: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, logit, summary.json của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    ckpt_dir: str = "checkpoints"     # checkpoint tốt nhất (không commit)
    curves_dir: str = "curves"
    cache_dir: str | None = None      # thư mục cache ảnh đã giải mã (dataset.build_cache), None = đọc JPEG
    expected_total: int | None = 17509  # tổng số ảnh phải có của ba tập (None chỉ khi chạy thử trên tập con)
    overwrite: bool = False           # False: lần chạy đã có summary.json thì không train lại
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def ckpt_path(cfg: Config) -> Path:
    return Path(cfg.ckpt_dir) / f"{cfg.exp_id}_seed{cfg.seed}.pt"


def curve_path(cfg: Config) -> Path:
    """curves/<exp_id>_<mota>.png; seed khác 0 thêm hậu tố _seed<k> (mỗi lần huấn luyện một ảnh)."""
    desc = cfg.desc or cfg.backbone.split(".")[0]
    suffix = "" if cfg.seed == 0 else f"_seed{cfg.seed}"
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{desc}{suffix}.png"


def training_signature(cfg: Config) -> dict:
    """Ignore labels/paths/IDs; keep every parameter that changes training, including seed."""
    ignore = {"exp_id", "desc", "images_dir", "labels_dir", "out_dir", "pred_dir", "ckpt_dir",
              "curves_dir", "cache_dir", "expected_total", "overwrite", "save_test_predictions", "num_workers"}
    return {k: v for k, v in dataclasses.asdict(cfg).items() if k not in ignore}


def ready(cfg: Config) -> bool:
    rd = run_dir(cfg)
    return all(p.exists() for p in (rd / "summary.json", rd / "val_logits.npz", rd / "history.csv", ckpt_path(cfg)))


def atomic_torch_save(obj, path) -> None:
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên: random, numpy, torch (CPU và CUDA).

    Seed của worker DataLoader đặt trong dataset.make_loader (generator + worker_init_fn).
    Để nhanh, dùng cudnn.benchmark=True và không ép thuật toán tất định, nên hai lần chạy cùng seed trên GPU
    có thể lệch nhau ở mức nhiễu số học (ghi trong báo cáo); thứ tự batch, khởi tạo head và augmentation
    thì lặp lại được.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    """AdamW (mặc định) hoặc SGD+momentum với 3 nhóm tham số (model.param_groups)."""
    import torch
    from model import param_groups

    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer={cfg.optimizer!r} không hợp lệ (adamw | sgd)")


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.01) -> float:
    """Hệ số nhân LR tại `step` (tính theo iteration): warmup tuyến tính rồi cosine về min_ratio."""
    import math

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính `warmup_epochs` rồi cosine về ~0 (slide trang 55). Cập nhật THEO BƯỚC (iteration)."""
    import torch

    total = cfg.epochs * steps_per_epoch
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total, warmup, cfg.min_lr_ratio))


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    - `self.module` là bản sao riêng của model (eval, không gradient) dùng để đánh giá.
    - decay được "khởi động": d_t = min(decay, (1 + t) / (10 + t)) để các bước đầu không bị kéo về trọng số
      khởi tạo.
    - buffer dạng số thực (running_mean/var của BatchNorm) cũng lấy trung bình động; buffer nguyên
      (num_batches_tracked) thì chép thẳng. Như vậy thống kê BN của bản EMA khớp với trọng số EMA.
    """

    def __init__(self, model, decay: float):
        import copy

        self.decay = float(decay)
        self.num_updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model) -> None:
        import torch

        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for e, m in zip(self.module.parameters(), model.parameters()):
                e.mul_(d).add_(m.detach(), alpha=1.0 - d)
            for e, m in zip(self.module.buffers(), model.buffers()):
                if e.dtype.is_floating_point:
                    e.mul_(d).add_(m.detach(), alpha=1.0 - d)
                else:
                    e.copy_(m)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.module.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "lr_backbone", "lr_head", "lrs", "time_s"}.

    `train_loss` là loss huấn luyện (trên batch đã trộn nếu dùng Mixup/CutMix), nên không so được trực tiếp
    giữa các công thức khác loss; để so sánh hãy dùng val.
    """
    import torch
    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)  # = model.train(), nhưng backbone đóng băng thì BatchNorm giữ ở eval
    use_amp = cfg.amp and device.type == "cuda"
    total, n, lrs = 0.0, 0, []
    t0 = time.perf_counter()
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        targets = None
        if cfg.mix and random.random() < cfg.mix_prob:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        loss = mixed_loss(criterion, logits, targets) if targets is not None else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss không hữu hạn ({loss.item()}) ở {cfg.exp_id} seed {cfg.seed}")
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lrs.append([g["lr"] for g in optimizer.param_groups])
        scheduler.step()
        if ema is not None:
            ema.update(model)
        total += loss.item() * x.size(0)
        n += x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"train_loss": total / max(1, n), "lr_backbone": lrs[-1][0], "lr_head": lrs[-1][-1],
            "lrs": lrs, "time_s": time.perf_counter() - t0}


def evaluate(model, loader, criterion, device, amp: bool = False):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float),
    theo đúng thứ tự của loader (loader đánh giá không xáo trộn) để ghép logit với tên file.
    """
    import numpy as np
    import torch

    model.eval()
    names, ys, outs = [], [], []
    total, n = 0.0, 0
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(x)
            logits = logits.float()
            yd = y.to(device)
            total += criterion(logits, yd).item() * x.size(0)
            n += x.size(0)
            names.extend(f)
            ys.append(y.numpy())
            outs.append(logits.cpu().numpy())
    return names, np.concatenate(ys), np.concatenate(outs).astype(np.float32), total / max(1, n)


def softmax_np(logits):
    import numpy as np

    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps=None) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    Ba ô: (1) loss train và loss val theo epoch, (2) macro-F1 và top-1 val theo epoch (đánh dấu epoch tốt
    nhất), (3) LR theo bước để thấy warmup + cosine.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    import numpy as np

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train (loss huấn luyện)")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val (cross-entropy)")
    if "val_loss_raw" in history[0]:
        ax[0].plot(ep, [h["val_loss_raw"] for h in history], "x--", alpha=0.7, label="val, trọng số không EMA")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    for a in ax[:2]:
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[0].legend(fontsize=8)

    f1 = [h["val_macro_f1"] for h in history]
    ax[1].plot(ep, f1, "o-", label="macro-F1 val")
    ax[1].plot(ep, [h["val_top1"] for h in history], "s--", alpha=0.8, label="top-1 val")
    if "val_macro_f1_raw" in history[0]:
        ax[1].plot(ep, [h["val_macro_f1_raw"] for h in history], "x--", alpha=0.7, label="macro-F1 val, không EMA")
    best = int(np.argmax(f1))
    ax[1].axvline(ep[best], color="gray", ls=":", lw=1)
    ax[1].annotate(f"tốt nhất: epoch {ep[best]}, F1 = {f1[best]:.4f}", (ep[best], f1[best]),
                   textcoords="offset points", xytext=(6, -14), fontsize=8)
    ax[1].set(xlabel="epoch", ylabel="chỉ số trên val", title="Macro-F1 / top-1 val")
    ax[1].legend(fontsize=8, loc="lower right")

    if lr_steps is not None and len(lr_steps):
        lr_steps = np.asarray(lr_steps)
        ax[2].plot(lr_steps[:, 0], label="backbone")
        if lr_steps.shape[1] > 1:
            ax[2].plot(lr_steps[:, -1], label="head")
        ax[2].set_yscale("log")
        ax[2].legend(fontsize=8)
    ax[2].set(xlabel="bước (iteration)", ylabel="learning rate", title="LR: warmup + cosine")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def describe(cfg: Config) -> str:
    """Các trường khác mặc định (trừ định danh/đường dẫn): dùng làm tiêu đề ảnh và cột 'khác T00 ở điểm nào'."""
    skip = {"exp_id", "seed", "desc", "backbone", "images_dir", "labels_dir", "out_dir", "pred_dir", "ckpt_dir",
            "curves_dir", "cache_dir", "expected_total", "overwrite", "save_test_predictions", "num_workers"}
    base = Config()
    diff = [f"{f.name}={getattr(cfg, f.name)}" for f in dataclasses.fields(cfg)
            if f.name not in skip and getattr(cfg, f.name) != getattr(base, f.name)]
    return ", ".join(diff) if diff else "công thức nền"


def _build_criterion(cfg: Config, train_df, device):
    from dataset import NUM_CLASSES
    from losses import build_criterion, class_weights
    import numpy as np

    weight = None
    if cfg.class_weight_beta is not None:
        counts = np.bincount(train_df["Label"].to_numpy(), minlength=NUM_CLASSES)  # chỉ số liệu của TRAIN
        weight = class_weights(counts, cfg.class_weight_beta)
    if cfg.loss == "ce":
        crit = build_criterion("ce")
    elif cfg.loss == "ls":
        crit = build_criterion("ls", smoothing=cfg.label_smoothing)
    elif cfg.loss == "focal":
        crit = build_criterion("focal", gamma=cfg.focal_gamma, alpha=weight)
    elif cfg.loss == "ce_weighted":
        if weight is None:
            raise ValueError("loss=ce_weighted cần class_weight_beta (0 hoặc >0)")
        crit = build_criterion("ce_weighted", weight=weight)
    else:
        raise ValueError(f"loss={cfg.loss!r} không hợp lệ")
    return crit.to(device)


def load_checkpoint_model(cfg: Config, device, raw: bool = False):
    """Dựng lại model của một lần chạy và nạp checkpoint tốt nhất (trọng số EMA nếu lần chạy dùng EMA;
    `raw=True` lấy trọng số không EMA của cùng epoch)."""
    import torch
    from model import build_model

    model = build_model(cfg.backbone, pretrained=False, drop_rate=cfg.drop_rate, init="finetune",
                        img_size=cfg.img_size)
    ck = torch.load(ckpt_path(cfg), map_location="cpu")
    model.load_state_dict(ck["model_raw"] if raw and "model_raw" in ck else ck["model"])
    return model.to(device).eval()


def test_once(cfg: Config, model, test_df, device) -> dict:
    """Đánh giá TEST đúng MỘT lần cho lần chạy này (1 view), ghi logit + file dự đoán + dấu TEST_DONE.json.

    Nếu dấu đã tồn tại thì KHÔNG chạy lại (quy tắc 'test một lần mỗi seed').
    """
    import numpy as np
    import torch
    from dataset import build_transforms, make_loader
    from eval import compute_metrics, save_predictions

    rd = run_dir(cfg)
    marker = rd / "TEST_DONE.json"
    if marker.exists():
        print(f"[{cfg.exp_id} seed{cfg.seed}] test đã chạy trước đó, không chạy lại.")
        return json.loads(marker.read_text(encoding="utf-8"))
    loader = make_loader(test_df, cfg.images_dir, build_transforms(False, cfg.img_size), cfg.batch_size * 2,
                         train=False, num_workers=cfg.num_workers, seed=cfg.seed, cache_dir=cfg.cache_dir)
    names, y, logits, loss = evaluate(model, loader, torch.nn.CrossEntropyLoss(), device, amp=cfg.amp)
    probs = softmax_np(logits)
    np.savez_compressed(rd / "test_logits.npz", filenames=np.array(names), y_true=y, logits=logits)
    save_predictions(pred_path(cfg, "test"), names, y, probs)
    m = compute_metrics(y, probs.argmax(1), probs)
    out = {"test_macro_f1": m["macro_f1"], "test_top1": m["top1"], "test_ece": m["ece"],
           "test_loss": loss, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "inference": "I00 (1 view)"}
    marker.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt (summary.json).

    1. set_seed; tạo run_dir(cfg); ghi config.json
    2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
    3. train/val loader (test loader CHỈ tạo khi cfg.save_test_predictions)
    4. model, criterion, optimizer, scheduler, GradScaler, EMA
    5. mỗi epoch: train_one_epoch -> evaluate(val) -> history; lưu checkpoint tốt nhất theo MACRO-F1 VAL
       (hòa thì giữ epoch sớm hơn; có EMA thì đánh giá và chọn bằng trọng số EMA)
    6. nạp checkpoint tốt nhất, lưu logit val + predictions/<exp_id>_seed<k>_val.csv
    7. nếu cfg.save_test_predictions (Bước 4): đánh giá test đúng MỘT lần
    8. history.csv, ảnh đường cong, summary.json
    Lần chạy đã có summary.json thì không train lại (trừ khi cfg.overwrite); khi đó nếu bật
    save_test_predictions mà test chưa chạy thì nạp checkpoint và chạy test một lần.
    """
    import numpy as np
    import pandas as pd
    import timm
    import torch
    import torchvision

    import model as model_lib
    from dataset import build_transforms, check_split, load_split, make_loader
    from eval import compute_metrics, save_predictions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd = run_dir(cfg)
    summary_path = rd / "summary.json"
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)

    if (rd / "TEST_DONE.json").exists() and cfg.overwrite:
        raise RuntimeError("Không ghi đè lần chạy đã mở test. Dùng thư mục mới cho một nghiên cứu độc lập.")
    if summary_path.exists() and not cfg.overwrite:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if training_signature(Config(**summary["config"])) != training_signature(cfg):
            raise RuntimeError(f"{rd}: cấu hình đã thay đổi; dùng RUN_NAME mới, không dùng log cũ.")
        required = [ckpt_path(cfg), rd / "val_logits.npz", rd / "history.csv"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError("Log có sẵn nhưng thiếu checkpoint/logit: " + ", ".join(missing)
                               + ". Khôi phục checkpoint từ Drive hoặc dùng RUN_NAME mới để train.")
        if cfg.save_test_predictions and not (rd / "TEST_DONE.json").exists():
            check_split(train_df, val_df, test_df, cfg.images_dir, cfg.expected_total, verbose=False)
            done_cfg = Config(**{**summary["config"], "save_test_predictions": True})
            model = load_checkpoint_model(done_cfg, device)
            summary.update(test_once(done_cfg, model, test_df, device))
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            print(f"[{cfg.exp_id} seed{cfg.seed}] đã có kết quả, bỏ qua (overwrite=True để chạy lại).")
        return summary

    # 1
    set_seed(cfg.seed)
    rd.mkdir(parents=True, exist_ok=True)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8")

    # 2, 3
    check_split(train_df, val_df, test_df, cfg.images_dir, cfg.expected_total, verbose=False)
    train_loader = make_loader(train_df, cfg.images_dir, build_transforms(True, cfg.img_size, cfg.aug),
                               cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers,
                               seed=cfg.seed, cache_dir=cfg.cache_dir)
    val_loader = make_loader(val_df, cfg.images_dir, build_transforms(False, cfg.img_size), cfg.batch_size * 2,
                             train=False, num_workers=cfg.num_workers, seed=cfg.seed, cache_dir=cfg.cache_dir)

    # 4
    model = model_lib.build_model(cfg.backbone, pretrained=True, drop_rate=cfg.drop_rate, init=cfg.init,
                                  img_size=cfg.img_size)
    weights_tag = model.weights_tag
    params_m = model_lib.count_params(model)
    gmacs = model_lib.count_gmacs(model, cfg.img_size)
    model.to(device)
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)
    criterion = _build_criterion(cfg, train_df, device)
    val_criterion = torch.nn.CrossEntropyLoss()  # loss val luôn là CE thường để so được giữa các công thức
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    title = f"{cfg.exp_id} | {weights_tag} | seed {cfg.seed} | {describe(cfg)}"
    print(f"=== {title} | {params_m:.1f}M tham số, {gmacs:.2f} GMAC | {device} ===", flush=True)

    # 5
    history, lr_steps = [], []
    best = {"f1": -1.0, "epoch": 0}
    t_start = time.perf_counter()
    resume_path = Path(cfg.ckpt_dir) / f"{cfg.exp_id}_seed{cfg.seed}_last.pt"
    start_epoch, prior_time = 1, 0.0
    if resume_path.exists() and not cfg.overwrite:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["signature"] != training_signature(cfg):
            raise RuntimeError("Resume checkpoint khác cấu hình; hãy dùng RUN_NAME mới.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None:
            ema.module.load_state_dict(state["ema"])
            ema.num_updates = state["ema_updates"]
        history, lr_steps, best = state["history"], state["lr_steps"], state["best"]
        start_epoch, prior_time = state["epoch"] + 1, state["elapsed_s"]
        print(f"Resume {cfg.exp_id} seed {cfg.seed} từ epoch {start_epoch}.", flush=True)
    for epoch in range(start_epoch, cfg.epochs + 1):
        # Recreate workers and seed per epoch: interrupted runs restart only the incomplete epoch.
        epoch_seed = cfg.seed + epoch * 100003
        set_seed(epoch_seed)
        del train_loader
        train_loader = make_loader(train_df, cfg.images_dir, build_transforms(True, cfg.img_size, cfg.aug),
                                   cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers,
                                   seed=epoch_seed, cache_dir=cfg.cache_dir)
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        lr_steps.extend(tr.pop("lrs"))
        t0 = time.perf_counter()
        eval_model = ema.module if ema is not None else model
        _, y, logits, val_loss = evaluate(eval_model, val_loader, val_criterion, device, amp=cfg.amp)
        probs = softmax_np(logits)
        m = compute_metrics(y, probs.argmax(1), probs)
        row = {"epoch": epoch, "train_loss": tr["train_loss"], "val_loss": val_loss,
               "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "lr_backbone": tr["lr_backbone"], "lr_head": tr["lr_head"], "train_time_s": tr["time_s"]}
        if ema is not None:  # ghi thêm trọng số không EMA để so sánh (I06); KHÔNG dùng để chọn checkpoint
            _, _, logits_raw, loss_raw = evaluate(model, val_loader, val_criterion, device, amp=cfg.amp)
            pr = softmax_np(logits_raw)
            row["val_macro_f1_raw"] = compute_metrics(y, pr.argmax(1), pr)["macro_f1"]
            row["val_loss_raw"] = loss_raw
        row["val_time_s"] = time.perf_counter() - t0
        history.append(row)
        is_best = m["macro_f1"] > best["f1"]  # '>' chặt: hòa thì giữ epoch sớm hơn
        if is_best:
            best = {"f1": m["macro_f1"], "epoch": epoch}
            ck = {"model": {k: v.cpu() for k, v in eval_model.state_dict().items()}, "epoch": epoch,
                  "val_macro_f1": m["macro_f1"], "config": dataclasses.asdict(cfg), "weights_tag": weights_tag}
            if ema is not None:
                ck["model_raw"] = {k: v.cpu() for k, v in model.state_dict().items()}
            atomic_torch_save(ck, ckpt_path(cfg))
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
        atomic_torch_save({"signature": training_signature(cfg), "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                           "scaler": scaler.state_dict(), "ema": ema.module.state_dict() if ema else None,
                           "ema_updates": ema.num_updates if ema else 0, "history": history,
                           "lr_steps": lr_steps, "best": best, "epoch": epoch,
                           "elapsed_s": prior_time + time.perf_counter() - t_start}, resume_path)
        print(f"[{cfg.exp_id} s{cfg.seed}] epoch {epoch:2d}/{cfg.epochs} | train_loss {tr['train_loss']:.4f} | "
              f"val_loss {val_loss:.4f} | val_macroF1 {m['macro_f1']:.4f} | val_top1 {m['top1']:.4f} | "
              f"{tr['time_s']:.0f}s{' *' if is_best else ''}", flush=True)
    total_time = prior_time + time.perf_counter() - t_start

    # 6
    del train_loader
    model = load_checkpoint_model(cfg, device)
    names, y, logits, val_loss = evaluate(model, val_loader, val_criterion, device, amp=cfg.amp)
    probs = softmax_np(logits)
    m = compute_metrics(y, probs.argmax(1), probs)
    np.savez_compressed(rd / "val_logits.npz", filenames=np.array(names), y_true=y, logits=logits)
    save_predictions(pred_path(cfg, "val"), names, y, probs)
    np.save(rd / "lr_steps.npy", np.asarray(lr_steps, dtype=np.float32))

    # 8 (ghi trước bước 7 để lần chạy được coi là xong kể cả khi test chạy sau)
    plot_curves(history, curve_path(cfg), title, lr_steps)
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weights_tag": weights_tag,
        "desc": cfg.desc, "diff": describe(cfg), "params_m": params_m, "gmacs": gmacs,
        "gmac_tool": model_lib.GMAC_TOOL, "img_size": cfg.img_size, "epochs": cfg.epochs,
        "best_epoch": best["epoch"], "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
        "val_balanced_acc": m["balanced_acc"], "val_ece": m["ece"], "val_loss": val_loss,
        "val_f1_per_class": [float(v) for v in m["f1"]], "val_recall_per_class": [float(v) for v in m["recall"]],
        "train_time_per_epoch_s": float(np.mean([h["train_time_s"] for h in history])),
        "total_time_s": total_time, "curve": str(curve_path(cfg).name),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU ({platform.processor()})",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "timm": timm.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "config": dataclasses.asdict(cfg),
    }
    if ema is not None:
        best_row = history[best["epoch"] - 1]
        summary["val_macro_f1_raw_same_epoch"] = best_row["val_macro_f1_raw"]
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 7
    if cfg.save_test_predictions:
        summary.update(test_once(cfg, model, test_df, device))
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    resume_path.unlink(missing_ok=True)
    del model, val_loader, optimizer, ema
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    hints = typing.get_type_hints(Config)
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"tham số {pair!r} phải có dạng KEY=VALUE")
        key, raw = pair.split("=", 1)
        if key not in hints:
            raise KeyError(f"Config không có trường {key!r}; các trường hợp lệ: {sorted(hints)}")
        tp = hints[key]
        args = typing.get_args(tp)
        optional = type(None) in args
        base = next((a for a in args if a is not type(None)), tp) if args else tp
        if raw.lower() in ("none", "null", ""):
            if not optional:
                raise ValueError(f"{key} không nhận None")
            out[key] = None
        elif base is bool:
            if raw.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"{key} cần giá trị bool, nhận {raw!r}")
            out[key] = raw.lower() in ("true", "1", "yes")
        elif base is int:
            out[key] = int(raw)
        elif base is float:
            out[key] = float(raw)
        else:
            out[key] = raw
    return out


def config_to_overrides(cfg: Config) -> list[str]:
    """Ngược của parse_overrides: các trường khác mặc định, dạng KEY=VALUE (để gọi train.py bằng tiến trình con)."""
    base = Config()
    return [f"{f.name}={'none' if getattr(cfg, f.name) is None else getattr(cfg, f.name)}"
            for f in dataclasses.fields(cfg) if getattr(cfg, f.name) != getattr(base, f.name)]


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè trường của Config")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    summary = run(cfg)
    keys = ("exp_id", "seed", "weights_tag", "best_epoch", "val_macro_f1", "val_top1",
            "train_time_per_epoch_s", "params_m", "gmacs")
    print(json.dumps({k: summary.get(k) for k in keys}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# Tham khảo implementation: picuisme/K4-Track4-Day2-Deeplearning-Advance @ 941d9fb.
# Bản cho Nguyen Tuan Thanh: sửa optimizer/resume/report/Colab; số liệu phải chạy riêng.
