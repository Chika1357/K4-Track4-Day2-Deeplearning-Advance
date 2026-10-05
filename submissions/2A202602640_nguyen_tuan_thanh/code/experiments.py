"""experiments.py - điều phối toàn bộ bài lab: Bước 0 -> Bước 4 (GUIDE.md).

Mọi lần huấn luyện đều đi qua MỘT hàm `train.run(Config(...))`; file này chỉ định nghĩa danh sách cấu hình,
các QUY TẮC CHỌN (khai báo trước, chỉ dùng số liệu VAL) và phần suy luận/đo độ trễ.

Quy tắc chọn (ghi lại trong logs/step*_*.json để báo cáo trích dẫn):
  Bước 1  backbone đi tiếp = trong các backbone có macro-F1 val cách tốt nhất không quá `near_best_tol`
          (mặc định 0,005; mới 1 seed nên coi là "gần tương đương"), chọn cái có độ trễ batch-1 p50 thấp nhất.
  Bước 2  s = std (ddof=1) của macro-F1 val qua các seed của T00. Một yếu tố "thắng rõ" khi Δ > s.
          Kết hợp 1 = giá trị tốt nhất của MỖI trục nếu Δ > 0; kết hợp 2 = chỉ các yếu tố có Δ > s.
          Công thức chung kết = cấu hình có macro-F1 val (seed 0) cao nhất trong {T00, từng yếu tố, kết hợp}.
  Bước 3  suy luận chung kết (ngoại tuyến) = phương pháp một-mô-hình có macro-F1 val cao nhất, rồi temperature
          scaling (T khớp trên val của từng seed). Cấu hình thời gian thực = 1 view + biến thể nhanh nhất có
          p95 batch-1 <= 100 ms.
  Bước 4  test chạy đúng một lần mỗi seed (có dấu TEST_DONE.json chặn chạy lại).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import train  # noqa: E402  (thêm thư mục repo gốc vào sys.path để import eval)
from train import Config, run, run_dir  # noqa: E402
from dataset import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from eval import compute_metrics, save_predictions  # noqa: E402

LN9 = -math.log(1.0 / 9.0)
PAPER_TABLE1 = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]  # trích dẫn Table 1 (Olsen et al. 2019)
HARD = {"Chinee Apple": 0, "Snake Weed": 7}

# (exp_id, tag timm, mô tả ngắn, nhóm)
BACKBONES = [
    ("B01", "resnet50.tv_in1k", "resnet50", "ResNet"),
    ("B02", "resnext50_32x4d.tv_in1k", "resnext50_32x4d", "ResNeXt"),
    ("B03", "convnext_tiny.fb_in1k", "convnext_tiny", "ConvNeXt"),
    ("B04", "deit_small_patch16_224.fb_in1k", "deit_small", "Transformer (DeiT)"),
    ("B05", "swin_tiny_patch4_window7_224.ms_in1k", "swin_tiny", "Transformer (Swin)"),
    ("B06", "efficientnet_b0.ra_in1k", "efficientnet_b0", "Mạng nhẹ"),
]
DINOV2 = ("X01", "vit_small_patch14_dinov2.lvd142m", "dinov2_small_linear_probe", "Bonus: DINOv2 đóng băng")

# (exp_id, trục, mô tả ngắn, khác T00 ở đúng MỘT yếu tố)
ABLATIONS = [
    ("T01", "A", "scratch", {"init": "scratch"}),
    ("T02", "A", "frozen", {"init": "frozen"}),
    ("T03", "B", "colorjitter", {"aug": "color"}),
    ("T04", "B", "trivialaugment", {"aug": "trivial"}),
    ("T05", "B", "vflip", {"aug": "vflip"}),
    ("T06", "B", "cutmix", {"mix": "cutmix", "mix_alpha": 1.0}),
    ("T07", "B", "mixup", {"mix": "mixup", "mix_alpha": 0.2}),
    ("T08", "C", "labelsmooth", {"loss": "ls", "label_smoothing": 0.1}),
    ("T09", "C", "focal", {"loss": "focal", "focal_gamma": 2.0}),
    ("T10", "C", "classbalanced_ce", {"loss": "ce_weighted", "class_weight_beta": 0.999}),
    ("T11", "F", "ema", {"ema_decay": 0.998}),
]
AXIS_NAMES = {"A": "A. Khởi tạo", "B": "B. Augmentation", "C": "C. Hàm loss", "D": "D. Cân bằng mẫu",
              "E": "E. LR/optimizer", "F": "F. Chính quy hoá (EMA)", "G": "G. Độ phân giải/epoch",
              "-": "nền", "+": "kết hợp"}

# Phương pháp suy luận một-mô-hình: mã -> (mô tả, danh sách view, không gian gộp)
CROPS = ["center", "tl", "tr", "bl", "br"]
SINGLE_MODEL_METHODS = {
    "I00": ("1 view: center crop 224 (mốc)", ["center"], "prob"),
    "I01": ("TTA lật ngang (K=2), gộp xác suất", ["center", "center_f"], "prob"),
    "I02a": ("TTA 5 crop 224 (K=5), gộp xác suất", CROPS, "prob"),
    "I02b": ("TTA 5 crop + lật (K=10), gộp xác suất", CROPS + [c + "_f" for c in CROPS], "prob"),
    "I03a": ("TTA lật ngang (K=2), gộp logit", ["center", "center_f"], "logit"),
    "I03b": ("TTA 5 crop + lật (K=10), gộp logit", CROPS + [c + "_f" for c in CROPS], "logit"),
}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"không ghi được {type(o)} vào JSON")


@dataclass
class Plan:
    """Tham số chung của cả bài (đổi ở notebook nếu cần giảm ngân sách GPU; ghi lại trong báo cáo)."""
    epochs: int = 12
    batch_size: int = 64
    num_workers: int = 2
    seeds: tuple = (0, 1, 2)
    backbones: list = field(default_factory=lambda: list(BACKBONES))
    ablations: list = field(default_factory=lambda: list(ABLATIONS))
    dinov2: tuple | None = DINOV2          # bonus linear probe; None để bỏ
    chosen_backbone: str | None = None      # None = theo quy tắc Bước 1; hoặc ép một tag timm
    near_best_tol: float = 0.005
    parallel: bool = True                   # có >= 2 GPU thì chạy song song mỗi GPU một tiến trình
    bench_warmup: int = 10
    bench_iters: int = 100
    res_sweep: tuple = (224, 256, 288, 320)
    sanity_backbone: str = "resnet50.tv_in1k"
    overfit_steps: int = 100
    realtime_budget_ms: float = 100.0
    profile: str = "full"


@dataclass
class Paths:
    repo: Path            # thư mục gốc repo (có eval.py)
    sub: Path             # thư mục bài nộp: logs/, predictions/, curves/, figures/, eval_out/, results.xlsx
    images_dir: Path
    labels_dir: Path
    ckpt_dir: Path
    cache_dir: Path | None = None
    expected_total: int | None = 17509

    def __post_init__(self):
        for k in ("repo", "sub", "images_dir", "labels_dir", "ckpt_dir"):
            setattr(self, k, Path(getattr(self, k)))
        for d in (self.logs, self.pred, self.curves, self.figures, self.eval_out, self.ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)

    logs = property(lambda self: self.sub / "logs")
    pred = property(lambda self: self.sub / "predictions")
    curves = property(lambda self: self.sub / "curves")
    figures = property(lambda self: self.sub / "figures")
    eval_out = property(lambda self: self.sub / "eval_out")


def metrics_of(probs, y) -> dict:
    """Chỉ số vô hướng theo đúng định nghĩa của eval.py."""
    m = compute_metrics(np.asarray(y), np.asarray(probs).argmax(1), np.asarray(probs))
    return {"macro_f1": m["macro_f1"], "top1": m["top1"], "balanced_acc": m["balanced_acc"],
            "ece": m["ece"], "nll": m["nll"], "f1": m["f1"], "recall": m["recall"], "precision": m["precision"]}


class Lab:
    def __init__(self, paths: Paths, plan: Plan | None = None):
        self.paths = paths
        self.plan = plan or Plan()
        os.environ["LAB_REPO_DIR"] = str(paths.repo)

    # ------------------------------------------------------------------ tiện ích
    def cfg(self, exp_id: str, **kw) -> Config:
        p, pl = self.paths, self.plan
        base = dict(epochs=pl.epochs, batch_size=pl.batch_size, num_workers=pl.num_workers,
                    images_dir=str(p.images_dir), labels_dir=str(p.labels_dir), out_dir=str(p.logs),
                    pred_dir=str(p.pred), ckpt_dir=str(p.ckpt_dir), curves_dir=str(p.curves),
                    cache_dir=str(p.cache_dir) if p.cache_dir else None, expected_total=p.expected_total)
        base.update(kw)
        return Config(exp_id=exp_id, **base)

    def save(self, name: str, obj) -> Path:
        path = self.paths.logs / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        return path

    def load(self, name: str):
        path = self.paths.logs / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def splits(self):
        from dataset import load_split
        return load_split(self.paths.labels_dir, 0)

    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval_loader(self, df, batch_size: int | None = None):
        """Loader đánh giá trả ảnh gốc 256x256 đã chuẩn hoá, đúng thứ tự df (các view cắt/resize trên GPU)."""
        from dataset import build_transforms, make_loader
        return make_loader(df, self.paths.images_dir, build_transforms(False, None), batch_size or self.plan.batch_size,
                           train=False, num_workers=self.plan.num_workers, cache_dir=self.paths.cache_dir)

    def val_logits(self, cfg: Config):
        """(filenames, y_true, logits) 1-view trên val đã lưu khi train xong."""
        d = np.load(run_dir(cfg) / "val_logits.npz", allow_pickle=False)
        return [str(f) for f in d["filenames"]], d["y_true"], d["logits"]

    # ------------------------------------------------------------------ chạy nhiều cấu hình
    def run_many(self, cfgs: list[Config]) -> list[dict]:
        """Chạy danh sách cấu hình. Có >= 2 GPU: mỗi GPU một tiến trình `python train.py --set ...`;
        ngược lại chạy lần lượt trong tiến trình này. Lần chạy đã xong (có summary.json) được bỏ qua."""
        import torch

        todo = [c for c in cfgs if not train.ready(c)]
        n_gpu = torch.cuda.device_count()
        if self.plan.parallel and n_gpu >= 2 and len(todo) >= 2:
            self._run_parallel(todo, n_gpu)
        return [run(c) for c in cfgs]

    def _run_parallel(self, todo: list[Config], n_gpu: int) -> None:
        queue, running, failed = list(todo), {}, []
        while queue or running:
            for gpu in range(n_gpu):
                if gpu not in running and queue:
                    c = queue.pop(0)
                    run_dir(c).mkdir(parents=True, exist_ok=True)
                    log = open(run_dir(c) / "stdout.log", "w", encoding="utf-8")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                           "LAB_REPO_DIR": str(self.paths.repo)}
                    cmd = [sys.executable, str(CODE_DIR / "train.py"), "--set", *train.config_to_overrides(c)]
                    running[gpu] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), c, log)
                    print(f"[GPU {gpu}] bắt đầu {c.exp_id} seed {c.seed} ({c.backbone}; {train.describe(c)})", flush=True)
            time.sleep(5)
            for gpu, (proc, c, log) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[gpu]
                tail = (run_dir(c) / "stdout.log").read_text(encoding="utf-8", errors="replace").splitlines()
                if proc.returncode != 0:
                    failed.append(c.exp_id)
                    print(f"[GPU {gpu}] LỖI {c.exp_id} seed {c.seed}:\n" + "\n".join(tail[-25:]), flush=True)
                else:
                    epochs = [ln for ln in tail if "epoch" in ln and "val_macroF1" in ln]
                    print(f"[GPU {gpu}] xong {c.exp_id} seed {c.seed}: {epochs[-1] if epochs else ''}", flush=True)
        if failed:
            raise RuntimeError(f"các lần chạy bị lỗi: {failed} (xem logs/<exp_id>/seed<k>/stdout.log)")

    # ================================================================== BƯỚC 0
    def step0_data(self) -> dict:
        """Đọc + kiểm tra split (README 2.1), EDA: phân bố lớp, ảnh mẫu, thống kê ảnh. Lưu logs/step0_data.json."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        from dataset import check_split

        train_df, val_df, test_df = self.splits()
        info = check_split(train_df, val_df, test_df, self.paths.images_dir, self.paths.expected_total)
        info["paper_table1"] = PAPER_TABLE1
        info["matches_paper_table1"] = info["class_total"] == PAPER_TABLE1
        # Đối chiếu nhãn của ba file fold 0 với labels.csv (chỉ GHI NHẬN; theo S1 không sửa file CSV của fold)
        info["label_mismatch_vs_labels_csv"] = []
        labels_csv = self.paths.labels_dir / "labels.csv"
        if labels_csv.exists():
            ref = pd.read_csv(labels_csv).set_index("Filename")["Label"]
            for split, d in (("train", train_df), ("val", val_df), ("test", test_df)):
                other = ref.reindex(d["Filename"]).to_numpy()
                for f, a, b in zip(d["Filename"], d["Label"], other):
                    if a != b:
                        info["label_mismatch_vs_labels_csv"].append(
                            {"split": split, "Filename": f, "label_in_fold_csv": int(a), "label_in_labels_csv": int(b)})
            print("Ảnh có nhãn trong file fold khác labels.csv:", info["label_mismatch_vs_labels_csv"] or "không có")

        # Kích thước và số kênh của MỌI ảnh trong ba tập (chỉ đọc header)
        sizes: dict = {}
        for f in pd.concat([train_df, val_df, test_df])["Filename"]:
            with Image.open(self.paths.images_dir / f) as im:
                key = f"{im.size[0]}x{im.size[1]} {im.mode}"
            sizes[key] = sizes.get(key, 0) + 1
        info["image_formats"] = sizes
        # mean/std theo kênh trên 500 ảnh TRAIN (chỉ để tham khảo; bài dùng mean/std ImageNet của trọng số)
        sample = train_df.sample(min(500, len(train_df)), random_state=0)["Filename"]
        px = np.stack([np.asarray(Image.open(self.paths.images_dir / f).convert("RGB"), dtype=np.float32) / 255.0
                       for f in sample])
        info["train_channel_mean"] = [round(float(v), 4) for v in px.mean(axis=(0, 1, 2))]
        info["train_channel_std"] = [round(float(v), 4) for v in px.std(axis=(0, 1, 2))]
        print("Định dạng ảnh:", sizes)
        print("mean/std theo kênh (500 ảnh train):", info["train_channel_mean"], info["train_channel_std"])
        print("Khớp Table 1 của bài báo:", info["matches_paper_table1"], "| tổng theo lớp:", info["class_total"])

        # Biểu đồ phân bố lớp
        fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [3, 2]})
        x = np.arange(NUM_CLASSES)
        for i, (k, c) in enumerate(zip(("train", "val", "test"), ("#2a78b5", "#f08c2e", "#3aa655"))):
            bars = ax[0].bar(x + (i - 1) * 0.27, info["per_class"][k], 0.27, label=f"{k} ({info['n'][k]})", color=c)
            ax[0].bar_label(bars, fontsize=6.5, rotation=90, padding=2)
        ax[0].set_xticks(x, CLASS_NAMES, rotation=25, ha="right")
        ax[0].set(ylabel="số ảnh", title="Phân bố lớp theo tập (fold 0)", yscale="log")
        ax[0].legend()
        ax[1].barh(CLASS_NAMES[::-1], info["class_total"][::-1], color="#6b7a8f")
        for i, v in enumerate(info["class_total"][::-1]):
            ax[1].text(v, i, f" {v}", va="center", fontsize=8)
        ax[1].set(xlabel="số ảnh (cả ba tập)", title=f"Tổng theo lớp; lớn nhất/nhỏ nhất = {info['imbalance_ratio']:.2f}")
        ax[1].set_xlim(0, max(info["class_total"]) * 1.15)
        fig.tight_layout()
        fig.savefig(self.paths.figures / "eda_class_distribution.png", dpi=130)
        plt.close(fig)

        # Ảnh mẫu: 5 ảnh mỗi lớp (lấy từ train)
        n_per = 5
        fig, ax = plt.subplots(NUM_CLASSES, n_per, figsize=(n_per * 2.0, NUM_CLASSES * 2.0))
        for c in range(NUM_CLASSES):
            pool = train_df[train_df["Label"] == c]
            files = pool.sample(min(n_per, len(pool)), random_state=0)["Filename"].tolist()
            for j in range(n_per):
                ax[c, j].axis("off")
                if j < len(files):
                    ax[c, j].imshow(Image.open(self.paths.images_dir / files[j]))
                    if j == 0:
                        ax[c, j].set_title(f"{c}: {CLASS_NAMES[c]}", fontsize=9, loc="left")
        fig.suptitle("Ảnh mẫu mỗi lớp (train, fold 0)", fontsize=11)
        fig.tight_layout()
        fig.savefig(self.paths.figures / "eda_samples.png", dpi=72)
        plt.close(fig)
        self.save("step0_data", info)
        return info

    def step0_sanity(self) -> dict:
        """Kiểm tra pipeline trước khi chạy thật (GUIDE 1.3): seed, loss ban đầu, overfit batch nhỏ, ảnh sau
        augmentation, chế độ train/eval. Lưu logs/step0_sanity.json + figures/sanity_*.png."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        import torch.nn.functional as F
        import model as model_lib
        from dataset import build_transforms, denormalize, make_loader
        from losses import mix_batch

        dev = self.device()
        train_df, val_df, _ = self.splits()
        out: dict = {"ln9": LN9}

        # 1) cố định seed: hai loader cùng seed phải cho batch đầu giống hệt nhau
        def first_batch(seed):
            train.set_seed(seed)
            ld = make_loader(train_df, self.paths.images_dir, build_transforms(True, 224, "basic"), 16, train=True,
                             num_workers=self.plan.num_workers, seed=seed, cache_dir=self.paths.cache_dir)
            return next(iter(ld))
        (xa, ya, fa), (xb, yb, fb), (xc, _, fc) = first_batch(0), first_batch(0), first_batch(1)
        out["seed_same_files"] = list(fa) == list(fb)
        out["seed_same_tensors"] = bool(torch.equal(xa, xb))
        out["other_seed_differs"] = list(fa) != list(fc)
        print(f"1) Seed: cùng seed -> cùng thứ tự file: {out['seed_same_files']}, cùng tensor sau augmentation: "
              f"{out['seed_same_tensors']}; khác seed -> khác batch: {out['other_seed_differs']}")
        assert out["seed_same_files"] and out["seed_same_tensors"], "DataLoader không tái lập được với cùng seed"

        # 2) loss ban đầu của head mới ~ ln 9 (mọi backbone, 256 ảnh val, eval mode)
        sub = val_df.sample(min(256, len(val_df)), random_state=0)
        ld = make_loader(sub, self.paths.images_dir, build_transforms(False, 224), 64, train=False,
                         num_workers=self.plan.num_workers, cache_dir=self.paths.cache_dir)
        out["initial_loss"] = {}
        names = [b[1] for b in self.plan.backbones] + ([self.plan.dinov2[1]] if self.plan.dinov2 else [])
        for name in names:
            train.set_seed(0)
            m = model_lib.build_model(name, pretrained=True, img_size=224).to(dev)
            _, _, _, loss = train.evaluate(m, ld, torch.nn.CrossEntropyLoss(), dev)
            out["initial_loss"][name] = loss
            print(f"2) Loss ban đầu {name:45s}: {loss:.4f}  (kỳ vọng {LN9:.4f}, lệch {loss - LN9:+.4f})")
            assert abs(loss - LN9) < 0.3, f"loss ban đầu của {name} lệch xa ln 9: kiểm tra head mới"
            del m

        # 3) overfit một batch nhỏ (16 ảnh, không augmentation) tới loss gần 0
        train.set_seed(0)
        small = train_df.groupby("Label").head(2).head(16)
        xs, ys, _ = next(iter(make_loader(small, self.paths.images_dir, build_transforms(False, 224), 16, train=False,
                                          num_workers=0, cache_dir=self.paths.cache_dir)))
        xs, ys = xs.to(dev), ys.to(dev)
        m = model_lib.build_model(self.plan.sanity_backbone, pretrained=True).to(dev)
        opt = torch.optim.AdamW(model_lib.param_groups(m, 1e-3, 1e-2, 0.0))
        m.train()
        curve = []
        for _ in range(self.plan.overfit_steps):
            opt.zero_grad()
            loss = F.cross_entropy(m(xs), ys)
            loss.backward()
            opt.step()
            curve.append(loss.item())
            if loss.item() < 0.01:
                break
        # Recalibrate BN on this diagnostic batch before assessing eval-mode overfit.
        bn_layers = [b for b in m.modules() if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
        old_momentum = [b.momentum for b in bn_layers]
        for b in bn_layers:
            b.momentum = 1.0
        with torch.no_grad():
            m(xs)
        for b, old in zip(bn_layers, old_momentum):
            b.momentum = old
        m.eval()
        with torch.no_grad():
            acc = (m(xs).argmax(1) == ys).float().mean().item()
        out["overfit"] = {"backbone": self.plan.sanity_backbone, "n_images": int(len(ys)), "steps": len(curve),
                          "first_loss": curve[0], "final_loss": curve[-1], "final_acc_eval_mode": acc, "curve": curve}
        print(f"3) Overfit {len(ys)} ảnh ({self.plan.sanity_backbone}): loss {curve[0]:.3f} -> {curve[-1]:.4f} "
              f"sau {len(curve)} bước; accuracy trên chính batch đó (eval) = {acc:.2f}")
        assert curve[-1] < 0.1, "không overfit được một batch nhỏ: lỗi nằm ở code hoặc model, chưa chạy tiếp"

        # 5) train()/eval(): ở eval, đầu ra của một ảnh không phụ thuộc các ảnh khác trong batch; ở train thì có
        with torch.no_grad():
            m.eval()
            d_eval = (m(xs)[:1] - m(xs[:1])).abs().max().item()
            has_bn = any(isinstance(k, torch.nn.modules.batchnorm._BatchNorm) for k in m.modules())
            m.train()
            d_train = (m(xs)[:2] - m(xs[:2])).abs().max().item()
        model_lib.freeze_backbone(m)
        model_lib.set_train_mode(m)
        bn_train = sum(k.training for k in m.modules() if isinstance(k, torch.nn.modules.batchnorm._BatchNorm))
        out["mode_check"] = {"has_batchnorm": has_bn, "eval_batch_dependence": d_eval, "train_batch_dependence": d_train,
                             "frozen_bn_layers_in_train_mode": int(bn_train)}
        print(f"5) eval(): đầu ra 1 ảnh lệch {d_eval:.2e} khi đổi các ảnh cùng batch (phải ~0); "
              f"train(): lệch {d_train:.2e} (BatchNorm dùng thống kê batch). "
              f"Backbone đóng băng: {bn_train} tầng BN còn ở train mode (phải = 0).")
        assert d_eval < 1e-3 and bn_train == 0
        del m, opt

        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(range(1, len(curve) + 1), curve)
        ax.axhline(LN9, color="gray", ls=":", label=f"ln 9 = {LN9:.3f}")
        ax.set(xlabel="bước", ylabel="loss CE", yscale="log",
               title=f"Overfit {len(ys)} ảnh: {curve[0]:.2f} → {curve[-1]:.4f}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.paths.figures / "sanity_overfit.png", dpi=130)
        plt.close(fig)

        # 4) ảnh sau augmentation (đã giải chuẩn hoá) + nhãn; hàng dưới: CutMix và Mixup kèm lam
        train.set_seed(0)
        fig, ax = plt.subplots(4, 8, figsize=(16, 8.6))
        for r, aug in enumerate(("basic", "trivial")):
            xb_, yb_, _ = next(iter(make_loader(train_df, self.paths.images_dir, build_transforms(True, 224, aug), 8,
                                                train=True, num_workers=0, seed=0, cache_dir=self.paths.cache_dir)))
            for j in range(8):
                ax[r, j].imshow(denormalize(xb_[j]).permute(1, 2, 0).numpy())
                ax[r, j].set_title(f"[{aug}] {CLASS_NAMES[int(yb_[j])]}", fontsize=8)
        lams = {}
        for r, mode in ((2, "cutmix"), (3, "mixup")):
            for _ in range(50):  # chỉ để minh hoạ: lấy mẫu lại tới khi lam ở khoảng giữa cho dễ nhìn
                xm, (y_a, y_b, lam) = mix_batch(xb_, yb_, 1.0, mode)
                if 0.3 < lam < 0.7:
                    break
            lams[mode] = lam
            for j in range(8):
                ax[r, j].imshow(denormalize(xm[j]).permute(1, 2, 0).numpy())
                ax[r, j].set_title(f"[{mode}] {lam:.2f}·{CLASS_NAMES[int(y_a[j])]}\n+ {1 - lam:.2f}·{CLASS_NAMES[int(y_b[j])]}",
                                   fontsize=7)
        for a in ax.ravel():
            a.axis("off")
        fig.suptitle("Ảnh sau augmentation (đã giải chuẩn hoá) và nhãn tương ứng", fontsize=11)
        fig.tight_layout()
        fig.savefig(self.paths.figures / "sanity_augmentation.png", dpi=110)
        plt.close(fig)
        out["mix_lams"] = lams
        print("4) Đã lưu figures/sanity_augmentation.png (ảnh + nhãn sau augmentation, CutMix/Mixup kèm lam).")
        self.save("step0_sanity", out)
        return out

    # ================================================================== BƯỚC 1
    def backbone_cfgs(self) -> list[Config]:
        cfgs = [self.cfg(e, backbone=tag, desc=desc) for e, tag, desc, _ in self.plan.backbones]
        if self.plan.dinov2:
            e, tag, desc, _ = self.plan.dinov2
            cfgs.append(self.cfg(e, backbone=tag, desc=desc, init="frozen"))
        return cfgs

    def quick_latency(self, backbone: str, img_size: int = 224) -> dict:
        """Độ trễ batch-1 FP32 của một kiến trúc (không phụ thuộc trọng số đã train)."""
        import model as model_lib
        from benchmark import latency_report
        m = model_lib.build_model(backbone, pretrained=False, img_size=img_size)
        return latency_report(m, 1, img_size, "fp32", self.device().type, warmup=self.plan.bench_warmup,
                              iters=max(50, self.plan.bench_iters // 2), label=backbone)

    def step1_backbones(self) -> pd.DataFrame:
        """Huấn luyện mọi backbone bằng công thức nền T00 (seed 0), đo độ trễ sơ bộ, chọn backbone đi tiếp."""
        cfgs = self.backbone_cfgs()
        summaries = self.run_many(cfgs)
        groups = {b[0]: b[3] for b in self.plan.backbones + ([self.plan.dinov2] if self.plan.dinov2 else [])}
        prev = {r["exp_id"]: r for r in (self.load("step1_backbones") or {}).get("rows", [])}
        rows = []
        for c, s in zip(cfgs, summaries):
            lat = prev[c.exp_id]["latency"] if c.exp_id in prev else self.quick_latency(c.backbone)
            rows.append({"exp_id": s["exp_id"], "group": groups[s["exp_id"]], "backbone": c.backbone,
                         "weights_tag": s["weights_tag"], "params_m": s["params_m"], "gmacs": s["gmacs"],
                         "img_size": s["img_size"], "epochs": s["epochs"], "seed": s["seed"],
                         "best_epoch": s["best_epoch"], "val_macro_f1": s["val_macro_f1"], "val_top1": s["val_top1"],
                         "train_time_per_epoch_s": s["train_time_per_epoch_s"], "latency_b1_p50_ms": lat["p50"],
                         "latency_b1_p95_ms": lat["p95"], "latency": lat, "init": c.init, "curve": s["curve"],
                         "f1_chinee_apple": s["val_f1_per_class"][0], "f1_snake_weed": s["val_f1_per_class"][7],
                         "device": s["device"], "gmac_tool": s["gmac_tool"]})
        cand = [r for r in rows if r["init"] == "finetune"]   # linear probe DINOv2 (bonus) không tham gia chọn
        best = max(cand, key=lambda r: r["val_macro_f1"])
        near = [r for r in cand if best["val_macro_f1"] - r["val_macro_f1"] <= self.plan.near_best_tol]
        if self.plan.chosen_backbone:
            chosen = next(r for r in cand if r["backbone"] == self.plan.chosen_backbone)
            rule = "ép bằng Plan.chosen_backbone"
        else:
            chosen = min(near, key=lambda r: r["latency_b1_p50_ms"])
            rule = (f"trong các backbone có macro-F1 val cách tốt nhất <= {self.plan.near_best_tol} "
                    f"(gần tương đương ở mức 1 seed), chọn độ trễ batch-1 p50 thấp nhất")
        choice = {"chosen": chosen["backbone"], "chosen_exp_id": chosen["exp_id"], "rule": rule,
                  "best_exp_id": best["exp_id"], "best_val_macro_f1": best["val_macro_f1"],
                  "near_best": [r["exp_id"] for r in near], "tol": self.plan.near_best_tol}
        self.save("step1_backbones", {"rows": rows, "choice": choice})
        print(f"Backbone tốt nhất về macro-F1 val: {best['exp_id']} ({best['backbone']}, {best['val_macro_f1']:.4f}). "
              f"Gần tương đương: {choice['near_best']}. CHỌN đi tiếp: {chosen['exp_id']} {chosen['backbone']} "
              f"(F1 {chosen['val_macro_f1']:.4f}, p50 {chosen['latency_b1_p50_ms']:.1f} ms). Quy tắc: {rule}.")
        return pd.DataFrame(rows).drop(columns=["latency"])

    def chosen_backbone(self) -> str:
        s1 = self.load("step1_backbones")
        if s1 is None:
            raise RuntimeError("chưa chạy Bước 1")
        return s1["choice"]["chosen"]

    # ================================================================== BƯỚC 2
    def t00_cfg(self, seed: int = 0) -> Config:
        return self.cfg("T00", backbone=self.chosen_backbone(), desc="baseline", seed=seed)

    def reuse_identical(self, source: Config, target: Config) -> bool:
        import shutil
        if train.ready(target) or not train.ready(source):
            return False
        if train.training_signature(source) != train.training_signature(target):
            return False
        src, dst = run_dir(source), run_dir(target)
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("history.csv", "lr_steps.npy", "val_logits.npz"):
            shutil.copy2(src / name, dst / name)
        shutil.copy2(train.ckpt_path(source), train.ckpt_path(target))
        shutil.copy2(train.pred_path(source, "val"), train.pred_path(target, "val"))
        summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
        summary.update(exp_id=target.exp_id, desc=target.desc, config=dataclasses.asdict(target),
                       curve=train.curve_path(target).name, total_time_s=0.0,
                       reused_from=f"{source.exp_id}/seed{source.seed}", diff=train.describe(target))
        h = pd.read_csv(dst / "history.csv").to_dict("records")
        train.plot_curves(h, train.curve_path(target),
                          f"{target.exp_id} | {target.backbone} | seed {target.seed} | reuse {source.exp_id}",
                          np.load(dst / "lr_steps.npy"))
        (dst / "config.json").write_text(json.dumps(dataclasses.asdict(target), indent=2), encoding="utf-8")
        (dst / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Reuse {source.exp_id} seed {source.seed} -> {target.exp_id}: cấu hình training giống hệt.")
        return True

    def step2_training(self) -> pd.DataFrame:
        """T00 x các seed (ước lượng nhiễu s) + mỗi lần đổi đúng một yếu tố so với T00 + kết hợp."""
        bb = self.chosen_backbone()
        source = next(c for c in self.backbone_cfgs() if c.backbone == bb)
        self.reuse_identical(source, self.t00_cfg(self.plan.seeds[0]))
        t00 = self.run_many([self.t00_cfg(s) for s in self.plan.seeds])
        abl_cfgs = [self.cfg(e, backbone=bb, desc=desc, **ov) for e, _, desc, ov in self.plan.ablations]
        abl = self.run_many(abl_cfgs)

        f1s = [s["val_macro_f1"] for s in t00]
        noise = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else float("nan")
        base = t00[0]

        def row(s, axis, diff, overrides, note=""):
            f1c = s["val_f1_per_class"]
            return {"exp_id": s["exp_id"], "backbone": bb, "axis": axis, "axis_name": AXIS_NAMES[axis], "diff": diff,
                    "overrides": overrides, "seed": s["seed"], "best_epoch": s["best_epoch"],
                    "val_macro_f1": s["val_macro_f1"], "val_top1": s["val_top1"],
                    "delta_vs_t00": s["val_macro_f1"] - base["val_macro_f1"],
                    "f1_chinee_apple": f1c[0], "f1_snake_weed": f1c[7], "f1_min_weed": float(min(f1c[:8])),
                    "f1_negatives": f1c[8], "curve": s["curve"], "note": note}

        rows = [row(s, "-", "công thức nền T00", {}, f"seed {s['seed']}") for s in t00]
        singles = [row(s, axis, ", ".join(f"{k}={v}" for k, v in ov.items()), ov)
                   for s, (_, axis, _, ov) in zip(abl, self.plan.ablations)]
        rows += singles

        # Kết hợp: giá trị tốt nhất của mỗi trục
        best_per_axis = {}
        for r in singles:
            if r["axis"] not in best_per_axis or r["val_macro_f1"] > best_per_axis[r["axis"]]["val_macro_f1"]:
                best_per_axis[r["axis"]] = r
        positive = [r for r in best_per_axis.values() if r["delta_vs_t00"] > 0]
        clear = [r for r in best_per_axis.values() if r["delta_vs_t00"] > (noise if math.isfinite(noise) else 0.0)]
        if len(positive) < 2:  # không đủ 2 yếu tố có lợi: vẫn thử kết hợp 2 yếu tố đứng đầu để đo cộng dồn/triệt tiêu
            positive = sorted(best_per_axis.values(), key=lambda r: -r["delta_vs_t00"])[:2]
        combos = [("greedy: tốt nhất mỗi trục (Δ>0)", positive)]
        if len(clear) >= 2 and {r["exp_id"] for r in clear} != {r["exp_id"] for r in positive}:
            combos.append(("chỉ các yếu tố thắng rõ (Δ>s)", clear))
        next_id = max(int(e[0][1:]) for e in self.plan.ablations) + 1 if self.plan.ablations else 1
        combo_cfgs, combo_meta = [], []
        for i, (label, parts) in enumerate(combos):
            ov = {}
            for r in parts:
                ov.update(r["overrides"])
            desc = "combo_" + "+".join(next(a[2] for a in self.plan.ablations if a[0] == r["exp_id"]) for r in parts)
            combo_cfgs.append(self.cfg(f"T{next_id + i:02d}", backbone=bb, desc=desc[:60], **ov))
            combo_meta.append((label, parts, ov))
        for s, (label, parts, ov) in zip(self.run_many(combo_cfgs), combo_meta):
            parts_ids = [r["exp_id"] for r in parts]
            sum_delta = sum(r["delta_vs_t00"] for r in parts)
            r = row(s, "+", " + ".join(parts_ids), ov,
                    f"{label}; tổng Δ của từng yếu tố = {sum_delta:+.4f}")
            r["sum_of_single_deltas"] = sum_delta
            rows.append(r)

        cands = [rows[0]] + [r for r in rows if r["axis"] != "-"]
        final = max(cands, key=lambda r: r["val_macro_f1"])
        choice = {"backbone": bb, "noise_std_t00": noise, "t00_val_macro_f1": f1s,
                  "t00_mean": float(np.mean(f1s)), "final_recipe_exp_id": final["exp_id"],
                  "final_overrides": final["overrides"], "final_val_macro_f1_seed0": final["val_macro_f1"],
                  "final_delta_vs_t00": final["delta_vs_t00"],
                  "best_per_axis": {a: r["exp_id"] for a, r in best_per_axis.items()},
                  "rule": "macro-F1 val (seed 0) cao nhất trong {T00, từng yếu tố, kết hợp}; "
                          "kết hợp = tốt nhất mỗi trục có Δ>0 (tham lam theo trục, song song trên cùng nền T00)"}
        self.save("step2_training", {"rows": rows, "choice": choice})
        print(f"T00 macro-F1 val qua {len(f1s)} seed: {np.round(f1s, 4).tolist()} -> mean {np.mean(f1s):.4f}, std s = {noise:.4f}")
        print(f"Công thức chung kết: {final['exp_id']} ({final['diff']}), macro-F1 val {final['val_macro_f1']:.4f}, "
              f"Δ = {final['delta_vs_t00']:+.4f} so với T00 seed 0 (s = {noise:.4f}).")
        return pd.DataFrame(rows).drop(columns=["overrides"])

    def final_cfg(self, seed: int = 0) -> Config:
        s2 = self.load("step2_training")
        if s2 is None:
            raise RuntimeError("chưa chạy Bước 2")
        return self.cfg("F01", backbone=s2["choice"]["backbone"], desc="final", seed=seed,
                        **s2["choice"]["final_overrides"])

    def train_finalists(self) -> pd.DataFrame:
        """Huấn luyện cấu hình chung kết F01 với mọi seed (CHƯA đụng tới test)."""
        target = self.final_cfg(self.plan.seeds[0])
        selected = self.load("step2_training")["choice"]["final_recipe_exp_id"]
        source = dataclasses.replace(target, exp_id=selected)
        self.reuse_identical(source, target)
        if selected == "T00":
            for seed in self.plan.seeds:
                self.reuse_identical(self.t00_cfg(seed), self.final_cfg(seed))
        out = self.run_many([self.final_cfg(s) for s in self.plan.seeds])
        df = pd.DataFrame([{k: s[k] for k in ("exp_id", "seed", "best_epoch", "val_macro_f1", "val_top1")} for s in out])
        print(f"F01 macro-F1 val (1 view): mean {df.val_macro_f1.mean():.4f} ± {df.val_macro_f1.std(ddof=1):.4f}")
        return df

    # ================================================================== BƯỚC 3
    def view_fns(self, sizes=()) -> dict:
        """Mọi view đơn dùng ở Bước 3/4: tên -> hàm (batch 256x256 trên GPU -> batch đưa vào model)."""
        import inference as inf
        fns = {"center": lambda x: inf.center_crop(x, 224)}
        for i, k in enumerate(CROPS):
            if i:
                fns[k] = lambda x, i=i: inf.views_multicrop(x, 224)[i]
        for k in list(fns):
            fns[k + "_f"] = lambda x, f=fns[k]: inf.view_hflip(f(x))
        for s in sizes:
            fns[f"full{s}"] = lambda x, s=s: inf.resize(x, s)
            fns[f"full{s}_f"] = lambda x, s=s: inf.view_hflip(inf.resize(x, s))
        return fns

    def methods(self) -> dict:
        m = dict(SINGLE_MODEL_METHODS)
        if self.plan.profile == "deadline":
            m = {k: v for k, v in m.items() if k in {"I00", "I01", "I02a", "I03a"}}
        for i, s in enumerate(self.plan.res_sweep):
            m[f"I04{'abcdefgh'[i]}"] = (f"Độ phân giải kiểm tra {s} (resize toàn ảnh, 1 view)", [f"full{s}"], "prob")
        return m

    def step3_inference(self) -> pd.DataFrame:
        """So sánh các phương pháp suy luận trên VAL với mô hình F01 seed đầu tiên; đo độ trễ; chọn phương pháp."""
        completed = self.load("step3_inference")
        if completed is not None:
            print("Suy luận đã chốt; giữ nguyên lựa chọn khi resume.")
            return pd.DataFrame(completed["rows"])
        import copy
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        import inference as inf
        import model as model_lib
        from benchmark import bench, latency_report, tta_latency

        dev = self.device()
        pl = self.plan
        seed0 = pl.seeds[0]
        fcfg = self.final_cfg(seed0)
        model = train.load_checkpoint_model(fcfg, dev)
        _, val_df, _ = self.splits()
        loader = self.eval_loader(val_df)
        required_views = {v for _, views, _ in self.methods().values() for v in views}
        fns = {k: v for k, v in self.view_fns(pl.res_sweep).items() if k in required_views}
        names, y, V = inf.predict_views(model, loader, dev, fns, amp=False)
        assert names == val_df["Filename"].tolist(), "thứ tự file của loader val bị đổi"
        saved_names, _, saved_logits = self.val_logits(fcfg)
        assert saved_names == names
        np.savez_compressed(self.paths.logs / "step3_val_views.npz", filenames=np.array(names), y_true=y,
                            **{k: v for k, v in V.items() if v is not None})

        bw, bi = pl.bench_warmup, pl.bench_iters
        lat_rows: list[dict] = []

        def lat(label, **kw):
            fn = tta_latency if "k_views" in kw else latency_report
            args = dict(device=dev.type, warmup=bw, iters=bi, label=label)
            args.update(kw)
            if fn is latency_report:
                r = latency_report(args.pop("model", model), args.pop("batch_size", 1), args.pop("img_size", 224), **args)
            else:
                r = tta_latency(args.pop("model", model), args.pop("k_views"), **args)
            lat_rows.append(r)
            return r

        base_lat = lat("I00 1 view 224 FP32 batch 1")
        base_thr = lat("I00 1 view 224 FP32 batch 32", batch_size=32)
        rows = []

        def add(code, method, probs, k, lat_r, note="", model_desc=None, thr=None):
            m = metrics_of(probs, y)
            rows.append({"exp_id": code, "method": method,
                         "model": model_desc or f"F01 seed {seed0} ({fcfg.backbone})", "k": k,
                         "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": m["ece"], "val_nll": m["nll"],
                         "lat_p50_ms": lat_r["p50"] if lat_r else None, "lat_p95_ms": lat_r["p95"] if lat_r else None,
                         "lat_p99_ms": lat_r["p99"] if lat_r else None,
                         "throughput_img_s": thr, "rel_cost": (lat_r["p50"] / base_lat["p50"]) if lat_r else None,
                         "delta_f1_vs_i00": None, "note": note})
            return m

        # --- I00-I04: một mô hình, đổi view
        probs_of = {}
        for code, (desc, views, space) in self.methods().items():
            if any(V.get(v) is None for v in views):
                rows.append({"exp_id": code, "method": desc, "model": f"F01 seed {seed0} ({fcfg.backbone})",
                             "k": len(views), "note": "KHÔNG chạy được: kiến trúc cố định độ phân giải 224"})
                continue
            p = inf.aggregate_views([V[v] for v in views], space)
            probs_of[code] = p
            if code == "I00":
                lr_ = base_lat
            elif views[0].startswith("full"):
                lr_ = lat(f"{code} 1 view {views[0][4:]} FP32 batch 1", img_size=int(views[0][4:]))
            else:
                lr_ = lat(f"{code} TTA K={len(views)} FP32 batch 1", k_views=len(views))
            if code == "I00":
                throughput = base_thr["images_per_s"]
            elif views[0].startswith("full"):
                throughput = lat(f"{code} FP32 batch 32", batch_size=32, img_size=int(views[0][4:]))["images_per_s"]
            else:
                xbatch = torch.randn(32, 3, 224, 224, device=dev)
                def batch_views():
                    with torch.inference_mode():
                        for _ in views:
                            model(xbatch)
                measured = bench(batch_views, warmup=bw, iters=bi,
                                 sync=torch.cuda.synchronize if dev.type == "cuda" else None)
                throughput = 32000.0 / measured["p50"]
                lat_rows.append({**base_thr, **measured, "config": f"{code} K={len(views)} FP32 batch 32",
                                 "images_per_s": throughput, "k_views": len(views)})
            add(code, desc, p, len(views), lr_, thr=throughput)
        d_saved = float(np.abs(inf.softmax(saved_logits) - probs_of["I00"]).max())

        # --- I05: ensemble (trung bình xác suất)
        seed_probs = []
        for s in pl.seeds:
            n_s, _, lg = self.val_logits(self.final_cfg(s))
            assert n_s == names
            seed_probs.append(inf.softmax(lg))
        if len(seed_probs) > 1:
            add("I05a", f"Ensemble {len(seed_probs)} seed của F01 (1 view mỗi mô hình)", inf.ensemble_probs(seed_probs),
                len(seed_probs), lat(f"I05a ensemble {len(seed_probs)} mô hình cùng kiến trúc, batch 1", k_views=len(seed_probs)),
                model_desc=f"F01 seed {list(pl.seeds)}", thr=base_thr["images_per_s"] / len(seed_probs),
                note="Thông lượng batch 32 ước tính từ K lượt cùng kiến trúc; xem phép đo I00 batch 32.")
        s1 = self.load("step1_backbones")
        top = sorted([r for r in s1["rows"] if r["init"] == "finetune"], key=lambda r: -r["val_macro_f1"])[:3]
        if len(top) >= 2:
            bb_probs, bb_models = [], []
            for r in top:
                n_b, _, lg = self.val_logits(self.cfg(r["exp_id"], backbone=r["backbone"]))
                assert n_b == names
                bb_probs.append(inf.softmax(lg))
                bb_models.append(model_lib.build_model(r["backbone"], pretrained=False).to(dev).eval())
            x1 = torch.randn(1, 3, 224, 224, device=dev)

            def ens_fn():
                with torch.inference_mode():
                    for bm in bb_models:
                        bm(x1)
            r_ = bench(ens_fn, warmup=bw, iters=bi, sync=torch.cuda.synchronize if dev.type == "cuda" else None)
            lr_ = {**base_lat, **r_, "config": f"I05b ensemble {len(top)} backbone khác nhau, batch 1",
                   "images_per_s": 1000.0 / r_["p50"]}
            lat_rows.append(lr_)
            x1 = torch.randn(32, 3, 224, 224, device=dev)
            thr_ = bench(ens_fn, warmup=bw, iters=bi, sync=torch.cuda.synchronize if dev.type == "cuda" else None)
            lat_rows.append({**base_thr, **thr_, "config": f"I05b ensemble {len(top)} backbone, batch 32",
                             "images_per_s": 32000.0 / thr_["p50"]})
            add("I05b", f"Ensemble {len(top)} backbone tốt nhất Bước 1 (công thức T00, 1 view)", inf.ensemble_probs(bb_probs),
                len(top), lr_, model_desc=" + ".join(r["exp_id"] for r in top),
                thr=32000.0 / thr_["p50"])
            del bb_models

        # --- I06: trọng số EMA so với trọng số thường (cùng epoch, cùng lần chạy)
        s2 = self.load("step2_training")
        ema_cfg = fcfg if fcfg.ema_decay else next(
            (self.cfg(e, backbone=fcfg.backbone, desc=d, **ov) for e, _, d, ov in pl.ablations if ov.get("ema_decay")), None)
        if ema_cfg is not None and (run_dir(ema_cfg) / "summary.json").exists():
            for raw, code, label in ((True, "I06a", "Trọng số thường (không EMA), cùng epoch"), (False, "I06b", "Trọng số EMA")):
                m_ = train.load_checkpoint_model(ema_cfg, dev, raw=raw)
                _, _, lg = inf.predict_logits(m_, loader, dev, view=lambda x: inf.center_crop(x, 224))
                add(code, label, inf.softmax(lg), 1, base_lat, model_desc=f"{ema_cfg.exp_id} seed {ema_cfg.seed} ({ema_cfg.backbone})",
                    note="cùng kiến trúc nên cùng độ trễ với I00; EMA không tốn thêm khi suy luận", thr=base_thr["images_per_s"])
                del m_

        # --- I07: temperature scaling (T khớp trên val)
        z0 = V["center"]
        T = inf.fit_temperature(z0, y)
        half = np.random.default_rng(0).permutation(len(y))
        a, b = half[: len(y) // 2], half[len(y) // 2:]
        ece_cf_before = np.mean([metrics_of(inf.softmax(z0[i]), y[i])["ece"] for i in (a, b)])
        ece_cf_after = np.mean([metrics_of(inf.apply_temperature(z0[j], inf.fit_temperature(z0[i], y[i])), y[j])["ece"]
                                for i, j in ((a, b), (b, a))])
        ece_before = metrics_of(inf.softmax(z0), y)["ece"]
        p_ts = inf.apply_temperature(z0, T)
        m_ts = add("I07", f"Temperature scaling trên I00 (T = {T:.3f}, khớp trên val)", p_ts, 1, base_lat,
                   note=f"ECE val trước {ece_before:.4f} -> sau {metrics_of(p_ts, y)['ece']:.4f} (khớp và đo trên cùng val "
                        f"nên lạc quan); cross-fit 2 nửa val: {ece_cf_before:.4f} -> {ece_cf_after:.4f}", thr=base_thr["images_per_s"])
        calib = {"T": T, "ece_before": ece_before, "ece_after_insample": m_ts["ece"],
                 "ece_crossfit_before": float(ece_cf_before), "ece_crossfit_after": float(ece_cf_after)}

        # --- I08: gộp BatchNorm, AMP, FP16 (độ chính xác trên val + độ trễ)
        center = lambda x: inf.center_crop(x, 224)  # noqa: E731
        variants = {"fp32": (model, "fp32", False)}
        fuse_info = {"has_batchnorm": inf.has_batchnorm(model)}
        if fuse_info["has_batchnorm"]:
            fused = inf.fuse_conv_bn(model)
            fuse_info.update(pairs=fused.fuse_pairs, max_abs_diff_logits=fused.fuse_max_abs_diff)
            _, _, lg = inf.predict_logits(fused, loader, dev, view=center)
            lr_ = lat("I08a gộp BN, FP32 batch 1", model=fused, fused_bn=True)
            fused_thr = lat("I08a gộp BN, FP32 batch 32", model=fused, fused_bn=True, batch_size=32)["images_per_s"]
            add("I08a", f"Gộp BatchNorm vào conv ({fused.fuse_pairs} cặp), FP32", inf.softmax(lg), 1, lr_,
                note=f"logit lệch tối đa {fused.fuse_max_abs_diff:.2e} trên batch ngẫu nhiên; "
                     f"xác suất val lệch tối đa {np.abs(inf.softmax(lg) - probs_of['I00']).max():.2e}", thr=fused_thr)
            variants["fused_fp32"] = (fused, "fp32", True)
        else:
            rows.append({"exp_id": "I08a", "method": "Gộp BatchNorm vào conv", "model": fcfg.backbone, "k": 1,
                         "note": "KHÔNG áp dụng: kiến trúc không có BatchNorm2d (dùng LayerNorm)"})
        if dev.type == "cuda":
            _, _, lg = inf.predict_logits(model, loader, dev, view=center, amp=True)
            lr_ = lat("I08b AMP (autocast) batch 1", dtype="amp")
            amp_thr = lat("I08b AMP (autocast) batch 32", dtype="amp", batch_size=32)["images_per_s"]
            add("I08b", "AMP (autocast FP16/FP32)", inf.softmax(lg), 1, lr_,
                note=f"xác suất val lệch tối đa {np.abs(inf.softmax(lg) - probs_of['I00']).max():.2e} so với FP32", thr=amp_thr)
            variants["amp"] = (model, "amp", False)
            half_model = copy.deepcopy(model).half()
            _, _, lg = inf.predict_logits(half_model, loader, dev, view=lambda x: center(x).half())
            lr_ = lat("I08c FP16 (model.half()) batch 1", dtype="fp16")
            half_thr = lat("I08c FP16 (model.half()) batch 32", dtype="fp16", batch_size=32)["images_per_s"]
            add("I08c", "FP16 (model.half())", inf.softmax(lg), 1, lr_,
                note=f"xác suất val lệch tối đa {np.abs(inf.softmax(lg) - probs_of['I00']).max():.2e} so với FP32", thr=half_thr)
            variants["fp16"] = (model, "fp16", False)
            del half_model
            if fuse_info["has_batchnorm"]:
                lat("I08d gộp BN + FP16 batch 1", model=fused, dtype="fp16", fused_bn=True)
                variants["fused_fp16"] = (fused, "fp16", True)
        else:
            rows.append({"exp_id": "I08b", "method": "AMP / FP16", "model": fcfg.backbone, "k": 1,
                         "note": "không đo: không có GPU"})

        i00 = next(r for r in rows if r["exp_id"] == "I00")
        for r in rows:
            if r.get("val_macro_f1") is not None and not r["exp_id"].startswith("I06"):
                r["delta_f1_vs_i00"] = r["val_macro_f1"] - i00["val_macro_f1"]
        i06 = {r["exp_id"]: r for r in rows if r["exp_id"].startswith("I06")}
        if len(i06) == 2:  # I06 có thể dùng mô hình khác F01 (lần chạy có EMA): chỉ so EMA với không EMA của chính nó
            d = i06["I06b"]["val_macro_f1"] - i06["I06a"]["val_macro_f1"]
            i06["I06b"]["note"] += f"; Δ macro-F1 so với I06a (không EMA) = {d:+.4f}"

        # --- chọn: (a) ngoại tuyến = phương pháp một-mô-hình có macro-F1 val cao nhất (hòa: K nhỏ hơn)
        single = [r for r in rows if r["exp_id"] in self.methods() and r.get("val_macro_f1") is not None]
        best = max(single, key=lambda r: (r["val_macro_f1"], -r["k"]))
        _, views, space = self.methods()[best["exp_id"]]
        flips = {"wrong_to_right": int(((probs_of["I00"].argmax(1) != y) & (probs_of[best["exp_id"]].argmax(1) == y)).sum()),
                 "right_to_wrong": int(((probs_of["I00"].argmax(1) == y) & (probs_of[best["exp_id"]].argmax(1) != y)).sum())}
        # (b) thời gian thực = 1 view, biến thể có p95 batch-1 thấp nhất
        b1 = [r for r in lat_rows if r["batch"] == 1 and r["img_size"] == 224 and "k_views" not in r
              and (r["config"].startswith("I00") or r["config"].startswith("I08"))]
        rt = min(b1, key=lambda r: r["p95"])
        choice = {"model": f"F01 seed {seed0}", "offline_method": best["exp_id"], "offline_desc": best["method"],
                  "offline_views": views, "offline_space": space, "offline_k": len(views),
                  "offline_val_macro_f1": best["val_macro_f1"], "offline_delta_vs_i00": best["delta_f1_vs_i00"],
                  "offline_flips_vs_i00": flips, "offline_p95_ms": best["lat_p95_ms"],
                  "realtime_config": rt["config"], "realtime_p50_ms": rt["p50"], "realtime_p95_ms": rt["p95"],
                  "realtime_p99_ms": rt["p99"], "realtime_ok": rt["p95"] <= pl.realtime_budget_ms,
                  "i00_p95_ms": base_lat["p95"], "calibration_i00": calib, "fuse": fuse_info,
                  "i00_matches_training_time_val_probs_max_abs_diff": d_saved,
                  "latency_conditions": {"gpu": base_lat["gpu"], "torch": base_lat["torch"], "warmup": bw, "iters": bi,
                                         "preprocessing_included": False, "input": "tensor ngẫu nhiên đã nằm trên thiết bị"},
                  "rule": "ngoại tuyến: macro-F1 val cao nhất trong các phương pháp một-mô-hình (I00-I04) + temperature "
                          "scaling; thời gian thực: 1 view, biến thể dtype/gộp BN có p95 batch-1 thấp nhất"}
        self.save("step3_inference", {"rows": rows, "latency": lat_rows, "choice": choice})

        # --- biểu đồ đánh đổi độ chính xác - độ trễ
        fig, ax = plt.subplots(figsize=(9, 5.6))
        pts = [r for r in rows if r.get("val_macro_f1") is not None and r.get("lat_p50_ms")]
        for r in pts:
            single_model = r["exp_id"] in self.methods() or r["exp_id"].startswith(("I06", "I07", "I08"))
            ax.scatter(r["lat_p50_ms"], r["val_macro_f1"], s=46, color="#2a78b5" if single_model else "#d9662b", zorder=3)
            ax.annotate(r["exp_id"], (r["lat_p50_ms"], r["val_macro_f1"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.axvline(pl.realtime_budget_ms, color="gray", ls=":", lw=1)
        ax.text(pl.realtime_budget_ms, ax.get_ylim()[0], f" ngân sách {pl.realtime_budget_ms:.0f} ms", fontsize=8, va="bottom", color="gray")
        ax.set(xscale="log", xlabel="độ trễ p50 batch 1 (ms, thang log; không tính tiền xử lý)", ylabel="macro-F1 val",
               title=f"Đánh đổi độ chính xác - độ trễ ({fcfg.backbone}, {base_lat['gpu']})\nxanh: một mô hình · cam: ensemble")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(self.paths.figures / "inference_tradeoff.png", dpi=130)
        plt.close(fig)
        print(f"Chọn suy luận ngoại tuyến: {best['exp_id']} ({best['method']}), macro-F1 val {best['val_macro_f1']:.4f} "
              f"(Δ {best['delta_f1_vs_i00']:+.4f} so với I00; đổi sai->đúng {flips['wrong_to_right']}, đúng->sai {flips['right_to_wrong']}).")
        print(f"Cấu hình thời gian thực: {rt['config']}: p50 {rt['p50']:.1f} / p95 {rt['p95']:.1f} / p99 {rt['p99']:.1f} ms "
              f"({'ĐẠT' if choice['realtime_ok'] else 'KHÔNG đạt'} ngân sách {pl.realtime_budget_ms:.0f} ms).")
        print(f"Temperature scaling (I00): T = {T:.3f}, ECE val {calib['ece_before']:.4f} -> {calib['ece_after_insample']:.4f} "
              f"(cross-fit: {ece_cf_before:.4f} -> {ece_cf_after:.4f}).")
        del model
        return pd.DataFrame(rows)

    # ================================================================== BƯỚC 4
    def _eval_cli(self, *args) -> str:
        r = subprocess.run([sys.executable, str(self.paths.repo / "eval.py"), *map(str, args)],
                           capture_output=True, text=True, encoding="utf-8",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode != 0:
            raise RuntimeError(f"eval.py thoát mã {r.returncode}:\n{r.stdout}\n{r.stderr}")
        return r.stdout

    def refresh_scores(self) -> dict:
        P, L = self.paths.pred, self.paths.labels_dir
        common = ["--test-csv", L / "test_subset0.csv", "--labels", L / "labels.csv"]
        texts = {}
        for tag in ("F01", "F01_uncal", "F02", "T00"):
            texts[f"score_{tag}"] = self._eval_cli("score", "--pred", P / f"{tag}_seed*_test.csv", *common,
                                                   "--tag", tag, "--out", self.paths.eval_out)
        ch = self.load("step3_inference")["choice"]
        texts["grade_F01_vs_T00"] = self._eval_cli(
            "grade", "--final", P / "F01_seed*_test.csv", "--baseline", P / "T00_seed*_test.csv",
            "--uncal", P / "F01_uncal_seed*_test.csv", "--final-val", P / "F01_seed*_val.csv",
            "--val-csv", L / "val_subset0.csv", "--latency-p95-ms", str(ch["i00_p95_ms"]),
            "--latency-method", "proper", *common, "--out", self.paths.eval_out)
        for key, val in texts.items():
            (self.paths.eval_out / f"{key}.md").write_text(val, encoding="utf-8")
        return texts

    def step4_final(self) -> dict:
        """Mở TEST đúng một lần cho mỗi seed: mốc T00 (1 view) và chung kết F01 (suy luận đã chốt trên val).

        Ghi predictions/: T00_seed<k>_test.csv, F01_seed<k>_{val,test}.csv (đã temperature scaling),
        F01_uncal_seed<k>_test.csv (chưa scaling), F02_seed<k>_{val,test}.csv (cấu hình thời gian thực: cùng mô
        hình F01, 1 view + temperature scaling). Sau đó chạy eval.py score/grade.
        """
        import inference as inf

        done = self.load("step4_final")
        if done is not None:
            print("Bước 4 đã chạy: test KHÔNG được chạy lại. Trả về kết quả đã lưu (logs/step4_final.json).")
            done["eval_text"] = self.refresh_scores()
            return done
        s3 = self.load("step3_inference")
        if s3 is None:
            raise RuntimeError("chưa chạy Bước 3 (chưa chốt phương pháp suy luận trên val)")
        ch = s3["choice"]
        dev = self.device()
        _, val_df, test_df = self.splits()
        needed = sorted(set(ch["offline_views"]) | {"center"})
        sizes = sorted({int(v[4:].split("_")[0]) for v in needed if v.startswith("full")})
        fns = {k: f for k, f in self.view_fns(sizes).items() if k in needed}
        per_seed = []
        for seed in self.plan.seeds:
            # --- mốc: T00 + I00
            t00 = run(dataclasses.replace(self.t00_cfg(seed), save_test_predictions=True))
            # --- chung kết
            fcfg = self.final_cfg(seed)
            rd = run_dir(fcfg)
            marker, cache = rd / "TEST_DONE.json", rd / "test_views.npz"
            model = train.load_checkpoint_model(fcfg, dev)
            vn, vy, vV = inf.predict_views(model, self.eval_loader(val_df), dev, fns)
            if cache.exists():   # Bước 4 bị ngắt giữa chừng: dùng lại logit test đã lưu, KHÔNG forward test lần hai
                d = np.load(cache, allow_pickle=False)
                tn, ty, tV = [str(f) for f in d["filenames"]], d["y_true"], {k: d[k] for k in needed}
            else:
                assert not marker.exists(), f"{fcfg.exp_id} seed {seed}: test đã chạy trước đó, không chạy lại"
                tn, ty, tV = inf.predict_views(model, self.eval_loader(test_df), dev, fns)  # lượt test DUY NHẤT của seed
                np.savez_compressed(cache, filenames=np.array(tn), y_true=ty, **tV)
                marker.write_text(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "views": needed,
                                              "inference": ch["offline_method"]}), encoding="utf-8")
            del model
            agg = lambda V: inf.aggregate_views([V[v] for v in ch["offline_views"]], ch["offline_space"])  # noqa: E731
            pv, pt = agg(vV), agg(tV)
            T = inf.fit_temperature(inf.probs_to_logits(pv), vy)            # T khớp trên VAL của seed này
            pv_cal = inf.apply_temperature(inf.probs_to_logits(pv), T)
            pt_cal = inf.apply_temperature(inf.probs_to_logits(pt), T)
            T_rt = inf.fit_temperature(vV["center"], vy)
            rt_val, rt_test = inf.apply_temperature(vV["center"], T_rt), inf.apply_temperature(tV["center"], T_rt)
            P = self.paths.pred
            save_predictions(P / f"F01_seed{seed}_val.csv", vn, vy, pv_cal)
            save_predictions(P / f"F01_seed{seed}_test.csv", tn, ty, pt_cal)
            save_predictions(P / f"F01_uncal_seed{seed}_test.csv", tn, ty, pt)
            save_predictions(P / f"F02_seed{seed}_val.csv", vn, vy, rt_val)
            save_predictions(P / f"F02_seed{seed}_test.csv", tn, ty, rt_test)
            save_predictions(P / f"F02_uncal_seed{seed}_test.csv", tn, ty, inf.softmax(tV["center"]))
            mv, mt, mu = metrics_of(pv_cal, vy), metrics_of(pt_cal, ty), metrics_of(pt, ty)
            mrv, mrt = metrics_of(rt_val, vy), metrics_of(rt_test, ty)
            per_seed.append({
                "seed": seed, "T_offline": T, "T_realtime": T_rt,
                "F01": {"val_macro_f1": mv["macro_f1"], "test_macro_f1": mt["macro_f1"], "test_top1": mt["top1"],
                        "test_ece": mt["ece"], "test_ece_uncal": mu["ece"], "val_ece": mv["ece"]},
                "F02": {"val_macro_f1": mrv["macro_f1"], "test_macro_f1": mrt["macro_f1"], "test_top1": mrt["top1"],
                        "test_ece": mrt["ece"], "test_ece_uncal": metrics_of(inf.softmax(tV["center"]), ty)["ece"]},
                "T00": {"val_macro_f1": t00["val_macro_f1"], "test_macro_f1": t00["test_macro_f1"],
                        "test_top1": t00["test_top1"], "test_ece": t00["test_ece"]}})
            print(f"seed {seed}: T00 test macro-F1 {t00['test_macro_f1']:.4f} | F01 test macro-F1 {mt['macro_f1']:.4f} "
                  f"(top-1 {mt['top1']:.4f}, ECE {mu['ece']:.4f} -> {mt['ece']:.4f}, T = {T:.3f}) | F02 {mrt['macro_f1']:.4f}", flush=True)

        # --- eval.py: score từng nhóm, grade chung kết so với mốc
        P, L = self.paths.pred, self.paths.labels_dir
        common = ["--test-csv", L / "test_subset0.csv", "--labels", L / "labels.csv"]
        texts = {}
        for tag, pattern in (("F01", "F01_seed*_test.csv"), ("F01_uncal", "F01_uncal_seed*_test.csv"),
                             ("F02", "F02_seed*_test.csv"), ("T00", "T00_seed*_test.csv")):
            texts[f"score_{tag}"] = self._eval_cli("score", "--pred", P / pattern, *common, "--tag", tag,
                                                   "--out", self.paths.eval_out)
        texts["grade_F01_vs_T00"] = self._eval_cli(
            "grade", "--final", P / "F01_seed*_test.csv", "--baseline", P / "T00_seed*_test.csv",
            "--uncal", P / "F01_uncal_seed*_test.csv", "--final-val", P / "F01_seed*_val.csv",
            "--val-csv", L / "val_subset0.csv", "--latency-p95-ms", f"{ch['i00_p95_ms']:.2f}",
            "--latency-method", "proper", *common, "--out", self.paths.eval_out)
        for k, v in texts.items():
            (self.paths.eval_out / f"{k}.md").write_text(v, encoding="utf-8")
        out = {"per_seed": per_seed, "inference": ch, "eval_text": texts,
               "note": "test chạy một lần mỗi seed; F01 và F02 dùng chung một lượt forward trên test (F02 = view center, "
                       "FP32, không gộp BN; p95 truyền cho eval.py grade là của đúng cấu hình này: I00 FP32 batch 1)"}
        self.save("step4_final", out)
        print(texts["score_F01"], "\n", texts["score_T00"], "\n", texts["grade_F01_vs_T00"])
        return out

    def error_analysis(self, seed: int | None = None) -> dict:
        """Ma trận nhầm lẫn trên test (cộng qua seed) + lưới ảnh bị đoán sai của cặp lớp khó. Chạy SAU Bước 4."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        seed = self.plan.seeds[0] if seed is None else seed
        out = {}
        fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))
        for ax, tag, title in zip(axes, ("T00", "F01"), ("Mốc T00 + I00", "Chung kết F01")):
            cm = pd.read_csv(self.paths.eval_out / f"{tag}_confusion_sum.csv", index_col=0).to_numpy()
            norm = cm / cm.sum(1, keepdims=True)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(NUM_CLASSES):
                for j in range(NUM_CLASSES):
                    if cm[i, j]:
                        ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.1f}%", ha="center", va="center", fontsize=6.5,
                                color="white" if norm[i, j] > 0.5 else "black")
            ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=8)
            ax.set(xlabel="dự đoán", ylabel="nhãn thật",
                   title=f"{title}: ma trận nhầm lẫn TEST (cộng {len(self.plan.seeds)} seed; % theo hàng)")
            off = cm.copy()
            np.fill_diagonal(off, 0)
            pairs = sorted(((int(off[i, j]), i, j) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if off[i, j]),
                           reverse=True)[:8]
            out[f"{tag}_top_confusions"] = [{"true": CLASS_NAMES[i], "pred": CLASS_NAMES[j], "count_sum_seeds": c,
                                             "pct_of_true_class": float(norm[i, j] * 100)} for c, i, j in pairs]
        fig.tight_layout()
        fig.savefig(self.paths.figures / "confusion_test.png", dpi=120)
        plt.close(fig)

        df = pd.read_csv(self.paths.pred / f"F01_seed{seed}_test.csv")
        wrong = df[df.y_true != df.y_pred].copy()
        wrong["conf"] = wrong[[f"p{i}" for i in range(NUM_CLASSES)]].max(1)
        out["n_wrong_seed"] = int(len(wrong))
        hard = wrong[wrong.y_true.isin([0, 7]) & wrong.y_pred.isin([0, 7])].sort_values("conf", ascending=False)
        other = wrong.drop(hard.index).sort_values("conf", ascending=False)
        pick = pd.concat([hard.head(12), other.head(max(0, 18 - min(12, len(hard))))])
        out["n_chinee_snake_confusions_seed"] = int(len(hard))
        out["misclassified_shown"] = pick[["Filename", "y_true", "y_pred", "conf"]].to_dict("records")
        if len(pick):
            cols = 6
            rws = int(math.ceil(len(pick) / cols))
            fig, ax = plt.subplots(rws, cols, figsize=(cols * 2.5, rws * 2.9), squeeze=False)
            for a in ax.ravel():
                a.axis("off")
            for a, (_, r) in zip(ax.ravel(), pick.iterrows()):
                a.imshow(Image.open(self.paths.images_dir / r.Filename))
                a.set_title(f"thật: {CLASS_NAMES[int(r.y_true)]}\nđoán: {CLASS_NAMES[int(r.y_pred)]} ({r.conf:.2f})\n{r.Filename}",
                            fontsize=7)
            fig.suptitle(f"F01 seed {seed}: ảnh TEST bị đoán sai (trước: cặp Chinee Apple ↔ Snake Weed; sau: lỗi tự tin nhất)",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(self.paths.figures / "errors_test.png", dpi=110)
            plt.close(fig)
        self.save("step4_errors", out)
        return out

# Tham khảo implementation: picuisme/K4-Track4-Day2-Deeplearning-Advance @ 941d9fb.
# Bản cho Nguyen Tuan Thanh: sửa optimizer/resume/report/Colab; số liệu phải chạy riêng.
