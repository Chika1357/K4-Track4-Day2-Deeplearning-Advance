"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md của repo, mục 2.1.

Giao diện (giữ nguyên theo starter để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Phần thêm (không đổi hợp đồng cũ):
    build_cache(images_dir, filenames, cache_dir)  -> nạp trước ảnh đã giải mã vào một file memmap uint8
    denormalize(x)                                 -> đưa tensor đã chuẩn hoá về [0, 1] để vẽ
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # mọi tag trọng số dùng trong bài đều có mean/std này (model.py kiểm tra)
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509
NATIVE_SIZE = 256                       # mọi ảnh DeepWeeds là RGB 256x256 (đã kiểm ở EDA)
AUG_CHOICES = ("basic", "color", "trivial", "randaug", "vflip")


# --------------------------------------------------------------------------- #
# Đọc và kiểm tra split
# --------------------------------------------------------------------------- #
def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Chỉ đọc: KHÔNG sửa, lọc hay chia lại. Trả về (train_df, val_df, test_df), mỗi bảng có
    cột `Filename, Label` (file của tác giả không có cột Species; tên lớp lấy từ labels.csv).
    """
    labels_dir = Path(labels_dir)
    out = []
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_subset{fold}.csv"
        if not path.exists():
            raise FileNotFoundError(f"thiếu file chia dữ liệu: {path}")
        df = pd.read_csv(path)
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
        df["Label"] = df["Label"].astype(int)
        out.append(df.reset_index(drop=True))
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, expected_total: int | None = TOTAL_IMAGES,
                verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Dừng ngay (AssertionError) nếu:
      1. có Filename trùng trong một tập, hoặc nhãn ngoài 0..8
      2. giao của một cặp tập theo Filename khác rỗng (train∩val, train∩test, val∩test)
      3. hợp ba tập khác `expected_total` (17.509; đặt None để bỏ qua khi chạy thử trên tập con)
      4. có Filename không tồn tại trong `images_dir`
      5. tỉ lệ lệch khỏi 60/20/20 quá 1 điểm phần trăm (bỏ qua cùng với mục 3 khi expected_total=None)
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(d["Filename"]) for k, d in splits.items()}
    n = {k: int(len(d)) for k, d in splits.items()}
    total = sum(n.values())

    for k, d in splits.items():
        assert not d["Filename"].duplicated().any(), f"{k}: có Filename bị trùng"
        assert d["Label"].between(0, NUM_CLASSES - 1).all(), f"{k}: nhãn ngoài 0..{NUM_CLASSES - 1}"

    per_class = {k: [int((d["Label"] == c).sum()) for c in range(NUM_CLASSES)] for k, d in splits.items()}
    overlap = {"train∩val": len(names["train"] & names["val"]),
               "train∩test": len(names["train"] & names["test"]),
               "val∩test": len(names["val"] & names["test"])}
    union = len(names["train"] | names["val"] | names["test"])
    frac = {k: v / total for k, v in n.items()}

    on_disk = {p.name for p in Path(images_dir).iterdir()} if Path(images_dir).is_dir() else set()
    missing_files = sorted(f for s in names.values() for f in s if f not in on_disk)

    class_total = [sum(per_class[k][c] for k in splits) for c in range(NUM_CLASSES)]
    weeds = [x for x in class_total[:-1] if x > 0]
    out = {"n": n, "total": total, "frac": {k: round(v, 4) for k, v in frac.items()},
           "per_class": per_class, "class_total": class_total, "class_names": CLASS_NAMES,
           "overlap": overlap, "union": union, "missing_files": len(missing_files),
           "imbalance_ratio": (max(class_total) / min(weeds)) if weeds else float("nan"),
           "n_images_on_disk": len(on_disk)}

    if verbose:
        print(f"Số ảnh: train {n['train']} | val {n['val']} | test {n['test']} | tổng {total}")
        print("Tỉ lệ : " + " / ".join(f"{frac[k] * 100:.2f}%" for k in splits))
        table = pd.DataFrame(per_class, index=CLASS_NAMES)
        table["tổng"] = class_total
        print(table.to_string())
        print(f"Giao  : {overlap}   | hợp ba tập = {union}")
        print(f"File thiếu trong thư mục ảnh: {len(missing_files)}")
        print(f"Tỉ lệ lớp nhiều nhất / lớp ít nhất: {out['imbalance_ratio']:.2f}")

    assert all(v == 0 for v in overlap.values()), f"các tập bị giao nhau: {overlap}"
    assert union == total, "hợp ba tập khác tổng số dòng (có ảnh trùng giữa các tập)"
    if expected_total is not None:
        assert union == expected_total, f"hợp ba tập = {union}, kỳ vọng {expected_total}"
    assert not missing_files, f"{len(missing_files)} file không có trong {images_dir}, ví dụ {missing_files[:3]}"
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)) if expected_total is not None else ():
        assert abs(frac[k] - target) <= 0.01, (
            f"tỉ lệ {k} = {frac[k]:.3f} lệch khỏi {target} hơn 1 điểm phần trăm: báo giảng viên trước khi train")
    if verbose:
        print("=> ĐẠT mọi kiểm tra chia dữ liệu (S1-S4).")
    return out


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def build_transforms(train: bool, img_size: int | None = 224, aug: str = "basic"):
    """Tạo transform (torchvision).

    Train:
      - "basic"  : RandomResizedCrop(img_size) + lật ngang                       (công thức nền T00)
      - "color"  : basic + ColorJitter (sáng, tương phản, bão hoà, một chút hue)
      - "trivial": basic + TrivialAugmentWide
      - "randaug": basic + RandAugment(2, 9)
      - "vflip"  : basic + lật dọc. Ảnh DeepWeeds chụp từ trên xuống nên không có "chiều trên/dưới"
                   cố định; lật dọc giữ nguyên nhãn. Có giúp hay không thì đo ở trục B.
    Val/test (`train=False`): ảnh gốc 256x256 -> CenterCrop(img_size) + ToTensor + Normalize, KHÔNG
      có augmentation ngẫu nhiên. `img_size=None` giữ nguyên 256 (dùng cho TTA/dò độ phân giải:
      các view được cắt/resize trên GPU trong inference.py).
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.
    """
    from torchvision import transforms as T

    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        if img_size is None or img_size == NATIVE_SIZE:
            return T.Compose(norm)
        if img_size < NATIVE_SIZE:
            return T.Compose([T.CenterCrop(img_size), *norm])
        return T.Compose([T.Resize(img_size, antialias=True), *norm])

    if aug not in AUG_CHOICES:
        raise ValueError(f"aug={aug!r} không hợp lệ, chọn một trong {AUG_CHOICES}")
    ops = [T.RandomResizedCrop(img_size or NATIVE_SIZE, antialias=True), T.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug == "vflip":
        ops.append(T.RandomVerticalFlip())
    return T.Compose([*ops, *norm])


def denormalize(x):
    """Tensor đã chuẩn hoá (C,H,W) hoặc (N,C,H,W) -> [0, 1] để vẽ."""
    import torch

    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


# --------------------------------------------------------------------------- #
# Cache ảnh đã giải mã (tuỳ chọn, để DataLoader không nghẽn ở giải mã JPEG)
# --------------------------------------------------------------------------- #
def build_cache(images_dir: str | Path, filenames, cache_dir: str | Path) -> Path:
    """Giải mã mọi ảnh một lần vào `cache_dir/images_u8.npy` (N, 256, 256, 3) + `index.csv`.

    Pixel giống hệt `PIL.Image.open(...).convert("RGB")`, chỉ khác là không phải giải mã lại mỗi epoch.
    File memmap dùng chung được giữa các tiến trình (2 GPU chạy song song trên Kaggle).
    """
    from PIL import Image

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filenames = sorted(set(filenames))
    idx_path, arr_path = cache_dir / "index.csv", cache_dir / "images_u8.npy"
    if idx_path.exists() and arr_path.exists() and list(pd.read_csv(idx_path)["Filename"]) == filenames:
        return cache_dir
    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8,
                                    shape=(len(filenames), NATIVE_SIZE, NATIVE_SIZE, 3))
    for i, f in enumerate(filenames):
        with Image.open(Path(images_dir) / f) as im:
            a = np.asarray(im.convert("RGB"))
        assert a.shape == (NATIVE_SIZE, NATIVE_SIZE, 3), f"{f}: kích thước {a.shape} khác 256x256x3"
        arr[i] = a
    arr.flush()
    del arr
    pd.DataFrame({"Filename": filenames}).to_csv(idx_path, index=False)
    return cache_dir


# --------------------------------------------------------------------------- #
# Dataset và DataLoader
# --------------------------------------------------------------------------- #
try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # cho phép import module khi chưa cài torch (chỉ dùng load_split/check_split)
    _TorchDataset = object


class DeepWeedsDataset(_TorchDataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str). Tên file cần có để ghi
    `predictions/*.csv` đúng định dạng của eval.py. Nếu có `cache_dir` (xem build_cache) thì đọc
    ảnh đã giải mã từ memmap thay vì mở file JPEG.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None,
                 cache_dir: str | Path | None = None):
        self.filenames = df["Filename"].astype(str).tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self._mm = None
        self._cache_file = None
        self._cache_rows = None
        if cache_dir is not None and (Path(cache_dir) / "images_u8.npy").exists():
            index = pd.read_csv(Path(cache_dir) / "index.csv")["Filename"]
            pos = {f: i for i, f in enumerate(index)}
            if all(f in pos for f in self.filenames):
                self._cache_file = str(Path(cache_dir) / "images_u8.npy")
                self._cache_rows = [pos[f] for f in self.filenames]

    def __len__(self) -> int:
        return len(self.filenames)

    def _load(self, i: int):
        from PIL import Image

        if self._cache_file is not None:
            if self._mm is None:  # mở lười trong từng worker
                self._mm = np.load(self._cache_file, mmap_mode="r")
            return Image.fromarray(np.array(self._mm[self._cache_rows[i]]))
        with Image.open(self.images_dir / self.filenames[i]) as im:
            return im.convert("RGB")

    def __getitem__(self, i: int):
        img = self._load(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.filenames[i]


def _seed_worker(worker_id: int) -> None:
    """Mỗi worker lấy seed riêng suy ra từ seed của DataLoader (tái lập được augmentation)."""
    import torch

    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, cache_dir: str | Path | None = None):
    """Tạo DataLoader.

    - train=True : xáo trộn (hoặc sampler), drop_last=True để batch cuối nhỏ không làm BatchNorm nhiễu.
    - train=False: không xáo trộn, giữ đúng thứ tự `df` (để ghép logit với Filename).
    - sampler="balanced": WeightedRandomSampler, trọng số mỗi ảnh = 1 / (số ảnh của lớp nó) (trục D).
    - `seed` cố định thứ tự batch và seed của worker.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform, cache_dir=cache_dir)
    g = torch.Generator()
    g.manual_seed(seed)
    smp = None
    if train and sampler == "balanced":
        counts = np.bincount(df["Label"].to_numpy(), minlength=NUM_CLASSES).astype(np.float64)
        w = 1.0 / counts[df["Label"].to_numpy()]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                    replacement=True, generator=g)
    elif sampler not in (None, "none", "balanced"):
        raise ValueError(f"sampler={sampler!r} không hợp lệ (None | 'balanced')")
    return DataLoader(ds, batch_size=batch_size, shuffle=train and smp is None, sampler=smp,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train and len(ds) > batch_size, worker_init_fn=_seed_worker, generator=g,
                      persistent_workers=num_workers > 0)

# Tham khảo implementation: picuisme/K4-Track4-Day2-Deeplearning-Advance @ 941d9fb.
# Bản cho Nguyen Tuan Thanh: sửa optimizer/resume/report/Colab; số liệu phải chạy riêng.
