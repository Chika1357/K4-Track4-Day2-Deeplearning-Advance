"""Colab entry point. Runtime data stays local; logs and checkpoints stay on Drive.

python colab_runner.py --stage prepare --work /content/deepweeds --output <Drive/run>
python colab_runner.py --stage 0 1 2 3 4 5 --work ... --output ...
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiments as E

IDENTITY = {"student_name": "Nguyen Tuan Thanh", "student_id": "2A202602640"}
DEADLINE_BACKBONES = [
    ("B01", "resnet18.tv_in1k", "resnet18", "ResNet"),
    ("B02", "resnext50_32x4d.tv_in1k", "resnext50", "ResNeXt"),
    ("B03", "convnext_tiny.fb_in1k", "convnext_tiny", "ConvNeXt"),
    ("B04", "deit_tiny_patch16_224.fb_in1k", "deit_tiny", "Transformer (DeiT)"),
    ("B05", "mobilenetv3_small_100.lamb_in1k", "mobilenetv3_small", "Mạng nhẹ"),
]
DEADLINE_ABLATIONS = [
    ("T01", "A", "frozen", {"init": "frozen"}),
    ("T02", "B", "cutmix", {"mix": "cutmix", "mix_alpha": 1.0}),
    ("T03", "C", "labelsmooth", {"loss": "ls", "label_smoothing": 0.1}),
]


def make_plan(profile="deadline", epochs=None, batch_size=64, workers=2):
    if profile == "deadline":
        p = E.Plan(epochs=12, batch_size=batch_size, num_workers=workers,
                   backbones=list(DEADLINE_BACKBONES), ablations=list(DEADLINE_ABLATIONS),
                   dinov2=None, parallel=False, bench_iters=50, res_sweep=(224, 256), profile=profile)
    else:
        p = E.Plan(batch_size=batch_size, num_workers=workers, parallel=False, profile=profile)
    if epochs is not None:
        if epochs < 1:
            raise ValueError("epochs phải >= 1")
        p.epochs = epochs
    return p


def download(url, dest, md5=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    def checksum(path):
        h = hashlib.md5()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    if dest.exists() and (md5 is None or checksum(dest) == md5):
        return
    partial = dest.with_suffix(dest.suffix + ".partial")
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DeepWeeds-Lab/1.0"})
            with urllib.request.urlopen(request, timeout=120) as src, partial.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            if md5 is not None and checksum(partial) != md5:
                raise ValueError(f"Checksum sai cho {dest.name}")
            partial.replace(dest)
            return
        except Exception as exc:
            error = exc
            print(f"Tải {dest.name}: lần {attempt + 1} lỗi {exc}; thử lại.", flush=True)
            time.sleep(2)
    raise RuntimeError(f"Không tải được {url}: {error}")


def prepare(work):
    work = Path(work)
    labels, images = work / "labels", work / "images"
    for name in ("labels.csv", "train_subset0.csv", "val_subset0.csv", "test_subset0.csv"):
        download(f"https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/{name}", labels / name)
    archive = work / "images.zip"
    download("https://zenodo.org/records/7939060/files/images.zip?download=1", archive,
             "b7b30f96d466fba86016aa5a26606e0f")
    images.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            # Dataset archives may contain a directory prefix; CSV names refer to the basename.
            target = images / Path(info.filename).name
            if target.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if not target.exists() or target.stat().st_size != info.file_size:
                with z.open(info) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
    print("Dữ liệu đã sẵn sàng:", images, flush=True)


def create_lab(work, output, plan):
    work, output = Path(work), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = output / "identity.json"
    identity.write_text(json.dumps(IDENTITY, indent=2), encoding="utf-8")
    config = output / "plan.json"
    signature = json.loads(json.dumps(dataclasses.asdict(plan)))
    if config.exists() and json.loads(config.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("RUN_NAME đã dùng với cấu hình khác. Đổi RUN_NAME để giữ các thí nghiệm công bằng.")
    config.write_text(json.dumps(signature, indent=2, ensure_ascii=False), encoding="utf-8")
    if not (output / "README.md").exists():
        shutil.copy2(HERE.parent / "README.md", output / "README.md")
    # Copy exactly the source used to produce these runs into the final submission.
    shutil.copytree(HERE, output / "code", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    repo = E.train.find_repo_root()
    if repo != output / "code":
        shutil.copy2(repo / "eval.py", output / "code/eval.py")
    lab = E.Lab(E.Paths(repo=repo, sub=output, images_dir=work / "images", labels_dir=work / "labels",
                         ckpt_dir=output / "checkpoints", cache_dir=work / "cache"), plan)
    return lab


def run_stages(lab, stages, notebook_link=None, bonus=False):
    import torch
    if not torch.cuda.is_available() and any(s in {"0", "1", "2", "3", "4"} for s in stages):
        raise RuntimeError("Chưa bật GPU: Colab > Runtime > Change runtime type > T4 GPU.")
    for stage in stages:
        started = time.perf_counter()
        print(f"\nBƯỚC {stage} — {time.strftime('%H:%M:%S')}", flush=True)
        if stage == "0":
            import dataset
            tr, va, te = lab.splits()
            dataset.build_cache(lab.paths.images_dir, __import__('pandas').concat([tr, va, te])["Filename"], lab.paths.cache_dir)
            if lab.load("step0_data") is None:
                lab.step0_data()
            if lab.load("step0_sanity") is None:
                lab.step0_sanity()
        elif stage == "1":
            lab.step1_backbones()
            s = lab.load("step1_backbones")
            epoch_s = next(r['train_time_per_epoch_s'] for r in s['rows'] if r['backbone'] == s['choice']['chosen'])
            # Two additional baseline seeds + ablations + at least one combo + two finalist seeds.
            estimated = (5 + len(lab.plan.ablations)) * lab.plan.epochs * epoch_s / 60
            print(f"ƯỚC LƯỢNG training còn lại ~{estimated:.0f} phút (chưa gồm inference/Drive I/O).", flush=True)
        elif stage == "2":
            lab.step2_training()
            lab.train_finalists()
        elif stage == "3":
            lab.step3_inference()
        elif stage == "4":
            lab.step4_final()
            lab.error_analysis()
            if bonus:
                import bonus as extras
                extras.robustness(lab)
                extras.gradcam_errors(lab)
        elif stage == "5":
            import make_results
            if lab.load("step4_final") is None:
                raise RuntimeError("Cần chạy xong bước 4 trước khi xuất bài nộp.")
            lab.refresh_scores()
            result = make_results.build_all(lab, notebook_link=notebook_link)
            if not result['ok']:
                raise RuntimeError(f"Kiểm tra bài nộp chưa đạt: {result}")
            package_submission(lab.paths.sub)
        else:
            raise ValueError(f"Bước không hợp lệ: {stage}")
        print(f"Bước {stage} xong trong {(time.perf_counter()-started)/60:.1f} phút.", flush=True)


def package_submission(output):
    output = Path(output)
    dest = output.parent / "submission_2A202602640.zip"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob("*")):
            if not p.is_file() or any(x in p.parts for x in ("checkpoints", "__pycache__")):
                continue
            if p.suffix in {".pt", ".pth", ".tmp", ".pyc"}:
                continue
            z.write(p, str(Path("2A202602640_nguyen_tuan_thanh") / p.relative_to(output)))
    print("BÀI NỘP:", dest, flush=True)
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["deadline", "full"], default="deadline")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage", nargs="+", default=["prepare", "0", "1", "2", "3", "4", "5"])
    parser.add_argument("--notebook-link")
    args = parser.parse_args()
    if "prepare" in args.stage:
        prepare(args.work)
    stages = [s for s in args.stage if s != "prepare"]
    if stages:
        lab = create_lab(args.work,args.output,make_plan(args.profile,args.epochs,args.batch_size,args.workers))
        run_stages(lab, stages, args.notebook_link)


if __name__ == "__main__":
    main()
