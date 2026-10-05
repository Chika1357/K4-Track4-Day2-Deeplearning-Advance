"""selftest.py - kiểm tra tự viết cho các phần dễ sai (RUBRIC mục C và H). Không cần GPU, không cần dữ liệu.

Chạy:  python selftest.py        (hoặc: python -m unittest selftest -v)
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark  # noqa: E402
import inference  # noqa: E402
import losses  # noqa: E402
import model as model_lib  # noqa: E402
import train  # noqa: E402
from dataset import build_transforms  # noqa: E402


def _data(n=64, k=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=g) * 3, torch.randint(0, k, (n,), generator=g)


class TestLosses(unittest.TestCase):
    def test_focal_gamma0_is_cross_entropy(self):
        logits, y = _data()
        self.assertLess(abs(losses.FocalLoss(gamma=0.0)(logits, y) - F.cross_entropy(logits, y)).item(), 1e-6)

    def test_focal_gamma2_downweights_easy_examples(self):
        logits, y = _data()
        self.assertLess(losses.FocalLoss(gamma=2.0)(logits, y).item(), F.cross_entropy(logits, y).item())
        pt = F.softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
        manual = (-(1 - pt) ** 2 * pt.log()).mean()
        self.assertLess(abs(losses.FocalLoss(gamma=2.0)(logits, y) - manual).item(), 1e-6)

    def test_focal_alpha_weights_classes(self):
        logits, y = _data()
        alpha = torch.linspace(0.5, 1.5, 9)
        manual = (alpha[y] * F.cross_entropy(logits, y, reduction="none")).mean()
        self.assertLess(abs(losses.FocalLoss(gamma=0.0, alpha=alpha)(logits, y) - manual).item(), 1e-6)

    def test_label_smoothing_eps0_is_cross_entropy_and_matches_torch(self):
        logits, y = _data()
        self.assertLess(abs(losses.LabelSmoothingCE(0.0)(logits, y) - F.cross_entropy(logits, y)).item(), 1e-6)
        ref = F.cross_entropy(logits, y, label_smoothing=0.1)
        self.assertLess(abs(losses.LabelSmoothingCE(0.1)(logits, y) - ref).item(), 1e-6)

    def test_build_criterion_kinds(self):
        logits, y = _data()
        w = losses.class_weights([675, 637, 618, 613, 637, 605, 644, 609, 5463], beta=0.999)
        for kind, kw in [("ce", {}), ("ls", {"smoothing": 0.1}), ("focal", {"gamma": 2.0}), ("ce_weighted", {"weight": w})]:
            self.assertTrue(torch.isfinite(losses.build_criterion(kind, **kw)(logits, y)))
        with self.assertRaises(ValueError):
            losses.build_criterion("khong_co")

    def test_class_weights(self):
        counts = [675, 637, 618, 613, 637, 605, 644, 609, 5463]
        w0 = losses.class_weights(counts, beta=0.0)
        self.assertAlmostEqual(w0.mean().item(), 1.0, places=5)
        self.assertAlmostEqual((w0[0] / w0[8]).item(), 5463 / 675, places=3)   # tỉ lệ nghịch với số ảnh
        wb = losses.class_weights(counts, beta=0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)
        self.assertEqual(int(wb.argmin()), 8)                                    # Negatives nhẹ nhất
        self.assertLess((wb.max() / wb.min()).item(), (w0.max() / w0.min()).item())  # class-balanced dịu hơn 1/n

    def test_cutmix_lam_is_true_area_and_labels_are_mixed(self):
        torch.manual_seed(0)
        checked = 0
        for _ in range(50):
            y = torch.arange(8)
            x = y.float().view(8, 1, 1, 1).expand(8, 3, 32, 32).clone()   # ảnh i có mọi pixel = i
            xm, (ya, yb, lam) = losses.mix_batch(x, y, alpha=1.0, mode="cutmix")
            self.assertTrue(0.0 <= lam <= 1.0)
            self.assertTrue(torch.equal(ya, y))
            for i in range(8):
                if yb[i] == ya[i]:
                    continue                                              # hoán vị trỏ về chính nó
                pasted = (xm[i] != float(i))
                self.assertAlmostEqual(pasted.float().mean().item(), 1.0 - lam, places=6)  # lam = 1 - diện tích thật/(H*W)
                if pasted.any():
                    self.assertTrue((xm[i][pasted] == float(yb[i])).all())  # vùng dán đến từ đúng ảnh mang nhãn y_b
                checked += 1
        self.assertGreater(checked, 100)

    def test_cutmix_box_clipped_at_border(self):
        torch.manual_seed(1)
        for _ in range(200):
            y1, y2, x1, x2 = losses.rand_bbox(32, 32, lam=0.3)
            self.assertTrue(0 <= y1 <= y2 <= 32 and 0 <= x1 <= x2 <= 32)

    def test_mixup_and_mixed_loss(self):
        torch.manual_seed(0)
        x = torch.randn(8, 3, 16, 16)
        y = torch.randint(0, 9, (8,))
        xm, (ya, yb, lam) = losses.mix_batch(x, y, alpha=0.4, mode="mixup")
        perm_src = (xm - lam * x) / (1 - lam)
        self.assertTrue(any(torch.allclose(perm_src[0], x[j], atol=1e-4) for j in range(8)))
        logits = torch.randn(8, 9)
        ce = nn.CrossEntropyLoss()
        expect = lam * ce(logits, ya) + (1 - lam) * ce(logits, yb)
        self.assertLess(abs(losses.mixed_loss(ce, logits, (ya, yb, lam)) - expect).item(), 1e-7)
        soft = lam * F.one_hot(ya, 9) + (1 - lam) * F.one_hot(yb, 9)       # tương đương nhãn mềm
        soft_ce = -(soft * F.log_softmax(logits, 1)).sum(1).mean()
        self.assertLess(abs(expect - soft_ce).item(), 1e-6)


class TinyNet(nn.Module):
    """Mạng nhỏ có đủ conv thường, depthwise, BN và head để kiểm tra nhóm tham số / gộp BN."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.act = nn.ReLU()
        self.block = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=True), nn.BatchNorm2d(8), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 9)

    def get_classifier(self):
        return self.fc

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        return self.fc(self.pool(self.block(x)).flatten(1))


def _randomize_bn(m):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.normal_()
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.normal_()


class TestModel(unittest.TestCase):
    def test_param_groups_three_groups(self):
        m = TinyNet()
        groups = model_lib.param_groups(m, 1e-4, 1e-3, 0.05)
        by = {g["name"]: g for g in groups}
        self.assertEqual(set(by), {"backbone_decay", "backbone_no_decay", "head", "head_no_decay"})
        self.assertTrue(all(p.ndim > 1 for p in by["backbone_decay"]["params"]))
        self.assertTrue(all(p.ndim <= 1 for p in by["backbone_no_decay"]["params"]))
        self.assertEqual(by["backbone_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(by["backbone_decay"]["weight_decay"], 0.05)
        self.assertEqual((by["backbone_decay"]["lr"], by["head"]["lr"]), (1e-4, 1e-3))
        n = sum(p.numel() for g in groups for p in g["params"])
        self.assertEqual(n, sum(p.numel() for p in m.parameters()))          # không sót, không trùng

    def test_freeze_backbone_keeps_bn_in_eval(self):
        m = TinyNet()
        model_lib.freeze_backbone(m)
        self.assertEqual([n for n, p in m.named_parameters() if p.requires_grad], ["fc.weight", "fc.bias"])
        groups = model_lib.param_groups(m, 1e-4, 1e-3, 0.05)
        self.assertEqual([g["name"] for g in groups], ["head", "head_no_decay"])               # bỏ qua tham số đóng băng
        model_lib.set_train_mode(m)
        self.assertTrue(m.training and m.fc.training)
        self.assertFalse(m.bn1.training)
        before = m.bn1.running_mean.clone()
        m(torch.randn(4, 3, 16, 16))
        self.assertTrue(torch.equal(before, m.bn1.running_mean))              # thống kê BN không bị cập nhật

    def test_timm_models(self):
        import timm  # noqa: F401
        m = model_lib.build_model("efficientnet_b0", pretrained=False, init="frozen")
        self.assertTrue(all(p.requires_grad for p in model_lib.head_parameters(m)))
        self.assertEqual(sum(p.requires_grad for p in m.parameters()), 2)
        self.assertAlmostEqual(model_lib.count_params(m), 4.019, places=2)
        m = model_lib.build_model("resnet50", pretrained=False)
        self.assertEqual(m(torch.zeros(1, 3, 64, 64)).shape, (1, 9))
        self.assertAlmostEqual(model_lib.count_gmacs(m, 224), 4.1, delta=0.15)  # slide: ResNet-50 4,1 GMAC
        for name in ("deit_small", "swin_tiny", "convnext_tiny"):               # pos_embed/cls_token/bias bảng: wd = 0
            m = model_lib.build_model(name, pretrained=False)
            by = {g["name"]: g for g in model_lib.param_groups(m, 1e-4, 1e-3, 0.05)}
            names = {id(p): n for n, p in m.named_parameters()}
            decayed = {names[id(p)] for p in by["backbone_decay"]["params"]}
            self.assertFalse({n for n in decayed if "pos_embed" in n or "cls_token" in n or "bias_table" in n})
            self.assertTrue(all(p.ndim > 1 for p in by["backbone_decay"]["params"]))


class TestTrain(unittest.TestCase):
    def test_lr_schedule_warmup_then_cosine(self):
        total, warm = 1000, 100
        f = [train.lr_factor(s, total, warm, 0.01) for s in range(total)]
        self.assertTrue(all(b > a for a, b in zip(f[:warm - 1], f[1:warm])))       # warmup tăng dần
        self.assertAlmostEqual(f[warm - 1], 1.0, places=6)
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(f[warm:], f[warm + 1:])))  # cosine giảm dần
        self.assertLess(f[-1], 0.011)
        self.assertAlmostEqual(f[warm + (total - warm) // 2], 0.01 + 0.99 * 0.5, places=2)

    def test_scheduler_applies_to_all_groups(self):
        m = TinyNet()
        cfg = train.Config(epochs=2, warmup_epochs=1.0)
        opt = train.build_optimizer(m, cfg)
        sch = train.build_scheduler(opt, cfg, steps_per_epoch=10)
        lrs = []
        for _ in range(20):
            lrs.append([g["lr"] for g in opt.param_groups])
            opt.step()
            sch.step()
        lrs = np.array(lrs)
        np.testing.assert_allclose(lrs[:, -1] / lrs[:, 0], 10.0, rtol=1e-6)        # head luôn gấp 10 lần backbone
        self.assertAlmostEqual(lrs[9, 0], 1e-4, places=9)

    def test_ema(self):
        m = TinyNet()
        ema = train.EMA(m, decay=0.9)
        w0 = m.fc.weight.detach().clone()
        with torch.no_grad():
            m.fc.weight.add_(1.0)
            m.bn1.running_mean.add_(2.0)
            m.bn1.num_batches_tracked.add_(5)
        ema.update(m)
        d = min(0.9, 2 / 11)                                                       # decay khởi động ở bước 1
        self.assertTrue(torch.allclose(ema.module.fc.weight, d * w0 + (1 - d) * (w0 + 1.0), atol=1e-6))
        self.assertAlmostEqual(ema.module.bn1.running_mean.mean().item(), (1 - d) * 2.0, places=5)
        self.assertEqual(int(ema.module.bn1.num_batches_tracked), 5)
        self.assertFalse(ema.module.training)
        self.assertFalse(any(p.requires_grad for p in ema.module.parameters()))
        for _ in range(300):
            ema.update(m)
        self.assertTrue(torch.allclose(ema.module.fc.weight, m.fc.weight, atol=1e-4))  # hội tụ về trọng số hiện tại

    def test_parse_overrides(self):
        d = train.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "amp=false", "lr_head=3e-4",
                                   "mix=cutmix", "class_weight_beta=0.999", "sampler=none"])
        self.assertEqual(d, {"seed": 1, "loss": "focal", "ema_decay": None, "amp": False, "lr_head": 3e-4,
                             "mix": "cutmix", "class_weight_beta": 0.999, "sampler": None})
        with self.assertRaises(KeyError):
            train.parse_overrides(["khong_co=1"])
        with self.assertRaises(ValueError):
            train.parse_overrides(["seed=none"])
        cfg = train.Config(exp_id="T09", loss="focal", ema_decay=0.998, save_test_predictions=True)
        self.assertEqual(train.Config(**train.parse_overrides(train.config_to_overrides(cfg))), cfg)

    def test_baseline_defaults_and_paths(self):
        c = train.Config()
        self.assertEqual((c.epochs, c.batch_size, c.lr_backbone, c.lr_head, c.weight_decay), (12, 64, 1e-4, 1e-3, 0.05))
        self.assertFalse(c.save_test_predictions)
        self.assertEqual(train.pred_path(train.Config(exp_id="F01", seed=2), "test"), Path("predictions/F01_seed2_test.csv"))
        self.assertEqual(train.run_dir(train.Config(exp_id="T03", seed=1)), Path("runs/T03/seed1"))
        self.assertEqual(train.curve_path(train.Config(exp_id="B01", backbone="resnet50.tv_in1k")).name, "B01_resnet50.png")


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_exact(self):
        torch.manual_seed(0)
        m = TinyNet().eval()
        _randomize_bn(m)
        x = torch.randn(4, 3, 24, 24)
        fused = inference.fuse_conv_bn(m, check_input=x)
        self.assertEqual(fused.fuse_pairs, 2)
        self.assertFalse(inference.has_batchnorm(fused))
        self.assertLess((m(x) - fused(x)).abs().max().item(), 1e-5)
        self.assertLess(fused.fuse_max_abs_diff, 1e-5)
        self.assertTrue(inference.has_batchnorm(m))                              # model gốc không bị sửa

    def test_fuse_conv_bn_timm(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 96, 96)
        for name, has_bn in [("resnet50", True), ("efficientnet_b0", True), ("mobilenetv3", True), ("convnext_tiny", False)]:
            m = model_lib.build_model(name, pretrained=False).eval()
            _randomize_bn(m)
            fused = inference.fuse_conv_bn(m, check_input=x)
            self.assertEqual(fused.fuse_pairs > 0, has_bn, name)
            self.assertFalse(inference.has_batchnorm(fused), name)
            rel = (m(x) - fused(x)).abs().max().item() / (m(x).abs().max().item() + 1e-12)
            self.assertLess(rel, 1e-4, name)

    def test_views(self):
        x = torch.arange(2 * 3 * 256 * 256, dtype=torch.float32).reshape(2, 3, 256, 256)
        self.assertTrue(torch.equal(inference.view_hflip(inference.view_hflip(x)), x))
        self.assertTrue(torch.equal(inference.view_hflip(x)[..., 0], x[..., -1]))
        crops = inference.views_multicrop(x, 224)
        self.assertEqual(len(crops), 5)
        self.assertTrue(all(c.shape == (2, 3, 224, 224) for c in crops))
        self.assertTrue(torch.equal(crops[0], x[..., 16:240, 16:240]))            # crop giữa
        self.assertEqual(len(inference.views_multicrop(x, 224, flips=True)), 10)
        self.assertEqual([v.shape[-1] for v in inference.views_multiscale(x, [224, 256, 288])], [224, 256, 288])
        from PIL import Image
        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
        full = build_transforms(False, None)(img)[None]
        ref = build_transforms(False, 224)(img)[None]
        self.assertTrue(torch.equal(inference.build_views("center")[0](full), ref))  # I00 == CenterCrop(224) lúc val
        self.assertEqual([len(inference.build_views(k)) for k in ("center", "hflip", "5crop", "10crop", "full288", "full256+hflip")],
                         [1, 2, 5, 10, 1, 2])

    def test_aggregate_and_ensemble(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(20, 9)), rng.normal(size=(20, 9))
        p = inference.aggregate_views([a, b], "prob")
        q = inference.aggregate_views([a, b], "logit")
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(q.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(p, (inference.softmax(a) + inference.softmax(b)) / 2)
        np.testing.assert_allclose(q, inference.softmax((a + b) / 2))
        np.testing.assert_allclose(inference.aggregate_views([a], "prob"), inference.softmax(a))
        np.testing.assert_allclose(inference.ensemble_probs([p, q]), (p + q) / 2)

    def test_temperature_scaling(self):
        rng = np.random.default_rng(0)
        n, true_t = 4000, 2.5
        clean = rng.normal(size=(n, 9)) * 2
        y = np.array([rng.choice(9, p=inference.softmax(clean[i:i + 1])[0]) for i in range(n)])
        over_confident = clean * true_t                                           # logit bị phóng đại 2,5 lần
        T = inference.fit_temperature(over_confident, y)
        self.assertAlmostEqual(T, true_t, delta=0.25)
        self.assertLess(inference.nll(over_confident, y, T), inference.nll(over_confident, y, 1.0))
        p = inference.apply_temperature(over_confident, T)
        np.testing.assert_array_equal(p.argmax(1), over_confident.argmax(1))      # accuracy không đổi
        np.testing.assert_allclose(inference.apply_temperature(inference.probs_to_logits(inference.softmax(clean)), 1.0),
                                   inference.softmax(clean), atol=1e-9)


class TestBenchmark(unittest.TestCase):
    def test_bench_and_reports(self):
        calls = {"fn": 0, "sync": 0}

        def fn():
            calls["fn"] += 1

        def sync():
            calls["sync"] += 1

        r = benchmark.bench(fn, warmup=10, iters=50, sync=sync)
        self.assertEqual(calls["fn"], 60)                 # 10 lần warmup bị bỏ + 50 lần đo
        self.assertEqual(calls["sync"], 101)              # đồng bộ trước và sau MỖI lần đo
        self.assertEqual(r["n"], 50)
        self.assertTrue(r["p50"] <= r["p95"] <= r["p99"])
        rep = benchmark.latency_report(TinyNet(), 2, 32, "fp32", "cpu", warmup=2, iters=5)
        self.assertEqual((rep["batch"], rep["img_size"], rep["dtype"], rep["n"]), (2, 32, "fp32", 5))
        self.assertAlmostEqual(rep["images_per_s"], 2 / (rep["p50"] / 1000), places=6)
        tta = benchmark.tta_latency(TinyNet(), 4, img_size=32, device="cpu", warmup=2, iters=5)
        self.assertEqual(tta["k_views"], 4)
        self.assertGreater(tta["p50"], 0)


if __name__ == "__main__":
    print("-ln(1/9) =", round(-math.log(1 / 9), 4))
    unittest.main(verbosity=2)

# Tham khảo implementation: picuisme/K4-Track4-Day2-Deeplearning-Advance @ 941d9fb.
# Bản cho Nguyen Tuan Thanh: sửa optimizer/resume/report/Colab; số liệu phải chạy riêng.
