"""Regression checks for resume/provenance. Synthetic data lives only in a TemporaryDirectory."""
from pathlib import Path
import dataclasses
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

import train
import model as model_lib
import experiments


class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, padding=1)
        self.bn = nn.BatchNorm2d(6)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(6, 9)
        self.weights_tag = 'synthetic-only'
        self.frozen_backbone = False

    def get_classifier(self):
        return self.fc

    def forward(self, x):
        return self.fc(self.dropout(self.pool(self.bn(self.conv(x))).flatten(1)))


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='deepweeds_test_')
        self.root = Path(self.tmp.name)
        self.images = self.root / 'images'
        self.labels = self.root / 'labels'
        self.images.mkdir()
        self.labels.mkdir()
        for split in ('train','val','test'):
            rows = []
            for label in range(9):
                name = f'{split}_{label}.jpg'
                pixel = np.random.default_rng(label).integers(0,256,(32,32,3),dtype=np.uint8)
                Image.fromarray(pixel).save(self.images / name)
                rows.append({'Filename':name,'Label':label})
            pd.DataFrame(rows).to_csv(self.labels / f'{split}_subset0.csv',index=False)
        self.patches = [patch.object(model_lib,'build_model',side_effect=lambda *a,**k: MiniNet()),
                        patch.object(model_lib,'count_gmacs',return_value=0.00001),
                        patch.object(train,'plot_curves',side_effect=lambda history,path,*a,**k: Path(path).write_bytes(b'CPU synthetic test'))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def cfg(self, folder='run'):
        out = self.root / folder
        (out / 'curves').mkdir(parents=True,exist_ok=True)
        return train.Config(backbone='synthetic',exp_id='T00',epochs=2,batch_size=9,img_size=16,
                            num_workers=0,amp=False,images_dir=str(self.images),labels_dir=str(self.labels),
                            out_dir=str(out / 'logs'),pred_dir=str(out / 'predictions'),
                            ckpt_dir=str(out / 'checkpoints'),curves_dir=str(out / 'curves'),expected_total=None)

    def test_every_bias_has_zero_decay(self):
        m = MiniNet()
        names = {id(p):n for n,p in m.named_parameters()}
        groups = model_lib.param_groups(m,1e-4,1e-3,0.05)
        for g in groups:
            for p in g['params']:
                if names[id(p)].endswith('bias') or p.ndim <= 1:
                    self.assertEqual(g['weight_decay'],0.0)
        ids = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))

    def test_resume_matches_uninterrupted_training(self):
        uninterrupted = self.cfg('complete')
        train.run(uninterrupted)
        resumed = self.cfg('interrupted')
        real_save = train.atomic_torch_save
        def interrupt_after_first_epoch(obj,path):
            real_save(obj,path)
            if str(path).endswith('_last.pt') and obj['epoch']==1:
                raise RuntimeError('simulated disconnect')
        with patch.object(train,'atomic_torch_save',side_effect=interrupt_after_first_epoch):
            with self.assertRaisesRegex(RuntimeError,'simulated disconnect'):
                train.run(resumed)
        train.run(resumed)
        left = torch.load(train.ckpt_path(uninterrupted),weights_only=False)['model']
        right = torch.load(train.ckpt_path(resumed),weights_only=False)['model']
        self.assertEqual(set(left),set(right))
        for key in left:
            torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
        self.assertEqual(len(pd.read_csv(train.run_dir(resumed)/'history.csv')),2)

    def test_stale_summary_without_checkpoint_is_rejected(self):
        c = self.cfg()
        train.run(c)
        train.ckpt_path(c).unlink()
        with self.assertRaisesRegex(RuntimeError,'thiếu checkpoint'):
            train.run(c)

    def test_changed_configuration_is_rejected(self):
        c = self.cfg()
        train.run(c)
        with self.assertRaisesRegex(RuntimeError,'cấu hình đã thay đổi'):
            train.run(dataclasses.replace(c,loss='ls',label_smoothing=0.1))

    def test_training_signature_preserves_seed_and_recipe(self):
        c = self.cfg()
        self.assertEqual(train.training_signature(c),train.training_signature(dataclasses.replace(c,exp_id='F01',out_dir='elsewhere')))
        self.assertNotEqual(train.training_signature(c),train.training_signature(dataclasses.replace(c,seed=1)))
        self.assertNotEqual(train.training_signature(c),train.training_signature(dataclasses.replace(c,mix='cutmix')))

    def test_test_marker_blocks_overwrite(self):
        c = self.cfg()
        train.run(c)
        (train.run_dir(c)/'TEST_DONE.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'Không ghi đè'):
            train.run(dataclasses.replace(c,overwrite=True))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
