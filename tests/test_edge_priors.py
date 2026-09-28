"""Run directly with Python after installing SDFStudio and EGSA requirements."""
from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

MODULE_PATH = Path(__file__).parents[1] / 'nerfstudio/data/edge_priors.py'
SPEC = importlib.util.spec_from_file_location('egsa_edge_priors', MODULE_PATH)
EDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EDGE)


def prepare(img):
    img = img.float()
    if img.max() > 1:
        img = img / 255.
    return (0.2989*img[..., 0]+0.5870*img[..., 1]+0.1140*img[..., 2])[None, None]


def canny_reference(img):
    # Independent reference for the Canny-magnitude computation.
    import kornia.filters as KF
    mag, _ = KF.canny(prepare(img), low_threshold=0.1, high_threshold=0.2)
    mag = mag[0, 0]
    mag = mag - mag.min()
    return (mag / mag.max().clamp(min=1e-12)).clamp_min(1e-6)


def previous_sobel_reference(img):
    gray = prepare(img)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
    gx, gy = F.conv2d(gray,kx,padding=1), F.conv2d(gray,ky,padding=1)
    mag = torch.sqrt(gx*gx+gy*gy)[0,0]
    mag = mag-mag.min()
    return (mag/max(mag.max().item(),1e-12)).clamp_min(1e-6)


class EdgePriorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.inputs = [torch.rand(29,37,3), torch.randint(0,256,(29,37,3),dtype=torch.uint8),
                       torch.zeros(29,37,3), torch.ones(29,37,3)]

    def test_sobel_bitwise_unchanged(self):
        for img in self.inputs:
            self.assertTrue(torch.equal(EDGE.compute_edge_map_cpu(img), previous_sobel_reference(img)))

    def test_canny_bitwise_matches_reference(self):
        for img in self.inputs:
            out = EDGE.compute_edge_map_cpu(img,'canny')
            self.assertTrue(torch.equal(out,canny_reference(img)))
            self.assertTrue(torch.isfinite(out).all())
            self.assertGreaterEqual(float(out.min()),0.)
            self.assertLessEqual(float(out.max()),1.)
            self.assertEqual(out.shape,img.shape[:2])
            self.assertEqual(out.device.type,'cpu')

    def test_canny_not_sobel(self):
        img=self.inputs[0]
        self.assertFalse(torch.equal(EDGE.compute_edge_map_cpu(img,'canny'),EDGE.compute_edge_map_cpu(img,'sobel')))

    def test_missing_kornia_is_explicit_error(self):
        original_import=builtins.__import__
        def blocked(name,*args,**kwargs):
            if name.startswith('kornia'):
                raise ModuleNotFoundError('Kornia intentionally unavailable')
            return original_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=blocked):
            with self.assertRaisesRegex(ImportError,'requires Kornia'):
                EDGE.compute_edge_map_cpu(self.inputs[0],'canny')
            self.assertTrue(torch.equal(EDGE.compute_edge_map_cpu(self.inputs[0]),previous_sobel_reference(self.inputs[0])))

    def test_canny_failure_is_not_hidden(self):
        with patch('kornia.filters.canny',side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError,'test failure'):
                EDGE.compute_edge_map_cpu(self.inputs[0],'canny')

    def test_unknown_detector_rejected(self):
        with self.assertRaises(ValueError):
            EDGE.compute_edge_map_cpu(self.inputs[0],'typo')


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
