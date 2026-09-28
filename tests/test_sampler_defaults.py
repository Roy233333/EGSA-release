"""Keep CLI and direct sampler defaults consistent with the paper settings."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'nerfstudio'
EXPECTED = dict(edge_type='sobel', edge_weight=1.3, uniform_mix=0.6,
                ema_decay=0.9, hardness_enabled=True, block_size=8,
                blockwise_exact=False)


class SamplerDefaultsTests(unittest.TestCase):
    def test_cli_defaults(self):
        tree = ast.parse((ROOT / 'data/datamanagers/base_datamanager.py').read_text(encoding='utf-8'))
        config = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == 'VanillaDataManagerConfig')
        values = {n.target.id: ast.literal_eval(n.value) for n in config.body
                  if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                  and n.target.id.startswith('adaptive_')}
        self.assertEqual(values, {'adaptive_' + key: value for key, value in EXPECTED.items()})

    def test_direct_sampler_defaults(self):
        tree = ast.parse((ROOT / 'data/pixel_samplers.py').read_text(encoding='utf-8'))
        sampler = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                       and n.name == 'AdaptivePixelSampler')
        init = next(n for n in sampler.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        values = {a.arg: ast.literal_eval(d) for a, d in zip(init.args.kwonlyargs, init.args.kw_defaults)
                  if a.arg in EXPECTED}
        self.assertEqual(values, EXPECTED)


if __name__ == '__main__':
    unittest.main()
