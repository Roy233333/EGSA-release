"""CPU regressions for incremental caches against the currently stored q map."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nerfstudio.data.pixel_samplers import AdaptivePixelSampler
from nerfstudio.pipelines.adaptive_feedback import submit_adaptive_feedback


def cell_reference(q, size):
    height, width = q.shape
    return torch.stack([
        torch.stack([
            q[y:y + size, x:x + size].double().sum()
            for x in range(0, width, size)
        ])
        for y in range(0, height, size)
    ])


class SamplerMassCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_sampler(self, size=4, decay=0.9, shape=(9, 11), masked=False):
        generator = torch.Generator().manual_seed(726)
        image = torch.rand((*shape, 3), generator=generator)
        edge = torch.exp(-12 * torch.rand(shape, generator=generator))
        edge[::3] = 0.8
        with redirect_stdout(io.StringIO()):
            sampler = AdaptivePixelSampler(64, device='cpu', block_size=size, ema_decay=decay)
        mask = torch.ones(shape)
        if masked:
            mask[:, :4] = 0
        with patch.object(sampler, '_compute_edge_map_cpu', return_value=edge):
            sampler._ensure_state(7, image, mask)
        return sampler, image

    def assert_consistent(self, sampler, gid=7):
        q = sampler.prob_maps[gid]
        self.assertEqual(q.dtype, torch.float32)
        self.assertEqual(sampler.loss_weights[gid].dtype, torch.float32)
        z = float(q.double().sum())
        self.assertAlmostEqual(sampler.prob_sums[gid], max(z, 1e-12), delta=max(z * 1e-7, 2e-13))
        if sampler.block_size > 1:
            cells = sampler.block_sums[gid]
            self.assertEqual(cells.dtype, torch.float64)
            torch.testing.assert_close(cells, cell_reference(q, sampler.block_size), rtol=1e-7, atol=2e-13)
            self.assertGreaterEqual(float(cells.min()), 0.0)

    def test_initialisation_uses_float64_reductions_and_is_lazy(self):
        sampler, image = self.make_sampler(masked=True)
        self.assert_consistent(sampler)
        before_q = sampler.prob_maps[7]
        with patch.object(sampler, '_rebuild_block_sums', side_effect=AssertionError('unexpected rebuild')):
            sampler._ensure_state(7, image, None)
        self.assertIs(sampler.prob_maps[7], before_q)

    def test_duplicate_observations_partial_cells_and_decay_endpoints(self):
        for decay in (0.0, 0.9, 1.0):
            for size in (1, 4, 8):
                with self.subTest(decay=decay, size=size):
                    sampler, _ = self.make_sampler(size=size, decay=decay, masked=True)
                    index = torch.tensor([[7, 8, 10], [7, 0, 0], [7, 8, 10], [7, 4, 6], [7, 8, 9]])
                    values = torch.tensor([1e-8, 0.1, 2e-8, 3e-6, 0.2])
                    expected = sampler.loss_weights[7].clone()
                    for (_, y, x), value in zip(index.tolist(), values):
                        expected[y, x] = decay * expected[y, x] + (1 - decay) * value
                    with redirect_stdout(io.StringIO()), patch.object(
                        sampler, '_rebuild_block_sums', side_effect=AssertionError('unexpected rebuild')
                    ):
                        sampler.update(index, values)
                    torch.testing.assert_close(sampler.loss_weights[7], expected)
                    self.assert_consistent(sampler)

    def test_low_error_dense_trace_does_not_corrupt_mass_caches(self):
        sampler, _ = self.make_sampler(shape=(16, 17), size=4)
        linear = torch.arange(16 * 17)
        indices = torch.stack([torch.full_like(linear, 7), linear // 17, linear % 17], -1)
        indices = indices.repeat_interleave(2, dim=0)
        generator = torch.Generator().manual_seed(451)
        values = (0.5 + torch.rand(len(indices), generator=generator)) * 1e-8
        with redirect_stdout(io.StringIO()), patch.object(
            sampler, '_rebuild_block_sums', side_effect=AssertionError('unexpected rebuild')
        ):
            for _ in range(1000):
                sampler.update(indices, values)
        self.assert_consistent(sampler)

    def test_all_zero_prior_stays_zero_with_image_sampling_guard(self):
        sampler, image = self.make_sampler()
        sampler._ensure_state(8, image, torch.zeros(image.shape[:2]))
        with redirect_stdout(io.StringIO()):
            sampler.update(torch.tensor([[8, 2, 3], [8, 2, 3]]), torch.tensor([0.1, 0.2]))
        self.assert_consistent(sampler, 8)
        self.assertEqual(sampler.prob_sums[8], 1e-12)

    def test_image_allocation_uses_updated_float64_masses(self):
        sampler, image = self.make_sampler()
        with patch.object(sampler, '_compute_edge_map_cpu', return_value=torch.full(image.shape[:2], 0.05)):
            sampler._ensure_state(23, image, None)
        with redirect_stdout(io.StringIO()):
            sampler.update(
                torch.tensor([[7, 8, 10], [23, 0, 0], [23, 8, 10], [7, 8, 10]]),
                torch.tensor([1e-8, 0.2, 0.03, 2e-8]),
            )
        self.assert_consistent(sampler, 7)
        self.assert_consistent(sampler, 23)

        # Independent full-map sums check both the cache and its use in allocation.
        masses = torch.stack([sampler.prob_maps[gid].double().sum() for gid in (7, 23)])
        self.assertNotEqual(float(masses[0]), float(masses[1]))
        expected = (1 - sampler.uniform_mix) * masses / masses.sum() + sampler.uniform_mix / 2
        with torch.random.fork_rng(devices=[]), patch('torch.multinomial', wraps=torch.multinomial) as draw:
            torch.manual_seed(937)
            sampled = sampler.sample({
                'image': torch.stack([image, image]),
                'image_idx': torch.tensor([7, 23]),
            })
        image_probabilities = draw.call_args_list[0][0][0]
        self.assertEqual(image_probabilities.dtype, torch.float64)
        torch.testing.assert_close(image_probabilities, expected, rtol=1e-12, atol=1e-15)
        self.assertEqual(sampled['indices'].shape, (sampler.num_rays_per_batch, 3))
        self.assertTrue(set(sampled['indices'][:, 0].tolist()).issubset({7, 23}))

    def test_sample_and_detached_feedback_use_new_cache_path(self):
        sampler, image = self.make_sampler()
        sampled = sampler.sample({'image': image.unsqueeze(0), 'image_idx': torch.tensor([7])})

        class Manager:
            train_pixel_sampler = sampler

            def update_from_per_ray_loss(self, batch, losses, key):
                self.received = losses[key]
                self.train_pixel_sampler.update(batch['indices'], losses[key])

        manager = Manager()
        rgb = (sampled['image'] + 0.001).requires_grad_()
        with redirect_stdout(io.StringIO()):
            submitted = submit_adaptive_feedback(manager, sampled, {'rgb': rgb})
        self.assertTrue(submitted)
        self.assertFalse(manager.received.requires_grad)
        self.assertEqual(sampler.feedback_update_calls, 1)
        self.assert_consistent(sampler)


if __name__ == '__main__':
    unittest.main()
