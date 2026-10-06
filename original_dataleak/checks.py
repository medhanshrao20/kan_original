"""Focused method checks; no metric/checkpoint artifacts are written."""
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from decomposition import decompose
from model import KANInformer, ProbAttention
from run import Config, clean_data, feature_matrix, select_features, split_masks


class MethodChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = Config(d_model=8, n_heads=2, kan_hidden=4, ewt_modes=3)
        rng = np.random.default_rng(1)
        self.x = 2 + np.sin(np.arange(256) * 0.3) + rng.normal(0, 0.1, 256)

    def test_odd_length_and_ewt_synthesis(self):
        parts, report = decompose(self.x[:255], 3, self.config, fixed_high=[1, 2])
        self.assertEqual(parts.shape, (255, 4))
        self.assertTrue(np.isfinite(parts).all())
        self.assertLess(report["ewt_synthesis_max_error"], 1e-8)

    def test_full_size_paper_and_author_forward_backward(self):
        for architecture in ["paper", "author"]:
            config = Config(architecture=architecture)
            model = KANInformer(17, config, torch.device("cpu"))
            x = torch.rand(2, 7, 17)
            output = model(x)
            self.assertEqual(output.shape, (2, 17) if architecture == "author" else (2, 3))
            output.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            with torch.no_grad():
                self.assertEqual(model.forecast(x[:1]).shape, (1, 3))

    def test_probattention_matches_published_sampling_equations(self):
        q, k, v = [torch.rand(2, 2, 32, 4) for _ in range(3)]
        attention = ProbAttention(False, factor=3)
        torch.manual_seed(17)
        actual = attention(q, k, v)
        torch.manual_seed(17)
        sample_k = min(32, 3 * int(np.ceil(np.log(32))))
        index_sample = torch.randint(32, (32, sample_k))
        sampled = k[:, :, index_sample, :]
        scores_sample = torch.matmul(q.unsqueeze(-2), sampled.transpose(-2, -1)).squeeze(-2)
        measure = scores_sample.max(-1).values - scores_sample.sum(-1) / 32
        indices = measure.topk(sample_k, sorted=False).indices
        batch, heads = torch.arange(2)[:, None, None], torch.arange(2)[None, :, None]
        reduced = q[batch, heads, indices]
        scores = reduced @ k.transpose(-2, -1) / 2
        expected = v.mean(-2, keepdim=True).expand(-1, -1, 32, -1).clone()
        expected[batch, heads, indices] = scores.softmax(-1) @ v
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_future_suffix_cannot_change_forecast_inputs(self):
        from run import VARIABLES
        rng = np.random.default_rng(18)
        frame = pd.DataFrame(rng.normal(size=(256, len(VARIABLES))), columns=VARIABLES)
        frame["WS"] = self.x
        altered = frame.copy()
        altered.iloc[220:] = 1000
        clean, _ = clean_data(frame, 180, True)
        other, _ = clean_data(altered, 180, True)
        pd.testing.assert_frame_equal(clean.iloc[:220], other.iloc[:220])
        self.assertEqual(select_features(clean.iloc[:180])[0], select_features(other.iloc[:180])[0])
        a, report = decompose(clean["WS"].iloc[:180].to_numpy(), 3, self.config)
        high = report["high_indices"]
        for data in [clean, other]:
            parts, _ = decompose(data["WS"].iloc[:220].to_numpy(), 3, self.config, fixed_high=high)
            inputs = feature_matrix(data.iloc[:220], ["WS"], parts)[-7:]
            if data is clean:
                reference = inputs
            else:
                np.testing.assert_array_equal(inputs, reference)

    def test_targets_do_not_cross_split_boundaries(self):
        origins = np.arange(7, 98)
        train, val, test = split_masks(origins, self.config, 80, 90)
        self.assertTrue(np.all(origins[train] + 2 < 80))
        self.assertTrue(np.all(origins[val] >= 80))
        self.assertTrue(np.all(origins[val] + 2 < 90))
        self.assertTrue(np.all(origins[test] >= 90))


if __name__ == "__main__":
    unittest.main(verbosity=2)
