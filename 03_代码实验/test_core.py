"""Independent numerical/property checks for the teaching implementations."""
import itertools
import math
import unittest
import numpy as np
from core_algorithms import *


class CoreTests(unittest.TestCase):
    def test_softmax_shift_and_extremes(self):
        x = np.array([[1000., 1001., -1000.]])
        p = np.exp(log_softmax(x))
        np.testing.assert_allclose(p.sum(-1), [1.0])
        np.testing.assert_allclose(log_softmax(x), log_softmax(x - 4000))

    def test_mask_and_empty_rows(self):
        p = masked_softmax([2., 100., 0.], [True, False, True])
        self.assertEqual(p[1], 0.)
        self.assertAlmostEqual(p.sum(), 1.)
        with self.assertRaises(ValueError):
            masked_softmax([1., 2.], [False, False])

    def test_ce_gradient_finite_difference(self):
        x = np.array([0.2, -0.4, 0.7])
        target = 1
        analytic = np.exp(log_softmax(x))
        analytic[target] -= 1
        numeric = np.zeros_like(x)
        h = 1e-6
        for i in range(len(x)):
            plus, minus = x.copy(), x.copy()
            plus[i] += h
            minus[i] -= h
            numeric[i] = (-log_softmax(plus)[target] + log_softmax(minus)[target]) / (2*h)
        np.testing.assert_allclose(analytic, numeric, atol=1e-8)

    def test_sequence_mask_and_mean(self):
        logits = np.zeros((1, 3, 5))
        sums, means, lengths = sequence_logprob(logits, np.array([[-100, 2, 3]]), [[False, True, True]])
        self.assertAlmostEqual(sums[0], -2 * math.log(5))
        self.assertAlmostEqual(means[0], -math.log(5))
        self.assertEqual(lengths[0], 2)
        with self.assertRaises(ValueError):
            sequence_logprob(logits, np.array([[0, 0, 0]]), [[False, False, False]])

    def test_attention_no_future_leakage(self):
        rng = np.random.default_rng(3)
        q = rng.normal(size=(2, 3, 4))
        k = rng.normal(size=q.shape)
        v = rng.normal(size=q.shape)
        out, weights = causal_attention(q, k, v)
        changed_v = v.copy()
        changed_v[:, 2] += 1000
        changed_out, _ = causal_attention(q, k, changed_v)
        np.testing.assert_allclose(out[:, :2], changed_out[:, :2])
        self.assertTrue(np.all(weights[:, 0, 1:] == 0))
        np.testing.assert_allclose(weights.sum(-1), 1.)

    def test_lora_merge_equivalence(self):
        rng = np.random.default_rng(1)
        x, w, a, b = (rng.normal(size=s) for s in [(4, 5), (7, 5), (2, 5), (7, 2)])
        np.testing.assert_allclose(lora_forward(x, w, a, b, 4), x @ (w + 2*b@a).T, atol=1e-12)
        np.testing.assert_allclose(lora_forward(x, w, a, b*0, 4), x@w.T)

    def test_dpo_initial_and_gradient(self):
        self.assertAlmostEqual(float(dpo_loss(-2., -3., -2., -3.)), math.log(2))
        c, r, rc, rr, beta = -2., -3., -2.5, -3.1, 0.3
        h = 1e-6
        numeric = (dpo_loss(c+h, r, rc, rr, beta)-dpo_loss(c-h, r, rc, rr, beta))/(2*h)
        z = beta*((c-r)-(rc-rr))
        self.assertAlmostEqual(float(numeric), -beta/(1+math.exp(z)), places=8)

    def test_simpo_example_and_length_invariance(self):
        value = float(simpo_loss(-100, -30, 100, 20))
        self.assertAlmostEqual(value, 0.4740769841801067)
        self.assertAlmostEqual(value, float(simpo_loss(-200, -60, 200, 40)))

    def test_grpo_groups(self):
        adv = grpo_advantages([[0, 0, 1, 1], [1, 1, 1, 1]])
        np.testing.assert_allclose(adv[0], [-1, -1, 1, 1], atol=1e-7)
        np.testing.assert_allclose(adv[1], [0, 0, 0, 0])
        np.testing.assert_allclose(adv.mean(axis=1), 0.)

    def test_kto_reference_and_gradient_directions(self):
        np.testing.assert_allclose(kto_loss([0, 0], [True, False]), [.5, .5])
        h = 1e-5
        good_gradient = (kto_loss(h, True)-kto_loss(-h, True))/(2*h)
        bad_gradient = (kto_loss(h, False)-kto_loss(-h, False))/(2*h)
        self.assertAlmostEqual(float(good_gradient), -.025, places=8)
        self.assertAlmostEqual(float(bad_gradient), .025, places=8)
        self.assertAlmostEqual(float(kto_loss(1000, True)), 0.)
        self.assertAlmostEqual(float(kto_loss(-1000, False)), 0.)

    def test_clip_signs(self):
        np.testing.assert_allclose(ppo_surrogate([1.5, 0.5, 1.5, 0.5], [1, 1, -1, -1]),
                                   [1.2, 0.5, -1.5, -0.8])

    def test_loss_normalization(self):
        # Short response has per-token loss 1, long response has per-token loss 3.
        values = np.array([[1, 1, 0, 0], [3, 3, 3, 3]])
        mask = [[1, 1, 0, 0], [1, 1, 1, 1]]
        self.assertAlmostEqual(aggregate_token_losses(values, mask, "grpo"), 2.)
        self.assertAlmostEqual(aggregate_token_losses(values, mask, "dapo"), 14/6)
        self.assertAlmostEqual(aggregate_token_losses(values, mask, "dr_grpo", 4), 14/8)

    def test_pass_k_against_enumeration(self):
        for n in range(1, 9):
            for c in range(n+1):
                for k in range(1, n+1):
                    groups = list(itertools.combinations(range(n), k))
                    exact = sum(any(i < c for i in group) for group in groups)/len(groups)
                    self.assertAlmostEqual(pass_at_k(n, c, k), exact)
        with self.assertRaises(ValueError):
            pass_at_k(2, 1, 3)

    def test_metrics(self):
        m = classification_metrics(80, 20, 40)
        self.assertAlmostEqual(m["precision"], 0.8)
        self.assertAlmostEqual(m["recall"], 2/3)
        self.assertAlmostEqual(m["f1"], 8/11)

    def test_paired_bootstrap(self):
        scores = [1, 0, 1, 1, 0]
        self.assertEqual(paired_bootstrap(scores, scores)["ci95"], [0., 0.])
        shifted = paired_bootstrap([0]*6, [1]*6)
        self.assertEqual(shifted["delta"], 1.)
        self.assertEqual(shifted["ci95"], [1., 1.])

    def test_rrf_rank_and_duplicates(self):
        result = dict(reciprocal_rank_fusion([["A", "B", "C"], ["C", "B", "A"]]))
        self.assertAlmostEqual(result["A"], 1/61+1/63)
        self.assertEqual(reciprocal_rank_fusion([["A", "A"]]), [("A", 1/61)])

    def test_kv_units(self):
        self.assertEqual(kv_cache_bytes(32, 1, 8192, 8, 128, 2), 2**30)

    def test_timeout_after_write_no_duplicate(self):
        service = MockRefundService()
        self.assertEqual(recover_refund(service, "order-demo", "request-1"), "refunded")
        self.assertEqual(recover_refund(service, "order-demo", "request-1"), "refunded")
        self.assertEqual(service.writes, 1)
        with self.assertRaises(ValueError):
            service.refund("another-order", "request-1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
