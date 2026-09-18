"""Independent finite differences and invariants for the new teaching examples."""
import unittest
import numpy as np
from posttraining_pretraining_examples import (
    reverse_kl_and_grad, enumerated_opd_grad, gspo_ratio, sapo_gate,
    sapo_gate_derivative, gdpo_advantages, causal_training_example,
    token_weighted_mean, training_budget)


class PosttrainingPretrainingTests(unittest.TestCase):
    def test_opd_gradient_matches_kl_finite_difference(self):
        for z, teacher in [([0.3, -0.2, 0.8], [0.7, 0.4, -0.1]),
                           ([4, -3, 1], [-2, 2, 0.5])]:
            z = np.array(z, dtype=float)
            _, direct = reverse_kl_and_grad(z, teacher)
            numerical = np.zeros_like(z)
            for i in range(len(z)):
                step = np.zeros_like(z); step[i] = 1e-5
                numerical[i] = (reverse_kl_and_grad(z+step, teacher)[0]
                                - reverse_kl_and_grad(z-step, teacher)[0]) / 2e-5
            np.testing.assert_allclose(direct, numerical, atol=1e-9)
            np.testing.assert_allclose(enumerated_opd_grad(z, teacher), numerical, atol=1e-9)
            self.assertAlmostEqual(float(direct.sum()), 0)

    def test_matching_teacher_has_zero_loss_and_gradient(self):
        z = [1, -2, 0.5]
        loss, grad = reverse_kl_and_grad(z, z)
        self.assertEqual(loss, 0)
        np.testing.assert_array_equal(grad, np.zeros(3))
        np.testing.assert_array_equal(enumerated_opd_grad(z, z), np.zeros(3))

    def test_gspo_is_geometric_not_arithmetic(self):
        self.assertAlmostEqual(gspo_ratio(np.log([2, 0.5])), 1)
        self.assertNotEqual(gspo_ratio(np.log([2, 0.5])), np.mean([2, 0.5]))
        # Accumulating the log ratios avoids overflowing the product.
        self.assertAlmostEqual(gspo_ratio([800, -800]), 1)

    def test_sapo_unit_slope_and_soft_decay(self):
        for tau in [0.5, 1, 2, 4]:
            self.assertAlmostEqual(float(sapo_gate_derivative(1, tau)), 1)
            for r in [0.2, 1, 3]:
                fd = (sapo_gate(r+1e-5, tau)-sapo_gate(r-1e-5, tau)) / 2e-5
                self.assertAlmostEqual(float(fd), float(sapo_gate_derivative(r, tau)), places=8)
            self.assertLess(float(sapo_gate_derivative(4, tau)), 1)

    def test_gdpo_reward_scale_invariance_and_constant_group(self):
        r = np.array([[[0, 0], [50, 0], [100, 1]], [[25, 1], [75, 0], [50, 1]]], dtype=float)
        scaled = r.copy(); scaled[:, :, 0] = scaled[:, :, 0]*7+13
        a = gdpo_advantages(r)
        np.testing.assert_allclose(a, gdpo_advantages(scaled), atol=1e-10)
        self.assertAlmostEqual(float(a.mean()), 0)
        self.assertAlmostEqual(float(a.std()), 1)
        np.testing.assert_array_equal(gdpo_advantages(np.ones((2, 3, 2))), np.zeros((2, 3)))

    def test_ntp_shift_and_no_future_target_visibility(self):
        tokens = ['我', '喜欢', '机器', '学习']
        inputs, targets, allowed = causal_training_example(tokens)
        self.assertEqual(inputs, tokens[:-1]); self.assertEqual(targets, tokens[1:])
        for i in range(len(inputs)):
            self.assertTrue(allowed[i, i])
            self.assertFalse(allowed[i, i+1:].any())
            visible = np.array(inputs)[allowed[i]].tolist()
            self.assertEqual(visible, tokens[:i+1])

    def test_token_weighting_and_budget_units(self):
        self.assertEqual(token_weighted_mean([1, 3], [2, 6]), 2.5)
        b = training_budget(8, 2, 4, 1024, 1000, 10**9)
        self.assertEqual(b['tokens_per_update'], 65536)
        self.assertEqual(b['total_tokens'], 65536000)
        self.assertEqual(b['approximate_dense_flops'], 393216000000000000)
        with self.assertRaises(ValueError):
            token_weighted_mean([1, 3], [0, 0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
