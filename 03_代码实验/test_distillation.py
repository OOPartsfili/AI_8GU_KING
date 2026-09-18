"""Numerical and masking checks for the distillation teaching examples."""
import unittest
import numpy as np
from distillation_examples import log_softmax, kl_from_logs, soft_kd, masked_nll, finite_difference


class DistillationTests(unittest.TestCase):
    def test_temperature_and_shift_stability(self):
        z = np.array([3., 1, -1])
        np.testing.assert_allclose(log_softmax(z), log_softmax(z + 10000), atol=1e-12)
        warm = np.exp(log_softmax(z, 4))
        self.assertLess(warm.max(), np.exp(log_softmax(z)).max())
        self.assertAlmostEqual(warm.sum(), 1.)

    def test_known_kl_values_and_identity(self):
        p, q = np.log([0.8, 0.2]), np.log([0.6, 0.4])
        self.assertAlmostEqual(float(kl_from_logs(p, q)), 0.09151622184943578)
        self.assertAlmostEqual(float(kl_from_logs(q, p)), 0.10464962875290948)
        self.assertAlmostEqual(float(kl_from_logs(p, p)), 0.)

    def test_kd_gradient_multiple_temperatures(self):
        z, teacher = np.array([0.2, -0.7, 1.3]), np.array([2., 0.1, -1.])
        original = teacher.copy()
        for temperature in [0.5, 1., 2., 8.]:
            _, analytic = soft_kd(z, teacher, temperature)
            numerical = finite_difference(lambda v: soft_kd(v, teacher, temperature)[0], z)
            np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=1e-7)
            self.assertAlmostEqual(float(analytic.sum()), 0.)
        np.testing.assert_array_equal(teacher, original)

    def test_mask_excludes_direct_targets_but_not_assistant(self):
        z = np.array([[3., 0, 0], [0, 2., 0], [0, 0, 1.]])
        target, mask = [0, -100, 2], [1, 0, 1]
        expected = (np.log(1+2*np.exp(-3)) + np.log(1+2*np.exp(-1))) / 2
        self.assertAlmostEqual(masked_nll(z, target, mask), expected)
        changed = z.copy()
        changed[1] = [100, -50, 0]
        self.assertAlmostEqual(masked_nll(changed, target, mask), expected)
        changed[2] = [10, 0, 0]
        self.assertGreater(masked_nll(changed, target, mask), expected)

    def test_invalid_temperature_or_no_supervision_rejected(self):
        for temperature in [0, -1, float("nan")]:
            with self.assertRaises(ValueError):
                log_softmax([0, 1], temperature)
        with self.assertRaises(ValueError):
            masked_nll([[0, 1]], [-100], [0])
        with self.assertRaises(ValueError):
            masked_nll([[0, 1]], [5], [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
