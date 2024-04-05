import unittest

from code_cov import metrics


class MetricsTest(unittest.TestCase):
    def test_calculate_pass_at_k__too_many_correct__calculates_correctly(self):
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(200, 101, 100))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(200, 150, 100))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(200, 200, 100))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(200, 191, 10))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(200, 195, 10))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(200, 200, 10))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(200, 200, 1))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(20, 11, 10))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(20, 15, 10))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(20, 20, 10))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(20, 20, 1))
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(2, 2, 1))

    def test_calculate_pass_at_k__few_correct__calculates_correctly(self):
        self.assertAlmostEqual(0.0, metrics.calculate_pass_at_k(200, 0, 100))
        self.assertAlmostEqual(0.5, metrics.calculate_pass_at_k(200, 1, 100))
        self.assertAlmostEqual(
            0.9999996678, metrics.calculate_pass_at_k(200, 20, 100))
        # The following is actually less than 1.0, but by a very small amount.
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(200, 50, 100))
        # The following is actually less than 1.0, but by a very small amount.
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(200, 100, 100))
        self.assertAlmostEqual(0.0, metrics.calculate_pass_at_k(200, 0, 10))
        self.assertAlmostEqual(0.05, metrics.calculate_pass_at_k(200, 1, 10))
        self.assertAlmostEqual(
            0.6602256238, metrics.calculate_pass_at_k(200, 20, 10))
        self.assertAlmostEqual(
            0.9987150482, metrics.calculate_pass_at_k(200, 95, 10))
        # The following is actually less than 1.0, but by a very small amount.
        self.assertAlmostEqual(1.0, metrics.calculate_pass_at_k(200, 190, 10))
        self.assertAlmostEqual(0.0, metrics.calculate_pass_at_k(200, 0, 1))
        self.assertAlmostEqual(0.005, metrics.calculate_pass_at_k(200, 1, 1))
        self.assertAlmostEqual(0.1, metrics.calculate_pass_at_k(200, 20, 1))
        self.assertAlmostEqual(0.5, metrics.calculate_pass_at_k(200, 100, 1))
        self.assertAlmostEqual(0.995, metrics.calculate_pass_at_k(200, 199, 1))
        self.assertAlmostEqual(0.0, metrics.calculate_pass_at_k(20, 0, 10))
        self.assertAlmostEqual(0.5, metrics.calculate_pass_at_k(20, 1, 10))
        self.assertAlmostEqual(
            0.7631578947, metrics.calculate_pass_at_k(20, 2, 10))
        self.assertAlmostEqual(
            0.9837461300, metrics.calculate_pass_at_k(20, 5, 10))
        self.assertAlmostEqual(
            0.9999945875, metrics.calculate_pass_at_k(20, 10, 10))
        self.assertAlmostEqual(0.0, metrics.calculate_pass_at_k(20, 0, 1))
        self.assertAlmostEqual(0.05, metrics.calculate_pass_at_k(20, 1, 1))
        self.assertAlmostEqual(0.1, metrics.calculate_pass_at_k(20, 2, 1))
        self.assertAlmostEqual(0.5, metrics.calculate_pass_at_k(20, 10, 1))
        self.assertAlmostEqual(0.95, metrics.calculate_pass_at_k(20, 19, 1))
        self.assertAlmostEqual(0.0, metrics.calculate_pass_at_k(2, 0, 1))
        self.assertAlmostEqual(0.5, metrics.calculate_pass_at_k(2, 1, 1))
