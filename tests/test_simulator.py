import unittest

import helpers  # noqa: F401
from generator import TRUE_PARAMS, generate
from simulator import marginal_roi, modeled_sales, simulate_shift


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.rows, _ = generate()

    def test_marginal_ranking_pinned(self):
        marg = marginal_roi(self.rows, TRUE_PARAMS)
        self.assertEqual(next(iter(marg)), "radio")
        self.assertEqual(min(marg, key=marg.get), "tv")
        self.assertAlmostEqual(marg["radio"], 3.2889, places=3)
        self.assertAlmostEqual(marg["tv"], 1.8360, places=3)

    def test_shift_low_to_high_marginal_gains(self):
        sim = simulate_shift(self.rows, TRUE_PARAMS, "tv", "radio", 2000.0)
        self.assertGreater(sim["delta_sales"], 0)
        self.assertAlmostEqual(sim["delta_sales"], 167596.03, places=1)

    def test_shift_reverse_loses(self):
        sim = simulate_shift(self.rows, TRUE_PARAMS, "radio", "tv", 2000.0)
        self.assertLess(sim["delta_sales"], 0)

    def test_shift_validation(self):
        with self.assertRaises(ValueError):
            simulate_shift(self.rows, TRUE_PARAMS, "tv", "tv", 100.0)
        with self.assertRaises(ValueError):
            simulate_shift(self.rows, TRUE_PARAMS, "tv", "radio", 0.0)
        with self.assertRaises(ValueError):
            simulate_shift(self.rows, TRUE_PARAMS, "tv", "nope", 100.0)

    def test_modeled_sales_no_override_is_baseline_plus_media(self):
        total = modeled_sales(self.rows, TRUE_PARAMS)
        self.assertAlmostEqual(total, 29859743.55, places=1)


if __name__ == "__main__":
    unittest.main()
