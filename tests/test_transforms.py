import unittest

import helpers  # noqa: F401
from transforms import adstock_series, hill, hill_series, transform_channel


class TestAdstock(unittest.TestCase):
    def test_zero_decay_is_identity(self):
        self.assertEqual(adstock_series([10.0, 20.0, 5.0], 0.0), [10.0, 20.0, 5.0])

    def test_hand_computed_decay(self):
        out = adstock_series([100.0, 0.0, 0.0], 0.5)
        self.assertEqual(out, [100.0, 50.0, 25.0])

    def test_carryover_accumulates(self):
        out = adstock_series([100.0, 100.0], 0.5)
        self.assertEqual(out, [100.0, 150.0])

    def test_decay_validation(self):
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                adstock_series([1.0], bad)

    def test_negative_spend_rejected(self):
        with self.assertRaises(ValueError):
            adstock_series([-1.0], 0.5)


class TestHill(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(hill(0.0, 1.5, 30000.0), 0.0)

    def test_half_at_gamma(self):
        self.assertAlmostEqual(hill(30000.0, 1.5, 30000.0), 0.5, places=12)
        self.assertAlmostEqual(hill(123.0, 2.0, 123.0), 0.5, places=12)

    def test_monotonic_and_bounded(self):
        vals = hill_series([0.0, 100.0, 1000.0, 10000.0, 100000.0], 1.2, 5000.0)
        self.assertEqual(vals, sorted(vals))
        self.assertTrue(all(0.0 <= v < 1.0 for v in vals))

    def test_diminishing_returns(self):
        a = hill(2000.0, 1.2, 5000.0) - hill(1000.0, 1.2, 5000.0)
        b = hill(3000.0, 1.2, 5000.0) - hill(2000.0, 1.2, 5000.0)
        self.assertGreater(a, b)

    def test_param_validation(self):
        with self.assertRaises(ValueError):
            hill(1.0, 0.0, 100.0)
        with self.assertRaises(ValueError):
            hill(1.0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            hill(-1.0, 1.0, 100.0)

    def test_transform_channel_composes(self):
        params = {"decay": 0.5, "alpha": 1.0, "gamma": 100.0}
        out = transform_channel([100.0, 0.0], params)
        self.assertAlmostEqual(out[0], 0.5, places=12)
        self.assertAlmostEqual(out[1], 50.0 / 150.0, places=12)


if __name__ == "__main__":
    unittest.main()
