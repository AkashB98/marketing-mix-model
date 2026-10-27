import unittest

import helpers  # noqa: F401
from generator import TRUE_PARAMS, generate
from model import channel_totals, decompose, fit_ols


class TestDecompose(unittest.TestCase):
    def setUp(self):
        self.rows, _ = generate()
        self.decomp = decompose(self.rows, TRUE_PARAMS)

    def test_conservation(self):
        for d in self.decomp:
            self.assertAlmostEqual(
                d["baseline"] + sum(d[f"contrib_{c}"] for c in
                                     ["tv", "paid_search", "paid_social", "display", "radio"])
                + d["residual"], d["sales"], places=6)

    def test_residual_is_noise_scale(self):
        residuals = [abs(d["residual"]) for d in self.decomp]
        self.assertLess(sum(residuals) / len(residuals), 3000.0)

    def test_totals_pinned(self):
        totals = channel_totals(self.decomp, self.rows)
        self.assertAlmostEqual(totals["tv"]["spend"], 1911610.75, places=2)
        self.assertAlmostEqual(totals["tv"]["contribution"], 6239857.75, places=1)
        self.assertAlmostEqual(totals["radio"]["roi"], 5.13, places=2)
        self.assertAlmostEqual(totals["media_contribution_total"], 16528343.55, places=1)

    def test_shares_sum_to_100(self):
        totals = channel_totals(self.decomp, self.rows)
        chans = ["tv", "paid_search", "paid_social", "display", "radio"]
        self.assertAlmostEqual(sum(totals[c]["share_pct"] for c in chans), 100.0, places=1)


class TestFit(unittest.TestCase):
    def setUp(self):
        self.rows, _ = generate()
        self.fit = fit_ols(self.rows, TRUE_PARAMS)

    def test_r2_floor(self):
        self.assertGreater(self.fit["r2"], 0.99)

    def test_recovery_within_20pct(self):
        for c, rec in self.fit["recovery"].items():
            self.assertLess(abs(rec["error_pct"]), 20.0, c)

    def test_recovery_pinned(self):
        self.assertAlmostEqual(self.fit["recovery"]["tv"]["fitted"], 91924.04, places=1)
        self.assertAlmostEqual(self.fit["recovery"]["paid_search"]["error_pct"], 2.78, places=1)


if __name__ == "__main__":
    unittest.main()
