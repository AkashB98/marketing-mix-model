import hashlib
import unittest

import helpers  # noqa: F401
from generator import CHANNELS, generate, load_rows, rows_to_csv, rows_to_jsonl


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.rows, self.truth = generate()

    def test_shape(self):
        self.assertEqual(len(self.rows), 104)
        for r in self.rows:
            for c in CHANNELS:
                self.assertIn(f"spend_{c}", r)
            self.assertGreater(r["sales"], 0)

    def test_deterministic(self):
        rows2, _ = generate()
        self.assertEqual(rows_to_jsonl(self.rows), rows_to_jsonl(rows2))

    def test_seed_changes_data(self):
        other, _ = generate(seed=7)
        self.assertNotEqual(rows_to_jsonl(self.rows), rows_to_jsonl(other))

    def test_jsonl_md5_pinned(self):
        md5 = hashlib.md5(rows_to_jsonl(self.rows).encode()).hexdigest()
        self.assertEqual(md5, "6fd0afa1ee1298ec96f18ba6d21da488")

    def test_totals_pinned(self):
        total_spend = sum(r[f"spend_{c}"] for r in self.rows for c in CHANNELS)
        self.assertAlmostEqual(total_spend, 4371284.92, places=2)
        self.assertAlmostEqual(sum(r["sales"] for r in self.rows), 29868573.29, places=2)

    def test_csv_round_trip(self):
        loaded = load_rows(rows_to_csv(self.rows), "csv")
        self.assertEqual(loaded, self.rows)

    def test_jsonl_round_trip(self):
        loaded = load_rows(rows_to_jsonl(self.rows), "jsonl")
        self.assertEqual(loaded, self.rows)

    def test_truth_matches_params(self):
        self.assertEqual(self.truth["params"]["tv"]["coefficient"], 95000.0)
        self.assertEqual(self.truth["noise_sigma"], 1500.0)

    def test_spend_non_negative(self):
        for r in self.rows:
            for c in CHANNELS:
                self.assertGreaterEqual(r[f"spend_{c}"], 0.0)


if __name__ == "__main__":
    unittest.main()
