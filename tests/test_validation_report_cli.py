import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401
from generator import generate, rows_to_jsonl
from report import build_report
from validation import validate

ROOT = Path(__file__).parent.parent


class TestValidation(unittest.TestCase):
    def test_generated_clean(self):
        rows, _ = generate()
        self.assertEqual(validate(rows), [])

    def test_flags_defects(self):
        rows, _ = generate(weeks=10)
        bad = [dict(r) for r in rows]
        bad[2]["spend_tv"] = -5.0
        bad[3]["sales"] = -1.0
        bad[4] = dict(bad[3])  # duplicate week
        issues = validate(bad)
        text = " ".join(issues)
        self.assertIn("negative spend_tv", text)
        self.assertIn("negative sales", text)
        self.assertIn("duplicate week", text)

    def test_missing_field(self):
        rows, _ = generate(weeks=3)
        del rows[0]["sales"]
        self.assertTrue(any("missing field sales" in i for i in validate(rows)))


class TestReport(unittest.TestCase):
    def test_report_shape(self):
        rows, _ = generate()
        rep = build_report(rows, __import__("generator").TRUE_PARAMS)
        self.assertEqual(rep["weeks"], 104)
        self.assertEqual(rep["best_roi_channel"], "radio")
        self.assertEqual(rep["worst_roi_channel"], "tv")
        self.assertEqual(rep["top_marginal_channel"], "radio")
        self.assertGreater(rep["fit_r2"], 0.99)


class TestCLI(unittest.TestCase):
    def _run(self, *argv):
        return subprocess.run([sys.executable, str(ROOT / "cli.py"), *argv],
                              capture_output=True, text=True, cwd=ROOT)

    def test_check_exit_codes(self):
        rows, _ = generate()
        with tempfile.TemporaryDirectory() as td:
            good = Path(td) / "good.jsonl"
            good.write_text(rows_to_jsonl(rows))
            self.assertEqual(self._run("check", "--input", str(good)).returncode, 0)
            breach = self._run("check", "--input", str(good), "--min-r2", "0.999999")
            self.assertEqual(breach.returncode, 1)
            bad = Path(td) / "bad.jsonl"
            bad.write_text('{"week": 0, "sales": -5}\n')
            self.assertEqual(self._run("check", "--input", str(bad)).returncode, 2)
            missing = self._run("check", "--input", str(Path(td) / "nope.jsonl"))
            self.assertEqual(missing.returncode, 2)

    def test_report_cli(self):
        rows, _ = generate()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.jsonl"
            p.write_text(rows_to_jsonl(rows))
            out = self._run("report", "--input", str(p))
            self.assertEqual(out.returncode, 0)
            self.assertEqual(json.loads(out.stdout)["sales_total"], 29868573.29)


class TestNoNetwork(unittest.TestCase):
    def test_demo_logic_runs_with_sockets_blocked(self):
        import socket

        original = socket.socket

        class Blocked(socket.socket):
            def connect(self, *a, **k):
                raise AssertionError("network access attempted")

        socket.socket = Blocked
        try:
            rows, _ = generate()
            rep = build_report(rows, __import__("generator").TRUE_PARAMS)
            self.assertEqual(rep["weeks"], 104)
        finally:
            socket.socket = original


if __name__ == "__main__":
    unittest.main()
