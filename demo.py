"""End-to-end demo: generate -> decompose -> fit -> simulate -> serve."""
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from generator import TRUE_PARAMS, generate  # noqa: E402
from model import decompose, fit_ols  # noqa: E402
from report import build_report  # noqa: E402
from simulator import marginal_roi, simulate_shift  # noqa: E402


def main():
    rows, _ = generate(weeks=104, seed=42)
    rows2, _ = generate(weeks=104, seed=42)
    assert rows == rows2, "generator must be deterministic"
    print("1. generated 104 deterministic weeks (seed 42)")

    decomp = decompose(rows, TRUE_PARAMS)
    err = max(abs(d["sales"] - d["modeled"] - d["residual"]) for d in decomp)
    assert err < 1e-9, err
    print("2. decomposition conserves sales (baseline + channels + residual)")

    report = build_report(rows, TRUE_PARAMS)
    print(f"3. report: sales total ${report['sales_total']:,.2f}, "
          f"media contribution ${report['media_contribution_total']:,.2f}")
    for c, t in report["channels"].items():
        print(f"   {c:<12} spend ${t['spend']:>12,.2f}  contrib ${t['contribution']:>12,.2f}  "
              f"share {t['share_pct']:>5.2f}%  ROI {t['roi']:.2f}")

    fit = fit_ols(rows, TRUE_PARAMS)
    assert fit["r2"] > 0.90, fit["r2"]
    print(f"4. OLS fit R^2 = {fit['r2']:.4f}; recovered coefficients within "
          f"{max(abs(v['error_pct']) for v in fit['recovery'].values()):.1f}% of truth")

    marg = marginal_roi(rows, TRUE_PARAMS)
    worst = min(marg, key=marg.get)
    best = next(iter(marg))
    sim = simulate_shift(rows, TRUE_PARAMS, worst, best, 2000.0)
    assert sim["delta_sales"] > 0, sim
    print(f"5. simulate: move $2,000/week {worst} -> {best}: "
          f"modeled sales delta ${sim['delta_sales']:,.2f} over {sim['weeks']} weeks")

    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / "data.jsonl"
        subprocess.run([sys.executable, str(ROOT / "cli.py"), "generate",
                        "--out", str(data)], check=True, cwd=ROOT)
        chk = subprocess.run([sys.executable, str(ROOT / "cli.py"), "check",
                              "--input", str(data)], capture_output=True, text=True, cwd=ROOT)
        assert chk.returncode == 0, chk.stdout + chk.stderr
        print("6. CLI check gate: PASS (exit 0)")

        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "cli.py"), "serve", "--input", str(data),
             "--port", "8791"], cwd=ROOT, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        try:
            for _ in range(50):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8791/health", timeout=1) as r:
                        assert json.load(r)["status"] == "ok"
                    break
                except Exception:
                    time.sleep(0.1)
            with urllib.request.urlopen("http://127.0.0.1:8791/report", timeout=2) as r:
                served = json.load(r)
            assert served["sales_total"] == report["sales_total"]
            req = urllib.request.Request(
                "http://127.0.0.1:8791/simulate",
                data=json.dumps({"from_channel": worst, "to_channel": best,
                                 "amount_per_week": 2000.0}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=2) as r:
                served_sim = json.load(r)
            assert served_sim["delta_sales"] == sim["delta_sales"]
            print("7. serve round-trip: /report and /simulate match local exactly")
        finally:
            proc.terminate()
    print("demo: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
