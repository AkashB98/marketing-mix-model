"""Golden evals for marketing-mix-model. Deterministic; run 3x, md5 must match."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from generator import TRUE_PARAMS, generate, load_rows, rows_to_jsonl  # noqa: E402
from model import channel_totals, decompose, fit_ols  # noqa: E402
from report import build_report  # noqa: E402
from simulator import marginal_roi, simulate_shift  # noqa: E402
from transforms import adstock_series, hill  # noqa: E402
from validation import validate  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"eval": name, "pass": bool(ok), "detail": detail})


def main():
    rows, _ = generate()

    md5 = hashlib.md5(rows_to_jsonl(rows).encode()).hexdigest()
    check("E1_generator_determinism", md5 == "6fd0afa1ee1298ec96f18ba6d21da488", md5)

    total_spend = sum(r[f"spend_{c}"] for r in rows for c in
                      ["tv", "paid_search", "paid_social", "display", "radio"])
    check("E2_seeded_totals",
          len(rows) == 104 and abs(total_spend - 4371284.92) < 0.01
          and abs(sum(r["sales"] for r in rows) - 29868573.29) < 0.01,
          f"spend={total_spend:.2f}")

    check("E3_adstock_math", adstock_series([100.0, 0.0, 0.0], 0.5) == [100.0, 50.0, 25.0])

    check("E4_hill_math",
          hill(0.0, 1.5, 30000.0) == 0.0
          and abs(hill(30000.0, 1.5, 30000.0) - 0.5) < 1e-12
          and hill(1000.0, 1.2, 5000.0) < hill(2000.0, 1.2, 5000.0))

    decomp = decompose(rows, TRUE_PARAMS)
    err = max(abs(d["sales"] - d["modeled"] - d["residual"]) for d in decomp)
    check("E5_decomposition_conservation", err < 1e-9, f"max_err={err:.2e}")

    totals = channel_totals(decomp, rows)
    check("E6_roi_ranking",
          totals["radio"]["roi"] > totals["tv"]["roi"]
          and abs(totals["radio"]["roi"] - 5.13) < 0.01
          and abs(totals["tv"]["contribution"] - 6239857.75) < 1.0,
          f"radio_roi={totals['radio']['roi']}")

    fit = fit_ols(rows, TRUE_PARAMS)
    max_err_pct = max(abs(v["error_pct"]) for v in fit["recovery"].values())
    check("E7_fit_recovery", fit["r2"] > 0.99 and max_err_pct < 20.0,
          f"r2={fit['r2']:.4f} max_err_pct={max_err_pct:.2f}")

    marg = marginal_roi(rows, TRUE_PARAMS)
    check("E8_marginal_roi_ranking",
          next(iter(marg)) == "radio" and min(marg, key=marg.get) == "tv"
          and abs(marg["radio"] - 3.2889) < 0.001,
          f"radio={marg['radio']} tv={marg['tv']}")

    sim = simulate_shift(rows, TRUE_PARAMS, "tv", "radio", 2000.0)
    check("E9_simulator_delta", abs(sim["delta_sales"] - 167596.03) < 1.0
          and simulate_shift(rows, TRUE_PARAMS, "radio", "tv", 2000.0)["delta_sales"] < 0,
          f"delta={sim['delta_sales']:.2f}")

    bad = [dict(r) for r in rows[:5]]
    bad[1]["spend_tv"] = -1.0
    bad[2]["sales"] = -1.0
    check("E10_validation_flags_and_clean",
          validate(rows) == [] and len(validate(bad)) >= 2)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "d.jsonl"
        p.write_text(rows_to_jsonl(rows))
        ok = subprocess.run([sys.executable, str(ROOT / "cli.py"), "check",
                             "--input", str(p)], capture_output=True, cwd=ROOT)
        breach = subprocess.run([sys.executable, str(ROOT / "cli.py"), "check",
                                 "--input", str(p), "--min-r2", "0.999999"],
                                capture_output=True, cwd=ROOT)
        badp = Path(td) / "bad.jsonl"
        badp.write_text('{"week": 0, "sales": -5}\n')
        badr = subprocess.run([sys.executable, str(ROOT / "cli.py"), "check",
                               "--input", str(badp)], capture_output=True, cwd=ROOT)
        check("E11_cli_check_exit_codes",
              (ok.returncode, breach.returncode, badr.returncode) == (0, 1, 2),
              f"{ok.returncode}/{breach.returncode}/{badr.returncode}")

    fixture = load_rows((ROOT / "fixtures" / "sample_26w.jsonl").read_text(), "jsonl")
    fmd5 = hashlib.md5(rows_to_jsonl(fixture).encode()).hexdigest()
    check("E12_fixture_pinned",
          len(fixture) == 26 and fmd5 == "caeb475884af0438a809af475c6e63d4"
          and abs(sum(r["sales"] for r in fixture) - 7312842.30) < 0.01,
          fmd5)

    report = build_report(rows, TRUE_PARAMS)
    check("E13_report_consistency",
          report["best_roi_channel"] == "radio"
          and report["top_marginal_channel"] == "radio"
          and abs(report["media_contribution_total"] - 16528343.55) < 1.0)

    passed = sum(1 for r in RESULTS if r["pass"])
    out = {"passed": passed, "total": len(RESULTS), "results": RESULTS}
    (ROOT / "evals" / "eval_report.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"evals: {passed}/{len(RESULTS)} passed")
    for r in RESULTS:
        print(("PASS " if r["pass"] else "FAIL ") + r["eval"] + f"  {r['detail']}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
