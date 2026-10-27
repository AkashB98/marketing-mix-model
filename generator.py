"""Deterministic synthetic weekly MMM dataset for fictional Helios Home.

Synthetic data ONLY. Sales are built from known ground-truth parameters
so the model's recovery of them can be scored honestly:
  sales = baseline(trend + annual seasonality + promo spikes)
        + sum_c coefficient_c * Hill(adstock(spend_c))
        + seeded Gaussian noise
"""
import csv
import io
import json
import math
import random

CHANNELS = ["tv", "paid_search", "paid_social", "display", "radio"]

# Ground-truth channel parameters (the answer key the fit is scored against).
TRUE_PARAMS = {
    "tv":          {"decay": 0.60, "alpha": 1.5, "gamma": 30000.0, "coefficient": 95000.0},
    "paid_search": {"decay": 0.20, "alpha": 1.2, "gamma": 12000.0, "coefficient": 78000.0},
    "paid_social": {"decay": 0.35, "alpha": 1.3, "gamma": 10000.0, "coefficient": 56000.0},
    "display":     {"decay": 0.30, "alpha": 1.1, "gamma": 8000.0,  "coefficient": 38000.0},
    "radio":       {"decay": 0.45, "alpha": 1.4, "gamma": 6000.0,  "coefficient": 34000.0},
}

BASELINE_INTERCEPT = 120000.0
BASELINE_TREND = 150.0          # dollars of baseline sales growth per week
SEASONALITY_AMPLITUDE = 8000.0  # annual sine wave on baseline
PROMO_WEEKS = {24: 15000.0, 50: 18000.0, 76: 15000.0}  # week index -> spike $
NOISE_SIGMA = 1500.0

_BASE_SPEND = {
    "tv": 16000.0, "paid_search": 9000.0, "paid_social": 7000.0,
    "display": 4500.0, "radio": 3500.0,
}


def _weekly_spend(channel, week, rng):
    base = _BASE_SPEND[channel]
    if channel == "tv":
        # Flighting: 4-weeks-on / 2-weeks-off pulses, heavier in Q4.
        mult = 1.5 if (week % 6) < 4 else 0.25
        if (week % 52) >= 44:
            mult *= 1.3
    elif channel == "paid_search":
        # Always-on: flat base, holiday step in Q4 weeks of each year.
        # (Deliberately NOT a smooth annual sine — a sinusoidal spend
        # pattern is collinear with the baseline seasonality features
        # and the OLS fit cannot separate channel from season.)
        mult = 1.25 if (week % 52) >= 44 else 1.0
    elif channel == "paid_social":
        # Seasonal steps: holiday push + a mid-year burst, flat otherwise.
        wk = week % 52
        if wk >= 44:
            mult = 1.45
        elif 22 <= wk <= 28:
            mult = 1.2
        else:
            mult = 0.9
    elif channel == "display":
        mult = 0.9 + 0.3 * math.cos(2 * math.pi * week / 26.0)
    else:  # radio
        mult = 1.1 if (week % 8) < 5 else 0.6
    return round(base * mult * (0.92 + 0.16 * rng.random()), 2)


def baseline_at(week):
    seasonal = SEASONALITY_AMPLITUDE * math.sin(2 * math.pi * week / 52.0)
    return BASELINE_INTERCEPT + BASELINE_TREND * week + seasonal + PROMO_WEEKS.get(week, 0.0)


def generate(weeks=104, seed=42):
    """Return (rows, truth) — rows are weekly dicts; truth holds parameters."""
    from transforms import transform_channel

    spend_rng = {c: random.Random(seed * 1000 + i) for i, c in enumerate(CHANNELS)}
    spend = {c: [_weekly_spend(c, w, spend_rng[c]) for w in range(weeks)] for c in CHANNELS}
    saturated = {c: transform_channel(spend[c], TRUE_PARAMS[c]) for c in CHANNELS}
    noise_rng = random.Random(seed)
    rows = []
    for w in range(weeks):
        row = {"week": w}
        for c in CHANNELS:
            row[f"spend_{c}"] = spend[c][w]
        contrib = {c: TRUE_PARAMS[c]["coefficient"] * saturated[c][w] for c in CHANNELS}
        noise = noise_rng.gauss(0.0, NOISE_SIGMA)
        row["sales"] = round(baseline_at(w) + sum(contrib.values()) + noise, 2)
        rows.append(row)
    truth = {
        "params": {c: dict(TRUE_PARAMS[c]) for c in CHANNELS},
        "intercept": BASELINE_INTERCEPT,
        "trend": BASELINE_TREND,
        "seasonality_amplitude": SEASONALITY_AMPLITUDE,
        "promo_weeks": {str(k): v for k, v in PROMO_WEEKS.items()},
        "noise_sigma": NOISE_SIGMA,
        "seed": seed,
    }
    return rows, truth


def rows_to_jsonl(rows):
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def rows_to_csv(rows):
    buf = io.StringIO()
    fields = ["week"] + [f"spend_{c}" for c in CHANNELS] + ["sales"]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def load_rows(text, fmt="jsonl"):
    """Load weekly rows from JSONL or CSV text, coercing numbers explicitly."""
    rows = []
    if fmt == "jsonl":
        for line in text.strip().splitlines():
            if line.strip():
                rows.append(_coerce(json.loads(line)))
    elif fmt == "csv":
        for raw in csv.DictReader(io.StringIO(text)):
            rows.append(_coerce(raw))
    else:
        raise ValueError(f"unknown format: {fmt}")
    return rows


def _coerce(raw):
    row = {"week": int(float(raw["week"])), "sales": float(raw["sales"])}
    for c in CHANNELS:
        row[f"spend_{c}"] = float(raw[f"spend_{c}"])
    return row
