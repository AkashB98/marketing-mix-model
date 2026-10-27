"""Decomposition and OLS coefficient recovery for the mix model.

Decomposition — splitting observed sales into baseline + one slice per
channel, using a parameter set. With the true parameters the slices plus
the residual (noise) sum back to observed sales exactly.

Fit — ordinary least squares (OLS, the classic linear-regression fit) on
transformed features, solved via the normal equations with Gaussian
elimination. Scope limit, stated honestly: the adstock/saturation
transform parameters are assumed (the true ones); only the linear
coefficients are fitted.
"""
import math

from generator import (
    BASELINE_INTERCEPT,
    BASELINE_TREND,
    CHANNELS,
    PROMO_WEEKS,
    SEASONALITY_AMPLITUDE,
)
from transforms import transform_channel


def _promo(week):
    return PROMO_WEEKS.get(week, 0.0)


def _seasonal(week):
    return SEASONALITY_AMPLITUDE * math.sin(2 * math.pi * week / 52.0)


def decompose(rows, params):
    """Split each week's sales into baseline / per-channel / residual."""
    spend = {c: [r[f"spend_{c}"] for r in rows] for c in CHANNELS}
    saturated = {c: transform_channel(spend[c], params[c]) for c in CHANNELS}
    out = []
    for i, r in enumerate(rows):
        baseline = (
            BASELINE_INTERCEPT
            + BASELINE_TREND * r["week"]
            + _seasonal(r["week"])
            + _promo(r["week"])
        )
        parts = {"week": r["week"], "baseline": baseline}
        modeled = baseline
        for c in CHANNELS:
            parts[f"contrib_{c}"] = params[c]["coefficient"] * saturated[c][i]
            modeled += parts[f"contrib_{c}"]
        parts["modeled"] = modeled
        parts["residual"] = r["sales"] - modeled
        parts["sales"] = r["sales"]
        out.append(parts)
    return out


def channel_totals(decomp, rows):
    """Aggregate spend / contribution / share / ROI per channel."""
    totals = {}
    total_contrib = sum(
        sum(d[f"contrib_{c}"] for d in decomp) for c in CHANNELS
    )
    for c in CHANNELS:
        spend_total = sum(r[f"spend_{c}"] for r in rows)
        contrib_total = sum(d[f"contrib_{c}"] for d in decomp)
        totals[c] = {
            "spend": round(spend_total, 2),
            "contribution": round(contrib_total, 2),
            "share_pct": round(100.0 * contrib_total / total_contrib, 2) if total_contrib else 0.0,
            "roi": round(contrib_total / spend_total, 4) if spend_total else 0.0,
        }
    totals["baseline_total"] = round(sum(d["baseline"] for d in decomp), 2)
    totals["sales_total"] = round(sum(d["sales"] for d in decomp), 2)
    totals["media_contribution_total"] = round(total_contrib, 2)
    return totals


def _features(rows, params):
    spend = {c: [r[f"spend_{c}"] for r in rows] for c in CHANNELS}
    saturated = {c: transform_channel(spend[c], params[c]) for c in CHANNELS}
    X, y = [], []
    for i, r in enumerate(rows):
        w = r["week"]
        X.append(
            [1.0, float(w), math.sin(2 * math.pi * w / 52.0),
             math.cos(2 * math.pi * w / 52.0), _promo(w)]
            + [saturated[c][i] for c in CHANNELS]
        )
        y.append(r["sales"])
    return X, y


def _solve(A, b):
    """Gaussian elimination with partial pivoting for A x = b."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("singular design matrix")
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                factor = M[r][col] / M[col][col]
                for k in range(col, n + 1):
                    M[r][k] -= factor * M[col][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_ols(rows, params):
    """Fit linear coefficients on transformed features; report R^2 and
    recovered-vs-true channel coefficients."""
    X, y = _features(rows, params)
    p = len(X[0])
    XtX = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(p)] for i in range(p)]
    Xty = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(p)]
    beta = _solve(XtX, Xty)
    mean_y = sum(y) / len(y)
    ss_res = ss_tot = 0.0
    for k in range(len(X)):
        pred = sum(beta[j] * X[k][j] for j in range(p))
        ss_res += (y[k] - pred) ** 2
        ss_tot += (y[k] - mean_y) ** 2
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    names = ["intercept", "trend", "season_sin", "season_cos", "promo"] + CHANNELS
    coefficients = {names[i]: beta[i] for i in range(p)}
    recovery = {}
    for c in CHANNELS:
        true_c = params[c]["coefficient"]
        recovery[c] = {
            "true": true_c,
            "fitted": round(coefficients[c], 2),
            "error_pct": round(100.0 * (coefficients[c] - true_c) / true_c, 2),
        }
    return {"r2": r2, "coefficients": coefficients, "recovery": recovery}
