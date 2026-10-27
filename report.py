"""Channel readout: spend, contribution, share, ROI + fit summary."""
from generator import CHANNELS
from model import channel_totals, decompose, fit_ols
from simulator import marginal_roi


def build_report(rows, params):
    decomp = decompose(rows, params)
    totals = channel_totals(decomp, rows)
    fit = fit_ols(rows, params)
    marg = marginal_roi(rows, params)
    channels = {c: totals[c] for c in CHANNELS}
    best_roi = max(channels, key=lambda c: channels[c]["roi"])
    worst_roi = min(channels, key=lambda c: channels[c]["roi"])
    return {
        "weeks": len(rows),
        "sales_total": totals["sales_total"],
        "baseline_total": totals["baseline_total"],
        "media_contribution_total": totals["media_contribution_total"],
        "channels": channels,
        "fit_r2": round(fit["r2"], 4),
        "coefficient_recovery": fit["recovery"],
        "marginal_roi_per_dollar": marg,
        "best_roi_channel": best_roi,
        "worst_roi_channel": worst_roi,
        "top_marginal_channel": next(iter(marg)),
    }
