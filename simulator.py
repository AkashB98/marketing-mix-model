"""Budget reallocation simulator.

Recomputes modeled sales under a changed spend plan, using the same
adstock + saturation machinery as the decomposition — so a reallocation
that pushes a channel deeper into saturation honestly shows shrinking
returns instead of a linear fantasy.
"""
from generator import CHANNELS, baseline_at
from transforms import transform_channel


def modeled_sales(rows, params, spend_override=None):
    """Total modeled sales (baseline + media, no noise) for a spend plan.

    spend_override: {channel: [weekly spend, ...]} replacing observed spend.
    """
    total = sum(baseline_at(r["week"]) for r in rows)
    for c in CHANNELS:
        spend = (
            spend_override[c] if spend_override and c in spend_override
            else [r[f"spend_{c}"] for r in rows]
        )
        saturated = transform_channel(spend, params[c])
        total += params[c]["coefficient"] * sum(saturated)
    return total


def simulate_shift(rows, params, from_channel, to_channel, amount_per_week):
    """Move a fixed $/week from one channel to another; report the delta."""
    if from_channel not in CHANNELS or to_channel not in CHANNELS:
        raise ValueError("unknown channel")
    if from_channel == to_channel:
        raise ValueError("from_channel and to_channel must differ")
    if amount_per_week <= 0:
        raise ValueError("amount_per_week must be > 0")
    base_total = modeled_sales(rows, params)
    override = {}
    for c in (from_channel, to_channel):
        override[c] = [r[f"spend_{c}"] for r in rows]
    for i in range(len(rows)):
        cut = min(amount_per_week, override[from_channel][i])
        override[from_channel][i] -= cut
        override[to_channel][i] += cut
    new_total = modeled_sales(rows, params, override)
    return {
        "from_channel": from_channel,
        "to_channel": to_channel,
        "amount_per_week": amount_per_week,
        "weeks": len(rows),
        "base_modeled_sales": round(base_total, 2),
        "new_modeled_sales": round(new_total, 2),
        "delta_sales": round(new_total - base_total, 2),
    }


def marginal_roi(rows, params, bump=1000.0):
    """Extra modeled sales from sustaining +$bump/week in one channel,
    per dollar of extra spend — the ranking a planner actually reallocates on."""
    base_total = modeled_sales(rows, params)
    out = {}
    for c in CHANNELS:
        override = {c: [r[f"spend_{c}"] + bump for r in rows]}
        gain = modeled_sales(rows, params, override) - base_total
        out[c] = round(gain / (bump * len(rows)), 4)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
