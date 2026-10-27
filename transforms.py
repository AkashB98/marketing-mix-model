"""Adstock (carryover) and saturation (diminishing returns) transforms.

Adstock — the carryover effect: an ad seen this week keeps influencing
buyers for several weeks, fading geometrically.
Saturation — diminishing returns: each extra dollar in a channel buys
less response than the previous one, modelled with a Hill curve.
"""
import math


def adstock_series(spend, decay):
    """Geometric adstock: out[t] = spend[t] + decay * out[t-1]."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    out = []
    prev = 0.0
    for x in spend:
        if x < 0:
            raise ValueError("spend must be non-negative")
        prev = float(x) + decay * prev
        out.append(prev)
    return out


def hill(x, alpha, gamma):
    """Hill saturation: x^alpha / (x^alpha + gamma^alpha).

    Returns 0.0 at x == 0, exactly 0.5 at x == gamma, approaches 1.0.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if x < 0:
        raise ValueError("x must be non-negative")
    if x == 0:
        return 0.0
    xa = math.pow(x, alpha)
    ga = math.pow(gamma, alpha)
    return xa / (xa + ga)


def hill_series(values, alpha, gamma):
    return [hill(v, alpha, gamma) for v in values]


def transform_channel(spend, params):
    """Spend -> adstock -> Hill saturation, using a channel param dict."""
    return hill_series(
        adstock_series(spend, params["decay"]), params["alpha"], params["gamma"]
    )
