# marketing-mix-model

**A marketing mix model (MMM) toolkit — adstock, saturation, contribution decomposition, coefficient recovery, and a budget reallocation simulator — as one transparent, runnable artifact.**

Every growth team eventually asks the MMM question: of everything we spent across every channel, what actually drove sales — and where should the next dollar go? This project rebuilds the whole method as code a reviewer can run and argue with: a deterministic synthetic weekly dataset for a fictional smart-home store (**Helios Home**, 104 weeks, five channels), the two transforms that make MMM work — **adstock** (carryover — an ad keeps influencing buyers after it runs, fading over weeks) and **saturation** (diminishing returns — each extra dollar in a channel buys less than the one before) — a decomposition that splits every week's sales into baseline + one slice per channel, an OLS fit (**ordinary least squares** — the classic linear-regression fit) that recovers the channel coefficients from data and is scored against the known ground truth, and a simulator that moves budget between channels and honestly shows the shrinking returns.

> **Synthetic data only.** Every spend and sales figure is fabricated by a seeded generator with known ground-truth parameters. No real marketing, customer, or revenue data is used anywhere in this project.

Stdlib-only Python. Zero network, zero API keys, zero external dependencies.

## Quickstart

```bash
# 1. Generate the synthetic Helios Home dataset (104 weeks, seed 42)
python3 cli.py generate --out dataset.jsonl

# 2. Decompose sales into baseline + per-channel contributions
python3 cli.py decompose --input dataset.jsonl --limit 5

# 3. Fit OLS coefficients on the transformed features; score vs truth
python3 cli.py fit --input dataset.jsonl

# 4. Channel readout: spend / contribution / share / ROI + fit summary
python3 cli.py report --input dataset.jsonl

# 5. Simulate moving $2,000/week from TV to radio
python3 cli.py simulate --input dataset.jsonl --from-channel tv --to-channel radio --amount 2000

# 6. CI gate: exit 0 = pass, 1 = metric breach (R^2 floor / conservation),
#    2 = data-quality / IO error
python3 cli.py check --input dataset.jsonl

# 7. One-command end-to-end demo (generate -> decompose -> fit -> simulate -> serve)
python3 demo.py

# 8. Tiny localhost API: POST /simulate, GET /report, GET /health on 127.0.0.1
python3 cli.py serve --port 8765 --input dataset.jsonl
```

## The seeded dataset at a glance (seed 42, 104 weeks)

**$29,868,573.29** of sales on **$4,371,284.92** of media spend. Baseline (the sales that happen with no media — trend, seasonality, promos) accounts for $13,331,400.00; media contributes $16,528,343.55:

| Channel | Spend | Contribution | Share | ROI |
|---|---|---|---|---|
| tv | $1,911,610.75 | $6,239,857.75 | 37.75% | 3.26 |
| paid_search | $963,806.43 | $3,953,024.71 | 23.92% | 4.10 |
| paid_social | $745,379.00 | $3,048,203.63 | 18.44% | 4.09 |
| display | $421,630.22 | $1,599,784.16 | 9.68% | 3.79 |
| radio | $328,858.52 | $1,687,473.30 | 10.21% | 5.13 |

**The readout that starts the budget argument:** TV eats 43.7% of spend, earns the biggest contribution — and has the *worst* ROI (3.26) and the worst **marginal ROI** (the sales bought by the *next* sustained $1/week: TV 1.84 vs radio 3.29). Moving $2,000/week from TV to radio for the 104 weeks adds **$167,596.03** of modeled sales on the *same total budget* — the simulator proves it by recomputing adstock + saturation, not by multiplying an average ROI. These numbers are pinned by golden evals, so any change to the generator, transforms, or model shows up as an eval failure, not a surprise.

**Does the fit recover the truth?** The generator builds sales from known coefficients; the OLS fit, given only the spend + sales series (and the transform parameters — see Scope), recovers them with R² = 0.9931 and a worst channel error of 10.2% (display; TV −3.2%, paid search +2.8%). Recovery *error* is reported per channel in the report — an MMM that won't show its own error bars against ground truth is asking to be trusted blind.

## How the model works

| Piece | Rule |
|---|---|
| **Adstock** (carryover) | `adstock_t = spend_t + decay × adstock_{t−1}`; decay per channel (TV 0.60 — a TV spot lingers; paid search 0.20 — a click is spent fast) |
| **Saturation** (Hill curve) | `response = x^α / (x^α + γ^α)` over adstocked spend; exactly 0.5 at x = γ, bounded below 1.0, so contribution = coefficient × response can never grow linearly forever |
| **Baseline** | intercept + $150/week trend + $8,000 annual sine seasonality + documented promo spikes (weeks 24 / 50 / 76) |
| **Decomposition** | baseline + Σ channel contributions + residual (the seeded noise, σ = $1,500) == observed sales, exactly, every week |
| **Fit** | OLS on [intercept, trend, sin, cos, promo, Hill(adstock(spend)) per channel] via the normal equations, solved with Gaussian elimination — no numpy |
| **Simulator** | a spend plan replaces observed spend; modeled sales are recomputed end-to-end, so moving money *into* a saturated channel shows the flattening |

**The generator.** Seeded PRNGs draw weekly spend per channel with realistic shapes — TV flights (4 weeks on / 2 off, heavier in Q4), always-on search with a Q4 step, seasonal social steps, a 26-week display cycle, pulsed radio — then sales are assembled from the ground-truth parameter table plus seeded Gaussian noise. Same seed, same dataset, byte-for-byte (eval E1 pins the JSONL md5).

## Architecture

| Module | Job |
|---|---|
| `generator.py` | Deterministic synthetic weekly dataset: per-channel spend patterns, ground-truth parameter table, baseline + contributions + noise assembly; JSONL / CSV serialisers + loaders with explicit numeric coercion |
| `transforms.py` | Geometric adstock and Hill saturation, with parameter validation (decay ∈ [0, 1), α > 0, γ > 0) |
| `model.py` | Decomposition (conserving, residual explicit) + OLS fit via normal equations / Gaussian elimination + recovered-vs-true scoring |
| `simulator.py` | Budget reallocation (fixed $/week shifts, floored at zero spend) + marginal-ROI ranking by finite difference on the full transform chain |
| `report.py` | Channel readout (spend / contribution / share / ROI), fit summary, marginal-ROI ranking, best/worst channel callouts |
| `validation.py` | Data-quality gate: required fields, week continuity / duplicates, non-negative spend and sales — reports issues, never raises on bad data |
| `cli.py` | `generate` / `decompose` / `fit` / `simulate` / `report` / `check` (CI exit 0/1/2) / `serve` (stdlib `http.server` on 127.0.0.1: `POST /simulate`, `GET /report`, `GET /health`) |
| `demo.py` | End-to-end narrative with asserts: determinism, conservation, fit floor, simulator direction, CLI exit codes, live localhost serve round-trip matching local results exactly |
| `fixtures/` | 26-week sample dataset (seed 42, $7,312,842.30 of sales) in JSONL + CSV |

## Evals

`python3 evals/run_evals.py` → **13/13 golden evals**, report committed at `evals/eval_report.json` (md5 `e773d0a06dfe8aa8e068b36c639fb754`, byte-identical across 3 verified runs):

- **E1** generator determinism — seed 42 JSONL bytes pinned by md5
- **E2** seeded totals pinned (104 weeks, $4,371,284.92 spend, $29,868,573.29 sales)
- **E3** adstock math pinned (100 at decay 0.5 → 100 / 50 / 25)
- **E4** Hill math pinned (0 at 0, exactly 0.5 at γ, monotonic)
- **E5** decomposition conservation (max error < 1e-9)
- **E6** ROI ranking pinned (radio 5.13 > TV 3.26; TV contribution $6,239,857.75)
- **E7** fit recovery (R² > 0.99; every channel coefficient within 20% of truth)
- **E8** marginal-ROI ranking pinned (radio top 3.2889, TV bottom 1.8360)
- **E9** simulator delta pinned (TV → radio $2,000/week = +$167,596.03; the reverse shift loses)
- **E10** validation flags planted defects; generated dataset validates clean
- **E11** CLI `check` exit codes 0 / 1 / 2
- **E12** bundled fixture pinned (26 weeks, md5, $7,312,842.30 sales)
- **E13** report consistency (best-ROI and top-marginal channel both radio)

Unit tests: `python3 -m unittest discover -s tests` → **39 hermetic tests**, including a no-network enforcement test, hand-computed adstock/Hill cases, generator determinism and round-trips, decomposition conservation, fit recovery bounds, simulator direction and validation, data-quality defects, and CLI exit codes.

## Development loop — what the checks validated

1. **Spend patterns that mimic seasonality make the fit lie.** The first generator draft gave paid search and paid social smooth annual *sine* spend patterns — the same shape as the baseline seasonality features. A sine spend column is **collinear** (it moves in lockstep with another feature, so the fit cannot tell which one earned the sales) with the seasonality columns, and coefficient recovery degrades badly. The fix was realistic planning shapes instead of smooth curves — always-on search with a Q4 step, social in seasonal steps — after which every channel recovers within ~10% of truth. Pinned by E7 so a future "prettier" spend pattern can't silently reintroduce it.
2. **Short windows make the OLS design singular.** Fitting a window with no promo week leaves the promo feature all-zero and the normal equations have no solution; the solver raises rather than inventing a coefficient, and the no-network test now runs on the full series for exactly this reason. Documented as a scope limit below, not papered over.

## Scope and honest limits

- **Transform parameters are assumed, not fitted.** The fit uses the true adstock decays and Hill (α, γ); a production MMM searches or puts priors over them (Bayesian MMMs like Robyn / Meridian / PyMC-Marketing). This project fits the linear coefficients exactly and says so — joint transform fitting is the named next step, not a hidden gap.
- **Recovery is scored on synthetic ground truth.** The coefficients "work" because the generator put them there; the honest claim is that the pipeline recovers known truth reproducibly and reports its error, not that any channel finding transfers to a real store.
- **Weekly, single-geo, no external regressors.** No pricing, distribution, competitor, or weather effects; no geo-level variation (geo experiments are how real MMMs get validated). The promo feature is a known-spike dummy, not a learned promotion model.
- **The simulator holds the world still.** Reallocation deltas assume every other channel, the baseline, and the truth parameters stay fixed — a planning estimate, not a forecast guarantee.
