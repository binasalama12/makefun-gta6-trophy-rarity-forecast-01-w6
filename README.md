# GTA VI Trophy Rarity Forecast — wallet w6

Forecast each of the 55 GTA VI PS5 trophies' Exophase earn rate (%) at the
case target time. One program, `main.py` (Python 3.13, standard library
only, no network), reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else
`/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Approach: EWMA trend on the logit scale, with a dilution branch

All modelling happens in logit space, `logit(p) = ln(p / (100 - p))`,
because earn rates move multiplicatively and the contest scores pinball
loss on that scale.

**Trend branch — history actually moves.** Take the reads from the last
24 hours and fit a weighted least-squares line of `logit(rate)` against
hours since the first read, with exponential weights of half-life 6 hours
(recent reads count more). Read the line off at the target time for the
median. Spread = weighted residual standard deviation +
`0.5 * sqrt(horizon in days)`, floored at 0.35 logit units. This is the
branch that matters for the final case, whose history spans ~2 days of
10-minute reads.

**Flat-history branch — history is stale.** With fewer than 3 distinct
rates for a trophy (or zero variance), the page has not moved at all, and
Exophase refreshes its counts on its own schedule: on launch night the
page sat unchanged at 638 tracked players, then refreshed to 3,868 with
every rate lower. So instead of trusting the stale page, the median is
30% of the last rate and the spread is a wide 1.35 logit units, so the
90% band covers a several-fold move either way. One exception: a trophy
at exactly 100% keeps its median at 100% — the launch-night refresh left
that rate unchanged. This is the branch active on both practice cases.

**Quantiles.** Standard-normal z-scores (-1.645, -0.674, 0, 0.674, 1.645)
around the median on the logit scale, clamped to [0.01, 99.99], rounded
to 4 decimals (half up), forced non-decreasing. Output is deterministic:
same input bytes give same output bytes.

## How to run

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=/tmp/w6-out python3 main.py
```

No dependencies, no credentials, no network.
