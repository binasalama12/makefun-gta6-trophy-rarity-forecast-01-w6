"""Forecast GTA VI PS5 trophy earn rates on Exophase (make.fun w6 entry).

Model: EWMA trend on the logit scale, with a flat-history dilution branch.

For each trophy we work in logit space, logit(p) = ln(p / (100 - p)),
because earn rates are percentages that move multiplicatively and the
contest scores pinball loss on that scale.

Trend branch (history actually moves):
  Take the reads from the last 24 hours, fit a weighted least-squares
  line of logit(rate) against hours since the first read, with
  exponential weights of half-life 6 hours (a read 6h older counts half
  as much as the newest one), and read the line off at the target time
  for the median. The spread is the weighted residual standard
  deviation plus a random-walk term, 0.5 * sqrt(horizon in days),
  floored at 0.35 logit units.

Flat-history branch (fewer than 3 distinct rates for the trophy, or
zero variance across reads):
  Exophase refreshes its counts on its own schedule, so a page that has
  sat unchanged for hours can jump at the next refresh: on launch night
  the tracked-player count went 638 -> 3868 while every rate fell.
  Anticipate that dilution instead of trusting the stale page: the
  median is 30% of the last rate and the spread is a wide 1.35 logit
  units, so the 90% band covers a several-fold move either way.
  Exception: a trophy at exactly 100% (every tracked player has it)
  keeps its median at 100% -- the launch-night refresh left that one
  rate unchanged, and a rate that is already everyone cannot dilute
  the same way; the wide spread still applies.

Quantiles use the standard-normal z-scores (-1.645, -0.674, 0, 0.674,
1.645) around the median on the logit scale, are clamped to
[0.01, 99.99], rounded to 4 decimals (half up), and forced
non-decreasing. The program is pure standard library, reads every
*.in file in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output), and is fully deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

HALF_LIFE_HOURS = 6.0      # exponential weight half-life for trend reads
WINDOW_HOURS = 24.0        # only reads this recent feed the trend fit
DILUTION_FACTOR = 0.30     # flat-history median = last rate * this
FLAT_SPREAD = 1.35         # flat-history logit spread
SPREAD_FLOOR = 0.35        # minimum logit spread, trend branch
RANDOM_WALK_SIGMA = 0.5    # logit spread per sqrt(day), trend branch
LO = 0.01                  # clamp floor for rates (percent)
HI = 99.99                 # clamp ceiling for rates (percent)


def clamp_rate(pct):
    """Clamp a percent into [0.01, 99.99] so its logit is finite."""
    return min(HI, max(LO, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp (with 'Z' or an offset)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5)
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def last_known_rate(history, index):
    """Most recent non-null rate for trophy `index`, else None."""
    for read in reversed(history):
        rate = read["rates"][index]
        if rate is not None:
            return rate
    return None


def forecast_trophy(case, index):
    """Return (median_percent, spread_logit) for one trophy."""
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    last_rate = last_known_rate(history, index)
    if last_rate is None:
        # Trophy never observed: centre wide and admit we know nothing.
        return 50.0, FLAT_SPREAD

    # --- gather (hours since first read, logit rate) points, last 24h ---
    times = [parse_time(read["at"]) for read in history]
    t_first = times[0]
    t_last = times[-1]
    points = []
    for read, when in zip(history, times):
        rate = read["rates"][index]
        if rate is None:
            continue
        if (t_last - when).total_seconds() > WINDOW_HOURS * 3600.0:
            continue
        hours = (when - t_first).total_seconds() / 3600.0
        points.append((hours, logit(rate)))

    distinct = {round(lg, 12) for _, lg in points}
    if len(points) < 3 or len(distinct) < 3:
        return flat_branch(last_rate)

    # --- weighted least squares, weights decay from the newest read ---
    t_last_hours = (t_last - t_first).total_seconds() / 3600.0
    weighted = []
    for hours, lg in points:
        age = t_last_hours - hours
        weight = 0.5 ** (age / HALF_LIFE_HOURS)
        weighted.append((hours, lg, weight))

    sum_w = sum(w for _, _, w in weighted)
    mean_x = sum(w * x for x, _, w in weighted) / sum_w
    mean_y = sum(w * y for _, y, w in weighted) / sum_w
    var_x = sum(w * (x - mean_x) ** 2 for x, _, w in weighted)
    if var_x <= 0.0:
        return flat_branch(last_rate)
    cov_xy = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in weighted)
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    # Weighted residual spread around the fitted line.
    resid_var = sum(
        w * (y - (intercept + slope * x)) ** 2 for x, y, w in weighted
    ) / sum_w
    spread = math.sqrt(resid_var) + RANDOM_WALK_SIGMA * math.sqrt(horizon_days)
    spread = max(spread, SPREAD_FLOOR)

    # Extrapolate the line to the target instant.
    if case.get("target"):
        t_target = parse_time(case["target"])
    else:
        t_target = t_last
    target_hours = (t_target - t_first).total_seconds() / 3600.0
    median_logit = intercept + slope * target_hours
    median_logit = min(max(median_logit, logit(LO)), logit(HI))
    return inv_logit(median_logit), spread


def flat_branch(last_rate):
    """Stale-page branch: expect a refresh to dilute the rate."""
    if last_rate >= HI:
        # Already earned by everyone tracked; the launch-night refresh
        # left this rate at 100%, so keep the median there.
        return HI, FLAT_SPREAD
    return clamp_rate(last_rate * DILUTION_FACTOR), FLAT_SPREAD


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        median, spread = forecast_trophy(case, index)
        centre = logit(median)
        quantiles = []
        previous = None
        for z in Z_SCORES:
            q = round4(clamp_rate(inv_logit(centre + z * spread)))
            if previous is not None and q < previous:
                q = previous
            quantiles.append(q)
            previous = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
