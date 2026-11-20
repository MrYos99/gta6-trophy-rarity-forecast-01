#!/usr/bin/env python3
"""Scoring for the GTA VI trophy rarity forecast harness.

Implements the contest's scoring exactly as RULES.txt states it:
  pinball item loss = 2 * max(tau*d, (tau-1)*d), d = logit(y) - logit(q),
  logit(x) = ln(p/(1-p)) with p = x/100 after x is clamped to 0.01..99.99.
  Loss = mean over every trophy x level. Lower is better.

Also implements the two disclosed house baselines, faithfully from RULES.txt:
  - persistence (house baseline, starter/main.ts): median = last read's rate,
    q_tau = invLogit(logit(r) + z_tau * 0.5 * sqrt(d_days)), z = normal quantiles.
  - curve baseline (baseline-trend/main.ts, not in the kit): per trophy, OLS line of
    logit(rate) on ln(1 + hours since release) over the case's last 24h of reads,
    read at the target; spread = random walk + fit residual combined;
    fewer than 3 reads -> the house baseline.

Assumptions where RULES.txt is silent (documented, not hidden):
  - Curve baseline uses the case's last-24h reads AS RECORDED (frozen duplicates
    included, no dedup): the rules say "the case's last 24 hours of reads".
  - Curve spread = sqrt(resid_sd^2 + (0.5*sqrt(d_days))^2).
  - The case's `target` is used for the curve baseline's x; when a case file has no
    usable target timestamp we fall back to at + horizon_minutes.
"""

import json
import math
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
RELEASE_MS = 1795132800000  # 2026-11-19T00:00:00Z


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1 - p))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    return round(x * 10000) / 10000


def pinball_loss(answer_rates, forecast):
    """answer_rates: {id: rate}; forecast: {id: [q05..q95]}. Returns mean loss."""
    total = 0.0
    n = 0
    for tid, y in answer_rates.items():
        q = forecast[tid]
        ly = logit(y)
        for tau, qv in zip(LEVELS, q):
            d = ly - logit(qv)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return total / n if n else float("nan")


def _hours_between(release_ms, iso):
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return max(0.0, (dt.timestamp() * 1000 - release_ms) / 3_600_000)


def _parse_release(case):
    g = case.get("game", {}) or {}
    rd = g.get("release_date")
    if rd:
        from datetime import datetime
        try:
            return int(datetime.fromisoformat(rd + "T00:00:00+00:00").timestamp() * 1000)
        except ValueError:
            pass
    return RELEASE_MS


def house_baseline(case):
    """Persistence baseline. Returns {id: [q...]}."""
    last = case["history"][-1]
    d_days = max(0.0, case["horizon_minutes"] / 1440.0)
    out = {}
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i]
        if not isinstance(r, (int, float)):
            r = 1.0
        m = logit(r)
        q = [round4(clamp(inv_logit(m + z * 0.5 * math.sqrt(d_days)), 0, 100))
             for z in Z]
        for j in range(1, 5):
            q[j] = max(q[j], q[j - 1])
        out[t["id"]] = q
    return out


def _ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    resid_sd = math.sqrt(rss / (n - 2)) if n > 2 else 0.0
    return a, b, resid_sd


def curve_baseline(case):
    """House curve baseline. Returns {id: [q...]}."""
    release = _parse_release(case)
    from datetime import datetime, timezone
    at_dt = datetime.fromisoformat(case["at"].replace("Z", "+00:00"))
    hours_at = max(0.0, (at_dt.timestamp() * 1000 - release) / 3_600_000)
    x_at = math.log(1 + hours_at)
    target = case.get("target")
    if target:
        x_target = math.log(1 + _hours_between(release, target))
    else:
        x_target = x_at + math.log(1 + case["horizon_minutes"] / 60.0)
    d_days = max(0.0, case["horizon_minutes"] / 1440.0)
    rw = 0.5 * math.sqrt(d_days)

    last = case["history"][-1]
    # Reads in the last 24h, as recorded (no dedup — literal reading of the rules).
    x_cut = math.log(1 + max(0.0, hours_at - 24))
    out = {}
    for i, t in enumerate(case["trophies"]):
        pts = []
        for r in case["history"]:
            x = math.log(1 + _hours_between(release, r["at"]))
            if x < x_cut:
                continue
            rate = r["rates"][i]
            if isinstance(rate, (int, float)) and r["players_tracked"] > 0:
                pts.append((x, logit(rate)))
        fit = _ols([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 3 else None
        lr = last["rates"][i]
        if not isinstance(lr, (int, float)):
            lr = 1.0
        if fit is None:
            m = logit(lr)
            sigma = rw
        else:
            a, b, resid_sd = fit
            m = a + b * x_target
            sigma = math.sqrt(resid_sd ** 2 + rw * rw)
        q = [round4(clamp(inv_logit(m + z * sigma), 0, 100)) for z in Z]
        for j in range(1, 5):
            q[j] = max(q[j], q[j - 1])
        out[t["id"]] = q
    return out


def load_case(path):
    with open(path) as f:
        return json.load(f)


def load_forecast(path):
    with open(path) as f:
        o = json.load(f)
    return {t["id"]: t["q"] for t in o["trophies"]}


if __name__ == "__main__":
    # usage: score.py <case.in> <forecast.out> <answer.json>
    case = load_case(sys.argv[1])
    fc = load_forecast(sys.argv[2])
    with open(sys.argv[3]) as f:
        ans = {int(k): v for k, v in json.load(f).items()}
    print("ours       %.6f" % pinball_loss(ans, fc))
    print("persistence %.6f" % pinball_loss(ans, house_baseline(case)))
    print("curve      %.6f" % pinball_loss(ans, curve_baseline(case)))
