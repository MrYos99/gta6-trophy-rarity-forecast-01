#!/usr/bin/env python3
"""Synthetic final-case generator for the trophy rarity forecast harness.

Ground truth is generated as two moving parts, exactly like the real world:
  players tracked N(t): jumps at Exophase refreshes, frozen between them.
  earners E_i(t): grow continuously per trophy class.
The page the model sees is frozen between refreshes and jumps at a refresh —
the same shape as the house's real recording (launch night: frozen at 638
tracked from 23:48Z to 04:00Z, then refreshed at 04:30Z to 3,868).

Calibrated anchors (from RULES.txt + the practice cases, not invented):
  - First refresh: 04:30Z on 19 Nov, N 638 -> 3,868 (x6.06), every rate but
    the one at 100% lower. So the first multiplier m1 = 6.06.
  - Later multipliers decay: m_k = 1 + 5.06 * 0.45^(k-1) (k = refresh index).
    Total N growth converges to ~x100 over the first ~15 refreshes.
  - The trophy at 100% (id 2, "Just Like Paradise") stayed at 100% across the
    x6 refresh, so its earners grow WITH the base: E = N always, rate 100%.
  - Launch-night rates are the real ones from p00.in (E0_i = rate_i * 638).

Two truth families (the trend path that decides the final has never been scored
on real data — both must be survived):
  - linear:    E_i(t) = E0_i * (1 + t / h_i)          (earners keep growing)
  - saturating: E_i(t) = E0_i * (1 + A_i*(1-exp(-t/h_i)))  (earner growth stalls;
    the true logit-rate curve is concave in log-time, so a linear-in-log-time
    extrapolation over 4.5 days overshoots)

Earner timescales h_i (hours) by class (same classes as main.ts):
  story 10, activity 30, grind 120, platinum 60.
Saturating amplitudes A_i: story 8, activity 5, grind 3, platinum 1.5.

Scenarios: refresh cadence {2h, 4h, 8h, 12h} x family {linear, saturating}.
Checkpoint = the deadline 2026-11-21T11:30:00Z, target 2026-11-26T00:00:00Z
(horizon 6,510 min — the exact final shape). History: reads every 30 min from
2026-11-18T23:50:00Z, observed page = state at the last refresh <= read time.
Answer = rates at the last refresh <= the target (the page state at target;
the real answering read is the first valid read in 00:00-01:00 on 26 Nov, so
this is at most one refresh stale — documented, not hidden).

Usage: gen.py <outdir>  -> writes <scenario>/<id>.in + <scenario>/answer.json
for the 8 scenarios. Deterministic (no randomness).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

KIT_INPUT = os.path.expanduser(
    "~/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
    "gta6-trophy-rarity-forecast-01-kit/input")
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
FIRST_READ = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
CHECKPOINT = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
FIRST_REFRESH = datetime(2026, 11, 19, 4, 30, tzinfo=timezone.utc)
N0 = 638
M1 = 3868 / 638.0  # 6.063..., the observed first refresh
DECAY = 0.45

H = {"story": 10.0, "activity": 30.0, "grind": 120.0, "platinum": 60.0}
A = {"story": 8.0, "activity": 5.0, "grind": 3.0, "platinum": 1.5}


def trophy_class(t):
    if t["tier"] == "platinum":
        return "platinum"
    d = t.get("description", "") or ""
    import re
    if re.search(r"\d{2,}", d) or re.search(r"\$\d", d):
        return "grind"
    if re.search(r"chapter|final mission", d, re.I):
        return "story"
    return "activity"


def fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def refresh_times(cadence_h):
    """Refresh instants from the first observed one, every cadence_h hours."""
    times = [FIRST_REFRESH]
    t = FIRST_REFRESH
    while t < TARGET + timedelta(hours=2):
        t = t + timedelta(hours=cadence_h)
        times.append(t)
    return times


def build_case(trophies, base_rates, cadence_h, family, case_id, decay=DECAY):
    """Returns (case_dict, answer_dict)."""
    times = refresh_times(cadence_h)
    # N(t): multiplier per refresh index, applied at each refresh.
    n_at = []  # (time, N)
    n = float(N0)
    n_at.append((FIRST_READ, n))
    for k, rt in enumerate(times):
        m = 1.0 + (M1 - 1.0) * (decay ** k)
        n = n * m
        n_at.append((rt, n))

    def N(t):
        v = float(N0)
        for rt, nv in n_at[1:]:
            if rt <= t:
                v = nv
            else:
                break
        return v

    def E(i, t):
        """Earners of trophy i at time t (continuous)."""
        if base_rates[i] >= 99.995:
            return N(t)  # the 100% trophy: earners grow with the base
        cls = trophy_class(trophies[i])
        h = H[cls]
        hours = max(0.0, (t - RELEASE).total_seconds() / 3600.0)
        e0 = base_rates[i] / 100.0 * N0
        if family == "linear":
            return e0 * (1.0 + hours / h)
        return e0 * (1.0 + A[cls] * (1.0 - math.exp(-hours / h)))

    def page_at(t):
        """Observed page state at time t: last refresh <= t (frozen between)."""
        rt = FIRST_READ
        for cand in times:
            if cand <= t:
                rt = cand
            else:
                break
        n = N(rt)
        rates = []
        for i in range(len(trophies)):
            r = 100.0 * E(i, rt) / n if n > 0 else 0.0
            rates.append(round(max(0.0, min(100.0, r)), 2))
        earned = int(round(sum(E(i, rt) for i in range(len(trophies)))))
        return n, rates, earned

    # History: reads every 30 min from FIRST_READ to CHECKPOINT (exclusive of at).
    history = []
    t = FIRST_READ
    while t < CHECKPOINT:
        n, rates, earned = page_at(t)
        history.append({
            "slot": fmt(t),
            "at": fmt(t),
            "players_tracked": int(round(n)),
            "trophies_earned": earned,
            "rates": rates,
        })
        t += timedelta(minutes=30)

    # Answer: page state at the target (last refresh <= target).
    _, ans_rates, _ = page_at(TARGET)
    answer = {trophies[i]["id"]: ans_rates[i] for i in range(len(trophies))}

    case = {
        "id": case_id,
        "at": fmt(CHECKPOINT),
        "horizon_minutes": int((TARGET - CHECKPOINT).total_seconds() // 60),
        "target": fmt(TARGET),
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19",
                 "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
        "trophies": trophies,
        "history": history,
        "notes": ("Synthetic final case for validation only: refresh cadence "
                  f"{cadence_h}h, {family} earner growth."),
    }
    return case, answer


def main(outdir):
    with open(os.path.join(KIT_INPUT, "p00.in")) as f:
        p00 = json.load(f)
    trophies = p00["trophies"]
    base_rates = p00["history"][0]["rates"]
    assert len(trophies) == 55 and len(base_rates) == 55

    scenarios = []
    for cad in (2, 4, 8, 12):
        for fam in ("linear", "saturating"):
            scenarios.append((f"s-cad{cad}h-{fam}", cad, fam, DECAY))
    # N-growth regimes at the reference cadence: slow (0.30) and fast (0.60)
    # multiplier decay. Brackets how fast Exophase's tracked base grows after
    # the observed first x6.06 refresh.
    for dec in (0.30, 0.60):
        for fam in ("linear", "saturating"):
            scenarios.append((f"s-cad4h-{fam}-d{int(round(dec * 100)):03d}", 4, fam, dec))
    for name, cad, fam, dec in scenarios:
        d = os.path.join(outdir, name)
        os.makedirs(d, exist_ok=True)
        case, answer = build_case(trophies, base_rates, cad, fam, name, dec)
        with open(os.path.join(d, f"{name}.in"), "w") as f:
            json.dump(case, f)
        with open(os.path.join(d, "answer.json"), "w") as f:
            json.dump(answer, f)
        distinct = len({(r["players_tracked"], tuple(r["rates"]))
                        for r in case["history"]})
        print(f"{name}: history {len(case['history'])} reads, "
              f"{distinct} distinct, N(last)={case['history'][-1]['players_tracked']}, "
              f"answer id1={answer[1]} id2={answer[2]} id4={answer[4]}")


if __name__ == "__main__":
    main(sys.argv[1])
