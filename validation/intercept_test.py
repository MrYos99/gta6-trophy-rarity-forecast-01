#!/usr/bin/env python3
"""Minimal constructed test for the intercept fix (main.ts line 233).

Three trophies with EXACT linear logit trends plus small deterministic noise,
constant tracked base (no dilution), reads every 6h from h=6 to h=60, target
4.5 days out — the final's shape, but with unbiased slopes, so the only thing
being tested is the intercept geometry: a shrunk slope must pivot around the
fit's centroid (mx, my), not around (0, fit.a).

  A: logit = -4.0 + 1.5 * ln(1+h)   (steep; shrinkage bites: class mean ~0.6)
  B: logit = -2.0 + 0.2 * ln(1+h)
  C: logit = -2.5 + 0.2 * ln(1+h)

Prints, for the OLD code (git HEAD) and the NEW code (worktree): each
trophy's median, the TRUE rate at target, and the mean pinball loss vs truth.
The fixed code must be closer to truth on A (where shrinkage bites) and no
worse on B/C.
"""

import json
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import score as sc

RELEASE = "2026-11-19T00:00:00Z"


def inv_logit_pct(l):
    return 100.0 / (1.0 + math.exp(-l))


def build():
    rng = random.Random(7)
    trends = [(-4.0, 1.5), (-2.0, 0.2), (-2.5, 0.2)]
    trophies = [
        {"id": 1, "name": "T1", "description": "Do the first thing.",
         "tier": "bronze", "points": 15, "secret": False},
        {"id": 2, "name": "T2", "description": "Do the second thing.",
         "tier": "bronze", "points": 15, "secret": False},
        {"id": 3, "name": "T3", "description": "Do the third thing.",
         "tier": "bronze", "points": 15, "secret": False},
    ]
    history = []
    for h in range(6, 61, 6):
        rates = []
        for a, b in trends:
            y = a + b * math.log(1 + h) + rng.uniform(-0.05, 0.05)
            rates.append(round(max(0.01, min(99.99, inv_logit_pct(y))), 2))
        from datetime import datetime, timedelta, timezone
        at = datetime(2026, 11, 19, tzinfo=timezone.utc) + timedelta(hours=h)
        history.append({"slot": at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                        "at": at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                        "players_tracked": 10000,
                        "trophies_earned": 100,
                        "rates": rates})
    case = {
        "id": "intercept",
        "at": "2026-11-21T12:00:00.000Z",
        "horizon_minutes": 6480,
        "target": "2026-11-26T00:00:00.000Z",
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Test", "platform": "PS5", "release_date": "2026-11-19"},
        "trophies": trophies,
        "history": history,
    }
    truth = {}
    for i, (a, b) in enumerate(trends):
        truth[i + 1] = inv_logit_pct(a + b * math.log(1 + 168))
    return case, truth


def run(main_ts, case, outdir):
    os.makedirs(outdir, exist_ok=True)
    indir = os.path.join(outdir, "in")
    os.makedirs(indir, exist_ok=True)
    with open(os.path.join(indir, "intercept.in"), "w") as f:
        json.dump(case, f)
    env = dict(os.environ, MAKEFUN_INPUT_DIR=indir,
               MAKEFUN_OUTPUT_DIR=outdir)
    r = subprocess.run(["node", main_ts], env=env, capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1000:])
    return sc.load_forecast(os.path.join(outdir, "intercept.out"))


def main():
    case, truth = build()
    old = subprocess.run(["git", "show", "HEAD:main.ts"], cwd=REPO,
                         capture_output=True, text=True).stdout
    old_path = "/tmp/val/intercept-old-main.ts"
    with open(old_path, "w") as f:
        f.write(old)
    new_path = os.path.join(REPO, "main.ts")
    print("truth at target:",
          {k: round(v, 4) for k, v in truth.items()})
    for label, path in (("OLD (buggy intercept)", old_path),
                        ("NEW (centroid pivot)", new_path)):
        fc = run(path, case, f"/tmp/val/intercept-{label.split()[0].lower()}")
        loss = sc.pinball_loss(truth, fc)
        print(f"\n{label}: mean pinball vs truth = {loss:.4f}")
        for tid in (1, 2, 3):
            print(f"  id{tid}: median {fc[tid][2]:8.4f}  truth {truth[tid]:8.4f}  "
                  f"logit err {sc.logit(truth[tid]) - sc.logit(fc[tid][2]):+.3f}")


if __name__ == "__main__":
    main()
