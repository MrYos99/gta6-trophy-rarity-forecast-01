#!/usr/bin/env python3
"""Run the validation harness: generate synthetic finals, run a main.ts variant
on each, score ours + both house baselines, report per-scenario and worst-case.

Usage:
  run.py [--main <path/to/main.ts>] [--out <dir>] [--override NAME=val ...]
         [--scenarios s-cad4h-linear,...] [--skip-gen]

Defaults: --main = the repo's main.ts (this entry, as committed + worktree).
Overrides edit `const NAME = <number>;` lines in a scratch copy of main.ts
(the repo is never touched); e.g. --override SIGMA_SCALE=0.8.

The runner also scores the practice cases p00/p01 under the ASSUMED answer
(base x6.06, earners flat) — labelled ASSUMED, because the real practice
answers are not in the kit. Those numbers are a smoke check only, never a
tuning target.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
KIT_INPUT = os.path.expanduser(
    "~/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
    "gta6-trophy-rarity-forecast-01-kit/input")
sys.path.insert(0, HERE)
import gen as genmod
import score as sc

SCENARIOS = [f"s-cad{c}h-{f}" for c in (2, 4, 8, 12)
             for f in ("linear", "saturating")] + \
            [f"s-cad4h-{f}-d{d}" for d in ("030", "060")
             for f in ("linear", "saturating")]


def run_model(main_ts, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    env = dict(os.environ, MAKEFUN_INPUT_DIR=input_dir,
               MAKEFUN_OUTPUT_DIR=output_dir)
    r = subprocess.run(["node", main_ts], env=env, capture_output=True,
                       text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"node main.ts failed: {r.stderr[-2000:]}")
    return output_dir


def variant_main(main_ts, overrides, workdir):
    """Copy main.ts to workdir, applying const overrides. Returns the path."""
    os.makedirs(workdir, exist_ok=True)
    with open(main_ts) as f:
        src = f.read()
    for name, val in overrides.items():
        pat = re.compile(rf"(const {re.escape(name)}\s*=\s*)(-?[0-9.]+)(;)")
        src, n = pat.subn(rf"\g<1>{val}\g<3>", src)
        if n != 1:
            raise ValueError(f"override {name}: pattern not found exactly once")
    dst = os.path.join(workdir, "main.ts")
    with open(dst, "w") as f:
        f.write(src)
    return dst


def score_dir(case_path, out_path, answer_path):
    case = sc.load_case(case_path)
    fc = sc.load_forecast(out_path)
    with open(answer_path) as f:
        ans = {int(k): v for k, v in json.load(f).items()}
    return {
        "ours": sc.pinball_loss(ans, fc),
        "persistence": sc.pinball_loss(ans, sc.house_baseline(case)),
        "curve": sc.pinball_loss(ans, sc.curve_baseline(case)),
    }


def assumed_practice_answer(case):
    """p00/p01 answers under the flat-earner x6.06 assumption (ASSUMED, not real)."""
    last = case["history"][-1]
    n0 = last["players_tracked"]
    ans = {}
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i]
        if r is None:
            ans[t["id"]] = 1.0
        elif r >= 99.995:
            ans[t["id"]] = 100.0
        else:
            ans[t["id"]] = round(max(0.01, r * n0 / 3868.0), 2)
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=os.path.join(REPO, "main.ts"))
    ap.add_argument("--out", default="/tmp/val/harness")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--cases", default=None,
                    help="case directory (default: <out>/cases)")
    args = ap.parse_args()

    overrides = {}
    for o in args.override:
        k, v = o.split("=", 1)
        overrides[k] = v
    scen = args.scenarios or SCENARIOS
    root = args.out
    cases_dir = args.cases or os.path.join(root, "cases")
    if not args.skip_gen:
        shutil.rmtree(cases_dir, ignore_errors=True)
        os.makedirs(cases_dir, exist_ok=True)
        # Generate all scenarios (cheap), then score the requested subset.
        r = subprocess.run([sys.executable, os.path.join(HERE, "gen.py"),
                            cases_dir], capture_output=True, text=True)
        print(r.stdout, end="")
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            sys.exit(1)

    workdir = os.path.join(root, "variant")
    shutil.rmtree(workdir, ignore_errors=True)
    main_ts = variant_main(args.main, overrides, workdir) if overrides \
        else args.main
    print(f"model: {main_ts} overrides={overrides or 'none'}")

    rows = []
    for name in scen:
        d = os.path.join(cases_dir, name)
        outd = os.path.join(root, "outs", name)
        run_model(main_ts, d, outd)
        s = score_dir(os.path.join(d, f"{name}.in"),
                      os.path.join(outd, f"{name}.out"),
                      os.path.join(d, "answer.json"))
        rows.append((name, s))
        print(f"{name:22s} ours {s['ours']:.4f}  persistence {s['persistence']:.4f}  "
              f"curve {s['curve']:.4f}  margin-vs-best-house {min(s['persistence'], s['curve']) - s['ours']:+.4f}")

    worst = max(r[1]["ours"] for r in rows)
    worst_house = min(min(r[1]["persistence"], r[1]["curve"]) for r in rows)
    print(f"\nWORST-CASE ours: {worst:.4f} (scenario "
          f"{max(rows, key=lambda r: r[1]['ours'])[0]})")
    print(f"Best house loss on that same scenario: "
          f"{min(max(rows, key=lambda r: r[1]['ours'])[1]['persistence'], max(rows, key=lambda r: r[1]['ours'])[1]['curve']):.4f}")
    print(f"Win rate vs best house across scenarios: "
          f"{sum(1 for _, s in rows if s['ours'] < min(s['persistence'], s['curve']))}/{len(rows)}")

    # Practice cases under the ASSUMED answer (smoke check only).
    print("\nPractice under ASSUMED x6.06 flat-earner answer (NOT the real answers):")
    pout = os.path.join(root, "outs", "practice")
    run_model(main_ts, KIT_INPUT, pout)
    for pid in ("p00", "p01"):
        case = sc.load_case(os.path.join(KIT_INPUT, f"{pid}.in"))
        fc = sc.load_forecast(os.path.join(pout, f"{pid}.out"))
        ans = assumed_practice_answer(case)
        print(f"  {pid}: ours {sc.pinball_loss(ans, fc):.4f}  "
              f"persistence {sc.pinball_loss(ans, sc.house_baseline(case)):.4f}  "
              f"curve {sc.pinball_loss(ans, sc.curve_baseline(case)):.4f}")

    with open(os.path.join(root, "results.json"), "w") as f:
        json.dump({"model": main_ts, "overrides": overrides,
                   "scenarios": {n: s for n, s in rows}}, f, indent=2)


if __name__ == "__main__":
    main()
