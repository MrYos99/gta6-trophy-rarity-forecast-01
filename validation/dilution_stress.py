#!/usr/bin/env python3
"""Dilution stress scenarios — SEPARATE from the 12-scenario harness.

Same generator as validation/gen.py, but with slow multiplier decay (0.60) at
sparse cadences (8h, 12h): few refreshes have happened by the checkpoint, so
the base is still growing hard AFTER it (N multiplies ~x1.7 at 8h cadence and
~x4 at 12h cadence between checkpoint and target) while rates stay in a
scorable range. The 12 harness scenarios cannot test this — in all of them N
is essentially saturated by the checkpoint (future log-growth ~0.002–0.01) —
so a model that merely fits the recent rate trend looks identical to one that
models dilution. These scenarios separate them: the rate-fit model's
SLOPE_MIN = 0 floor cannot forecast the dilution-driven fall, while the
dilution model reads it off the jump pattern.

Usage:
  python3 validation/dilution_stress.py <outdir>   # writes <scenario>/<id>.in + answer.json
  python3 validation/run.py --main <main.ts> --out <dir> --cases <outdir> \
      --skip-gen --scenarios s-cad8h-linear-d060 s-cad8h-saturating-d060 \
      s-cad12h-linear-d060 s-cad12h-saturating-d060
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen as genmod


def main(outdir):
    with open(os.path.join(genmod.KIT_INPUT, "p00.in")) as f:
        p00 = json.load(f)
    trophies = p00["trophies"]
    base_rates = p00["history"][0]["rates"]
    assert len(trophies) == 55 and len(base_rates) == 55
    for cad in (8, 12):
        for fam in ("linear", "saturating"):
            name = f"s-cad{cad}h-{fam}-d060"
            d = os.path.join(outdir, name)
            os.makedirs(d, exist_ok=True)
            case, answer = genmod.build_case(trophies, base_rates, cad, fam, name, 0.60)
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
