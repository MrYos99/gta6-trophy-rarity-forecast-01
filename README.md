# GTA VI Trophy Rarity Forecaster

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast, for each of
Grand Theft Auto VI's 55 PlayStation 5 trophies, its earn rate on Exophase at
2026-11-26T00:00:00Z — five quantiles (5/25/50/75/95%) per trophy, scored by mean
pinball loss on the logit scale.

## Program

`main.ts` (Node 22, standard library only, no network). Reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `<id>.out` per case to
`$MAKEFUN_OUTPUT_DIR` (else `/output`):

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Output per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}` — one row
per trophy, quantiles non-decreasing, each 0–100. Deterministic: pure arithmetic, no
randomness, no clock. Two runs write the same bytes. Runs in ~1s for both practice
cases (cap is 8s per case).

## Model

Shrunk log-time trend on the rate itself, in logit space:

```
logit(r) = a + b * ln(1 + hours since release)
```

- **Local window.** The fit uses only the most recent reads (last 24h when they hold
  4+ distinct reads, else last 48h, else all). The local slope is what extrapolates;
  the launch transient would bend a longer fit, and logit(r) is concave in log-time,
  so a long window's average slope overstates the future.
- **Deduplication.** Exophase refreshes in jumps, so long frozen stretches carry no
  information and would fake certainty. A read is kept only when it changes.
- **Earner-weighted.** Each read is weighted by its earner count — a rate's logit is
  only as precise as its numerator.
- **Shrinkage.** Each trophy's slope is shrunk toward its class mean (story / activity /
  grind, classified from the trophy's own description, plus a platinum class), because a
  handful of noisy, quantised reads cannot support a free slope per trophy. Slopes are
  clamped to [-0.5, 3.5]: a rate can fall when the tracked base outgrows its earners
  (the platinum does exactly this).
- **Damped median.** The fitted line at the target is damped 0.85× back toward the
  last read.
- **Honest spread.** σ = fit residual + slope uncertainty × extrapolation distance +
  a 0.5·√days random-walk term, floored at 0.6 and capped at 2.5 logit units. A
  4.5-day launch-week extrapolation is genuinely uncertain, and pinball loss punishes
  a too-narrow band far more than a slightly-too-wide one.
- **Frozen pages.** When the history holds fewer than 3 distinct reads and the horizon
  is short, there is no trend signal at all — the next read is a refresh jump. The
  median then assumes the tracked base multiplies (~4×, a launch-week prior) while
  earners stay flat, and the spread is widened. Gated to short horizons, so it never
  fires on the final case (6,510 minutes).

## Why not a count decomposition?

An earlier version forecast earners and the tracked base separately (r = E/N). It was
dropped after synthetic validation: the base's growth is convex in log-time, so any
linear base fit under-forecasts it ~2× at a 4.5-day horizon, and that error divides
straight into every rate. One extrapolation beats two.

## Validation

- **Practice cases** (frozen launch-night history): under the E-frozen-across-refresh
  assumption the model scores mean pinball ≈ 0.35 vs the house persistence baseline's
  ≈ 1.88. The frozen-page rule moves medians down by the base jump, in the right
  direction and inside the band.
- **Synthetic finals**: four simulated final cases (different growth rates, base
  growth, and a different curve family — logistic in log-time) with refresh jumps,
  quantisation, and 2 days of history. The model beats both house baselines on all
  four (e.g. base case: 0.244 vs persistence 0.305 vs curve 0.252; logistic shape:
  0.271 vs 0.349 vs 0.287).
- **Edge cases**: empty history, single read, null rates, zero players_tracked — all
  produce valid output, never crash (a per-case fallback emits wide persistence).
- **Determinism**: two runs over the practice inputs and over the edge cases produce
  byte-identical outputs.
- **Kit check**: `competition_open`, `snapshot`, `license_osi`, `entrypoint` all pass.
  The only failing check is `transcripts_present`, which is satisfied at submit time
  when the parent bundles this session.

## License

MIT (see `LICENSE`).
