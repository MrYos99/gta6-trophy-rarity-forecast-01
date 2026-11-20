# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/MrYos99/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/MrYos99/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

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
  3+ distinct reads, else last 48h, else all). The local slope is what extrapolates;
  the launch transient would bend a longer fit, and logit(r) is concave in log-time,
  so a long window's average slope overstates the future.
- **Deduplication.** Exophase refreshes in jumps, so long frozen stretches carry no
  information and would fake certainty. A read is kept only when it changes.
- **Earner-weighted.** Each read is weighted by its earner count — a rate's logit is
  only as precise as its numerator.
- **Shrinkage.** Each trophy's slope is shrunk toward its class mean (story / activity /
  grind, classified from the trophy's own description, plus a platinum class), because a
  handful of noisy, quantised reads cannot support a free slope per trophy. Slopes are
  clamped to [0.0, 3.5]. The floor is 0, not negative: the observed history is a
  falling staircase — every refresh dilutes every rate — so a fitted negative slope is
  a measurement artifact of the refresh pattern, not a trend in player behaviour, and
  it is typically far steeper than any genuine post-checkpoint fall. The median never
  forecasts a decrease.
- **Centroid pivot (fixed 2026-11-20).** The shrunk slope pivots around the fit's
  centroid, not around the unshrunk intercept: `lineAtTarget = fit.a + fit.b * mx +
  slope * (xTarget - mx)`. Pivoting at x = 0 with the unshrunk intercept was a bug
  that accidentally compensated for the staircase-biased slopes; it is fixed, and the
  compensation is replaced by the SLOPE_MIN = 0.0 floor, which does the same job
  deliberately. See validation/README.md.
- **Damped median.** The fitted line at the target is damped 0.65× back toward the
  last read.
- **Honest spread.** σ = 0.6 × sqrt(fit residual² + slope uncertainty² × distance² +
  a 0.5·√days random-walk term), floored at 0.6 and capped at 2.5 logit units. A
  4.5-day launch-week extrapolation is genuinely uncertain, and pinball loss punishes
  a too-narrow band far more than a slightly-too-wide one. The 0.6 multiplier was
  tuned on the validation harness (validation/README.md).
- **Frozen pages.** When the history holds fewer than 3 distinct reads and the horizon
  is short, there is no trend signal at all — the next read is a refresh jump. The
  median then assumes the tracked base multiplies (~6×, the one refresh the rules
  disclose) while earners grow with it (~3.5×), so the median is roughly last × 0.58.
  A trophy already at (or near) 100% is exempt: it stayed at 100% across the disclosed
  x6.06 refresh, and it cannot fall by dilution. Gated to short horizons, so it never
  fires on the final case (6,510 minutes).

## Why not a count decomposition?

An earlier version forecast earners and the tracked base separately (r = E/N). It was
dropped after synthetic validation: the base's growth is convex in log-time, so any
linear base fit under-forecasts it ~2× at a 4.5-day horizon, and that error divides
straight into every rate. One extrapolation beats two.

## Validation

- **Practice cases** (frozen launch-night history): under the ASSUMED answer that
  every rate dilutes by exactly x6.06 with earners flat, the model scores mean
  pinball 0.694 (p00) / 0.695 (p01) vs the house persistence baseline's 1.688.
  That assumed answer is NOT the real practice answer (none exists), and its
  flat-earner premise is itself contradicted by the disclosed refresh, where
  earners plainly grew with the base. The number is a consistency check only,
  not a validation — do not compare it against the pre-fix code's 0.35 under
  the same assumption, because the two codes answer different questions about
  that invented world. The practice medians now follow the frozen rule:
  id1 (platinum, last 0.63%) -> 0.3675, id2 (100%) -> 99.99 (exempt), id3
  (last 1.25%) -> 0.7292.
- **Synthetic finals** (`validation/`, 12 simulated final cases): the trend path
  is scored across refresh cadences (2h/4h/8h/12h), rising and saturating earner
  growth, and fast/slow dilution. The tuned model beats the best house baseline
  on 11 of 12 scenarios, worst-case mean pinball 0.2861 (mean 0.2076), vs the
  pre-fix code's worst-case 0.6098 and 2/12 wins on the same scenarios. Full
  table, tuning sweeps, and the limits of these synthetic numbers are in
  `validation/README.md` — read the limits before trusting the table.
- **Edge cases**: empty history, single read, null rates, zero players_tracked — all
  produce valid output, never crash (a per-case fallback emits wide persistence).
- **Determinism**: two runs over the practice inputs produce byte-identical outputs
  (verified 2026-11-20 on the tuned code: 55 rows, quantiles non-decreasing).
- **Kit check**: 14 passed, 0 failed, 5 skipped, 5 checked at submission —
  measured 2026-11-20 against commit 483a812 + the existing bundle
  (`gta6-trophy-rarity-forecast-01-transcripts.tar.gz`). The 5 submission-time
  checks (repo_public, repo_owner, repo_created_after_open, commit_in_repo,
  duplicate_snapshot) need the public GitHub repo + push at submit time.
  Note: that check predates the fixes in this commit — re-run it after the
  parent re-bundles the fix session.

## License

MIT (see `LICENSE`).
