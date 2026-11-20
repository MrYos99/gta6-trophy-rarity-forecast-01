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

## Dilution variant (this commit)

This commit replaces the direct rate trend with a **count decomposition**:
earners E(t) and the tracked base N(t) are fitted separately in log-time, and
the rate is r = E/N. The rate dilutes because N grows faster than E after
launch — modelling the two counts separately is the mechanism, not a
re-parametrisation of the same trend.

- **Earners.** WLS of ln(E + 0.5) on ln(1 + hours), earner-weighted, slope
  shrunk toward the class mean, clamped [0, 1.0], centroid pivot. E is
  reconstructed from the printed rate × players_tracked. A 0.00% print is
  censored, not zero: it is reconstructed at the bin midpoint, E = 0.0025% ×
  N (a print of 0.00 means the true rate is in [0, 0.005%)). Rate-fit weights
  are unchanged.
- **Base.** Jump-decay model, not a linear fit: log-jumps
  g_k = ln(N_k / N_{k-1}) decay geometrically with ratio ρ = median of the
  last three consecutive ratios (clamped [0.2, 0.85], prior 0.5); cadence =
  median gap between refreshes (first gap excluded); future jumps are summed
  to the last refresh at or before the target — the answer is the page state
  at that refresh, since N is a step function. This is what handles the trap
  that killed the earlier count version: the base's growth is convex in
  log-time, so a linear base fit under-forecasts N ~2× at a 4.5-day horizon
  and that error divides straight into every rate. A jump model has no
  log-time linearity to violate.
- **Count-consistent median.** median = logit(last) + 0.65 ×
  (logit(r_line at answering refresh) − logit(r_line at last refresh)), where
  r_line comes from the same E line and N model at both instants, so the level
  residual cancels and only the move is used. At zero horizon (target = last
  refresh) the forecast is the last read exactly — verified on a synthetic
  case (54/55 medians identical; the 100% trophy reads 99.99, the output
  clamp). The spread adds an N-uncertainty term: each extrapolated jump is
  uncertain by ~half its size, summed in quadrature.
- **Kept from the baseline:** deduplication of frozen reads, earner-weighted
  fits, class shrinkage, frozen-page rule (×6 base / ×3.5 earners, 100%
  exempt), and the direct rate fit as fallback when the earner fit or the N
  model is unavailable.
- **Harness (12 scenarios):** worst-case 0.2512, wins 11/12, mean 0.1960 —
  vs the baseline's 0.2861 / 11/12 / 0.2076. The variant wins worst-case by
  0.035 and mean by 0.012; it loses only s-cad2h-saturating (0.2506 vs
  persistence 0.2329, margin −0.0177 — a smaller loss than the baseline's
  −0.0296 there). It is worse than the baseline on three scenarios
  (d030-sat 0.1524 vs 0.1473, d060-lin 0.1508 vs 0.1458, d060-sat 0.1176 vs
  0.0778) — all fast-dilution worlds where floor-pinned trophies dominate
  the loss. Full table in `validation/README.md`.
- **Practice:** identical to the baseline (0.6937 / 0.6945 under the same
  ASSUMED answer; medians id1 0.3675, id2 99.99, id3 0.7292) — the frozen
  rule fires on the practice history, so the dilution path is not exercised
  there.
- **Limits.** The 12 scenarios have N essentially saturated by the
  checkpoint, so they under-test dilution; see the separate stress scenarios
  in `validation/README.md` (8h/12h cadence, decay 0.60, N still growing
  ~×1.7–×4 after checkpoint): the variant wins the 12h pair big (0.0923 /
  0.1350 vs baseline 0.1835 / 0.5810) but loses the 8h pair (0.1703 / 0.1552
  vs 0.1042 / 0.1285). The 8h loss is floor-pinned trophies: a printed rate
  stuck at the 0.01 floor while N grows fakes earner growth in the
  reconstruction, and no estimator can separate the two from a flat print.
  A window-base-growth correction (subtract the window's own ln N slope from
  the earner slope) was tried and REJECTED with evidence: it fixed the
  stress 8h pair (3/4 wins) but collapsed the main harness worst-case from
  0.2512 to 0.4255 — for resolved rates the base growth is already netted
  out through the rate's own movement, so the correction double-counts it.
  Two other fixes were also tried and rejected: a pinned-rate guard (fixed
  d060, killed cad4h-sat/cad8h-linear gains) and dropping 0.00 reads from
  the E fit (all-zero trophies lost their anchor and floated up). The
  midpoint censoring above is the fix that survived.

## Why not a count decomposition? (history)

An earlier version forecast earners and the tracked base separately (r = E/N)
and was dropped after synthetic validation: the base's growth is convex in
log-time, so any linear base fit under-forecasts it ~2× at a 4.5-day horizon,
and that error divides straight into every rate. The dilution variant above
is the same idea with the trap handled — a jump-decay base model instead of
a linear base fit — and it now beats the direct trend on worst-case.

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
  pre-fix code's worst-case 0.6098 and 2/12 wins on the same scenarios. The
  dilution variant in this commit scores worst-case 0.2512, 11/12, mean 0.1960
  on the same 12. Full tables, tuning sweeps, and the limits of these synthetic
  numbers are in `validation/README.md` — read the limits before trusting the
  table.
- **Edge cases**: empty history, single read, null rates, zero players_tracked — all
  produce valid output, never crash (a per-case fallback emits wide persistence).
  Re-verified 2026-11-20 on the dilution variant (5 crafted cases, all 55 rows
  valid, quantiles non-decreasing). Count-consistency verified at zero horizon:
  target = last refresh gives medians equal to the last read (54/55; the 100%
  trophy reads 99.99, the output clamp).
- **Determinism**: two runs over the practice inputs produce byte-identical outputs
  (verified 2026-11-20 on the tuned code: 55 rows, quantiles non-decreasing).
  Re-verified on the dilution variant, 2026-11-20.
- **Kit check**: 4 passed, 1 failed, 14 skipped, 5 checked at submission —
  measured 2026-11-20 on the dilution-variant worktree (before commit). The
  failure is `transcripts_present`, EXPECTED before bundling (no bundle exists
  yet — bundling happens at seal time, not build time); the 14 skips are all
  transcript-dependent checks. The 5 submission-time checks (repo_public,
  repo_owner, repo_created_after_open, commit_in_repo, duplicate_snapshot)
  need the public GitHub repo + push at submit time. The 4 passes:
  competition_open, snapshot, license_osi, entrypoint. Note: the earlier
  check (14 passed, 0 failed, 5 skipped, 5 at submission, against commit
  483a812 + the existing bundle) predates the variant — re-run after the
  parent re-bundles the variant session.

## License

MIT (see `LICENSE`).
