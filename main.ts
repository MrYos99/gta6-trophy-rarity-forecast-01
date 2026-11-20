// GTA VI trophy rarity forecaster — gta6-trophy-rarity-forecast-01 (Node 22, standard library only).
//
// Model: shrunk log-time trend on the rate itself + honest spread.
//
// An Exophase rate is earners / players_tracked, and both parts move — but forecasting the
// two parts separately was tried and dropped: the tracked base's growth is convex in
// log-time, so any linear base fit under-forecasts it by ~2x at a 4.5-day horizon, and
// that error divides straight into every rate. One extrapolation beats two. So this
// program fits the rate directly, in logit space, against ln(1 + hours since release):
//
//   logit(r) = a + b * ln(1 + hours)
//
// over the most recent reads only (the last 24 hours when they hold 4+ distinct reads,
// else the last 48, else all). The local slope is what extrapolates: the launch transient
// would bend a longer fit, and logit(r) is concave in log-time, so a long window's average
// slope overstates the future. Reads are deduplicated first — Exophase refreshes in jumps,
// so long frozen stretches carry no information and would fake certainty (a flat stretch
// looks like zero variance). Each read is weighted by its earner count (a rate's logit is
// only as precise as its numerator).
//
// The per-trophy slope is shrunk toward its class mean (story / activity / grind, classified
// from the trophy's own description, plus a platinum class), because a handful of noisy,
// quantised reads cannot support a free slope per trophy. Slopes may go mildly negative
// (a rate can fall when the base outgrows its earners — the platinum does exactly this),
// clamped to [-0.5, 3.5]. The extrapolated move from the last read is damped (x0.85).
// The spread is the fit residual plus slope uncertainty times the extrapolation distance
// plus a random-walk term, floored and capped: a 4.5-day launch-week extrapolation is
// genuinely uncertain, and pinball loss punishes a too-narrow band far more than a
// slightly-too-wide one.
//
// Frozen pages: when the history holds fewer than 3 distinct reads (the page has not moved
// since launch night) and the horizon is short, there is no trend signal at all — the next
// read is a refresh jump. The median then assumes the tracked base multiplies (~6x, the one
// refresh the rules disclose) while earners grow with it (~3.5x), and the spread is widened
// a lot. A trophy already at 100% is exempt: it stayed at 100% across the disclosed x6.06
// refresh. This rule is gated to short horizons, so it can never fire on the final case
// (6,510 minutes).
//
// Deterministic: pure arithmetic, no randomness, no clock, no network. Two runs write the
// same bytes. Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else
// /output), one <id>.out per <id>.in, in name order.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Release instant: GTA VI launched 2026-11-19 (UTC). Overridden per case when the case names it. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** How much of the model's move away from the last read the median keeps.
 * DAMP = 0.65, tuned on the validation harness (validation/README.md): worst-case
 * is flat (0.29–0.30) for 0.5–0.75, and 0.65 sits in the middle of that region. */
const DAMP = 0.65;
/** Prior SD (logit-slope units) of a trophy's slope around its class mean. */
const SLOPE_PRIOR_SD = 0.8;
/** Random-walk spread, logit units per square-root day. */
const SIGMA_RW = 0.5;
/** Spread floor / cap, logit units. */
const SIGMA_MIN = 0.6;
const SIGMA_MAX = 2.5;
/** Multiplier on the computed spread (robust tuning knob). SIGMA_SCALE = 0.6, tuned
 * on the validation harness (validation/README.md): worst-case is flat (0.286–0.288)
 * for 0.5–0.7, 0.321 at 1.0 and 0.394 at 1.2; 0.6 also has the best mean loss
 * (0.208). The raw residual spread overstates the final's uncertainty once the
 * biased slope is clamped (see SLOPE_MIN), so narrower intervals win pinball when
 * the median is well placed. The floor (SIGMA_MIN 0.6) and cap (SIGMA_MAX 2.5)
 * stay as hard bounds. */
const SIGMA_SCALE = 0.6;
/** Distinct reads a window must hold to be used (robust tuning knob).
 * WINDOW_MIN_READS = 3, tuned on the validation harness (validation/README.md):
 * with ~4h refreshes the 24h window holds ~6 reads, so 3 keeps the short window
 * usable on sparse cadences (8h/12h) where 4 would force the longer, flatter 48h
 * window. 2 is worse (worst-case 0.410) and 4 is worse (0.645) — 3 is the
 * sweet spot. */
const WINDOW_MIN_READS = 3;
/** Slope clamp. SLOPE_MIN = 0.0, tuned on the validation harness
 * (validation/README.md) and the single highest-leverage parameter: worst-case
 * 0.645 at -0.5, 0.522 at -0.25, 0.431 at -0.1, 0.378 at 0.0 (all with DAMP 0.85;
 * the tuned DAMP/window/sigma bring it to 0.286). The reason is structural, not a
 * quirk of the scenarios: the observed history is a falling staircase — every
 * refresh dilutes every rate — so a fitted negative slope is a measurement artifact
 * of the refresh pattern, not a trend in player behaviour, and it is typically far
 * steeper than any genuine post-checkpoint fall. Clamping at 0 means the median
 * never forecasts a decrease. The harness includes mildly falling regimes
 * (saturating earners) and the 0.0 floor still wins worst-case there, because the
 * genuine falls are small next to the staircase bias. If the final's tracked base
 * jumps sharply right before the target, this floor is the wrong call — that risk
 * is disclosed in the README, and it is the price of the worst-case win. */
const SLOPE_MIN = 0.0;
const SLOPE_MAX = 3.5;
/** Frozen-page prior: the tracked base multiplies by this at the next refresh.
 * The one refresh the rules disclose is x6.06 (638 -> 3,868 players tracked). */
const FROZEN_BASE_JUMP = 6;
/** Frozen-page prior: earners grow by this across the same refresh, so the
 * median rate moves by FROZEN_EARNER_GROWTH / FROZEN_BASE_JUMP (~x0.58), not
 * last/4. The disclosed refresh took every rate but the 100% one down, and
 * earners plainly grew with the base rather than staying flat. */
const FROZEN_EARNER_GROWTH = 3.5;
/** Extra spread (logit) when the history is frozen. */
const FROZEN_SIGMA_EXTRA = 1.2;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** logit of a percent, clamped to the scorer's 0.01..99.99 window first. */
const logit = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Hours between two ISO timestamps. */
const hoursBetween = (fromMs: number, toStr: string): number =>
  (Date.parse(toStr) - fromMs) / 3_600_000;

/**
 * Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9).
 * Used so any levels list works, not just the contest's five.
 */
function normQuantile(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239e0];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
    -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
    3.754408661907416e0];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Weighted least squares of y on x. Returns null when the fit is undefined. */
function wls(
  xs: number[],
  ys: number[],
  ws: number[],
): { a: number; b: number; seB: number; residSd: number; n: number; mx: number } | null {
  const n = xs.length;
  if (n < 2) return null;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  if (sw === 0) return null;
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += ws[i] * (xs[i] - mx) * (xs[i] - mx);
    sxy += ws[i] * (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx === 0) return null;
  const b = sxy / sxx;
  const a = my - b * mx;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    rss += ws[i] * r * r;
  }
  // Weighted residual SD on the y scale (weights are relative, so normalise by mean weight).
  const meanW = sw / n;
  const residSd = n > 2 ? Math.sqrt(rss / meanW / (n - 2)) : 0;
  const seB = n > 2 ? Math.sqrt((rss / meanW / (n - 2)) / sxx) : Number.POSITIVE_INFINITY;
  return { a, b, seB, residSd, n, mx };
}

/** Growth class from the trophy's own description. Drives the shrinkage target. */
function trophyClass(t: Trophy): 'story' | 'activity' | 'grind' | 'platinum' {
  if (t.tier === 'platinum') return 'platinum';
  const d = t.description ?? '';
  // A count of 10+ or a money amount: a grind (250 stunts, 100 locations, $250,000, 100% completion).
  if (/\d{2,}/.test(d) || /\$\d/.test(d)) return 'grind';
  if (/chapter|final mission/i.test(d)) return 'story';
  return 'activity';
}

/** One trophy's forecast row. */
function forecastTrophy(
  t: Trophy,
  pts: Array<{ x: number; y: number; w: number }>,
  lastRate: number,
  earnersLast: number,
  nBase: number,
  classSlope: number,
  globalResid: number,
  xAt: number,
  xTarget: number,
  days: number,
  frozen: boolean,
): { id: number; q: number[] } {
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const ws = pts.map((p) => p.w);
  const fit = pts.length >= 2 ? wls(xs, ys, ws) : null;

  // Slope: shrink the per-trophy fit toward the class mean; no signal => class mean.
  let slope: number;
  let seSlope: number;
  let resid: number;
  if (fit !== null && pts.length >= 3 && Number.isFinite(fit.seB)) {
    const se = Math.max(fit.seB, 0.3);
    const wOwn = 1 / (se * se);
    const wPrior = 1 / (SLOPE_PRIOR_SD * SLOPE_PRIOR_SD);
    slope = (fit.b * wOwn + classSlope * wPrior) / (wOwn + wPrior);
    seSlope = 1 / Math.sqrt(wOwn + wPrior);
    resid = Math.max(fit.residSd, 0.15);
  } else if (fit !== null && pts.length >= 2) {
    // Two points define a line but say nothing about its error: trust it only halfway.
    slope = 0.5 * fit.b + 0.5 * classSlope;
    seSlope = 1.2;
    resid = Math.max(fit.residSd, 0.3);
  } else {
    slope = classSlope;
    seSlope = 1.2;
    resid = globalResid;
  }
  slope = clamp(slope, SLOPE_MIN, SLOPE_MAX);

  // Median: the fitted line at the target, damped back toward the last read.
  // The shrunk slope must pivot around the fit's centroid (mx, my): the line
  // is fit.a + fit.b*mx + slope*(x - mx). Using the unshrunk intercept with
  // the shrunk slope (fit.a + slope*x) is wrong by (fit.b - slope)*mx, which
  // is large when shrinkage bites and the window sits far from release.
  let medianLogit: number;
  if (fit !== null) {
    const lineAtTarget = fit.a + fit.b * fit.mx + slope * (xTarget - fit.mx);
    medianLogit = logit(lastRate) + DAMP * (lineAtTarget - logit(lastRate));
  } else {
    medianLogit = logit(lastRate) + DAMP * slope * Math.max(0, xTarget - xAt);
  }

  // Frozen page, short horizon: the next read is a refresh jump. The tracked
  // base multiplies (~6x, the one refresh the rules disclose) while earners
  // grow with it (~3.5x), so the median rate is roughly last x 0.58. A trophy
  // already at (or near) 100% cannot fall by dilution — the rules prove the
  // 100% trophy stayed at 100% across the x6.06 refresh — so it is exempt.
  let sigmaExtra = 0;
  if (frozen) {
    if (lastRate >= 99.9) {
      medianLogit = logit(lastRate);
    } else {
      medianLogit = logit(clamp(100 * earnersLast * FROZEN_EARNER_GROWTH /
        (nBase * FROZEN_BASE_JUMP), 0.01, 99.99));
    }
    sigmaExtra = FROZEN_SIGMA_EXTRA;
    resid = Math.max(resid, 0.3);
  }

  const dx = Math.max(0, xTarget - xAt);
  const sigma = clamp(
    SIGMA_SCALE * Math.sqrt(resid * resid + (seSlope * dx) * (seSlope * dx) +
      (SIGMA_RW * Math.sqrt(days)) * (SIGMA_RW * Math.sqrt(days)) + sigmaExtra * sigmaExtra),
    SIGMA_MIN,
    SIGMA_MAX,
  );
  const q = LEVELS.map((tau) =>
    round4(clamp(invLogit(medianLogit + normQuantile(tau) * sigma), 0, 100)));
  // Enforce non-decreasing after rounding (rounding can only tie, but be safe).
  for (let i = 1; i < q.length; i++) q[i] = Math.max(q[i], q[i - 1]);
  return { id: t.id, q };
}

function forecastCase(c: CaseFile): { trophies: Array<{ id: number; q: number[] }> } {
  const releaseMs = c.game?.release_date
    ? Date.parse(`${c.game.release_date}T00:00:00Z`)
    : RELEASE_MS;
  const release = Number.isFinite(releaseMs) ? releaseMs : RELEASE_MS;
  const hoursAt = Math.max(0, hoursBetween(release, c.at));
  const xAt = Math.log(1 + hoursAt);
  const targetMs = Date.parse(c.target);
  const xTarget = Number.isFinite(targetMs)
    ? Math.log(1 + Math.max(0, hoursBetween(release, c.target)))
    : xAt + Math.log(1 + c.horizon_minutes / 60);
  const days = Math.max(0, c.horizon_minutes / 1440);

  // Deduplicate the history: Exophase refreshes in jumps, so long frozen stretches carry
  // no information and would fake certainty. Keep a read only when it changes.
  const kept: Read[] = [];
  let lastSig = '';
  for (const r of c.history) {
    const sig = `${r.players_tracked}|${JSON.stringify(r.rates)}`;
    if (sig !== lastSig) {
      kept.push(r);
      lastSig = sig;
    }
  }
  const frozen = kept.length < 3 && c.horizon_minutes <= 180;
  const lastRead = c.history[c.history.length - 1];
  const nBase = lastRead ? lastRead.players_tracked : 1;

  // Per-trophy points: logit(rate) on ln(1 + hours), weighted by earner count.
  // Window: the last 24h when it holds 4+ distinct reads, else the last 48h, else all.
  const xCut24 = Math.log(1 + Math.max(0, hoursAt - 24));
  const xCut48 = Math.log(1 + Math.max(0, hoursAt - 48));
  const pointsByTrophy: Array<Array<{ x: number; y: number; w: number }>> = c.trophies.map(() => []);
  const lastRateByTrophy: number[] = c.trophies.map(() => 1);
  const earnersLastByTrophy: number[] = c.trophies.map(() => 0);
  for (const r of kept) {
    const x = Math.log(1 + Math.max(0, hoursBetween(release, r.at)));
    c.trophies.forEach((_, i) => {
      const rate = r.rates[i];
      if (typeof rate === 'number' && Number.isFinite(rate) && r.players_tracked > 0) {
        pointsByTrophy[i].push({
          x,
          y: logit(rate),
          w: (rate / 100) * r.players_tracked + 0.5,
        });
      }
    });
  }
  const windowed = pointsByTrophy.map((pts) => {
    const w24 = pts.filter((p) => p.x >= xCut24);
    if (w24.length >= WINDOW_MIN_READS) return w24;
    const w48 = pts.filter((p) => p.x >= xCut48);
    if (w48.length >= WINDOW_MIN_READS) return w48;
    return pts;
  });
  c.trophies.forEach((_, i) => {
    for (let k = c.history.length - 1; k >= 0; k--) {
      const rate = c.history[k].rates[i];
      if (typeof rate === 'number' && Number.isFinite(rate)) {
        lastRateByTrophy[i] = rate;
        earnersLastByTrophy[i] = (rate / 100) * (c.history[k].players_tracked || nBase);
        break;
      }
    }
  });

  // Class slope means from trophies with a usable fit (precision-weighted).
  const classSlopes = new Map<string, number>();
  const classes = ['story', 'activity', 'grind', 'platinum'];
  let globalNum = 0;
  let globalDen = 0;
  let residSum = 0;
  let residCount = 0;
  for (const cls of classes) {
    let num = 0;
    let den = 0;
    c.trophies.forEach((t, i) => {
      if (trophyClass(t) !== cls) return;
      const pts = windowed[i];
      if (pts.length < 3) return;
      const fit = wls(
        pts.map((p) => p.x),
        pts.map((p) => p.y),
        pts.map((p) => p.w),
      );
      if (fit === null || !Number.isFinite(fit.seB)) return;
      const se = Math.max(fit.seB, 0.3);
      const w = 1 / (se * se + SLOPE_PRIOR_SD * SLOPE_PRIOR_SD);
      num += clamp(fit.b, SLOPE_MIN, SLOPE_MAX) * w;
      den += w;
      globalNum += clamp(fit.b, SLOPE_MIN, SLOPE_MAX) * w;
      globalDen += w;
      residSum += Math.max(fit.residSd, 0.15);
      residCount += 1;
    });
    classSlopes.set(cls, den > 0 ? num / den : NaN);
  }
  const globalSlope = globalDen > 0 ? globalNum / globalDen : 0;
  for (const cls of classes) {
    if (!Number.isFinite(classSlopes.get(cls) as number)) classSlopes.set(cls, globalSlope);
  }
  const globalResid = residCount > 0 ? residSum / residCount : 0.4;

  const trophies = c.trophies.map((t, i) =>
    forecastTrophy(
      t,
      windowed[i],
      lastRateByTrophy[i],
      earnersLastByTrophy[i],
      nBase,
      classSlopes.get(trophyClass(t)) as number,
      globalResid,
      xAt,
      xTarget,
      days,
      frozen,
    ),
  );
  return { trophies };
}

function safeForecast(c: CaseFile): { trophies: Array<{ id: number; q: number[] }> } {
  try {
    return forecastCase(c);
  } catch {
    // Never crash a case: fall back to wide persistence around the last read.
    const last = c.history[c.history.length - 1];
    const trophies = c.trophies.map((t, i) => {
      const r = last && typeof last.rates[i] === 'number' ? (last.rates[i] as number) : 1;
      const m = logit(r);
      const q = LEVELS.map((tau) => round4(clamp(invLogit(m + normQuantile(tau) * 2.0), 0, 100)));
      for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
      return { id: t.id, q };
    });
    return { trophies };
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const out = safeForecast(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
