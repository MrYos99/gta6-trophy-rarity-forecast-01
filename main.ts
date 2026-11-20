// GTA VI trophy rarity forecaster — gta6-trophy-rarity-forecast-01 (Node 22, standard library only).
//
// Model (DILUTION variant): separate earner / tracked-base fits, count-consistent.
//
// An Exophase rate is r = E / N, and the two parts move differently: earners E_i(t)
// grow continuously as players play, while the tracked base N(t) jumps at Exophase
// refreshes and sits frozen between them. The rate dilutes whenever the base
// outgrows a trophy's earners. So this program forecasts the two parts separately
// and divides:
//
//   E_i:  ln(E_i) fit on x = ln(1 + hours since release), per trophy, over the
//         most recent reads only, earner-weighted, slope shrunk toward its class
//         mean and clamped to [0, 1]. (d ln E / dx <= 1 for any E growing at most
//         linearly in time; a saturating earner curve has slope -> 0.)
//   N:    the observed refresh jumps, g_k = ln(N_k / N_{k-1}), decay geometrically:
//         ln g_k is fit linearly in k (ratio rho), the refresh cadence is the median
//         gap between observed refreshes, and future jumps g_K * rho^j are summed
//         up to the last refresh at or before the target. N is a step function, so
//         the answer is the page state at that last refresh, not at the target
//         instant itself. This is the whole point of the variant: ln N in log-time
//         is strongly concave (the base's growth saturates — the first jump is x6,
//         later jumps decay toward x1), so any straight-line ln N fit is badly wrong
//         at a 4.5-day horizon, in either direction depending on the window. The
//         jump-decay model extrapolates the *jumps*, which is what actually moves N.
//   r:    r_line(x) = E_line(x) / N(x), in logit space.
//
// Count-consistency: the median is anchored to the last observed read, and only the
// MOVE comes from the decomposition: median = logit(last) + DAMP * (logit(r_line at
// the answering refresh) - logit(r_line at the last refresh)). At zero horizon the
// forecast is exactly the last read — the forecast counts reconcile with the observed
// counts at the checkpoint by construction, and the E/N lines only supply the change.
// A trophy already at (or near) 100% is exempt (E = N always, it cannot dilute).
//
// The same machinery as the rate-fit model is kept where it still applies:
// deduplication of frozen reads, earner weighting, class shrinkage, honest spread
// (now with an extra N-uncertainty term from the extrapolated jumps), and the frozen
// page rule (fewer than 3 distinct reads, short horizon: next read is a refresh
// jump, base ~x6, earners ~x3.5). When the N jump model cannot be estimated (too few
// distinct base values, no cadence), the program falls back to the direct shrunk
// log-time rate fit.
//
// Deterministic: pure arithmetic, no randomness, no clock, no network. Two runs write
// the same bytes. Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
// (else /output), one <id>.out per <id>.in, in name order.

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
/** Earner-slope clamp (dilution path). E_SLOPE_MAX = 1.0: for earners growing at
 * most linearly in time, d ln E / d ln(1+t) = (1+t)/(h+t) < 1; a saturating earner
 * curve has slope -> 0. The floor is 0 for the same staircase reason as the rate
 * fit: a fitted negative earner slope on refresh-sampled points is noise, not
 * players un-earning trophies. */
const E_SLOPE_MIN = 0.0;
const E_SLOPE_MAX = 1.0;
/** Base jump-decay model (dilution path). RHO is the per-refresh decay ratio of
 * the log-jumps, taken as the median of the last three observed ratios and
 * clamped here: below 0.2 the base is already saturated (future jumps ~0),
 * above 0.85 the jumps barely decay and the base keeps growing. RHO_PRIOR =
 * 0.5 is used when too few jumps exist to form a ratio (the disclosed first
 * refresh x6.06 followed by decaying jumps; 0.5 sits between the early
 * observed ratios ~0.6 and the late asymptote ~0.45 of a geometric decay).
 * N_JUMP_SIGMA_FRAC = 0.5: each extrapolated future log-jump is uncertain
 * by ~half its size, and those uncertainties add in quadrature to the spread. */
const N_RHO_MIN = 0.2;
const N_RHO_MAX = 0.85;
const N_RHO_PRIOR = 0.5;
const N_JUMP_SIGMA_FRAC = 0.5;
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

/** The tracked-base jump model: N extrapolated by decaying refresh jumps. */
interface NModel {
  /** ln N at the last observed refresh. */
  logNLast: number;
  /** ln N at the answering refresh (last refresh at or before the target). */
  logNAnswer: number;
  /** ms of the last observed refresh. */
  lastRefreshMs: number;
  /** ms of the answering refresh. */
  answerMs: number;
  /** Number of future refreshes at or before the target. */
  futureJumps: number;
  /** Spread contribution (logit units) of the extrapolated jumps. */
  sigmaN: number;
}

/**
 * Fit the base jump model from the deduplicated history. Returns null when the
 * jumps cannot be estimated (too few distinct base values, a non-positive jump,
 * or no refresh cadence) — the caller then falls back to the direct rate fit.
 */
function fitNModel(kept: Read[], targetMs: number): NModel | null {
  // Distinct base values with their refresh instants.
  const series: Array<{ tMs: number; n: number }> = [];
  for (const r of kept) {
    const tMs = Date.parse(r.at);
    if (!Number.isFinite(tMs) || !(r.players_tracked > 0)) continue;
    if (series.length === 0 || r.players_tracked !== series[series.length - 1].n) {
      series.push({ tMs, n: r.players_tracked });
    }
  }
  if (series.length < 3) return null;
  const K = series.length - 1;
  const g: number[] = [];
  for (let k = 1; k <= K; k++) {
    const gk = Math.log(series[k].n / series[k - 1].n);
    // The tracked base never shrinks; a non-positive jump is bad data.
    if (!(gk > 0) || !Number.isFinite(gk)) return null;
    g.push(gk);
  }
  const gLast = g[g.length - 1];
  // Decay ratio: median of the last up-to-three consecutive log-jump ratios.
  // Only the most recent ratios govern the next jumps, and early ratios
  // overstate the decay: for geometrically decaying multipliers, ln(1 + c*d^k)
  // ratios approach d from above, so an average over all jumps (or an OLS fit
  // on ln g_k) inherits the early transient and over-forecasts future base
  // growth. With fewer than two jumps there is no ratio; keep the prior.
  let rho = N_RHO_PRIOR;
  const ratios: number[] = [];
  for (let k = 1; k < g.length; k++) ratios.push(g[k] / g[k - 1]);
  const recent = ratios.slice(-3).sort((a, b) => a - b);
  if (recent.length > 0) {
    rho = recent.length % 2 === 1
      ? recent[(recent.length - 1) / 2]
      : (recent[recent.length / 2 - 1] + recent[recent.length / 2]) / 2;
    rho = clamp(rho, N_RHO_MIN, N_RHO_MAX);
  }
  // Cadence: median gap between refreshes, excluding the first gap — it spans
  // the pre-first-refresh freeze, not a refresh interval.
  const gaps: number[] = [];
  for (let k = 2; k <= K; k++) gaps.push(series[k].tMs - series[k - 1].tMs);
  gaps.sort((a, b) => a - b);
  const cadenceMs = gaps.length % 2 === 1
    ? gaps[(gaps.length - 1) / 2]
    : (gaps[gaps.length / 2 - 1] + gaps[gaps.length / 2]) / 2;
  if (!(cadenceMs > 0) || !Number.isFinite(cadenceMs)) return null;
  if (!Number.isFinite(targetMs)) return null;
  const tLast = series[K].tMs;
  let future = 0;
  let sumG = 0;
  let sumVar = 0;
  for (let j = 1; j <= 10000; j++) {
    if (tLast + j * cadenceMs > targetMs) break;
    const gj = gLast * Math.pow(rho, j);
    sumG += gj;
    sumVar += (N_JUMP_SIGMA_FRAC * gj) * (N_JUMP_SIGMA_FRAC * gj);
    future++;
  }
  const logNLast = Math.log(series[K].n);
  return {
    logNLast,
    logNAnswer: logNLast + sumG,
    lastRefreshMs: tLast,
    answerMs: tLast + future * cadenceMs,
    futureJumps: future,
    sigmaN: Math.sqrt(sumVar),
  };
}

/**
 * One trophy's forecast row — dilution path. The median is the last read moved
 * by the E/N decomposition's implied logit move between the last refresh and
 * the answering refresh, damped; the spread adds the N-extrapolation
 * uncertainty. Falls back to the direct rate fit when the earner fit or the
 * base jump model is unavailable.
 */
function forecastTrophy(
  t: Trophy,
  ePts: Array<{ x: number; y: number; w: number }>,
  ratePts: Array<{ x: number; y: number; w: number }>,
  lastRate: number,
  earnersLast: number,
  nBase: number,
  classESlope: number,
  classRateSlope: number,
  globalResidE: number,
  globalResidRate: number,
  xAt: number,
  xTarget: number,
  days: number,
  frozen: boolean,
  nModel: NModel | null,
  xLastRefresh: number,
  xAnswer: number,
): { id: number; q: number[] } {
  const eFit = ePts.length >= 2
    ? wls(ePts.map((p) => p.x), ePts.map((p) => p.y), ePts.map((p) => p.w))
    : null;
  // A 0.00% print is censored, not zero (see forecastCase), so the earner fit
  // may be unavailable even when the rate fit is not.
  if (eFit === null || nModel === null) {
    return forecastTrophyDirect(t, ratePts, lastRate, earnersLast, nBase,
      classRateSlope, globalResidRate, xAt, xTarget, days, frozen);
  }

  // Earner slope: shrink the per-trophy fit toward the class mean. No
  // correction for the base growth inside the window is applied, deliberately:
  // ln E = ln r + ln N, so the window's base growth is already netted out of
  // the earner fit through the rate's own movement — subtracting the ln N
  // slope again double-counts it (verified on the harness: the correction
  // collapses the 12h-linear worst case from 0.25 to 0.43). The correction
  // only helps for flat, censored prints, where the rate cannot move; there
  // the fit is left as-is and the residual bias is disclosed in the README.
  let eSlope: number;
  let seSlope: number;
  let resid: number;
  if (ePts.length >= 3 && Number.isFinite(eFit.seB)) {
    const se = Math.max(eFit.seB, 0.3);
    const wOwn = 1 / (se * se);
    const wPrior = 1 / (SLOPE_PRIOR_SD * SLOPE_PRIOR_SD);
    eSlope = (eFit.b * wOwn + classESlope * wPrior) / (wOwn + wPrior);
    seSlope = 1 / Math.sqrt(wOwn + wPrior);
    resid = Math.max(eFit.residSd, 0.15);
  } else {
    // Two points define a line but say nothing about its error.
    eSlope = 0.5 * eFit.b + 0.5 * classESlope;
    seSlope = 1.2;
    resid = Math.max(eFit.residSd, 0.3);
  }
  eSlope = clamp(eSlope, E_SLOPE_MIN, E_SLOPE_MAX);

  // The decomposition's implied rate at the last refresh and at the answering
  // refresh, both from the same E line (centroid pivot) and N model — so the
  // level residual cancels and only the move is used.
  const eAtLast = eFit.a + eFit.b * eFit.mx + eSlope * (xLastRefresh - eFit.mx);
  const eAtAnswer = eFit.a + eFit.b * eFit.mx + eSlope * (xAnswer - eFit.mx);
  const rLast = clamp(Math.exp(eAtLast - nModel.logNLast) * 100, 0.01, 99.99);
  const rAnswer = clamp(Math.exp(eAtAnswer - nModel.logNAnswer) * 100, 0.01, 99.99);
  let medianLogit = logit(lastRate) + DAMP * (logit(rAnswer) - logit(rLast));

  // Frozen page, short horizon: same rule as the direct fit — the next read is
  // a refresh jump, base ~6x, earners ~3.5x; the 100% trophy is exempt.
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

  const dx = Math.max(0, xAnswer - xLastRefresh);
  const sigma = clamp(
    SIGMA_SCALE * Math.sqrt(resid * resid + (seSlope * dx) * (seSlope * dx) +
      (SIGMA_RW * Math.sqrt(days)) * (SIGMA_RW * Math.sqrt(days)) +
      nModel.sigmaN * nModel.sigmaN + sigmaExtra * sigmaExtra),
    SIGMA_MIN,
    SIGMA_MAX,
  );
  const q = LEVELS.map((tau) =>
    round4(clamp(invLogit(medianLogit + normQuantile(tau) * sigma), 0, 100)));
  for (let i = 1; i < q.length; i++) q[i] = Math.max(q[i], q[i - 1]);
  return { id: t.id, q };
}

/** One trophy's forecast row — direct shrunk log-time rate fit (fallback). */
function forecastTrophyDirect(
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

  // Per-trophy points, two views of the same reads: logit(rate) on ln(1 + hours)
  // (direct-fit fallback) and ln(earners) on ln(1 + hours) (dilution path).
  // Earners are reconstructed count-consistently: E = rate * N / 100 from the
  // read's own rate and players_tracked, with +0.5 smoothing so a 0.00% rate
  // (zero observed earners) still has a finite log. Window: the last 24h when it
  // holds 3+ distinct reads, else the last 48h, else all.
  const xCut24 = Math.log(1 + Math.max(0, hoursAt - 24));
  const xCut48 = Math.log(1 + Math.max(0, hoursAt - 48));
  const ratePtsByTrophy: Array<Array<{ x: number; y: number; w: number }>> = c.trophies.map(() => []);
  const ePtsByTrophy: Array<Array<{ x: number; y: number; w: number }>> = c.trophies.map(() => []);
  const lastRateByTrophy: number[] = c.trophies.map(() => 1);
  const earnersLastByTrophy: number[] = c.trophies.map(() => 0);
  for (const r of kept) {
    const x = Math.log(1 + Math.max(0, hoursBetween(release, r.at)));
    c.trophies.forEach((_, i) => {
      const rate = r.rates[i];
      if (typeof rate === 'number' && Number.isFinite(rate) && r.players_tracked > 0) {
        // Earners are reconstructed from the print's midpoint. A 0.00% print is
        // censored, not zero: it only says the true rate is in [0, 0.005%),
        // whose midpoint is 0.0025% — using 0 (+0.5 smoothing) would fake a
        // huge ln E rise when the trophy first prints 0.01, and dropping the
        // read would erase the (real) information that the rate fell to the
        // floor. The midpoint keeps the fit honest about what a 0.00 print is.
        // (The rate fit keeps the baseline's own weight, rate x N / 100 + 0.5.)
        const earners = (rate / 100) * r.players_tracked;
        const effEarners = ((rate > 0 ? rate : 0.0025) / 100) * r.players_tracked;
        ratePtsByTrophy[i].push({ x, y: logit(rate), w: earners + 0.5 });
        ePtsByTrophy[i].push({ x, y: Math.log(effEarners + 0.5), w: effEarners + 0.5 });
      }
    });
  }
  const windowPts = (pts: Array<{ x: number; y: number; w: number }>) => {
    const w24 = pts.filter((p) => p.x >= xCut24);
    if (w24.length >= WINDOW_MIN_READS) return w24;
    const w48 = pts.filter((p) => p.x >= xCut48);
    if (w48.length >= WINDOW_MIN_READS) return w48;
    return pts;
  };
  const windowedRate = ratePtsByTrophy.map(windowPts);
  const windowedE = ePtsByTrophy.map(windowPts);
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

  // Class slope means from trophies with a usable fit (precision-weighted),
  // computed separately for the earner slopes and the rate slopes.
  const classMeans = (
    windowed: Array<Array<{ x: number; y: number; w: number }>>,
    lo: number,
    hi: number,
  ): { map: Map<string, number>; globalResid: number } => {
    const map = new Map<string, number>();
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
        num += clamp(fit.b, lo, hi) * w;
        den += w;
        globalNum += clamp(fit.b, lo, hi) * w;
        globalDen += w;
        residSum += Math.max(fit.residSd, 0.15);
        residCount += 1;
      });
      map.set(cls, den > 0 ? num / den : NaN);
    }
    const globalSlope = globalDen > 0 ? globalNum / globalDen : 0;
    for (const cls of classes) {
      if (!Number.isFinite(map.get(cls) as number)) map.set(cls, globalSlope);
    }
    return { map, globalResid: residCount > 0 ? residSum / residCount : 0.4 };
  };
  const eMeans = classMeans(windowedE, E_SLOPE_MIN, E_SLOPE_MAX);
  const rateMeans = classMeans(windowedRate, SLOPE_MIN, SLOPE_MAX);

  // Base jump model + the answering refresh (last refresh at or before target).
  const nModel = fitNModel(kept, targetMs);
  const xOfMs = (tMs: number): number =>
    Math.log(1 + Math.max(0, (tMs - release) / 3_600_000));
  const xLastRefresh = nModel !== null ? xOfMs(nModel.lastRefreshMs) : xAt;
  const xAnswer = nModel !== null ? xOfMs(nModel.answerMs) : xTarget;

  const trophies = c.trophies.map((t, i) =>
    forecastTrophy(
      t,
      windowedE[i],
      windowedRate[i],
      lastRateByTrophy[i],
      earnersLastByTrophy[i],
      nBase,
      eMeans.map.get(trophyClass(t)) as number,
      rateMeans.map.get(trophyClass(t)) as number,
      eMeans.globalResid,
      rateMeans.globalResid,
      xAt,
      xTarget,
      days,
      frozen,
      nModel,
      xLastRefresh,
      xAnswer,
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
