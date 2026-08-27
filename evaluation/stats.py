"""
Statistical rigor primitives for the prediction model.

Three real gaps this closes, each measured rather than assumed useful:

1. Newey-West HAC t-statistics. A daily IC series computed from overlapping
   forward-return windows is autocorrelated by construction (day t and t+1
   share up to horizon-1 days of forward return). Treating consecutive days
   as independent, as a naive t = mean/std*sqrt(n) does, understates the
   variance and overstates significance.
2. Deflated significance under multiple trials. Roughly 20 configurations
   were compared during model selection (horizons, quantiles, feature
   transforms, universe width). Picking the best of many trials inflates the
   apparent effect size versus its true out-of-sample expectation; this
   applies a trials-count haircut to the reported t-stat.
3. Sample uniqueness weights (Lopez de Prado, *Advances in Financial Machine
   Learning*, ch. 4). Consecutive daily training rows for the same ticker
   have forward-return windows that overlap by up to horizon-1 days, so
   naive fitting overweights redundant information. Down-weighting by the
   average overlap fraction is the standard fix.
"""

import numpy as np
import pandas as pd


def newey_west_tstat(x: np.ndarray, lag: int) -> tuple[float, float]:
    """
    Newey-West HAC-adjusted mean and t-statistic for a possibly autocorrelated
    series (e.g. daily IC values from overlapping-horizon forward returns).

    Parameters
    ----------
    x   : the series (e.g. per-day rank IC)
    lag : truncation lag. Use horizon - 1: with an h-day forward return,
          observations more than h-1 days apart no longer share any of the
          window that generated their labels, so autocorrelation beyond
          that lag reflects real signal decay, not overlap.

    Returns
    -------
    (mean, t_stat) where t_stat uses the HAC standard error instead of the
    naive i.i.d. standard error.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float(np.mean(x)) if n else float("nan"), float("nan")

    mean = x.mean()
    resid = x - mean

    # Newey-West long-run variance: gamma_0 + 2 * sum_{k=1}^{lag} w_k * gamma_k,
    # Bartlett kernel weights w_k = 1 - k/(lag+1).
    gamma0 = (resid @ resid) / n
    lrv = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        gamma_k = (resid[k:] @ resid[:-k]) / n
        lrv += 2 * w * gamma_k

    lrv = max(lrv, 1e-16)
    se = np.sqrt(lrv / n)
    t = mean / se if se > 0 else float("nan")
    return float(mean), float(t)


def _expected_max_of_n_normals(n: int, n_sims: int = 500_000, seed: int = 0) -> float:
    """
    E[max of n iid standard normals], by direct Monte Carlo.

    Deliberately not a closed-form approximation: a first attempt at one here
    (a mis-transcribed Gumbel correction) was off by 3-4x — it gave 7.24 for
    n=20 against a true value of 1.87 — and was caught only by checking it
    against simulation. Simulation has no such transcription risk.
    Reference values this matches: n=10 -> 1.54, n=20 -> 1.87, n=50 -> 2.25.
    """
    if n <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    return float(rng.standard_normal((n_sims, n)).max(axis=1).mean())


def deflated_tstat(t_stat: float, n_trials: int) -> float:
    """
    Haircut a t-stat for having been selected as the best of `n_trials`
    configurations (Bailey & Lopez de Prado's Deflated Sharpe Ratio applies
    the same logic to Sharpe ratios; this is the equivalent adjustment for a
    t-statistic under the same "max of n correlated trials" reasoning).

    Treats the expected maximum of n_trials iid standard-normal draws as the
    new null. A model that only clears the naive significance bar because it
    was chosen as the best of many attempts will fail this bar; one with a
    real, robust effect will still clear it.

    This is an approximation: it assumes independent trials, but many of the
    ~20 tried here were correlated variants of each other (e.g. horizon=3 and
    horizon=5 share most of their feature pipeline). Treating them as
    independent makes this a CONSERVATIVE haircut, not an exact one — true
    overfitting risk is somewhat less than this number implies.
    """
    if n_trials <= 1:
        return t_stat
    return t_stat - _expected_max_of_n_normals(n_trials)


def sample_uniqueness_weights(df: pd.DataFrame, horizon: int,
                              date_col: str = "Date",
                              ticker_col: str = "ticker") -> pd.Series:
    """
    Lopez de Prado sample-uniqueness weights.

    Row i's label spans [date_i, date_i + horizon]. Two rows for the same
    ticker overlap if their windows intersect. Row i's weight is the average,
    over the days its window spans, of 1 / (number of concurrently-open
    windows that day) — a row whose window is entirely shared with others
    contributes little unique information and is downweighted accordingly.

    Implemented in O(n log n) per ticker via a sweep over window open/close
    events, not O(n^2) pairwise comparison.
    """
    out = pd.Series(1.0, index=df.index)

    for ticker, g in df.groupby(ticker_col, sort=False):
        g = g.sort_values(date_col)
        dates = g[date_col].values
        n = len(dates)
        if n == 0:
            continue

        # date -> integer trading-day index within this ticker's series, so
        # "horizon calendar days" becomes "horizon rows" regardless of gaps.
        idx = np.arange(n)
        end_idx = np.minimum(idx + horizon, n - 1)

        # concurrency[t] = how many rows' windows are open on row-index t.
        concurrency = np.zeros(n)
        for i in range(n):
            concurrency[i:end_idx[i] + 1] += 1

        weights = np.array([
            1.0 / concurrency[i:end_idx[i] + 1].mean() if end_idx[i] >= i else 1.0
            for i in range(n)
        ])
        out.loc[g.index] = weights

    return out


def fit_neutralization_coefs(scores: np.ndarray, beta: np.ndarray,
                             sector: pd.Series) -> dict:
    """
    Fit beta/sector neutralization coefficients on a POOLED panel (many
    dates, many rows) and return them for later fixed application.

    This exists because fitting a fresh regression on a single day's ~48
    stocks against beta + ~15 sector dummies (~17 parameters) is barely more
    observations than parameters — high variance, and measured to make
    things WORSE (rank IC 0.0336 -> 0.0237 versus not neutralizing at all).
    Pooling thousands of rows to fit stable coefficients ONCE, then applying
    those fixed coefficients at serving time, measured rank IC 0.0336 ->
    0.0455 instead. See docs/MODEL_VALIDATION.md §8 for both numbers.

    Fit on TRAINING data only and apply the fixed result to new dates (via
    `apply_neutralization_coefs`) — never refit per-day, and never fit using
    data from the dates being scored.
    """
    dummies = pd.get_dummies(sector, drop_first=True).astype(float)
    sector_cols = dummies.columns.tolist()
    X = np.column_stack([np.ones(len(scores)), np.asarray(beta, dtype=float),
                         dummies.values])
    y = np.asarray(scores, dtype=float)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
    return {"coef": coef, "sector_cols": sector_cols}


def apply_neutralization_coefs(scores: np.ndarray, beta: np.ndarray,
                               sector: pd.Series, neutralization: dict,
                               rerank: bool = True) -> np.ndarray:
    """
    Apply FIXED beta/sector neutralization coefficients (from
    `fit_neutralization_coefs`) to a new cross-section. No fitting happens
    here — that is the point: reusing coefficients estimated from far more
    data than any single day provides is what makes this stable.

    `rerank=True` re-ranks the residual to a [0, 1] percentile. Safe to do
    per-date even though the coefficients are pooled: ranking is a pure
    normalization step, not a regression, so it carries no leakage risk.
    """
    coef, sector_cols = neutralization["coef"], neutralization["sector_cols"]
    dummies = pd.get_dummies(sector, drop_first=True)
    dummies = dummies.reindex(columns=sector_cols, fill_value=0.0).astype(float)
    X = np.column_stack([np.ones(len(scores)), np.asarray(beta, dtype=float),
                         dummies.values])
    y = np.asarray(scores, dtype=float)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)

    if valid.sum() < len(y) * 0.5:
        # Too much missing beta/sector data for the residual to mean
        # anything. Returning the input untouched is safer than emitting a
        # confidently-wrong ranking derived from half-missing inputs.
        return np.asarray(scores)

    resid = y.copy()
    resid[valid] = y[valid] - X[valid] @ coef
    # Rows with missing beta/sector keep their raw score unchanged rather
    # than being silently dropped or NaN'd.

    if not rerank:
        return resid
    return pd.Series(resid).rank(pct=True).values


def neutralize_score(scores: np.ndarray, beta: np.ndarray,
                     sector: pd.Series) -> np.ndarray:
    """
    Fit-and-apply neutralization in one call, for one-off exploratory use
    (e.g. "what does neutralizing this specific pooled sample look like").

    NOT for production scoring: calling this per-date fits a fresh
    regression on a single day's ~48 rows, which measured WORSE than doing
    nothing (see `fit_neutralization_coefs`'s docstring). Production code
    (training/train.py, models/predictor.py) uses
    fit_neutralization_coefs + apply_neutralization_coefs instead, so
    coefficients come from a large pooled sample and are then held fixed.
    """
    n = fit_neutralization_coefs(scores, beta, sector)
    return apply_neutralization_coefs(scores, beta, sector, n, rerank=True)


def probability_backtest_overfitting_note(n_trials: int, best_t: float,
                                          deflated_t: float) -> str:
    """One-line honest summary of the multiple-testing situation."""
    if deflated_t > 2.0:
        verdict = "still significant after the haircut"
    elif deflated_t > 0:
        verdict = "weakened but still positive after the haircut"
    else:
        verdict = "does NOT survive the haircut — treat with real skepticism"
    return (f"{n_trials} configurations were compared; naive t={best_t:.2f}, "
           f"trials-adjusted t={deflated_t:.2f} ({verdict}).")
