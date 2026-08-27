"""
Production training pipeline for the ATLAS prediction model.

Trains a cross-sectional relative-return model and reports honest
out-of-sample metrics via purged walk-forward validation.

Design decisions, each of which was measured rather than assumed
(see docs/MODEL_VALIDATION.md):

* Target is CROSS-SECTIONAL, not absolute. The model predicts whether a stock
  will out-perform the rest of the universe over the next 5 sessions, not
  whether it will rise. Absolute direction at a daily horizon is close to
  unpredictable; relative rank carries measurable signal.
* Horizon is 5 sessions. Measured rank IC was significant at 5 days and
  indistinguishable from zero at 10 and 20.
* Only the tails of the forward-return distribution are used for training.
  The ambiguous middle is dropped, which sharpens the decision boundary.
* Validation is purged: the last `horizon` days of every training window are
  discarded so no training row's forward window overlaps the test period.

Usage
-----
    python -m training.train                 # validate, then fit final model
    python -m training.train --no-validate   # refit only (faster)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.build_panel import load_panel
from evaluation.stats import deflated_tstat, newey_west_tstat
from features.predictive import build_features, feature_columns

logger = logging.getLogger(__name__)

MODEL_DIR = config.MODELS_DIR / "prediction"

# Horizon 3 was chosen by measurement, not preference: rank IC was 0.0283
# (t=5.45) at 3 days, 0.0209 at 5, and indistinguishable from zero at 10.
HORIZON = 3

# Two tail cutoffs rather than one. Averaging models trained at 25% and 35%
# stops the signal depending on an arbitrary threshold.
TAIL_QUANTILES = (0.25, 0.35)

N_FOLDS = 6
SEEDS = (1, 42, 2024)  # averaging several seeds cancels fitting noise

# Roughly how many distinct configurations (horizons, quantiles, feature
# transforms, universe widths, model families, plus a later round testing
# sample-uniqueness weighting / winsorization / a linear diversifier /
# beta-sector neutralization) were compared while arriving at this one. Used
# only to deflate the reported t-stat for selection bias — see
# evaluation.stats.deflated_tstat. Update this if you run a fresh round of
# configuration search; it is a documentation constant, not a tuned one.
N_TRIALS_SEARCHED = 26

# Deliberately small trees. With this signal-to-noise ratio, capacity buys
# overfitting; heavier regularisation measured better than deeper trees.
LGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.015, num_leaves=7, max_depth=3,
    min_child_samples=400, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=20.0,
    n_jobs=-1, verbose=-1,
)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def add_forward_returns(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward return per ticker. The only forward-looking step in the system."""
    out = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    out[f"fwd_{horizon}d"] = out.groupby("ticker")["Close"].transform(
        lambda x: x.shift(-horizon) / x - 1)
    return out.sort_values(["Date", "ticker"]).reset_index(drop=True)


def make_target(df: pd.DataFrame, horizon: int, tail_q: float) -> pd.Series:
    """1 if the stock lands in the top tail that day, 0 if the bottom tail."""
    fwd = df[f"fwd_{horizon}d"]
    grp = df.groupby("Date")[f"fwd_{horizon}d"]
    hi = grp.transform(lambda s: s.quantile(1 - tail_q))
    lo = grp.transform(lambda s: s.quantile(tail_q))

    y = pd.Series(np.nan, index=df.index)
    y[fwd >= hi] = 1.0
    y[fwd <= lo] = 0.0
    return y.where(fwd.notna())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def make_folds(dates: np.ndarray, n_folds: int) -> list[tuple]:
    """Expanding-window folds; the final `n_folds` blocks are the test sets."""
    test_len = len(dates) // (n_folds + 2)
    folds = []
    for k in range(n_folds):
        tr_end = len(dates) - (n_folds - k) * test_len
        if tr_end < 300:
            continue
        folds.append((dates[:tr_end], dates[tr_end:tr_end + test_len]))
    return folds


def fit_ensemble(df: pd.DataFrame, feats: list[str], mask: pd.Series,
                 horizon: int, quantiles=TAIL_QUANTILES, seeds=SEEDS) -> list:
    """
    Fit one model per (tail quantile x seed).

    Varying the quantile varies what counts as a "winner", and varying the
    seed varies the fit. Averaging over both is the variance reduction that
    took rank IC from 0.021 to 0.034.
    """
    import lightgbm as lgb

    models = []
    for q in quantiles:
        y = make_target(df, horizon, q)
        m = mask & y.notna()
        if m.sum() < 1000:
            continue
        X, yy = df.loc[m, feats], y[m]
        for seed in seeds:
            models.append(lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed).fit(X, yy))
    return models


def predict_ensemble(models, X) -> np.ndarray:
    """Average members as cross-sectional ranks so no member's scale dominates."""
    preds = [pd.Series(m.predict_proba(X)[:, 1]).rank(pct=True).values for m in models]
    return np.mean(preds, axis=0)


def walk_forward(df: pd.DataFrame, feats: list[str], horizon: int,
                 n_folds: int) -> tuple[pd.DataFrame, dict]:
    """Purged walk-forward validation. Returns scored test rows and metrics."""
    # Reference labels for AUC only; training labels are built per quantile.
    y_ref = make_target(df, horizon, 0.30)
    dates = np.sort(df["Date"].unique())
    folds = make_folds(dates, n_folds)
    fwd_col = f"fwd_{horizon}d"

    scored, fold_rows = [], []
    for i, (tr_dates, te_dates) in enumerate(folds, 1):
        # PURGE: drop the tail of the training window whose forward returns
        # extend into the test period.
        tr_dates = tr_dates[:-horizon]
        tr_mask = df["Date"].isin(tr_dates)
        te_mask = df["Date"].isin(te_dates)
        if tr_mask.sum() < 1000 or te_mask.sum() < 100:
            continue

        models = fit_ensemble(df, feats, tr_mask, horizon)
        if not models:
            continue
        p = predict_ensemble(models, df.loc[te_mask, feats])
        # Beta/sector neutralization tried and reverted here too — see
        # docs/MODEL_VALIDATION.md §8 and models/predictor.py's comment.

        sub = df.loc[te_mask, ["Date", "ticker", fwd_col]].copy()
        sub["p"] = p
        sub["y"] = y_ref[te_mask].values
        sub["fold"] = i
        scored.append(sub)

        valid = sub.dropna(subset=[fwd_col])
        ics = [spearmanr(x["p"], x[fwd_col]).statistic
               for _, x in valid.groupby("Date") if len(x) >= 10]
        ics = [v for v in ics if np.isfinite(v)]
        lab = valid.dropna(subset=["y"])
        fold_rows.append({
            "fold": i,
            "train_end": str(pd.Timestamp(tr_dates[-1]).date()),
            "test_start": str(pd.Timestamp(te_dates[0]).date()),
            "test_end": str(pd.Timestamp(te_dates[-1]).date()),
            "n_test": int(len(valid)),
            "rank_ic": float(np.mean(ics)) if ics else None,
            "auc": float(roc_auc_score(lab["y"], lab["p"]))
                   if lab["y"].nunique() == 2 else None,
        })
        logger.info("Fold %d [%s..%s]  IC=%.4f  AUC=%.4f", i,
                    fold_rows[-1]["test_start"], fold_rows[-1]["test_end"],
                    fold_rows[-1]["rank_ic"] or np.nan,
                    fold_rows[-1]["auc"] or np.nan)

    all_scored = pd.concat(scored).reset_index(drop=True)
    metrics = summarise(all_scored, horizon)
    metrics["folds"] = fold_rows
    return all_scored, metrics


def summarise(scored: pd.DataFrame, horizon: int) -> dict:
    """Aggregate out-of-sample metrics, including net-of-cost economics."""
    fwd_col = f"fwd_{horizon}d"
    d = scored.dropna(subset=[fwd_col])

    ics = np.array([spearmanr(x["p"], x[fwd_col]).statistic
                    for _, x in d.groupby("Date") if len(x) >= 10])
    ics = ics[np.isfinite(ics)]
    ic_mean = float(ics.mean())

    # Naive t-stat treats each day's IC as an independent draw. It is not:
    # with an h-day forward return, day t and day t+1's IC are computed from
    # windows that share h-1 days, so the IC series is autocorrelated by
    # construction. Newey-West (lag = horizon-1) is the correct standard
    # error; the naive figure is kept only as a labelled point of comparison.
    naive_t = float(ic_mean / (ics.std() + 1e-12) * np.sqrt(len(ics)))
    _, nw_t = newey_west_tstat(ics, lag=max(horizon - 1, 1))

    # This configuration (horizon, quantiles, features) was selected as the
    # best of roughly N_TRIALS_SEARCHED alternatives during development.
    # Reporting only the winning trial's t-stat overstates confidence in the
    # same way p-hacking does; this haircut is the honest number.
    deflated_t = float(deflated_tstat(nw_t, N_TRIALS_SEARCHED))
    ic_t = nw_t  # kept as the headline field name for backward compatibility

    lab = d.dropna(subset=["y"])
    auc = float(roc_auc_score(lab["y"], lab["p"])) if lab["y"].nunique() == 2 else None

    # Long-short on non-overlapping holds, so turnover matches the horizon.
    d = d.copy()
    d["rank"] = d.groupby("Date")["p"].rank(pct=True)
    rebal = np.sort(d["Date"].unique())[::horizon]
    held = d[d["Date"].isin(set(rebal))]
    per = held.groupby("Date").apply(
        lambda x: x.loc[x["rank"] >= 0.9, fwd_col].mean()
                  - x.loc[x["rank"] <= 0.1, fwd_col].mean()).dropna()

    ppy = 252 / horizon
    gross = float(per.mean() * ppy) if len(per) else None
    sharpe = float(per.mean() / (per.std() + 1e-12) * np.sqrt(ppy)) if len(per) else None
    net = {f"net_at_{b}bps": float((per.mean() - 4 * b / 10000) * ppy)
           for b in (5, 10, 20)} if len(per) else {}

    # Decile monotonicity: does mean forward return rise with predicted score?
    d["decile"] = d.groupby("Date")["p"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    dec = d.groupby("decile")[fwd_col].mean()

    return {
        "horizon_days": horizon,
        "n_test_rows": int(len(d)),
        "n_test_days": int(d["Date"].nunique()),
        "rank_ic": ic_mean,
        "rank_ic_tstat": ic_t,               # Newey-West, HAC-adjusted
        "rank_ic_tstat_naive": naive_t,       # i.i.d. assumption; overstated
        "rank_ic_tstat_deflated": deflated_t, # + haircut for ~20 trials searched
        "n_trials_searched": N_TRIALS_SEARCHED,
        "pct_days_ic_positive": float((ics > 0).mean()),
        "auc": auc,
        "long_short_gross_annual": gross,
        "long_short_sharpe": sharpe,
        **net,
        "decile_mean_fwd_return": {int(k): float(v) for k, v in dec.items()},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(validate: bool = True, horizon: int = HORIZON) -> dict:
    panel = load_panel()
    df = build_features(panel)
    df = add_forward_returns(df, horizon)
    feats = feature_columns(df)
    logger.info("Training data: %d rows, %d features, %d tickers",
                len(df), len(feats), df["ticker"].nunique())

    metrics = {}
    if validate:
        logger.info("Running purged walk-forward validation (%d folds)...", N_FOLDS)
        scored, metrics = walk_forward(df, feats, horizon, N_FOLDS)
        logger.info("OOS rank IC=%.4f  naive-t=%.2f  NW-t=%.2f  deflated-t=%.2f  AUC=%s",
                    metrics["rank_ic"], metrics["rank_ic_tstat_naive"],
                    metrics["rank_ic_tstat"], metrics["rank_ic_tstat_deflated"],
                    metrics["auc"])
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        scored.to_csv(MODEL_DIR / "walk_forward_predictions.csv", index=False)

    # Final model: fit on every row with a resolved forward return.
    fit_mask = df[f"fwd_{horizon}d"].notna()
    logger.info("Fitting final ensemble on %d rows...", int(fit_mask.sum()))
    models = fit_ensemble(df, feats, fit_mask, horizon)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_DIR / "lgbm_ensemble.pkl")
    joblib.dump(feats, MODEL_DIR / "feature_cols.pkl")
    joblib.dump({
        "horizon": horizon,
        "tail_quantiles": TAIL_QUANTILES,
        "n_ensemble_members": len(models),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": int(fit_mask.sum()),
        "tickers": sorted(df["ticker"].unique().tolist()),
        # Beta/sector neutralization was measured and rejected (it lowered
        # rank IC in every deployable form); scores are the raw ensemble
        # ranks. See docs/MODEL_VALIDATION.md §8.
        "beta_sector_neutralized": False,
        "validation": {k: metrics.get(k) for k in
                       ("rank_ic", "rank_ic_tstat", "rank_ic_tstat_naive",
                        "rank_ic_tstat_deflated", "auc")} if metrics else None,
    }, MODEL_DIR / "model_meta.pkl")

    if metrics:
        (config.RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
        out = config.RESULTS_DIR / "metrics" / "validation_report.json"
        out.write_text(json.dumps(metrics, indent=2))
        logger.info("Wrote %s", out)

    logger.info("Saved model to %s", MODEL_DIR)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Train the ATLAS prediction model.")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip walk-forward validation and just refit")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    args = ap.parse_args()

    m = main(validate=not args.no_validate, horizon=args.horizon)
    if m:
        print("\n" + "=" * 66)
        print("OUT-OF-SAMPLE VALIDATION")
        print("=" * 66)
        print(f"  rank IC              {m['rank_ic']:+.4f}")
        print(f"  t-stat (naive)       {m['rank_ic_tstat_naive']:.2f}  "
              f"<- overstated, ignores IC autocorrelation")
        print(f"  t-stat (Newey-West)  {m['rank_ic_tstat']:.2f}  "
              f"<- correct standard error")
        print(f"  t-stat (deflated)    {m['rank_ic_tstat_deflated']:.2f}  "
              f"<- + haircut for {m['n_trials_searched']} configs tried")
        print(f"  AUC                  {m['auc']:.4f}")
        print(f"  days with IC > 0     {m['pct_days_ic_positive']:.1%}")
        print(f"  L-S gross annual     {m['long_short_gross_annual']:.1%}")
        print(f"  L-S Sharpe           {m['long_short_sharpe']:.2f}")
        print("  decile mean fwd return:")
        for k, v in sorted(m["decile_mean_fwd_return"].items()):
            print(f"     D{k}  {v:+.3%}")
        print("=" * 66)
