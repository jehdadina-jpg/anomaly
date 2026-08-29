"""
Diagnose: production wiring (per-date neutralization) scored worse per-fold
than the standalone experiment claimed. Hypothesis: the experiment's
neutralize() was called on the WHOLE FOLD's test mask (~170 days pooled into
one regression, ~8000 rows) rather than per-date (~48 rows, ~17 parameters --
barely more observations than parameters, high variance).

Tests three variants against the same folds:
  per-date      : regress on each individual day's ~48 stocks (production, live-realistic)
  pooled-fold    : regress once on the whole fold's test period (what the experiment measured)
  pooled-train   : regress once on the TRAINING period, apply fixed coefficients to test
                   (live-realistic: coefficients come from data available before the test
                   period, and are stable because they're fit on thousands of rows)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\mdadi\Downloads\anomalydetection")
from data.build_panel import load_panel
from features.predictive import build_features, feature_columns
from evaluation.stats import newey_west_tstat
from training.train import (HORIZON, fit_ensemble, predict_ensemble, make_folds,
                            make_target)
from scipy.stats import spearmanr

df = build_features(load_panel())
feats = feature_columns(df)
df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
FWD = f"fwd_{HORIZON}d"
df[FWD] = df.groupby("ticker")["Close"].transform(lambda x: x.shift(-HORIZON) / x - 1)
df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)

dates = np.sort(df["Date"].unique())
folds = make_folds(dates, 6)


def fit_neutralize_coefs(p, beta, sector):
    """Fit ONE regression on pooled (p, beta, sector) rows; return coefficients."""
    dummies = pd.get_dummies(sector, drop_first=True).astype(float)
    cols = dummies.columns.tolist()
    X = np.column_stack([np.ones(len(p)), beta, dummies.values])
    valid = np.isfinite(X).all(axis=1) & np.isfinite(p)
    coef, *_ = np.linalg.lstsq(X[valid], p[valid], rcond=None)
    return coef, cols


def apply_neutralize_coefs(p, beta, sector, coef, cols):
    """Apply FIXED coefficients (no fitting) to residualize p."""
    dummies = pd.get_dummies(sector, drop_first=True)
    dummies = dummies.reindex(columns=cols, fill_value=0.0).astype(float)
    X = np.column_stack([np.ones(len(p)), beta, dummies.values])
    valid = np.isfinite(X).all(axis=1) & np.isfinite(p)
    resid = p.copy()
    resid[valid] = p[valid] - X[valid] @ coef
    return resid


def per_date_neutralize(p, dates_s, beta, sector):
    out = p.copy()
    frame = pd.DataFrame({"p": p, "date": dates_s.values, "beta": beta.values,
                          "sector": sector.values})
    for _, g in frame.groupby("date"):
        dummies = pd.get_dummies(g["sector"], drop_first=True).astype(float)
        X = np.column_stack([np.ones(len(g)), g["beta"].values, dummies.values])
        y = g["p"].values
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if valid.sum() < len(y) * 0.5:
            continue
        coef, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
        resid = y.copy()
        resid[valid] = y[valid] - X[valid] @ coef
        out[g.index] = resid
    return out


def ic_report(scored, fwd_col="p_final"):
    ics = np.array([spearmanr(x[fwd_col], x[FWD]).statistic
                    for _, x in scored.groupby("Date") if len(x) >= 10])
    ics = ics[np.isfinite(ics)]
    _, nw_t = newey_west_tstat(ics, lag=max(HORIZON - 1, 1))
    return ics.mean(), nw_t


results = {"per_date": [], "pooled_fold": [], "pooled_train": [], "raw": []}

for i, (tr_dates, te_dates) in enumerate(folds, 1):
    tr_dates2 = tr_dates[:-HORIZON]
    tr_mask = df["Date"].isin(tr_dates2)
    te_mask = df["Date"].isin(te_dates)

    models = fit_ensemble(df, feats, tr_mask, HORIZON)
    p_test = predict_ensemble(models, df.loc[te_mask, feats])
    p_train = predict_ensemble(models, df.loc[tr_mask, feats])

    sub = df.loc[te_mask, ["Date", "ticker", FWD, "beta_60d", "sector"]].copy()

    # raw (no neutralization)
    sub["p_final"] = p_test
    ic, t = ic_report(sub)
    results["raw"].append((ic, t))

    # per-date (what's wired into production)
    p_pd = per_date_neutralize(p_test, sub["Date"], sub["beta_60d"], sub["sector"])
    sub["p_final"] = p_pd
    ic, t = ic_report(sub)
    results["per_date"].append((ic, t))

    # pooled-fold (what the earlier experiment actually measured)
    dummies = pd.get_dummies(sub["sector"], drop_first=True).astype(float)
    coef, cols = fit_neutralize_coefs(p_test, sub["beta_60d"].values, sub["sector"])
    p_pf_raw = apply_neutralize_coefs(p_test, sub["beta_60d"].values, sub["sector"], coef, cols)
    # still re-rank per date at the end (a pure normalization step)
    sub["_tmp"] = p_pf_raw
    p_pf = sub.groupby("Date")["_tmp"].transform(lambda s: s.rank(pct=True)).values
    sub["p_final"] = p_pf
    ic, t = ic_report(sub)
    results["pooled_fold"].append((ic, t))

    # pooled-train (coefficients fit on TRAINING data only, applied fixed to test)
    tr_sub = df.loc[tr_mask, ["beta_60d", "sector"]].copy()
    tr_sub["p"] = p_train
    coef2, cols2 = fit_neutralize_coefs(tr_sub["p"].values, tr_sub["beta_60d"].values, tr_sub["sector"])
    p_pt_raw = apply_neutralize_coefs(p_test, sub["beta_60d"].values, sub["sector"], coef2, cols2)
    sub["_tmp"] = p_pt_raw
    p_pt = sub.groupby("Date")["_tmp"].transform(lambda s: s.rank(pct=True)).values
    sub["p_final"] = p_pt
    ic, t = ic_report(sub)
    results["pooled_train"].append((ic, t))

    print(f"fold {i}: raw={results['raw'][-1][0]:+.4f}  per_date={results['per_date'][-1][0]:+.4f}  "
          f"pooled_fold={results['pooled_fold'][-1][0]:+.4f}  pooled_train={results['pooled_train'][-1][0]:+.4f}")

print()
print("=" * 70)
for name, rows in results.items():
    ics = [r[0] for r in rows]
    print(f"{name:14s} mean IC = {np.mean(ics):+.4f}  (per-fold: {[round(x,4) for x in ics]})")
print("=" * 70)
