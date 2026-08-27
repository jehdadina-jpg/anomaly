"""
Compare candidate model additions against the shipped configuration.

This is the script that decided beta/sector neutralization belongs in
production and that sample-uniqueness weighting, feature winsorization, and a
linear diversifier do not (see docs/MODEL_VALIDATION.md §8). Kept as a real,
runnable script rather than a one-off — rerun it before adopting a new idea,
so "it sounds more rigorous" and "it measurably helps" stay separate
questions.

Every variant is scored on identical purged walk-forward folds, so the
comparison is apples-to-apples. Slow: fits roughly 200 LightGBM models.

Usage
-----
    python -m evaluation.compare_configs
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.build_panel import load_panel
from evaluation.stats import neutralize_score, newey_west_tstat, sample_uniqueness_weights
from features.predictive import build_features, feature_columns
from training.train import HORIZON, LGB_PARAMS, N_FOLDS, SEEDS, TAIL_QUANTILES, make_folds

logger = logging.getLogger(__name__)


def _make_y(data, fwd_col, q):
    g = data.groupby("Date")[fwd_col]
    hi, lo = g.transform(lambda s: s.quantile(1 - q)), g.transform(lambda s: s.quantile(q))
    y = pd.Series(np.nan, index=data.index)
    y[data[fwd_col] >= hi] = 1.0
    y[data[fwd_col] <= lo] = 0.0
    return y.where(data[fwd_col].notna())


def _fit_lgb(data, feats, fwd_col, mask, quantiles=TAIL_QUANTILES,
            seeds=SEEDS, sample_weight=None):
    import lightgbm as lgb
    models = []
    for q in quantiles:
        y = _make_y(data, fwd_col, q)
        m = mask & y.notna()
        if m.sum() < 1000:
            continue
        X, yy = data.loc[m, feats], y[m]
        w = sample_weight[m].values if sample_weight is not None else None
        for seed in seeds:
            models.append(lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
                          .fit(X, yy, sample_weight=w))
    return models


def _predict_lgb(models, X):
    preds = [pd.Series(m.predict_proba(X)[:, 1]).rank(pct=True).values for m in models]
    return np.mean(preds, axis=0)


def _fit_linear(data, feats, fwd_col, mask, q=0.30):
    y = _make_y(data, fwd_col, q)
    m = mask & y.notna()
    X = data.loc[m, feats].fillna(0).values
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.05, max_iter=500, n_jobs=-1).fit(scaler.transform(X), y[m])
    return scaler, clf


def evaluate(df, feats, fwd_col, folds, label, *, use_weights=False,
            winsorized_df=None, add_linear=False, do_neutralize=False,
            uniq_weights=None):
    data = winsorized_df if winsorized_df is not None else df
    out = []
    for tr_d, te_d in folds:
        tr_d2 = tr_d[:-HORIZON] if HORIZON < len(tr_d) else tr_d
        trm = data["Date"].isin(tr_d2)
        tem = data["Date"].isin(te_d)

        sw = uniq_weights if use_weights else None
        lgb_models = _fit_lgb(data, feats, fwd_col, trm, sample_weight=sw)
        p = _predict_lgb(lgb_models, data.loc[tem, feats])

        if add_linear:
            scaler, clf = _fit_linear(data, feats, fwd_col, trm)
            Xs = scaler.transform(data.loc[tem, feats].fillna(0).values)
            p_lin = pd.Series(clf.predict_proba(Xs)[:, 1]).rank(pct=True).values
            p = 0.7 * p + 0.3 * p_lin

        if do_neutralize:
            sub_beta = data.loc[tem, "beta_60d"]
            sub_sector = data.loc[tem, "sector"]
            sub_dates = data.loc[tem, "Date"]
            frame = pd.DataFrame({"p": p, "date": sub_dates.values,
                                  "beta": sub_beta.values, "sector": sub_sector.values})
            p = p.copy()
            for _, g in frame.groupby("date"):
                p[g.index] = neutralize_score(g["p"].values, g["beta"].values, g["sector"])

        sub = data.loc[tem, ["Date", "ticker", fwd_col, "beta_60d"]].copy()
        sub["p"] = p
        out.append(sub)

    d = pd.concat(out).dropna(subset=[fwd_col])
    ics = np.array([spearmanr(x["p"], x[fwd_col]).statistic
                    for _, x in d.groupby("Date") if len(x) >= 10])
    ics = ics[np.isfinite(ics)]
    _, nw_t = newey_west_tstat(ics, lag=max(HORIZON - 1, 1))

    d["rank"] = d.groupby("Date")["p"].rank(pct=True)
    top_beta = d.loc[d["rank"] >= 0.9, "beta_60d"].mean()
    bot_beta = d.loc[d["rank"] <= 0.1, "beta_60d"].mean()

    print(f"{label:32s} IC={ics.mean():+.4f}  NW-t={nw_t:5.2f}  "
         f"beta-tilt={top_beta - bot_beta:+.3f}")
    return ics.mean(), nw_t


def main():
    df = build_features(load_panel())
    feats = feature_columns(df)
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    fwd_col = f"fwd_{HORIZON}d"
    df[fwd_col] = df.groupby("ticker")["Close"].transform(
        lambda x: x.shift(-HORIZON) / x - 1)
    df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)

    dates = np.sort(df["Date"].unique())
    folds = make_folds(dates, N_FOLDS)

    print("Computing sample uniqueness weights...")
    uniq_w = sample_uniqueness_weights(df, HORIZON)

    print("Building winsorized features...")
    g = df.groupby("Date")[feats]
    lo_q = g.quantile(0.01).add_prefix("lo_").reset_index()
    hi_q = g.quantile(0.99).add_prefix("hi_").reset_index()
    dfw = df.merge(lo_q, on="Date", how="left").merge(hi_q, on="Date", how="left")
    for c in feats:
        dfw[c] = dfw[c].clip(dfw[f"lo_{c}"], dfw[f"hi_{c}"])
    dfw = dfw.drop(columns=[f"lo_{c}" for c in feats] + [f"hi_{c}" for c in feats])

    print("\n" + "=" * 90)
    print("CONFIGURATION COMPARISON  (identical purged walk-forward folds)")
    print("=" * 90)
    evaluate(df, feats, fwd_col, folds, "baseline")
    evaluate(df, feats, fwd_col, folds, "+ uniqueness weighting",
            use_weights=True, uniq_weights=uniq_w)
    evaluate(dfw, feats, fwd_col, folds, "+ winsorized features")
    evaluate(df, feats, fwd_col, folds, "+ beta/sector neutralize (SHIPPED)",
            do_neutralize=True)
    evaluate(df, feats, fwd_col, folds, "+ linear diversifier (30%)", add_linear=True)
    print("=" * 90)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
