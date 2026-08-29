"""
Experiment 9: test each proposed quant-rigor addition against the shipped
baseline (IC 0.0336, NW-t 4.78), using the SAME purged walk-forward folds so
comparisons are apples-to-apples. Every addition is measured; nothing is
kept just because it sounds more rigorous.

Variants:
  A baseline (shipped)
  B + sample uniqueness weighting (Lopez de Prado)
  C + feature winsorization (1st/99th pct, per-day cross-sectional)
  D + beta/sector neutralization of the OUTPUT score (post-hoc residualization)
  E + linear diversifier (ElasticNet) blended into the ensemble
  F all of the above combined
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\mdadi\Downloads\anomalydetection")
from data.build_panel import load_panel
from features.predictive import build_features, feature_columns
from evaluation.stats import newey_west_tstat, sample_uniqueness_weights

H = 3
QUANTILES = (0.25, 0.35)
SEEDS = (1, 42, 2024)
LGB_PARAMS = dict(n_estimators=600, learning_rate=0.015, num_leaves=7, max_depth=3,
                  min_child_samples=400, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=20.0,
                  n_jobs=-1, verbose=-1)

df = build_features(load_panel())
FEATS = feature_columns(df)
df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
df[f"fwd_{H}d"] = df.groupby("ticker")["Close"].transform(lambda x: x.shift(-H) / x - 1)
df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)
FWD = f"fwd_{H}d"

print("Computing sample uniqueness weights...")
UNIQ_W = sample_uniqueness_weights(df, H)
print(f"  weight range: [{UNIQ_W.min():.3f}, {UNIQ_W.max():.3f}], mean={UNIQ_W.mean():.3f}")

# Winsorized feature copy: clip each feature to its cross-sectional 1st/99th
# percentile WITHIN EACH DATE, so a single circuit-limit day doesn't dominate.
print("Building winsorized features (vectorized)...")
# Per-column .transform(closure) applies a Python callable once per
# (date, feature) pair -- ~137,500 calls, painfully slow in pandas. Computing
# the two quantile columns with a single groupby.quantile() call (vectorized,
# C-level) and broadcasting them back via merge is the same result orders of
# magnitude faster.
g = df.groupby("Date")[FEATS]
lo_q = g.quantile(0.01).add_prefix("lo_").reset_index()
hi_q = g.quantile(0.99).add_prefix("hi_").reset_index()
bounds = lo_q.merge(hi_q, on="Date")
dfw = df.merge(bounds, on="Date", how="left")
for c in FEATS:
    dfw[c] = dfw[c].clip(dfw[f"lo_{c}"], dfw[f"hi_{c}"])
dfw = dfw.drop(columns=[f"lo_{c}" for c in FEATS] + [f"hi_{c}" for c in FEATS])
print(f"  done, {len(dfw)} rows")

dates = np.sort(df["Date"].unique())
n_folds, test_len = 6, len(dates) // 8
folds = [(dates[:len(dates) - (n_folds - k) * test_len],
          dates[len(dates) - (n_folds - k) * test_len:
                len(dates) - (n_folds - k) * test_len + test_len])
         for k in range(n_folds)]


def make_y(data, q):
    g = data.groupby("Date")[FWD]
    hi, lo = g.transform(lambda s: s.quantile(1 - q)), g.transform(lambda s: s.quantile(q))
    y = pd.Series(np.nan, index=data.index)
    y[data[FWD] >= hi] = 1.0
    y[data[FWD] <= lo] = 0.0
    return y.where(data[FWD].notna())


def fit_lgb_ensemble(data, feats, mask, sample_weight=None):
    models = []
    for q in QUANTILES:
        y = make_y(data, q)
        m = mask & y.notna()
        if m.sum() < 1000:
            continue
        X, yy = data.loc[m, feats], y[m]
        w = sample_weight[m].values if sample_weight is not None else None
        for seed in SEEDS:
            models.append(lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
                          .fit(X, yy, sample_weight=w))
    return models


def predict_lgb(models, X):
    preds = [pd.Series(m.predict_proba(X)[:, 1]).rank(pct=True).values for m in models]
    return np.mean(preds, axis=0)


def fit_linear(data, feats, mask, q=0.30):
    y = make_y(data, q)
    m = mask & y.notna()
    X = data.loc[m, feats].fillna(0).values
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.05, max_iter=500, n_jobs=-1).fit(scaler.transform(X), y[m])
    return scaler, clf


def predict_linear(scaler, clf, X):
    Xs = scaler.transform(X.fillna(0).values)
    p = clf.predict_proba(Xs)[:, 1]
    return pd.Series(p).rank(pct=True).values


def neutralize(scores: np.ndarray, betas: np.ndarray, sectors: pd.Series) -> np.ndarray:
    """
    Regress the score on beta + sector dummies within the day, keep only the
    residual. Standard Barra-style neutralization: removes the part of the
    ranking explained by a known risk factor, leaving pure stock selection.
    """
    X = pd.get_dummies(sectors, drop_first=True).astype(float)
    X.insert(0, "beta", betas)
    X.insert(0, "const", 1.0)
    X = X.fillna(0).values
    y = scores
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        # Re-rank so output stays a valid percentile score.
        return pd.Series(resid).rank(pct=True).values
    except Exception:
        return scores


def evaluate(pred_col_builder, label, use_weights=False, winsorized=False,
            add_linear=False, do_neutralize=False):
    data = dfw if winsorized else df
    out = []
    for tr_d, te_d in folds:
        tr_d2 = tr_d[:-H]
        trm = data["Date"].isin(tr_d2)
        tem = data["Date"].isin(te_d)

        sw = UNIQ_W if use_weights else None
        lgb_models = fit_lgb_ensemble(data, FEATS, trm, sample_weight=sw)
        p_lgb = predict_lgb(lgb_models, data.loc[tem, FEATS])

        if add_linear:
            scaler, clf = fit_linear(data, FEATS, trm)
            p_lin = predict_linear(scaler, clf, data.loc[tem, FEATS])
            p = 0.7 * p_lgb + 0.3 * p_lin
        else:
            p = p_lgb

        if do_neutralize:
            p = neutralize(p, data.loc[tem, "beta_60d"].values, data.loc[tem, "sector"])

        sub = data.loc[tem, ["Date", "ticker", FWD, "beta_60d"]].copy()
        sub["p"] = p
        out.append(sub)
    d = pd.concat(out).dropna(subset=[FWD])

    ics = np.array([spearmanr(x["p"], x[FWD]).statistic
                    for _, x in d.groupby("Date") if len(x) >= 10])
    ics = ics[np.isfinite(ics)]
    naive_t = ics.mean() / (ics.std() + 1e-12) * np.sqrt(len(ics))
    _, nw_t = newey_west_tstat(ics, lag=H - 1)

    # beta tilt: mean beta of top decile minus bottom decile
    d["rank"] = d.groupby("Date")["p"].rank(pct=True)
    top_beta = d.loc[d["rank"] >= 0.9, "beta_60d"].mean()
    bot_beta = d.loc[d["rank"] <= 0.1, "beta_60d"].mean()

    print(f"{label:32s} IC={ics.mean():+.4f}  naive-t={naive_t:5.2f}  NW-t={nw_t:5.2f}  "
          f"beta-tilt={top_beta - bot_beta:+.3f}")
    return ics.mean(), nw_t, top_beta - bot_beta


print("\n" + "=" * 90)
print("QUANT RIGOR ADDITIONS  (each measured against the shipped baseline)")
print("=" * 90)

evaluate(None, "A baseline (shipped)")
evaluate(None, "B + uniqueness weighting", use_weights=True)
evaluate(None, "C + winsorized features", winsorized=True)
evaluate(None, "D + beta/sector neutralize", do_neutralize=True)
evaluate(None, "E + linear diversifier (30%)", add_linear=True)
evaluate(None, "F ALL combined", use_weights=True, winsorized=True,
        do_neutralize=True, add_linear=True)
print("=" * 90)
