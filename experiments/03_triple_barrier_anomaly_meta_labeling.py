"""
Experiment 10: the strongest remaining techniques, measured honestly.

Four ideas from the quant literature that have NOT been tried on this model,
each tested on the identical purged walk-forward folds:

  A baseline                    shipped config (rank IC 0.0336)
  B triple-barrier labels       Lopez de Prado AFML ch.3. Instead of "return
                                after exactly 3 days", label by which barrier
                                is hit first: profit-take (+k*vol), stop-loss
                                (-k*vol), or the 3-day time limit. Adapts the
                                target to each stock's own volatility, so a
                                1% move in a calm stock and a 3% move in a
                                wild one are treated as comparable events.
  C anomaly-score features      This repo already ships 4 unsupervised anomaly
                                detectors that the PREDICTOR never uses.
                                IsolationForest / LOF scores over the feature
                                panel are added as predictive features:
                                unusual microstructure often precedes moves.
  D meta-labeling               AFML ch.3. A second model predicts WHEN THE
                                FIRST MODEL IS RIGHT, using the primary's own
                                confidence plus features. Final score =
                                primary rank weighted by meta-confidence.
                                Designed to raise precision, which is what a
                                top-decile buy score actually needs.
  E C + D combined

Everything is fit strictly inside each fold's training window.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\mdadi\Downloads\anomalydetection")
from data.build_panel import load_panel
from features.predictive import build_features, feature_columns
from evaluation.stats import newey_west_tstat
from training.train import LGB_PARAMS, make_folds

H = 3
SEEDS = (1, 42)          # 2 seeds during exploration for speed
QUANTILES = (0.25, 0.35)

print("Building features...")
df = build_features(load_panel())
FEATS = feature_columns(df)
df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

# ---- forward returns + path data needed for triple-barrier ----
df["fwd_3d"] = df.groupby("ticker")["Close"].transform(lambda x: x.shift(-H) / x - 1)
# Running max/min of the forward path, for barrier detection.
for k in range(1, H + 1):
    df[f"_f{k}"] = df.groupby("ticker")["Close"].transform(lambda x: x.shift(-k) / x - 1)
path = df[[f"_f{k}" for k in range(1, H + 1)]]
df["_path_max"] = path.max(axis=1)
df["_path_min"] = path.min(axis=1)
df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)
FWD = "fwd_3d"


def triple_barrier_label(data, k_vol=1.0):
    """
    +1 if the upper barrier (+k*vol) is touched before the lower one,
     0 if the lower barrier is touched first,
    NaN if neither is touched before the time barrier (ambiguous -> dropped).

    Barriers scale with each stock's own 20-day volatility, so the label
    means the same thing across calm and volatile names.
    """
    vol = data["vol_20d"] * np.sqrt(H)
    upper, lower = k_vol * vol, -k_vol * vol
    hit_up = data["_path_max"] >= upper
    hit_dn = data["_path_min"] <= lower

    y = pd.Series(np.nan, index=data.index)
    y[hit_up & ~hit_dn] = 1.0
    y[hit_dn & ~hit_up] = 0.0
    # Both touched: use the final return's sign as the tie-break.
    both = hit_up & hit_dn
    y[both] = (data.loc[both, FWD] > 0).astype(float)
    return y.where(data[FWD].notna() & data["vol_20d"].notna())


def quantile_label(data, q):
    g = data.groupby("Date")[FWD]
    hi, lo = g.transform(lambda s: s.quantile(1 - q)), g.transform(lambda s: s.quantile(q))
    y = pd.Series(np.nan, index=data.index)
    y[data[FWD] >= hi] = 1.0
    y[data[FWD] <= lo] = 0.0
    return y.where(data[FWD].notna())


dates = np.sort(df["Date"].unique())
folds = make_folds(dates, 6)


def add_anomaly_features(data, feats, trm, tem):
    """
    Fit IsolationForest on TRAINING rows only, score train+test.
    Also a cheap cross-sectional 'how unusual is this stock today' measure.
    """
    core = [c for c in feats if not c.startswith(("xs_", "market_", "sector_"))][:40]
    X_tr = data.loc[trm, core].fillna(0).values
    iso = IsolationForest(n_estimators=100, contamination=0.05,
                          random_state=42, n_jobs=-1).fit(X_tr)

    out = {}
    for name, mask in (("tr", trm), ("te", tem)):
        X = data.loc[mask, core].fillna(0).values
        s = iso.score_samples(X)          # lower = more anomalous
        sub = pd.DataFrame({"iso_score": s}, index=data.loc[mask].index)
        sub["iso_rank"] = sub.groupby(data.loc[mask, "Date"])["iso_score"].rank(pct=True)
        out[name] = sub
    return out


def fit_primary(data, feats, trm, label_fn, seeds=SEEDS):
    models = []
    for q in QUANTILES:
        y = label_fn(data, q) if label_fn is quantile_label else label_fn(data)
        m = trm & y.notna()
        if m.sum() < 1000:
            continue
        for sd in seeds:
            models.append(lgb.LGBMClassifier(**LGB_PARAMS, random_state=sd)
                          .fit(data.loc[m, feats], y[m]))
    return models


def predict_primary(models, X):
    preds = [pd.Series(m.predict_proba(X)[:, 1]).rank(pct=True).values for m in models]
    return np.mean(preds, axis=0)


def report(scored, label):
    ics = np.array([spearmanr(x["p"], x[FWD]).statistic
                    for _, x in scored.groupby("Date") if len(x) >= 10])
    ics = ics[np.isfinite(ics)]
    _, nw_t = newey_west_tstat(ics, lag=H - 1)
    # top-decile average forward return: what the 10/10 score actually earns
    scored = scored.copy()
    scored["rk"] = scored.groupby("Date")["p"].rank(pct=True)
    top = scored.loc[scored["rk"] >= 0.9, FWD].mean()
    print(f"{label:34s} IC={ics.mean():+.4f}  NW-t={nw_t:5.2f}  top-decile={top:+.3%}")
    return ics.mean(), nw_t


# ----------------------------------------------------------------- variants
def run(label, *, use_triple=False, use_anomaly=False, use_meta=False):
    out = []
    for tr_d, te_d in folds:
        tr_d2 = tr_d[:-H]
        trm = df["Date"].isin(tr_d2)
        tem = df["Date"].isin(te_d)
        feats = list(FEATS)
        data = df

        if use_anomaly:
            anom = add_anomaly_features(df, FEATS, trm, tem)
            data = df.copy()
            data.loc[trm, "iso_score"] = anom["tr"]["iso_score"]
            data.loc[trm, "iso_rank"] = anom["tr"]["iso_rank"]
            data.loc[tem, "iso_score"] = anom["te"]["iso_score"]
            data.loc[tem, "iso_rank"] = anom["te"]["iso_rank"]
            feats = FEATS + ["iso_score", "iso_rank"]

        label_fn = triple_barrier_label if use_triple else quantile_label
        models = fit_primary(data, feats, trm, label_fn)
        if not models:
            continue
        p_te = predict_primary(models, data.loc[tem, feats])

        if use_meta:
            # Meta-label: was the primary right, in-sample on training rows?
            p_tr = predict_primary(models, data.loc[trm, feats])
            y_tr = quantile_label(data, 0.30)[trm]
            ok = y_tr.notna()
            # "correct" = primary ranked it in the top half AND it was a winner,
            # or bottom half AND it was a loser.
            primary_says_up = pd.Series(p_tr, index=data.loc[trm].index) > 0.5
            meta_y = (primary_says_up == (y_tr == 1.0)).astype(float)[ok]

            meta_X_tr = data.loc[trm, feats][ok.values].copy()
            meta_X_tr["_primary"] = pd.Series(p_tr, index=data.loc[trm].index)[ok]
            meta = lgb.LGBMClassifier(**LGB_PARAMS, random_state=7).fit(meta_X_tr, meta_y)

            meta_X_te = data.loc[tem, feats].copy()
            meta_X_te["_primary"] = p_te
            conf = meta.predict_proba(meta_X_te)[:, 1]
            # Shrink scores toward the middle where the meta-model expects the
            # primary to be wrong; keep them where it expects it to be right.
            p_te = 0.5 + (p_te - 0.5) * conf

        sub = data.loc[tem, ["Date", "ticker", FWD]].copy()
        sub["p"] = p_te
        out.append(sub)

    return report(pd.concat(out).dropna(subset=[FWD]), label)


print("\n" + "=" * 82)
print("ADVANCED TECHNIQUES  (purged walk-forward, out-of-sample)")
print("=" * 82)
run("A baseline")
run("B triple-barrier labels", use_triple=True)
run("C + anomaly features", use_anomaly=True)
run("D + meta-labeling", use_meta=True)
run("E anomaly + meta", use_anomaly=True, use_meta=True)
print("=" * 82)
