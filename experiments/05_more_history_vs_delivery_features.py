"""
Experiment 12: is the ceiling the MODEL or the DATA?

Twelve modelling techniques have now been tried on this dataset; every one
was neutral or negative. That pattern usually means the model is at a local
optimum for the information available, and the binding constraint is data,
not algorithm.

The panel currently starts 2021-01-01 (~5.5 years, 66k rows). yfinance holds
far more history. This fetches 2015-2021 as well and asks a single question:
does roughly doubling the training history improve out-of-sample rank IC?

Caveat handled explicitly: NSE delivery data (from bhavcopy) only exists from
2021, so pre-2021 rows have no delivery features. Rather than pretend
otherwise, this compares three configurations honestly:
  A  2021+ with all 100 features            (current shipped model)
  B  2021+ WITHOUT delivery features        (isolates what delivery is worth)
  C  2015+ WITHOUT delivery features        (double the data, fewer features)
If C beats B, more history helps and it is worth backfilling delivery data.
If C beats A, more history beats better features outright.
"""
import sys, warnings
from pathlib import Path

import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\mdadi\Downloads\anomalydetection")
from data.build_panel import SECTOR_MAP, load_panel
from features.predictive import build_features, feature_columns
from evaluation.stats import newey_west_tstat
from training.train import LGB_PARAMS, make_folds

H = 3
SEEDS = (1, 42)
QUANTILES = (0.25, 0.35)
CACHE = Path(__file__).parent / ".cache" / "long_history_2015_2026.csv"

# ---------------------------------------------------------------- fetch
if CACHE.exists():
    print(f"Using cached long history: {CACHE}")
    long_panel = pd.read_csv(CACHE, parse_dates=["Date"])
else:
    import yfinance as yf
    tickers = sorted(SECTOR_MAP.keys())
    print(f"Downloading 2015-2026 history for {len(tickers)} tickers...")
    raw = yf.download(tickers, start="2015-01-01", end="2026-07-23",
                      interval="1d", auto_adjust=True, progress=False,
                      group_by="ticker", threads=True)
    frames = []
    for t in tickers:
        try:
            sub = raw[t].dropna(subset=["Close"]).reset_index()
        except Exception:
            continue
        if sub.empty:
            continue
        sub["ticker"] = t
        sub["symbol"] = t.replace(".NS", "")
        sub["sector"] = SECTOR_MAP.get(t, "Other")
        frames.append(sub[["Date", "Close", "High", "Low", "Open", "Volume",
                           "ticker", "symbol", "sector"]])
    long_panel = pd.concat(frames, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    long_panel.to_csv(CACHE, index=False)
    print(f"Cached to {CACHE}")

long_panel = long_panel.sort_values(["ticker", "Date"]).reset_index(drop=True)
print(f"long panel: {len(long_panel)} rows, {long_panel['ticker'].nunique()} tickers, "
      f"{long_panel['Date'].min().date()} -> {long_panel['Date'].max().date()}")

# Merge real delivery data where it exists (2021+), leave NaN before that.
short = load_panel()[["Date", "ticker", "delivery_pct", "delivery_qty",
                      "n_trades", "nse_traded_qty", "turnover_lacs"]]
long_panel = long_panel.merge(short, on=["Date", "ticker"], how="left")
print(f"delivery coverage in long panel: {long_panel['delivery_pct'].notna().mean():.1%}")


def evaluate(panel, label, drop_delivery=False, start=None):
    p = panel.copy()
    if start:
        p = p[p["Date"] >= start]
    p = p.sort_values(["ticker", "Date"]).reset_index(drop=True)

    df = build_features(p)
    feats = feature_columns(df)
    if drop_delivery:
        feats = [c for c in feats if "delivery" in c.lower()
                 or "trade_size" in c or "n_trades" in c or "turnover" in c]
        feats = [c for c in feature_columns(df) if c not in feats]

    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    df["fwd_3d"] = df.groupby("ticker")["Close"].transform(lambda x: x.shift(-H) / x - 1)
    df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)

    # Evaluate on the SAME test period in every configuration (2022-06 on),
    # so the comparison is about training data, not about which years are graded.
    dates = np.sort(df["Date"].unique())
    eval_start = pd.Timestamp("2022-06-01")
    folds = [(tr, te) for tr, te in make_folds(dates, 6)
             if pd.Timestamp(te[0]) >= eval_start]

    out = []
    for tr_d, te_d in folds:
        tr_d2 = tr_d[:-H]
        trm = df["Date"].isin(tr_d2)
        tem = df["Date"].isin(te_d)
        models = []
        for q in QUANTILES:
            g = df.groupby("Date")["fwd_3d"]
            hi, lo = g.transform(lambda s: s.quantile(1 - q)), g.transform(lambda s: s.quantile(q))
            y = pd.Series(np.nan, index=df.index)
            y[df["fwd_3d"] >= hi] = 1.0
            y[df["fwd_3d"] <= lo] = 0.0
            m = trm & y.notna()
            if m.sum() < 1000:
                continue
            for sd in SEEDS:
                models.append(lgb.LGBMClassifier(**LGB_PARAMS, random_state=sd)
                              .fit(df.loc[m, feats], y[m]))
        if not models:
            continue
        preds = [pd.Series(mm.predict_proba(df.loc[tem, feats])[:, 1]).rank(pct=True).values
                 for mm in models]
        sub = df.loc[tem, ["Date", "ticker", "fwd_3d"]].copy()
        sub["p"] = np.mean(preds, axis=0)
        out.append(sub)

    d = pd.concat(out).dropna(subset=["fwd_3d"])
    ics = np.array([spearmanr(x["p"], x["fwd_3d"]).statistic
                    for _, x in d.groupby("Date") if len(x) >= 10])
    ics = ics[np.isfinite(ics)]
    _, nw_t = newey_west_tstat(ics, lag=H - 1)
    n_train = int(trm.sum())
    print(f"{label:44s} IC={ics.mean():+.4f}  NW-t={nw_t:5.2f}  "
          f"feats={len(feats):3d}  last-fold train rows={n_train}")
    return ics.mean(), nw_t


print("\n" + "=" * 96)
print("IS THE CEILING THE MODEL OR THE DATA?   (same 2022-06+ test period throughout)")
print("=" * 96)
evaluate(long_panel, "A 2021+, all features (shipped)", start="2021-01-01")
evaluate(long_panel, "B 2021+, no delivery features", drop_delivery=True, start="2021-01-01")
evaluate(long_panel, "C 2015+, no delivery features (2x data)", drop_delivery=True)
print("=" * 96)
