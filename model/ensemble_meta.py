# -*- coding: utf-8 -*-
"""
VinUni Datathon 2026 - Stacking Ensemble: LGBM + Prophet + Meta-Learner
=========================================================================
Architecture:
  Step 1: TimeSeriesSplit OOF (2018-2022)
    - For each fold: train LGBM & Prophet on train portion, predict val
    - Collect OOF predictions [lgbm_oof, prophet_oof]

  Step 2: Meta-Learner
    - Ridge regression (positive weights, no intercept)
    - Learns optimal alpha, beta from OOF vs true Revenue
    - Output: final = alpha*lgbm + beta*prophet

  Step 3: Full training + forecast 2023-2024
    - Train both models on full data (2012-2022)
    - Predict 2023-2024 recursively (LGBM) / directly (Prophet)
    - Combine with learned alpha, beta

  Step 4: Post-processing
    - Trend Decay + Spike Clipping
"""

import sys, io, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# =============================================================================
# CONFIG
# =============================================================================
BASE        = Path(r"d:\vinuni_datathon2026\vinuni_datathon2026\model")
DATA_PATH   = BASE / "processed_data.csv"
SAMPLE_PATH = BASE / "sample_submission.csv"
OUTPUT_PATH = BASE / "submission_meta.csv"

PEAK_DATE    = pd.Timestamp("2017-01-01")
YEAR_PERIOD  = 365.25
ROLLING_WIN  = 7
OOF_START    = "2018-01-01"   # OOF period start
N_SPLITS     = 4              # TimeSeriesSplit folds (each ~1yr)
SPIKE_CLIP_Q = 0.95

TET_DATES = {
    2012: pd.Timestamp("2012-01-23"), 2013: pd.Timestamp("2013-02-10"),
    2014: pd.Timestamp("2014-01-31"), 2015: pd.Timestamp("2015-02-19"),
    2016: pd.Timestamp("2016-02-08"), 2017: pd.Timestamp("2017-01-28"),
    2018: pd.Timestamp("2018-02-16"), 2019: pd.Timestamp("2019-02-05"),
    2020: pd.Timestamp("2020-01-25"), 2021: pd.Timestamp("2021-02-12"),
    2022: pd.Timestamp("2022-02-01"), 2023: pd.Timestamp("2023-01-22"),
    2024: pd.Timestamp("2024-02-10"),
}

def days_to_nearest_tet(date):
    cands = [abs((date - t).days) for yr, t in TET_DATES.items()
             if abs(date.year - yr) <= 1]
    return float(min(cands)) if cands else 60.0

# =============================================================================
# HELPERS
# =============================================================================
def rolling_yoy_lag(date, center, rmap, win=ROLLING_WIN):
    vals = [rmap[date - pd.Timedelta(days=center + d)]
            for d in range(-win, win + 1)
            if (date - pd.Timedelta(days=center + d)) in rmap]
    return float(np.mean(vals)) if vals else np.nan

def exact_lag(date, lag, rmap):
    v = rmap.get(date - pd.Timedelta(days=lag), np.nan)
    if np.isnan(v):
        for d in range(1, 8):
            for s in (1, -1):
                v = rmap.get(date - pd.Timedelta(days=lag + s * d), np.nan)
                if not np.isnan(v):
                    return float(v)
    return float(v) if not np.isnan(v) else np.nan

def build_lgbm_features(dates: pd.Series, rmap: dict) -> pd.DataFrame:
    """Same as v2: lag_7/14/28 + rolling YoY + Fourier + trend."""
    feats = pd.DataFrame(index=dates.index)
    d = pd.DatetimeIndex(dates)

    feats["trend_days_from_peak"] = (dates - PEAK_DATE).dt.days
    feats["dom_ratio"] = (d.day - 1) / (d.days_in_month - 1)
    feats["days_to_tet"] = dates.apply(days_to_nearest_tet)

    t = d.dayofyear / YEAR_PERIOD * 2 * np.pi
    for k in range(1, 6):
        feats[f"sin_year_{k}"] = np.sin(k * t)
        feats[f"cos_year_{k}"] = np.cos(k * t)

    t_week = d.dayofweek / 7 * 2 * np.pi
    for k in range(1, 3):
        feats[f"sin_week_{k}"] = np.sin(k * t_week)
        feats[f"cos_week_{k}"] = np.cos(k * t_week)

    r364 = dates.apply(lambda x: rolling_yoy_lag(x, 364, rmap))
    r728 = dates.apply(lambda x: rolling_yoy_lag(x, 728, rmap))
    e364 = dates.apply(lambda x: exact_lag(x, 364, rmap))
    e728 = dates.apply(lambda x: exact_lag(x, 728, rmap))

    lag364 = r364.fillna(e364)
    lag728 = r728.fillna(e728).fillna(lag364)

    feats["lag_364_roll"] = np.log1p(lag364.clip(lower=0))
    feats["lag_728_roll"] = np.log1p(lag728.clip(lower=0))
    feats["yoy_ratio"]    = (lag364 / lag728.replace(0, np.nan)).clip(0.3, 2.5).fillna(1.0)

    # Short-term lags (like v2)
    for short in [7, 14, 28]:
        vals = dates.apply(lambda x: exact_lag(x, short, rmap))
        feats[f"lag_{short}"] = np.log1p(vals.clip(lower=0).fillna(lag364.clip(lower=0)))

    return feats

def get_feat_cols(rmap_sample):
    tmp = build_lgbm_features(pd.Series([pd.Timestamp("2020-06-01")]), rmap_sample)
    return tmp.columns.tolist()

def make_weights(dates):
    w = np.ones(len(dates))
    for yr, wt in {2019: 2.0, 2020: 3.0, 2021: 4.0, 2022: 5.0}.items():
        w[dates.dt.year.values == yr] = wt
    mn, mx = w.min(), w.max()
    return 1.0 + 4.0 * (w - mn) / (mx - mn) if mx > mn else w

LGBM_PARAMS = dict(
    objective="regression", metric="mae",
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=6,
    min_child_samples=25, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=5,
    reg_alpha=0.08, reg_lambda=0.08,
    random_state=42, n_jobs=-1, verbose=-1,
)

def is_tet(ds):
    date = pd.Timestamp(ds)
    for yr, tet in TET_DATES.items():
        if abs((date - tet).days) <= 14:
            return 1.0
    return 0.0

def train_prophet(train_df):
    pdf = train_df[["Date", "Revenue"]].copy()
    pdf.columns = ["ds", "y"]
    pdf["y"] = np.log1p(pdf["y"])
    pdf["is_tet"] = pdf["ds"].apply(is_tet)
    m = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True, weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=10.0,
        changepoint_range=0.95,
    )
    m.add_regressor("is_tet", standardize=False)
    m.fit(pdf, algorithm="Newton")
    return m

def predict_prophet(model, dates):
    future = pd.DataFrame({"ds": dates})
    future["is_tet"] = future["ds"].apply(is_tet)
    fc = model.predict(future)
    return np.expm1(fc["yhat"].clip(lower=0).values)

def recursive_lgbm_forecast(model, feat_cols, fdates, rmap_seed):
    rolling = dict(rmap_seed)
    preds = {}
    for fdate in fdates:
        row = build_lgbm_features(pd.Series([fdate]), rolling)
        p = float(np.expm1(model.predict(row[feat_cols])[0]))
        yoy_b = rolling_yoy_lag(fdate, 364, rolling)
        if not np.isnan(yoy_b):
            yoy_r = float(row["yoy_ratio"].values[0])
            p = 0.20 * (yoy_b * yoy_r) + 0.80 * p
        p = max(p, 0.0)
        preds[fdate] = p
        rolling[fdate] = p
    return preds

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("Step 1: Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.rename(columns={"date": "Date"})
df = df[["Date", "Revenue"]].dropna().sort_values("Date").reset_index(drop=True)
rmap_full = dict(zip(df["Date"], df["Revenue"]))
TRAIN_END = df["Date"].max()

sub    = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"])
sub    = sub.sort_values("Date").reset_index(drop=True)
fdates = sub["Date"]

CLIP_HIGH = float(np.quantile(df["Revenue"], SPIKE_CLIP_Q))
decay_df  = df[df["Date"] >= "2017-01-01"].copy()
slope, *_ = stats.linregress(
    (decay_df["Date"] - PEAK_DATE).dt.days,
    np.log(decay_df["Revenue"].clip(1))
)
def decay_factor(date):
    return float(np.exp(slope * (date - TRAIN_END).days))

print(f"  Rows: {len(df)}  {df['Date'].min().date()} -> {TRAIN_END.date()}")
print(f"  Annual decay: {np.exp(slope*365.25):.4f}  Clip P95: {CLIP_HIGH:,.0f}")

# OOF subset (2018-2022)
oof_df   = df[df["Date"] >= OOF_START].reset_index(drop=True)
print(f"  OOF period: {oof_df['Date'].min().date()} -> {oof_df['Date'].max().date()} ({len(oof_df)} rows)")

# =============================================================================
# 2. OOF PREDICTIONS via TimeSeriesSplit
# =============================================================================
print("\n" + "=" * 60)
print(f"Step 2: OOF predictions ({N_SPLITS}-fold TimeSeriesSplit)...")
print("=" * 60)

FEAT_COLS = get_feat_cols(rmap_full)

oof_lgbm    = np.full(len(oof_df), np.nan)
oof_prophet = np.full(len(oof_df), np.nan)

tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=30)

for fold, (tr_idx, va_idx) in enumerate(tscv.split(oof_df)):
    tr_dates = oof_df.loc[tr_idx, "Date"]
    va_dates = oof_df.loc[va_idx, "Date"]
    print(f"\n  Fold {fold+1}: train {tr_dates.min().date()}->{tr_dates.max().date()} | val {va_dates.min().date()}->{va_dates.max().date()}")

    # --- Build full training set for this fold (pre-OOF + fold train) ---
    fold_train_df = df[df["Date"] <= tr_dates.max()].copy()
    fold_rmap     = dict(zip(fold_train_df["Date"], fold_train_df["Revenue"]))

    # --- LGBM ---
    X_fold = build_lgbm_features(fold_train_df["Date"], fold_rmap)
    y_fold = np.log1p(fold_train_df["Revenue"].values)
    valid_f = X_fold["lag_364_roll"].notna() & (X_fold["lag_364_roll"] > 0)
    X_f = X_fold[valid_f][FEAT_COLS].copy()
    y_f = y_fold[valid_f]
    dates_f = fold_train_df.loc[valid_f, "Date"].reset_index(drop=True)
    w_f = make_weights(dates_f)

    m_lgbm = lgb.LGBMRegressor(**LGBM_PARAMS)
    m_lgbm.fit(X_f, y_f, sample_weight=w_f)

    # Predict validation in-sample (no recursive needed - val dates in training history)
    X_va = build_lgbm_features(va_dates, fold_rmap)[FEAT_COLS]
    oof_lgbm[va_idx] = np.expm1(m_lgbm.predict(X_va))
    print(f"    LGBM OOF   MAE: {mean_absolute_error(oof_df.loc[va_idx,'Revenue'], oof_lgbm[va_idx]):,.0f}")

    # --- Prophet ---
    try:
        m_prophet = train_prophet(fold_train_df)
        oof_prophet[va_idx] = predict_prophet(m_prophet, va_dates.values)
        print(f"    Prophet OOF MAE: {mean_absolute_error(oof_df.loc[va_idx,'Revenue'], oof_prophet[va_idx]):,.0f}")
    except Exception as e:
        print(f"    Prophet failed: {e}  -> using LGBM")
        oof_prophet[va_idx] = oof_lgbm[va_idx]

# =============================================================================
# 3. META-LEARNER: Learn optimal weights from OOF
# =============================================================================
print("\n" + "=" * 60)
print("Step 3: Training Meta-Learner on OOF predictions...")
print("=" * 60)

valid_oof = ~(np.isnan(oof_lgbm) | np.isnan(oof_prophet))
y_oof     = oof_df.loc[valid_oof, "Revenue"].values
X_meta    = np.column_stack([oof_lgbm[valid_oof], oof_prophet[valid_oof]])

print(f"  OOF samples for meta-learner: {valid_oof.sum()}")
print(f"  LGBM OOF   overall MAE: {mean_absolute_error(y_oof, X_meta[:,0]):,.0f}")
print(f"  Prophet OOF overall MAE: {mean_absolute_error(y_oof, X_meta[:,1]):,.0f}")

# Method 1: Ridge regression (positive=True so weights are >= 0)
meta_ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
meta_ridge.fit(X_meta, y_oof)
w_lgbm_ridge, w_prophet_ridge = meta_ridge.coef_
total = w_lgbm_ridge + w_prophet_ridge
w_lgbm_ridge_n   = w_lgbm_ridge / total
w_prophet_ridge_n = w_prophet_ridge / total
print(f"\n  Ridge weights (normalized): LGBM={w_lgbm_ridge_n:.3f}  Prophet={w_prophet_ridge_n:.3f}")

oof_ridge_pred = X_meta @ meta_ridge.coef_
print(f"  Ridge OOF MAE: {mean_absolute_error(y_oof, oof_ridge_pred):,.0f}")

# Method 2: Constrained optimization (weights >= 0, sum to 1, minimize MAE)
def neg_mae_weights(w):
    blend = w[0] * X_meta[:,0] + w[1] * X_meta[:,1]
    return mean_absolute_error(y_oof, blend)

result = minimize(
    neg_mae_weights,
    x0=[0.5, 0.5],
    method="SLSQP",
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    constraints={"type": "eq", "fun": lambda w: w[0] + w[1] - 1},
    options={"ftol": 1e-9, "maxiter": 500},
)
w_lgbm_opt, w_prophet_opt = result.x
print(f"\n  Optimized weights (MAE-min): LGBM={w_lgbm_opt:.3f}  Prophet={w_prophet_opt:.3f}")
oof_opt_pred = w_lgbm_opt * X_meta[:,0] + w_prophet_opt * X_meta[:,1]
print(f"  Optimized OOF MAE: {mean_absolute_error(y_oof, oof_opt_pred):,.0f}")

# Pick best weighting method
if mean_absolute_error(y_oof, oof_ridge_pred) < mean_absolute_error(y_oof, oof_opt_pred):
    W_LGBM, W_PROPHET = w_lgbm_ridge_n, w_prophet_ridge_n
    method_name = "Ridge"
else:
    W_LGBM, W_PROPHET = w_lgbm_opt, w_prophet_opt
    method_name = "MAE-Optimized"

print(f"\n  >> Selected: {method_name}  LGBM={W_LGBM:.3f}  Prophet={W_PROPHET:.3f}")

# =============================================================================
# 4. FULL TRAINING on 2012-2022
# =============================================================================
print("\n" + "=" * 60)
print("Step 4: Full training on entire 2012-2022...")
print("=" * 60)

# LGBM full
X_all  = build_lgbm_features(df["Date"], rmap_full)
y_all  = np.log1p(df["Revenue"].values)
valid  = X_all["lag_364_roll"].notna() & (X_all["lag_364_roll"] > 0)
X_tr   = X_all[valid][FEAT_COLS].copy()
y_tr   = y_all[valid]
dates_tr = df.loc[valid, "Date"].reset_index(drop=True)
weights  = make_weights(dates_tr)

print("  Training LGBM (full)...")
lgbm_full = lgb.LGBMRegressor(**LGBM_PARAMS)
lgbm_full.fit(X_tr, y_tr, sample_weight=weights)

val22     = dates_tr.dt.year == 2022
y_v22     = np.expm1(y_tr[val22.values])
yp_v22    = np.expm1(lgbm_full.predict(X_tr[val22.values]))
print(f"  LGBM MAE (2022): {mean_absolute_error(y_v22, yp_v22):,.0f}")

print("  Training Prophet (full)...")
prophet_full = train_prophet(df)
pf22 = predict_prophet(prophet_full, df.loc[df["Date"].dt.year == 2022, "Date"].values)
print(f"  Prophet MAE (2022): {mean_absolute_error(df.loc[df['Date'].dt.year==2022,'Revenue'].values, pf22):,.0f}")

# =============================================================================
# 5. FORECAST 2023-2024
# =============================================================================
print("\n" + "=" * 60)
print("Step 5: Forecasting 2023-2024...")
print("=" * 60)

# LGBM recursive
print("  LGBM recursive forecasting...")
lgbm_preds = recursive_lgbm_forecast(lgbm_full, FEAT_COLS, fdates, rmap_full)
lgbm_s = pd.Series(lgbm_preds)
print(f"  LGBM: mean={lgbm_s.mean():,.0f}  min={lgbm_s.min():,.0f}  max={lgbm_s.max():,.0f}")

# Prophet direct
print("  Prophet forecasting...")
prophet_raw = predict_prophet(prophet_full, fdates.values)
prophet_preds = dict(zip(fdates, prophet_raw))
prophet_s = pd.Series(prophet_preds)
print(f"  Prophet: mean={prophet_s.mean():,.0f}  min={prophet_s.min():,.0f}  max={prophet_s.max():,.0f}")

# =============================================================================
# 6. ENSEMBLE with LEARNED WEIGHTS + POST-PROCESSING
# =============================================================================
print("\n" + "=" * 60)
print(f"Step 6: Ensemble (LGBM={W_LGBM:.3f}, Prophet={W_PROPHET:.3f}) + Post-processing...")
print("=" * 60)

final_preds = {}
for fdate in fdates:
    lv = lgbm_preds.get(fdate, np.nan)
    pv = prophet_preds.get(fdate, np.nan)
    if np.isnan(lv): lv = pv
    if np.isnan(pv): pv = lv

    blended = W_LGBM * lv + W_PROPHET * pv
    decayed = blended * decay_factor(fdate)
    final   = float(np.clip(decayed, 0.0, CLIP_HIGH))
    final_preds[fdate] = final

final_s  = pd.Series(final_preds)
last_avg = df[df["Date"].dt.year == 2022]["Revenue"].mean()
print(f"  Final: mean={final_s.mean():,.0f}  min={final_s.min():,.0f}  max={final_s.max():,.0f}")
print(f"  Ratio vs 2022 avg: {final_s.mean()/last_avg:.3f}")

# =============================================================================
# 7. WRITE SUBMISSION
# =============================================================================
sub["Revenue"] = sub["Date"].map(final_preds)
if sub["Revenue"].isna().any():
    sub["Revenue"] = sub["Revenue"].interpolate("linear")
sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub[["Date", "Revenue", "COGS"]].to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved -> {OUTPUT_PATH}")
print(sub[["Date", "Revenue", "COGS"]].head(10).to_string(index=False))

print("\n" + "=" * 60)
print("DONE! Meta-Learner Stacking Ensemble complete.")
print(f"  Meta method     : {method_name}")
print(f"  Learned weights : LGBM={W_LGBM:.3f}  Prophet={W_PROPHET:.3f}")
print(f"  Decay/year      : {np.exp(slope*365.25):.4f}")
print(f"  Clip P95        : {CLIP_HIGH:,.0f}")
print("=" * 60)
