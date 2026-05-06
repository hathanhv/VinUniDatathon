# -*- coding: utf-8 -*-
"""
VinUni Datathon 2026 - Ensemble: LightGBM + Prophet
=====================================================
Architecture:
  Branch 1 (40%): LightGBM
    - Features: Fourier k=5, dom_ratio, lag_364_rolling
    - Recursive forecasting (lag updated each step)
    - Log1p target

  Branch 2 (60%): Prophet
    - Captures long-term trend decline (~45% from 2017)
    - Yearly + weekly seasonality (multiplicative)
    - changepoint_prior_scale tuned for flexible trend

  Ensemble -> Post-processing:
    - Trend Decay Factor (log-linear from training data)
    - Spike Clipping at P95
"""

import sys, io, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# =============================================================================
# CONFIG
# =============================================================================
BASE        = Path(r"d:\vinuni_datathon2026\vinuni_datathon2026\model")
DATA_PATH   = BASE / "processed_data.csv"
SAMPLE_PATH = BASE / "sample_submission.csv"
OUTPUT_PATH = BASE / "submission_ensemble.csv"

PEAK_DATE    = pd.Timestamp("2017-01-01")
YEAR_PERIOD  = 365.25
ROLLING_WIN  = 7

LGBM_WEIGHT    = 0.40   # 40% LightGBM
PROPHET_WEIGHT = 0.60   # 60% Prophet
SPIKE_CLIP_Q   = 0.95   # clip at P95 of training Revenue

# Exact Tet dates
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

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("Step 1: Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.rename(columns={"date": "Date"})
df = df[["Date", "Revenue"]].dropna().sort_values("Date").reset_index(drop=True)
rmap_train = dict(zip(df["Date"], df["Revenue"]))
TRAIN_END  = df["Date"].max()
print(f"  Rows: {len(df)}  {df['Date'].min().date()} -> {TRAIN_END.date()}")

sub    = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"])
sub    = sub.sort_values("Date").reset_index(drop=True)
fdates = sub["Date"]
print(f"  Forecast: {fdates.min().date()} -> {fdates.max().date()} ({len(fdates)} days)")

# Post-processing setup
CLIP_HIGH = float(np.quantile(df["Revenue"], SPIKE_CLIP_Q))

# Trend decay from log-linear fit (2017-2022)
decay_df = df[df["Date"] >= "2017-01-01"].copy()
decay_df["t"] = (decay_df["Date"] - PEAK_DATE).dt.days
slope, _, _, _, _ = stats.linregress(decay_df["t"], np.log(decay_df["Revenue"].clip(1)))
print(f"  Decay slope: {slope:.6f}/day  ({np.exp(slope*365.25):.4f}/year)")
print(f"  Spike clip (P95): {CLIP_HIGH:,.0f}")

def decay_factor(date):
    return float(np.exp(slope * (date - TRAIN_END).days))

# =============================================================================
# BRANCH 1: LightGBM (Recursive)
# =============================================================================
print("\n" + "=" * 60)
print("BRANCH 1: LightGBM Pipeline")
print("=" * 60)

# --- Feature Engineering ---
def lgbm_features(dates: pd.Series, rmap: dict) -> pd.DataFrame:
    """
    Features per diagram:
      - Fourier k=1..5 (yearly seasonality)
      - dom_ratio
      - lag_364_rolling
    Plus extras that proved useful:
      - trend_days_from_peak
      - days_to_tet
      - lag_728_rolling
      - yoy_ratio
    """
    feats = pd.DataFrame(index=dates.index)
    d = pd.DatetimeIndex(dates)

    feats["trend_days_from_peak"] = (dates - PEAK_DATE).dt.days
    feats["dom_ratio"] = (d.day - 1) / (d.days_in_month - 1)
    feats["days_to_tet"] = dates.apply(days_to_nearest_tet)

    # Fourier k=1..5 (yearly)
    t = d.dayofyear / YEAR_PERIOD * 2 * np.pi
    for k in range(1, 6):
        feats[f"sin_year_{k}"] = np.sin(k * t)
        feats[f"cos_year_{k}"] = np.cos(k * t)

    # Rolling YoY lags
    r364 = dates.apply(lambda x: rolling_yoy_lag(x, 364, rmap))
    r728 = dates.apply(lambda x: rolling_yoy_lag(x, 728, rmap))
    e364 = dates.apply(lambda x: exact_lag(x, 364, rmap))
    e728 = dates.apply(lambda x: exact_lag(x, 728, rmap))

    lag364 = r364.fillna(e364)
    lag728 = r728.fillna(e728).fillna(lag364)

    feats["lag_364_roll"] = np.log1p(lag364.clip(lower=0))
    feats["lag_728_roll"] = np.log1p(lag728.clip(lower=0))

    safe728 = lag728.replace(0, np.nan)
    feats["yoy_ratio"] = (lag364 / safe728).clip(0.3, 2.5).fillna(1.0)

    return feats

LGBM_COLS = lgbm_features(
    pd.Series([pd.Timestamp("2022-06-01")]),
    {pd.Timestamp("2021-06-01"): 1e6, pd.Timestamp("2020-06-01"): 1e6}
).columns.tolist()

print(f"  LGBM features ({len(LGBM_COLS)}): {LGBM_COLS}")

# --- Build training matrix ---
print("\n  Building training matrix...")
X_lgbm = lgbm_features(df["Date"], rmap_train)
y_lgbm = np.log1p(df["Revenue"].values)
valid  = X_lgbm["lag_364_roll"].notna() & (X_lgbm["lag_364_roll"] > 0)
X_tr   = X_lgbm[valid][LGBM_COLS].copy()
y_tr   = y_lgbm[valid]
dates_tr = df.loc[valid, "Date"].reset_index(drop=True)
print(f"  Training samples: {len(X_tr)}")

# Sample weights
def make_weights(dates):
    w = np.ones(len(dates))
    for yr, wt in {2019: 2.0, 2020: 3.0, 2021: 4.0, 2022: 5.0}.items():
        w[dates.dt.year.values == yr] = wt
    mn, mx = w.min(), w.max()
    return 1.0 + 4.0 * (w - mn) / (mx - mn) if mx > mn else w

weights = make_weights(dates_tr)

# Train LightGBM
print("  Training LightGBM...")
lgbm_params = dict(
    objective="regression", metric="mae",
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=6,
    min_child_samples=25, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=5,
    reg_alpha=0.08, reg_lambda=0.08,
    random_state=42, n_jobs=-1, verbose=-1,
)
lgbm_model = lgb.LGBMRegressor(**lgbm_params)
lgbm_model.fit(X_tr, y_tr, sample_weight=weights)
print("  LightGBM trained!")

# Validate on 2022
vm22   = dates_tr.dt.year == 2022
yt22   = np.expm1(y_tr[vm22.values])
yp22   = np.expm1(lgbm_model.predict(X_tr[vm22.values]))
mae22  = mean_absolute_error(yt22, yp22)
mape22 = np.mean(np.abs((yp22 - yt22) / (yt22 + 1))) * 100
print(f"  LGBM MAE  (2022): {mae22:,.0f}")
print(f"  LGBM MAPE (2022): {mape22:.2f}%")

# Recursive forecasting
print("\n  Recursive forecasting (LGBM)...")
rolling_map = dict(rmap_train)
lgbm_preds  = {}

for i, fdate in enumerate(fdates):
    if (i + 1) % 100 == 0 or i == 0:
        print(f"    Day {i+1:3d}/{len(fdates)}: {fdate.date()}")

    row    = lgbm_features(pd.Series([fdate]), rolling_map)
    raw_p  = float(np.expm1(lgbm_model.predict(row[LGBM_COLS])[0]))

    # YoY blend
    yoy_b = rolling_yoy_lag(fdate, 364, rolling_map)
    if np.isnan(yoy_b):
        yoy_b = raw_p
    yoy_r = float(row["yoy_ratio"].values[0])
    blended = 0.20 * (yoy_b * yoy_r) + 0.80 * raw_p
    blended = max(blended, 0.0)

    lgbm_preds[fdate] = blended
    rolling_map[fdate] = blended

lgbm_series = pd.Series(lgbm_preds)
print(f"  LGBM forecast: mean={lgbm_series.mean():,.0f}  min={lgbm_series.min():,.0f}  max={lgbm_series.max():,.0f}")

# =============================================================================
# BRANCH 2: Prophet
# =============================================================================
print("\n" + "=" * 60)
print("BRANCH 2: Prophet Pipeline")
print("=" * 60)

try:
    from prophet import Prophet

    # Prepare Prophet training data
    prophet_df = df[["Date", "Revenue"]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df["y"] = np.log1p(prophet_df["y"])  # log-space for stability

    print("  Training Prophet...")
    prophet_model = Prophet(
        seasonality_mode="multiplicative",   # Revenue data: multiplicative seasonality
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,         # Flexible trend (capture 45% decline)
        seasonality_prior_scale=10.0,        # Strong seasonality
        changepoint_range=0.95,              # Allow changepoints near end of training
        interval_width=0.80,
    )

    # Add Vietnamese Tet as custom seasonality
    def is_tet(ds):
        date = pd.Timestamp(ds)
        for yr, tet in TET_DATES.items():
            if abs((date - tet).days) <= 14:
                return 1.0
        return 0.0

    prophet_model.add_regressor("is_tet", standardize=False)
    prophet_df["is_tet"] = prophet_df["ds"].apply(is_tet)

    prophet_model.fit(prophet_df)
    print("  Prophet trained!")

    # Forecast future dates
    future = pd.DataFrame({"ds": fdates.values})
    future["is_tet"] = future["ds"].apply(is_tet)
    forecast = prophet_model.predict(future)

    # Inverse log1p to get actual Revenue
    prophet_preds = {}
    for _, row in forecast.iterrows():
        dt = pd.Timestamp(row["ds"])
        prophet_preds[dt] = float(np.expm1(max(row["yhat"], 0.0)))

    prophet_series = pd.Series(prophet_preds)
    print(f"  Prophet forecast: mean={prophet_series.mean():,.0f}  min={prophet_series.min():,.0f}  max={prophet_series.max():,.0f}")

    # Validate Prophet on 2022 (backcast)
    hist_future = pd.DataFrame({"ds": df.loc[df["Date"].dt.year == 2022, "Date"].values})
    hist_future["is_tet"] = hist_future["ds"].apply(is_tet)
    hist_fc = prophet_model.predict(hist_future)
    y_prop22 = np.expm1(hist_fc["yhat"].clip(lower=0).values)
    y_true22 = df.loc[df["Date"].dt.year == 2022, "Revenue"].values
    mae_p22   = mean_absolute_error(y_true22, y_prop22)
    mape_p22  = np.mean(np.abs((y_prop22 - y_true22) / (y_true22 + 1))) * 100
    print(f"  Prophet MAE  (2022): {mae_p22:,.0f}")
    print(f"  Prophet MAPE (2022): {mape_p22:.2f}%")

    HAS_PROPHET = True

except Exception as e:
    print(f"  [WARN] Prophet failed: {e}")
    print("  Fallback: using LGBM-only (weight=1.0)")
    prophet_series = lgbm_series.copy()
    HAS_PROPHET = False

# =============================================================================
# ENSEMBLE: 40% LGBM + 60% Prophet
# =============================================================================
print("\n" + "=" * 60)
print("ENSEMBLE: Combining branches...")
print("=" * 60)

ensemble_preds = {}
for fdate in fdates:
    lgbm_v   = lgbm_preds.get(fdate, np.nan)
    prophet_v = prophet_preds.get(fdate, lgbm_v)

    if np.isnan(lgbm_v):
        lgbm_v = prophet_v
    if np.isnan(prophet_v):
        prophet_v = lgbm_v

    blended = LGBM_WEIGHT * lgbm_v + PROPHET_WEIGHT * prophet_v
    ensemble_preds[fdate] = max(blended, 0.0)

ensemble_series = pd.Series(ensemble_preds)
print(f"  Ensemble (raw): mean={ensemble_series.mean():,.0f}  min={ensemble_series.min():,.0f}  max={ensemble_series.max():,.0f}")

# =============================================================================
# POST-PROCESSING: Decay + Clip
# =============================================================================
print("\nPost-processing: Trend Decay + Spike Clipping...")

final_preds = {}
for fdate in fdates:
    raw = ensemble_preds[fdate]
    # Apply trend decay
    decayed = raw * decay_factor(fdate)
    # Clip spike
    clipped = min(max(decayed, 0.0), CLIP_HIGH)
    final_preds[fdate] = clipped

final_series = pd.Series(final_preds)
print(f"  Final forecast  : mean={final_series.mean():,.0f}  min={final_series.min():,.0f}  max={final_series.max():,.0f}")
print(f"  2022 train avg  : {df[df['Date'].dt.year==2022]['Revenue'].mean():,.0f}")
print(f"  Forecast/2022   : {final_series.mean() / df[df['Date'].dt.year==2022]['Revenue'].mean():.3f}")

# =============================================================================
# WRITE SUBMISSION
# =============================================================================
print("\nWriting submission...")
sub["Revenue"] = sub["Date"].map(final_preds)
if sub["Revenue"].isna().any():
    sub["Revenue"] = sub["Revenue"].interpolate("linear")
sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub[["Date", "Revenue", "COGS"]].to_csv(OUTPUT_PATH, index=False)

print(f"  Saved -> {OUTPUT_PATH}")
print("\nSample output (first 15 rows):")
print(sub[["Date", "Revenue", "COGS"]].head(15).to_string(index=False))

print("\nComponent comparison (first 5 dates):")
cmp = pd.DataFrame({
    "Date"   : [d.strftime("%Y-%m-%d") for d in list(fdates)[:5]],
    "LGBM"   : [f"{lgbm_preds[d]:,.0f}" for d in list(fdates)[:5]],
    "Prophet": [f"{prophet_preds.get(d, 0):,.0f}" for d in list(fdates)[:5]],
    "Final"  : [f"{final_preds[d]:,.0f}" for d in list(fdates)[:5]],
})
print(cmp.to_string(index=False))

print("\n" + "=" * 60)
print("DONE! Ensemble (LightGBM + Prophet) complete.")
print(f"  Weights: LGBM={LGBM_WEIGHT*100:.0f}%  Prophet={PROPHET_WEIGHT*100:.0f}%")
print(f"  Decay  : {np.exp(slope*365.25):.4f}/year")
print(f"  Clip   : P{int(SPIKE_CLIP_Q*100)} = {CLIP_HIGH:,.0f}")
print(f"  Prophet: {'OK' if HAS_PROPHET else 'FAILED (LGBM only)'}")
print("=" * 60)
