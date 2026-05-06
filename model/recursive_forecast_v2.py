# -*- coding: utf-8 -*-
"""
VinUni Datathon 2026 - Recursive Revenue Forecasting v2
Improvements over v1:
  1. Rolling YoY Lag (mean of +-7 days window) - kills outlier sensitivity
  2. Short-term lags (lag_7/14/28) anchored from training data
  3. Optuna Hyperparameter Tuning with TimeSeriesSplit
  4. YoY Baseline Blend (alpha*lgbm + (1-alpha)*yoy_scaled)
  5. Exact Vietnamese Tet dates -> days_to_tet feature
"""

import sys, io, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  [WARN] optuna not installed. Run: pip install optuna")

# =============================================================================
# 0. CONFIG
# =============================================================================
BASE        = Path(r"d:\vinuni_datathon2026\vinuni_datathon2026\model")
DATA_PATH   = BASE / "processed_data.csv"
SAMPLE_PATH = BASE / "sample_submission.csv"
OUTPUT_PATH = BASE / "submission_v2.csv"

PEAK_DATE   = pd.Timestamp("2017-01-01")
YEAR_PERIOD = 365.25
ROLLING_WIN = 7        # +-7 ngay quanh cung ky nam truoc
YOY_ALPHA   = 0.15     # blend: 80% lgbm + 20% yoy_scaled
N_OPTUNA    = 150       # so trial Optuna (tang len de tuning tot hon)
N_SPLITS    = 5        # TimeSeriesSplit folds

# Exact Tet (Lunar New Year) dates for each Gregorian year
TET_DATES = {
    2012: pd.Timestamp("2012-01-23"), 2013: pd.Timestamp("2013-02-10"),
    2014: pd.Timestamp("2014-01-31"), 2015: pd.Timestamp("2015-02-19"),
    2016: pd.Timestamp("2016-02-08"), 2017: pd.Timestamp("2017-01-28"),
    2018: pd.Timestamp("2018-02-16"), 2019: pd.Timestamp("2019-02-05"),
    2020: pd.Timestamp("2020-01-25"), 2021: pd.Timestamp("2021-02-12"),
    2022: pd.Timestamp("2022-02-01"), 2023: pd.Timestamp("2023-01-22"),
    2024: pd.Timestamp("2024-02-10"),
}

def days_to_nearest_tet(date: pd.Timestamp) -> int:
    """So ngay gan nhat den Tet (am hoac duong)."""
    candidates = []
    for yr in [date.year - 1, date.year, date.year + 1]:
        if yr in TET_DATES:
            candidates.append(abs((date - TET_DATES[yr]).days))
    return min(candidates) if candidates else 60

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("Step 1: Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.rename(columns={"date": "Date"})
df = df[["Date", "Revenue"]].dropna(subset=["Revenue"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"  Rows: {len(df)}  |  {df['Date'].min().date()} -> {df['Date'].max().date()}")

date_revenue_map = dict(zip(df["Date"], df["Revenue"]))

# =============================================================================
# 2. FEATURE ENGINEERING (v2: rolling lag + days_to_tet)
# =============================================================================
def rolling_yoy_lag(date: pd.Timestamp, lag_center: int,
                    revenue_map: dict, win: int = ROLLING_WIN) -> float:
    """Mean Revenue trong window [lag_center-win, lag_center+win] ngay truoc."""
    vals = []
    for delta in range(-win, win + 1):
        d = date - pd.Timedelta(days=lag_center + delta)
        v = revenue_map.get(d, np.nan)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def get_exact_lag(date: pd.Timestamp, lag_days: int, revenue_map: dict) -> float:
    """Lag chinh xac, fallback +-7 ngay neu khong co."""
    v = revenue_map.get(date - pd.Timedelta(days=lag_days), np.nan)
    if np.isnan(v):
        for d in range(1, 8):
            for sign in [1, -1]:
                v = revenue_map.get(date - pd.Timedelta(days=lag_days + sign * d), np.nan)
                if not np.isnan(v):
                    return v
    return v

def make_features(dates: pd.Series, revenue_map: dict,
                  use_short_lags: bool = True) -> pd.DataFrame:
    """
    Build feature dataframe cho danh sach dates.
    use_short_lags=True: them lag_7/14/28 (chi dung khi co trong revenue_map)
    """
    feats = pd.DataFrame(index=dates.index)
    d = pd.DatetimeIndex(dates)

    # -- Trend
    feats["trend_days_from_peak"] = (dates - PEAK_DATE).dt.days

    # -- DOM ratio
    feats["dom_ratio"] = (d.day - 1) / (d.days_in_month - 1)

    # -- Seasonal flags
    feats["is_peak_season"]   = d.month.isin([4, 5, 6]).astype(int)
    feats["is_low_season"]    = d.month.isin([11, 12, 1]).astype(int)
    feats["is_qtr_end_month"] = d.month.isin([3, 6, 9, 12]).astype(int)

    # -- Calendar
    feats["month"]        = d.month
    feats["day_of_week"]  = d.dayofweek
    feats["day_of_year"]  = d.dayofyear
    feats["is_weekend"]   = (d.dayofweek >= 5).astype(int)
    feats["week_of_year"] = d.isocalendar().week.astype(int)
    feats["quarter"]      = d.quarter
    feats["year"]         = d.year
    feats["days_in_month"]= d.days_in_month

    # -- Fourier yearly (k=1..5)
    t = d.dayofyear / YEAR_PERIOD * 2 * np.pi
    for k in range(1, 6):
        feats[f"sin_year_{k}"] = np.sin(k * t)
        feats[f"cos_year_{k}"] = np.cos(k * t)

    # -- Fourier weekly (k=1..3)
    t_week = d.dayofweek / 7 * 2 * np.pi
    for k in range(1, 4):
        feats[f"sin_week_{k}"] = np.sin(k * t_week)
        feats[f"cos_week_{k}"] = np.cos(k * t_week)

    # -- Exact Tet distance
    feats["days_to_tet"] = dates.apply(days_to_nearest_tet)
    feats["is_tet_week"]  = (feats["days_to_tet"] <= 7).astype(int)
    feats["is_tet_month"] = (feats["days_to_tet"] <= 30).astype(int)

    # -- Year-end window
    feats["is_year_end_window"] = (
        ((d.month == 12) & (d.day >= 20)) |
        ((d.month == 1)  & (d.day <= 5))
    ).astype(int)

    # -- ROLLING YoY Lags (KEY IMPROVEMENT v2)
    # Rolling mean +-7 ngay quanh cung ky nam truoc (khong bi outlier)
    roll_364 = dates.apply(lambda x: rolling_yoy_lag(x, 364, revenue_map))
    roll_728 = dates.apply(lambda x: rolling_yoy_lag(x, 728, revenue_map))

    # Fallback: neu rolling khong co thi dung exact lag
    exact_364 = dates.apply(lambda x: get_exact_lag(x, 364, revenue_map))
    exact_728 = dates.apply(lambda x: get_exact_lag(x, 728, revenue_map))

    lag364 = roll_364.fillna(exact_364)
    lag728 = roll_728.fillna(exact_728).fillna(lag364)  # fallback to lag364

    feats["lag_364_roll"] = np.log1p(lag364.clip(lower=0))
    feats["lag_728_roll"] = np.log1p(lag728.clip(lower=0))

    # Ratio: tang truong YoY
    safe_728 = lag728.replace(0, np.nan)
    feats["yoy_ratio"] = (lag364 / safe_728).clip(0.5, 2.0).fillna(1.0)

    # -- Short-term lags (chi co gia tri khi nam trong training / recursive map)
    if use_short_lags:
        for short_lag in [7, 14, 28]:
            vals = dates.apply(lambda x: get_exact_lag(x, short_lag, revenue_map))
            feats[f"lag_{short_lag}"] = np.log1p(vals.clip(lower=0).fillna(lag364.clip(lower=0)))

    return feats


def get_all_feat_cols(use_short_lags: bool = True):
    tmp = make_features(
        pd.Series([pd.Timestamp("2022-06-01")]),
        {pd.Timestamp("2022-06-01"): 1e6},
        use_short_lags=use_short_lags
    )
    return list(tmp.columns)

ALL_FEAT_COLS = get_all_feat_cols(use_short_lags=True)
print(f"  Total features: {len(ALL_FEAT_COLS)}")

# =============================================================================
# 3. BUILD TRAINING MATRIX
# =============================================================================
print("\nStep 2: Building training matrix...")
X_all = make_features(df["Date"], date_revenue_map, use_short_lags=True)
y_all = np.log1p(df["Revenue"].values)

# Chi giu hang co lag_364
valid_mask  = X_all["lag_364_roll"].notna() & (X_all["lag_364_roll"] > 0)
X_train     = X_all[valid_mask][ALL_FEAT_COLS].copy()
y_train     = y_all[valid_mask]
train_dates = df.loc[valid_mask, "Date"].reset_index(drop=True)
print(f"  Training samples: {len(X_train)}")

# =============================================================================
# 4. SAMPLE WEIGHTS
# =============================================================================
def make_weights(dates: pd.Series) -> np.ndarray:
    years   = dates.dt.year.values
    weights = np.ones(len(dates), dtype=float)
    for yr, w in {2019: 2.0, 2020: 3.0, 2021: 4.0, 2022: 5.0}.items():
        weights[years == yr] = w
    mn, mx = weights.min(), weights.max()
    return 1.0 + 4.0 * (weights - mn) / (mx - mn) if mx > mn else weights

sample_weights = make_weights(train_dates)

# =============================================================================
# 5. OPTUNA HYPERPARAMETER TUNING
# =============================================================================
print(f"\nStep 3: Hyperparameter tuning ({'Optuna' if HAS_OPTUNA else 'default'})...")

DEFAULT_PARAMS = {
    "objective": "regression", "metric": "mae",
    "n_estimators": 2000, "learning_rate": 0.025,
    "num_leaves": 63, "max_depth": 7,
    "min_child_samples": 30, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.1, "reg_lambda": 0.1,
    "random_state": 42, "n_jobs": -1, "verbose": -1,
}

if HAS_OPTUNA:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=30)

    def objective(trial):
        params = {
            "objective": "regression", "metric": "mae",
            "n_estimators": trial.suggest_int("n_estimators", 800, 3000, step=200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 5,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        maes = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            sw_tr = sample_weights[tr_idx]
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr, sample_weight=sw_tr,
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)],
                  eval_set=[(X_val, y_val)])
            pred = np.expm1(m.predict(X_val))
            true = np.expm1(y_val)
            maes.append(mean_absolute_error(true, pred))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    BEST_PARAMS = {**DEFAULT_PARAMS, **study.best_params}
    print(f"  Best CV MAE : {study.best_value:,.0f}")
    print(f"  Best params : {study.best_params}")
else:
    BEST_PARAMS = DEFAULT_PARAMS
    print("  Using default params.")

# =============================================================================
# 6. TRAIN FINAL MODEL
# =============================================================================
print("\nStep 4: Training final LightGBM model...")
model = lgb.LGBMRegressor(**BEST_PARAMS)
model.fit(X_train, y_train, sample_weight=sample_weights)
print("  Done!")

feat_imp = pd.Series(model.feature_importances_, index=ALL_FEAT_COLS).sort_values(ascending=False)
print("\n  Top 15 Feature Importances:")
print(feat_imp.head(15).to_string())

# =============================================================================
# 7. VALIDATION ON 2022
# =============================================================================
print("\nStep 5: Validation on 2022...")
val_mask   = train_dates.dt.year == 2022
X_val      = X_train[val_mask.values]
y_val_true = np.expm1(y_train[val_mask.values])
y_val_pred = np.expm1(model.predict(X_val))

mae_val  = mean_absolute_error(y_val_true, y_val_pred)
mape_val = np.mean(np.abs((y_val_pred - y_val_true) / (y_val_true + 1))) * 100
print(f"  MAE  (2022): {mae_val:,.0f}")
print(f"  MAPE (2022): {mape_val:.2f}%")

# =============================================================================
# 8. RECURSIVE FORECASTING + YoY BLEND
# =============================================================================
print("\nStep 6: Recursive forecasting with YoY blend...")

sub            = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"])
sub            = sub.sort_values("Date").reset_index(drop=True)
forecast_dates = sub["Date"]

print(f"  Period: {forecast_dates.min().date()} -> {forecast_dates.max().date()}")
print(f"  Days  : {len(forecast_dates)}")

rolling_map        = dict(date_revenue_map)  # grows as we predict
predicted_revenues = {}

for i, fdate in enumerate(forecast_dates):
    if (i + 1) % 100 == 0 or i == 0:
        print(f"  Day {i+1:3d}/{len(forecast_dates)}: {fdate.date()}")

    # Build single-row feature
    feat_row = make_features(pd.Series([fdate]), rolling_map, use_short_lags=True)
    x = feat_row[ALL_FEAT_COLS]

    # LightGBM prediction
    lgbm_pred = float(np.expm1(model.predict(x)[0]))
    lgbm_pred = max(lgbm_pred, 0.0)

    # YoY Naive Baseline: lay rolling mean cung ky nam truoc, scale theo yoy_ratio
    yoy_base = rolling_yoy_lag(fdate, 364, rolling_map)
    if np.isnan(yoy_base):
        yoy_base = lgbm_pred
    yoy_ratio = float(feat_row["yoy_ratio"].values[0])
    yoy_pred  = yoy_base * yoy_ratio

    # Blend: alpha * lgbm + (1-alpha) * yoy
    final_pred = YOY_ALPHA * yoy_pred + (1.0 - YOY_ALPHA) * lgbm_pred
    final_pred = max(final_pred, 0.0)

    predicted_revenues[fdate] = final_pred
    rolling_map[fdate]        = final_pred  # feed back into lag

print("  Recursive done!")

# =============================================================================
# 9. OUTPUT STATISTICS & SUBMISSION
# =============================================================================
pred_s = pd.Series(predicted_revenues)
print("\nStep 7: Forecast statistics:")
print(f"  Min : {pred_s.min():,.0f}")
print(f"  Max : {pred_s.max():,.0f}")
print(f"  Mean: {pred_s.mean():,.0f}")
print(f"  Med : {pred_s.median():,.0f}")

last_yr_avg = df[df["Date"].dt.year == 2022]["Revenue"].mean()
print(f"\n  2022 avg (train): {last_yr_avg:,.0f}")
print(f"  Forecast avg    : {pred_s.mean():,.0f}  (ratio={pred_s.mean()/last_yr_avg:.3f})")

# Write submission
sub["Revenue"] = sub["Date"].map(predicted_revenues)
if sub["Revenue"].isna().any():
    sub["Revenue"] = sub["Revenue"].interpolate("linear")

sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub[["Date", "Revenue", "COGS"]].to_csv(OUTPUT_PATH, index=False)

print(f"\nStep 8: Saved -> {OUTPUT_PATH}")
print(sub[["Date", "Revenue", "COGS"]].head(10).to_string(index=False))

print("\n" + "=" * 60)
print("DONE! recursive_forecast_v2.py complete.")
print(f"  v1 MAE (2022) was ~105,795 -- compare above!")
print("=" * 60)
