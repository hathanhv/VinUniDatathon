# -*- coding: utf-8 -*-
"""
=============================================================================
VinUni Datathon 2026 - Recursive Revenue Forecasting Pipeline
=============================================================================
Strategy:
  - Feature Engineering chi dua vao Date (deterministic + Fourier + Lags)
  - Recursive Forecasting: dung ket qua du bao de cap nhat Lag cho buoc tiep theo
  - LightGBM + log1p(Revenue)
  - Sample Weights: trong so cao hon cho 2019-2022
  - Output: submission.csv voi Revenue + giu nguyen COGS tu sample_submission.csv
=============================================================================
"""

import sys
import io
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output to avoid encoding errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# 0. PATHS
# =============================================================================
BASE = Path(r"d:\vinuni_datathon2026\vinuni_datathon2026\model")
DATA_PATH   = BASE / "processed_data.csv"
SAMPLE_PATH = BASE / "sample_submission.csv"
OUTPUT_PATH = BASE / "submission.csv"

# =============================================================================
# 1. LOAD & CLEAN TRAINING DATA
# =============================================================================
print("=" * 60)
print("Step 1: Loading training data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.rename(columns={"date": "Date"})
df = df[["Date", "Revenue"]].dropna(subset=["Revenue"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"  Training data: {df['Date'].min().date()} -> {df['Date'].max().date()}")
print(f"  Total rows   : {len(df)}")
print(f"  Revenue range: {df['Revenue'].min():,.0f} -> {df['Revenue'].max():,.0f}")

# =============================================================================
# 2. FEATURE ENGINEERING FUNCTION
# =============================================================================
PEAK_DATE   = pd.Timestamp("2017-01-01")  # moc tham chieu trend giam
YEAR_PERIOD = 365.25                       # ky han nam cho Fourier

def make_date_features(dates: pd.Series) -> pd.DataFrame:
    """
    Tao tat ca deterministic features tu cot Date.
    Day la cac features ta LUON biet truoc trong tuong lai.
    """
    feats = pd.DataFrame(index=dates.index)
    d = pd.DatetimeIndex(dates)

    # -- Trend -----------------------------------------------------------
    # So ngay tu peak 2017-01-01 (am = truoc, duong = sau)
    feats["trend_days_from_peak"] = (dates - PEAK_DATE).dt.days

    # -- Day-of-Month ratio ----------------------------------------------
    days_in_month = d.days_in_month
    feats["dom_ratio"] = (d.day - 1) / (days_in_month - 1)

    # -- Seasonal flags --------------------------------------------------
    feats["is_peak_season"]   = d.month.isin([4, 5, 6]).astype(int)    # T4-T6
    feats["is_low_season"]    = d.month.isin([11, 12, 1]).astype(int)  # T11-T1
    feats["is_qtr_end_month"] = d.month.isin([3, 6, 9, 12]).astype(int)

    # -- Calendar basics -------------------------------------------------
    feats["month"]        = d.month
    feats["day_of_week"]  = d.dayofweek        # 0=Mon...6=Sun
    feats["day_of_year"]  = d.dayofyear
    feats["is_weekend"]   = (d.dayofweek >= 5).astype(int)
    feats["week_of_year"] = d.isocalendar().week.astype(int)
    feats["quarter"]      = d.quarter
    feats["year"]         = d.year

    # -- Fourier Features (Yearly seasonality) ---------------------------
    # k = 1..4 cap sin/cos theo ky han nam de hoc tinh mua vu manh
    t = d.dayofyear / YEAR_PERIOD * 2 * np.pi
    for k in range(1, 5):
        feats[f"sin_year_{k}"] = np.sin(k * t)
        feats[f"cos_year_{k}"] = np.cos(k * t)

    # -- Fourier Features (Weekly seasonality) ---------------------------
    t_week = d.dayofweek / 7 * 2 * np.pi
    for k in range(1, 3):
        feats[f"sin_week_{k}"] = np.sin(k * t_week)
        feats[f"cos_week_{k}"] = np.cos(k * t_week)

    # -- Tet / Holiday proximity -----------------------------------------
    # Cuoi thang 1 dau thang 2 = mua Tet Nguyen Dan
    feats["is_tet_window"] = (
        ((d.month == 1) & (d.day >= 20)) |
        ((d.month == 2) & (d.day <= 20))
    ).astype(int)

    # Cuoi nam Christmas/New Year
    feats["is_year_end_window"] = (
        ((d.month == 12) & (d.day >= 20)) |
        ((d.month == 1)  & (d.day <= 5))
    ).astype(int)

    return feats


def get_date_feature_names():
    """Tra ve list ten feature khong bao gom lag."""
    tmp = make_date_features(pd.Series([pd.Timestamp("2022-01-01")]))
    return list(tmp.columns)


# =============================================================================
# 3. XAY DUNG TRAINING DATASET voi LAG FEATURES
# =============================================================================
print("\nStep 2: Building training dataset with lag features...")

LAG_DAYS = [364, 728]   # YoY lags (1 nam, 2 nam)

# Map tu Date -> Revenue de tra cuu nhanh khi tinh lag
date_revenue_map = dict(zip(df["Date"], df["Revenue"]))

def get_lag(date: pd.Timestamp, lag_days: int, revenue_map: dict) -> float:
    """Tra cuu Revenue tai (date - lag_days). Tra ve NaN neu khong co."""
    target_date = date - pd.Timedelta(days=lag_days)
    return revenue_map.get(target_date, np.nan)

# Build feature matrix cho tap train
date_feats = make_date_features(df["Date"])
for lag in LAG_DAYS:
    lag_col = f"lag_{lag}"
    df[lag_col] = df["Date"].apply(lambda d: get_lag(d, lag, date_revenue_map))
    date_feats[lag_col] = df[lag_col].values

# Log1p transform target
df["log_revenue"] = np.log1p(df["Revenue"])

# Danh sach tat ca feature columns (date + lag)
lag_cols       = [f"lag_{l}" for l in LAG_DAYS]
all_feat_cols  = get_date_feature_names() + lag_cols

# Ap dung log1p cho lag columns (phai giu nhat quan voi luc train)
for lag in LAG_DAYS:
    date_feats[f"lag_{lag}"] = np.log1p(date_feats[f"lag_{lag}"].clip(lower=0))

# Chi giu hang co du du lieu: lag_364 it nhat phai co
valid_mask  = date_feats["lag_364"].notna()
X_train     = date_feats[valid_mask].copy()
y_train     = df.loc[valid_mask, "log_revenue"].values
train_dates = df.loc[valid_mask, "Date"].reset_index(drop=True)

print(f"  Training samples: {len(X_train)}")
print(f"  Features        : {len(all_feat_cols)}")
print(f"  Feature list    : {all_feat_cols}")

# =============================================================================
# 4. SAMPLE WEIGHTS: tang trong so cho 2019-2022
# =============================================================================
print("\nStep 3: Computing sample weights...")

def compute_sample_weights(dates: pd.Series) -> np.ndarray:
    """
    Trong so tuyen tinh theo nam:
      - Truoc 2019  : weight = 1.0
      - 2019-2022   : weight tang dan tu 2.0 -> 5.0
      - Normalize ve khoang [1, 5]
    """
    years   = dates.dt.year.values
    weights = np.ones(len(dates), dtype=float)

    year_weight = {2019: 2.0, 2020: 3.0, 2021: 4.0, 2022: 5.0}
    for yr, w in year_weight.items():
        weights[years == yr] = w

    min_w, max_w = weights.min(), weights.max()
    if max_w > min_w:
        weights = 1.0 + 4.0 * (weights - min_w) / (max_w - min_w)

    return weights

sample_weights = compute_sample_weights(train_dates)
print(f"  Weight range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")

# =============================================================================
# 5. HUAN LUYEN LIGHTGBM
# =============================================================================
print("\nStep 4: Training LightGBM model...")

# Dien NaN lag_728 bang lag_364 (khi khong du 2 nam lich su)
X_train_filled = X_train.copy()
X_train_filled["lag_728"] = X_train_filled["lag_728"].fillna(X_train_filled["lag_364"])

lgb_params = {
    "objective"        : "regression",
    "metric"           : "mae",
    "n_estimators"     : 1500,
    "learning_rate"    : 0.03,
    "num_leaves"       : 63,
    "max_depth"        : 7,
    "min_child_samples": 30,
    "feature_fraction" : 0.8,
    "bagging_fraction" : 0.8,
    "bagging_freq"     : 5,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "verbose"          : -1,
}

model = lgb.LGBMRegressor(**lgb_params)
model.fit(
    X_train_filled[all_feat_cols],
    y_train,
    sample_weight=sample_weights,
)
print("  Model trained successfully!")

# Feature importance
feat_imp = pd.Series(
    model.feature_importances_,
    index=all_feat_cols
).sort_values(ascending=False)
print("\n  Top 15 Feature Importances:")
print(feat_imp.head(15).to_string())

# =============================================================================
# 6. KIEM TRA TREN TAP VALIDATION (nam 2022)
# =============================================================================
print("\nStep 5: In-sample validation on year 2022...")

val_mask   = train_dates.dt.year == 2022
X_val      = X_train_filled[val_mask.values][all_feat_cols]
y_val_true = df.loc[valid_mask, "Revenue"].values[val_mask.values]

y_val_pred_log = model.predict(X_val)
y_val_pred     = np.expm1(y_val_pred_log)

mae_val  = np.mean(np.abs(y_val_pred - y_val_true))
mape_val = np.mean(np.abs((y_val_pred - y_val_true) / (y_val_true + 1))) * 100
print(f"  MAE  (2022): {mae_val:,.0f}")
print(f"  MAPE (2022): {mape_val:.2f}%")

# =============================================================================
# 7. RECURSIVE FORECASTING (Du bao Cuon chieu)
# =============================================================================
print("\nStep 6: Recursive forecasting 2023-01-01 -> 2024-07-01...")

sub = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"])
sub = sub.sort_values("Date").reset_index(drop=True)
forecast_dates = sub["Date"]

print(f"  Forecast period : {forecast_dates.min().date()} -> {forecast_dates.max().date()}")
print(f"  Total days      : {len(forecast_dates)}")

# Khoi tao rolling map voi toan bo lich su train
rolling_revenue_map = dict(date_revenue_map)
predicted_revenues  = {}

for i, fdate in enumerate(forecast_dates):
    if (i + 1) % 100 == 0 or i == 0:
        print(f"  Day {i+1:3d}/{len(forecast_dates)}: {fdate.date()}")

    # 7a. Date features (luon biet truoc)
    feat_row = make_date_features(pd.Series([fdate]))

    # 7b. Lag features (dung rolling_revenue_map)
    for lag in LAG_DAYS:
        lag_val = get_lag(fdate, lag, rolling_revenue_map)
        if np.isnan(lag_val):
            # Fallback: tim lag gan nhat trong +-7 ngay
            for delta in range(1, 8):
                lag_val = rolling_revenue_map.get(
                    fdate - pd.Timedelta(days=lag + delta), np.nan
                )
                if not np.isnan(lag_val):
                    break
            if np.isnan(lag_val):
                for delta in range(1, 8):
                    lag_val = rolling_revenue_map.get(
                        fdate - pd.Timedelta(days=lag - delta), np.nan
                    )
                    if not np.isnan(lag_val):
                        break
        # log1p transform nhu khi train
        if not np.isnan(lag_val):
            feat_row[f"lag_{lag}"] = np.log1p(max(float(lag_val), 0.0))
        else:
            feat_row[f"lag_{lag}"] = 0.0

    # Dien NaN lag_728 bang lag_364
    if pd.isna(feat_row["lag_728"].values[0]):
        feat_row["lag_728"] = feat_row["lag_364"]

    # 7c. Predict
    x        = feat_row[all_feat_cols]
    log_pred = float(model.predict(x)[0])
    rev_pred = max(float(np.expm1(log_pred)), 0.0)

    # 7d. Cap nhat rolling map cho buoc tiep theo
    predicted_revenues[fdate]      = rev_pred
    rolling_revenue_map[fdate] = rev_pred

print(f"\n  Recursive forecasting complete!")

# =============================================================================
# 8. THONG KE DU BAO
# =============================================================================
pred_series = pd.Series(predicted_revenues)
print("\nStep 7: Forecast statistics:")
print(f"  Min Revenue   : {pred_series.min():,.0f}")
print(f"  Max Revenue   : {pred_series.max():,.0f}")
print(f"  Mean Revenue  : {pred_series.mean():,.0f}")
print(f"  Median Revenue: {pred_series.median():,.0f}")

last_train_rev = df[df["Date"].dt.year == 2022]["Revenue"].mean()
print(f"\n  Avg Revenue 2022 (train): {last_train_rev:,.0f}")
print(f"  Avg Revenue Forecast    : {pred_series.mean():,.0f}")
print(f"  Ratio                   : {pred_series.mean() / last_train_rev:.3f}")

# =============================================================================
# 9. TAO FILE SUBMISSION
# =============================================================================
print("\nStep 8: Creating submission file...")

# Dien Revenue du bao vao
sub["Revenue"] = sub["Date"].map(predicted_revenues)

missing_rev = sub["Revenue"].isna().sum()
if missing_rev > 0:
    print(f"  WARNING: {missing_rev} dates missing Revenue -> interpolating...")
    sub["Revenue"] = sub["Revenue"].interpolate(method="linear")

# Giu nguyen COGS tu sample_submission (khong can predict COGS)
sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)

sub[["Date", "Revenue", "COGS"]].to_csv(OUTPUT_PATH, index=False)

print(f"  Saved: {OUTPUT_PATH}")
print(f"  Shape: {sub.shape}")
print("\nSample output (first 10 rows):")
print(sub[["Date", "Revenue", "COGS"]].head(10).to_string(index=False))

print("\nSample output (Tet window ~Jan 25 - Feb 5):")
tet_mask = (sub["Date"] >= "2023-01-25") & (sub["Date"] <= "2023-02-05")
print(sub[tet_mask][["Date", "Revenue", "COGS"]].to_string(index=False))

print("\nSample output (April peak season):")
apr_mask = (sub["Date"] >= "2023-03-29") & (sub["Date"] <= "2023-04-05")
print(sub[apr_mask][["Date", "Revenue", "COGS"]].to_string(index=False))

print("\n" + "=" * 60)
print("DONE! Recursive Revenue Forecasting Pipeline Complete.")
print("=" * 60)
