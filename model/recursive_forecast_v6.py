# -*- coding: utf-8 -*-
"""
VinUni Datathon 2026 - v6: Exogenous YoY-Shifted Features
===========================================================
Breakthrough idea: Lay cac feature co MI cao tu processed_data.csv
va shift chung 364 ngay de tao input cho 2023-2024.

- Training: lag364_X[t] = X[t-364]  (du lieu that cua nam truoc)
- Test 2023: lag364_X[2023-xx] = X[2022-xx] (co trong training data)
- Test 2024: lag728_X[2024-xx] = X[2022-xx] (fallback 2 nam)

=> Khong can recursion cho exogenous features!
   Chi co Revenue van recursive nhu cu.
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

# =============================================================================
# CONFIG
# =============================================================================
BASE        = Path(r"d:\vinuni_datathon2026\vinuni_datathon2026\model")
DATA_PATH   = BASE / "processed_data.csv"
SAMPLE_PATH = BASE / "sample_submission.csv"
OUTPUT_PATH = BASE / "submission_v6.csv"

PEAK_DATE   = pd.Timestamp("2017-01-01")
YEAR_PERIOD = 365.25
ROLLING_WIN = 7
YOY_ALPHA   = 0.15
N_OPTUNA    = 80
N_SPLITS    = 5

# Top MI features from EDA (can shift by 364 days)
EXOG_FEATURES = [
    "order_id",                    # MI=1.165 — total orders/day
    "order_status_delivered",      # MI=1.100
    "device_type_mobile",          # MI=1.014
    "device_type_desktop",         # MI=1.011
    "order_source_organic_search", # MI=0.926
    "order_source_paid_search",    # MI=0.886
    "total_quantity",              # volume signal
    "sessions",                    # web traffic (real from 2019)
    "unique_visitors",             # web traffic (real from 2019)
]

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
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("Step 1: Loading data...")
df_raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
df_raw = df_raw.rename(columns={"date": "Date"}).sort_values("Date").reset_index(drop=True)

# Revenue series
df = df_raw[["Date", "Revenue"]].dropna(subset=["Revenue"]).reset_index(drop=True)
rmap_train = dict(zip(df["Date"], df["Revenue"]))
TRAIN_END  = df["Date"].max()

print(f"  Revenue rows: {len(df)}  {df['Date'].min().date()} -> {TRAIN_END.date()}")
print(f"  Exog features: {EXOG_FEATURES}")

# =============================================================================
# 2. BUILD EXOGENOUS LAG LOOKUP (vectorized)
#    Extend date range to include test dates so shift() works for 2023-2024
# =============================================================================
print("\nStep 2: Building exogenous lag features...")

sub     = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"])
sub     = sub.sort_values("Date").reset_index(drop=True)
fdates  = sub["Date"]
MAX_DATE = fdates.max()

# Continuous daily date range from training start to test end
all_dates = pd.date_range(df["Date"].min(), MAX_DATE, freq="D")

# Build exog feature df on full date range (NaN for test dates)
df_exog = df_raw.set_index("Date")[EXOG_FEATURES]
df_exog = df_exog.reindex(all_dates)   # NaN for 2023-2024 test dates

# Log1p transform, clip negatives
df_exog_log = np.log1p(df_exog.clip(lower=0))

# Shift + rolling mean (window=15 = +-7 days)
# shift(364) on continuous daily index = exactly 364 calendar days back
df_lag364 = df_exog_log.shift(364).rolling(15, center=True, min_periods=5).mean()
df_lag728 = df_exog_log.shift(728).rolling(15, center=True, min_periods=5).mean()

# Build lookup dicts for fast access during feature construction
lag364_maps = {f: df_lag364[f].dropna().to_dict() for f in EXOG_FEATURES}
lag728_maps = {f: df_lag728[f].dropna().to_dict() for f in EXOG_FEATURES}

# Global median fallback (for truly missing)
global_medians = {f: df_exog_log[f].median() for f in EXOG_FEATURES}

def get_exog_lag(date, feat):
    """Get lag_364 value for exog feature; fallback to lag_728 then global median."""
    v = lag364_maps[feat].get(date, np.nan)
    if np.isnan(v):
        v = lag728_maps[feat].get(date, np.nan)
    if np.isnan(v):
        v = global_medians[feat]
    return float(v)

# Validate: check coverage for training and test dates
train_coverage = np.mean([
    not np.isnan(lag364_maps[EXOG_FEATURES[0]].get(d, np.nan))
    for d in df["Date"].iloc[400:]  # skip first year (no lag_364 yet)
])
test_2023_cov = np.mean([
    not np.isnan(lag364_maps[EXOG_FEATURES[0]].get(d, np.nan))
    for d in fdates if d.year == 2023
])
test_2024_cov = np.mean([
    not np.isnan(lag728_maps[EXOG_FEATURES[0]].get(d, np.nan))
    for d in fdates if d.year == 2024
])
print(f"  Lag364 coverage: train={train_coverage:.1%}  2023={test_2023_cov:.1%}")
print(f"  Lag728 coverage: 2024={test_2024_cov:.1%}")

# =============================================================================
# 3. REVENUE LAG HELPERS (same as v2)
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
# 4. FEATURE ENGINEERING
# =============================================================================
def build_features(dates: pd.Series, rmap: dict) -> pd.DataFrame:
    feats = pd.DataFrame(index=dates.index)
    d = pd.DatetimeIndex(dates)

    # Trend
    feats["trend_days_from_peak"] = (dates - PEAK_DATE).dt.days

    # Calendar
    feats["dom_ratio"]        = (d.day - 1) / (d.days_in_month - 1)
    feats["is_peak_season"]   = d.month.isin([4, 5, 6]).astype(int)
    feats["is_low_season"]    = d.month.isin([11, 12, 1]).astype(int)
    feats["is_qtr_end_month"] = d.month.isin([3, 6, 9, 12]).astype(int)
    feats["month"]            = d.month
    feats["day_of_week"]      = d.dayofweek
    feats["is_weekend"]       = (d.dayofweek >= 5).astype(int)
    feats["quarter"]          = d.quarter

    # Fourier yearly k=1..5
    t = d.dayofyear / YEAR_PERIOD * 2 * np.pi
    for k in range(1, 6):
        feats[f"sin_year_{k}"] = np.sin(k * t)
        feats[f"cos_year_{k}"] = np.cos(k * t)

    # Fourier weekly k=1..2
    t_w = d.dayofweek / 7 * 2 * np.pi
    for k in range(1, 3):
        feats[f"sin_week_{k}"] = np.sin(k * t_w)
        feats[f"cos_week_{k}"] = np.cos(k * t_w)

    # Tet
    feats["days_to_tet"] = dates.apply(days_to_nearest_tet)

    # Revenue lags (YoY rolling)
    r364 = dates.apply(lambda x: rolling_yoy_lag(x, 364, rmap))
    r728 = dates.apply(lambda x: rolling_yoy_lag(x, 728, rmap))
    e364 = dates.apply(lambda x: exact_lag(x, 364, rmap))
    e728 = dates.apply(lambda x: exact_lag(x, 728, rmap))

    lag364 = r364.fillna(e364)
    lag728 = r728.fillna(e728).fillna(lag364)

    feats["rev_lag_364"] = np.log1p(lag364.clip(lower=0))
    feats["rev_lag_728"] = np.log1p(lag728.clip(lower=0))
    feats["yoy_ratio"]   = (lag364 / lag728.replace(0, np.nan)).clip(0.3, 2.5).fillna(1.0)

    # Short-term Revenue lags
    for short in [7, 14, 28]:
        vals = dates.apply(lambda x: exact_lag(x, short, rmap))
        feats[f"rev_lag_{short}"] = np.log1p(
            vals.clip(lower=0).fillna(lag364.clip(lower=0))
        )

    # === EXOGENOUS YoY-Shifted Features (KEY v6 addition) ===
    for feat in EXOG_FEATURES:
        feats[f"exog_{feat}"] = dates.apply(lambda x: get_exog_lag(x, feat))

    return feats

FEAT_COLS = build_features(
    pd.Series([pd.Timestamp("2022-06-01")]),
    {pd.Timestamp("2021-06-01"): 1e6, pd.Timestamp("2020-06-01"): 1e6}
).columns.tolist()

print(f"  Total features: {len(FEAT_COLS)}")
print(f"  = {len(FEAT_COLS)-len(EXOG_FEATURES)} time/revenue + {len(EXOG_FEATURES)} exogenous")

# =============================================================================
# 5. TRAINING MATRIX
# =============================================================================
print("\nStep 3: Building training matrix...")
X_all = build_features(df["Date"], rmap_train)
y_all = np.log1p(df["Revenue"].values)

valid    = X_all["rev_lag_364"].notna() & (X_all["rev_lag_364"] > 0)
X_tr     = X_all[valid][FEAT_COLS].copy()
y_tr     = y_all[valid]
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

# =============================================================================
# 6. OPTUNA TUNING
# =============================================================================
label = f"Optuna {N_OPTUNA} trials" if HAS_OPTUNA else "default params"
print(f"\nStep 4: Tuning ({label})...")

BASE_P = dict(objective="regression", metric="mae",
              n_estimators=2000, learning_rate=0.03,
              num_leaves=63, max_depth=6, min_child_samples=25,
              feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
              reg_alpha=0.08, reg_lambda=0.08, random_state=42, n_jobs=-1, verbose=-1)

if HAS_OPTUNA:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=30)

    def objective(trial):
        p = {**BASE_P,
             "n_estimators":      trial.suggest_int("ne", 800, 3500, step=200),
             "learning_rate":     trial.suggest_float("lr", 0.008, 0.1, log=True),
             "num_leaves":        trial.suggest_int("nl", 16, 96),
             "max_depth":         trial.suggest_int("md", 3, 8),
             "min_child_samples": trial.suggest_int("mc", 10, 60),
             "feature_fraction":  trial.suggest_float("ff", 0.5, 1.0),
             "bagging_fraction":  trial.suggest_float("bf", 0.5, 1.0),
             "reg_alpha":         trial.suggest_float("ra", 1e-4, 2.0, log=True),
             "reg_lambda":        trial.suggest_float("rl", 1e-4, 2.0, log=True)}
        maes = []
        for tri, vai in tscv.split(X_tr):
            m = lgb.LGBMRegressor(**p)
            m.fit(X_tr.iloc[tri], y_tr[tri], sample_weight=weights[tri],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
                  eval_set=[(X_tr.iloc[vai], y_tr[vai])])
            maes.append(mean_absolute_error(
                np.expm1(y_tr[vai]), np.expm1(m.predict(X_tr.iloc[vai]))))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    bp = study.best_params
    FINAL_P = {**BASE_P, "n_estimators": bp["ne"], "learning_rate": bp["lr"],
               "num_leaves": bp["nl"], "max_depth": bp["md"],
               "min_child_samples": bp["mc"], "feature_fraction": bp["ff"],
               "bagging_fraction": bp["bf"], "reg_alpha": bp["ra"], "reg_lambda": bp["rl"]}
    print(f"  Best CV MAE: {study.best_value:,.0f}")
    print(f"  Params: {bp}")
else:
    FINAL_P = BASE_P

# =============================================================================
# 7. TRAIN & VALIDATE
# =============================================================================
print("\nStep 5: Training final model...")
model = lgb.LGBMRegressor(**FINAL_P)
model.fit(X_tr, y_tr, sample_weight=weights)

imp = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
print("  Top 20 importances (exog features marked *):")
for fname, val in imp.head(20).items():
    mark = " [EXOG]" if fname.startswith("exog_") else ""
    print(f"    {fname:<40s} {val}{mark}")

vm     = dates_tr.dt.year == 2022
y_t    = np.expm1(y_tr[vm.values])
y_p    = np.expm1(model.predict(X_tr[vm.values]))

from sklearn.metrics import mean_squared_error, r2_score
mae_v  = mean_absolute_error(y_t, y_p)
rmse_v = np.sqrt(mean_squared_error(y_t, y_p))
r2_v   = r2_score(y_t, y_p)
mape_v = np.mean(np.abs((y_p - y_t) / (y_t + 1))) * 100

print(f"\n  === Validation 2022 ===")
print(f"  MAE  : {mae_v:,.0f}")
print(f"  RMSE : {rmse_v:,.0f}")
print(f"  R²   : {r2_v:.4f}")
print(f"  MAPE : {mape_v:.2f}%")

# =============================================================================
# 8. RECURSIVE FORECASTING (Revenue only recursive; exog is pre-computed)
# =============================================================================
print("\nStep 6: Recursive forecasting 2023-2024...")
rolling_map = dict(rmap_train)
preds       = {}

for i, fdate in enumerate(fdates):
    if (i + 1) % 100 == 0 or i == 0:
        print(f"  Day {i+1:3d}/{len(fdates)}: {fdate.date()}")

    row    = build_features(pd.Series([fdate]), rolling_map)
    lgbm_p = float(np.expm1(model.predict(row[FEAT_COLS])[0]))
    lgbm_p = max(lgbm_p, 0.0)

    yoy_base = rolling_yoy_lag(fdate, 364, rolling_map)
    if not np.isnan(yoy_base):
        yoy_r  = float(row["yoy_ratio"].values[0])
        lgbm_p = YOY_ALPHA * (yoy_base * yoy_r) + (1.0 - YOY_ALPHA) * lgbm_p

    preds[fdate]        = lgbm_p
    rolling_map[fdate]  = lgbm_p

pred_s   = pd.Series(preds)
last_avg = df[df["Date"].dt.year == 2022]["Revenue"].mean()
print(f"\n  Forecast mean={pred_s.mean():,.0f}  median={pred_s.median():,.0f}")
print(f"  Ratio vs 2022: {pred_s.mean()/last_avg:.3f}")

# =============================================================================
# 9. WRITE SUBMISSION
# =============================================================================
sub["Revenue"] = sub["Date"].map(preds)
if sub["Revenue"].isna().any():
    sub["Revenue"] = sub["Revenue"].interpolate("linear")
sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub[["Date", "Revenue", "COGS"]].to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved -> {OUTPUT_PATH}")
print(sub[["Date", "Revenue", "COGS"]].head(10).to_string(index=False))

print("\n" + "=" * 60)
print("DONE! v6 (Exogenous YoY Lags) complete.")
print(f"  Exog features: {len(EXOG_FEATURES)} (lag_364/728 from training data)")
print("  Revenue: still recursive (short-term lags)")
print("=" * 60)
