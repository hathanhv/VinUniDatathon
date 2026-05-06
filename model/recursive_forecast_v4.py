# -*- coding: utf-8 -*-
"""
VinUni Datathon 2026 - Recursive Revenue Forecasting v4
Design: Lean + Trend-aware

Problems fixed vs v3:
  - Overpredict ~3x because lag_7/14/28 anchor on recent high values
    and accumulate error drift over 1.5-year horizon
  - Structural trend decline (post-2017) not captured in future predictions

v4 Strategy:
  1. Remove short-term lags entirely (only YoY lags as anchor)
  2. Compute Trend Decay rate from training data (fit log-linear slope)
  3. Apply exponential decay post-processing: pred *= decay^months
  4. Clip spikes at P95 of training data
  5. Fourier k=1..5 for smooth seasonality
  6. Heavier YoY blend (alpha=0.30) to anchor on actual historical scale
"""

import sys, io, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from scipy import stats
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
OUTPUT_PATH = BASE / "submission_v4.csv"

PEAK_DATE   = pd.Timestamp("2017-01-01")
YEAR_PERIOD = 365.25
ROLLING_WIN = 7

# Blend: lgbm vs yoy anchor (heavier yoy to prevent drift)
YOY_ALPHA   = 0.30   # 70% lgbm + 30% yoy

# Optuna
N_OPTUNA  = 80
N_SPLITS  = 5

# Clip spikes above this quantile of training Revenue
SPIKE_CLIP_Q = 0.95

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

def days_to_nearest_tet(date: pd.Timestamp) -> float:
    cands = [abs((date - t).days) for yr, t in TET_DATES.items()
             if abs(date.year - yr) <= 1]
    return float(min(cands)) if cands else 60.0

# =============================================================================
# LAG HELPERS
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
    return float(v)

# =============================================================================
# 10 FEATURES (no short-term lags)
# =============================================================================
FEATURES = [
    "lag_364_roll",        # 1. Rolling YoY lag (main anchor)
    "lag_728_roll",        # 2. Rolling 2-year lag
    "yoy_ratio",           # 3. YoY growth ratio
    "trend_days_from_peak",# 4. Trend since 2017 peak
    "dom_ratio",           # 5. Day-of-month position [0,1]
    "days_to_tet",         # 6. Distance to Tet
    "sin_year_1",          # 7-8. Fourier k=1
    "cos_year_1",
    "sin_year_2",          # 9-10. Fourier k=2
    "cos_year_2",
]
# Optional: add k=3..5 if helps
FOURIER_K = 5

def build_features(dates: pd.Series, rmap: dict) -> pd.DataFrame:
    feats = pd.DataFrame(index=dates.index)
    d = pd.DatetimeIndex(dates)

    feats["trend_days_from_peak"] = (dates - PEAK_DATE).dt.days
    feats["dom_ratio"] = (d.day - 1) / (d.days_in_month - 1)
    feats["days_to_tet"] = dates.apply(days_to_nearest_tet)

    # Fourier yearly k=1..FOURIER_K
    t = d.dayofyear / YEAR_PERIOD * 2 * np.pi
    for k in range(1, FOURIER_K + 1):
        feats[f"sin_year_{k}"] = np.sin(k * t)
        feats[f"cos_year_{k}"] = np.cos(k * t)

    # Rolling YoY lags
    roll364 = dates.apply(lambda x: rolling_yoy_lag(x, 364, rmap))
    roll728 = dates.apply(lambda x: rolling_yoy_lag(x, 728, rmap))
    ex364   = dates.apply(lambda x: exact_lag(x, 364, rmap))
    ex728   = dates.apply(lambda x: exact_lag(x, 728, rmap))

    lag364 = roll364.fillna(ex364)
    lag728 = roll728.fillna(ex728).fillna(lag364)

    feats["lag_364_roll"] = np.log1p(lag364.clip(lower=0))
    feats["lag_728_roll"] = np.log1p(lag728.clip(lower=0))

    safe728 = lag728.replace(0, np.nan)
    feats["yoy_ratio"] = (lag364 / safe728).clip(0.3, 2.5).fillna(1.0)

    # Build full feature list (Fourier k=1..K creates 2K cols + others)
    all_cols = (["lag_364_roll", "lag_728_roll", "yoy_ratio",
                 "trend_days_from_peak", "dom_ratio", "days_to_tet"] +
                [f"sin_year_{k}" for k in range(1, FOURIER_K + 1)] +
                [f"cos_year_{k}" for k in range(1, FOURIER_K + 1)])
    return feats[all_cols]

def get_feature_cols():
    return build_features(pd.Series([pd.Timestamp("2022-06-01")]),
                          {pd.Timestamp("2021-06-01"): 1e6,
                           pd.Timestamp("2020-06-01"): 1e6}).columns.tolist()

FEAT_COLS = get_feature_cols()

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
print(f"  Features ({len(FEAT_COLS)}): {FEAT_COLS}")

# =============================================================================
# 2. COMPUTE TREND DECAY FROM TRAINING DATA
#    Fit log(Revenue) ~ days_since_peak using recent data (2017-2022)
#    to get the actual daily decline rate
# =============================================================================
print("\nStep 2: Computing trend decay from training data...")

decay_df = df[df["Date"] >= "2017-01-01"].copy()
decay_df["t_days"] = (decay_df["Date"] - pd.Timestamp("2017-01-01")).dt.days
decay_df["log_rev"] = np.log(decay_df["Revenue"].clip(lower=1))

slope, intercept, r_val, p_val, _ = stats.linregress(
    decay_df["t_days"], decay_df["log_rev"]
)
daily_decay = slope        # negative = declining
monthly_decay = np.exp(daily_decay * 30.44)
annual_decay  = np.exp(daily_decay * 365.25)

print(f"  Log-linear slope: {daily_decay:.6f} per day")
print(f"  Monthly decay   : {monthly_decay:.4f}  ({(monthly_decay-1)*100:.2f}% per month)")
print(f"  Annual decay    : {annual_decay:.4f}   ({(annual_decay-1)*100:.2f}% per year)")
print(f"  R^2             : {r_val**2:.3f}")

# Decay applied from TRAIN_END onwards
def decay_factor(forecast_date: pd.Timestamp) -> float:
    """Multiplicative decay vs train_end date."""
    days_ahead = (forecast_date - TRAIN_END).days
    return float(np.exp(daily_decay * days_ahead))

# Spike clip threshold
CLIP_HIGH = float(np.quantile(df["Revenue"], SPIKE_CLIP_Q))
print(f"  Spike clip (P{int(SPIKE_CLIP_Q*100)}): {CLIP_HIGH:,.0f}")

# =============================================================================
# 3. BUILD TRAINING MATRIX
# =============================================================================
print("\nStep 3: Building training matrix...")
X_all = build_features(df["Date"], rmap_train)
y_all = np.log1p(df["Revenue"].values)

valid = X_all["lag_364_roll"].notna() & (X_all["lag_364_roll"] > 0)
X_tr  = X_all[valid][FEAT_COLS].copy()
y_tr  = y_all[valid]
dates_tr = df.loc[valid, "Date"].reset_index(drop=True)
print(f"  Samples: {len(X_tr)}")

# =============================================================================
# 4. SAMPLE WEIGHTS (upweight 2019-2022)
# =============================================================================
def make_weights(dates):
    w = np.ones(len(dates))
    for yr, wt in {2019: 2.0, 2020: 3.0, 2021: 4.0, 2022: 5.0}.items():
        w[dates.dt.year.values == yr] = wt
    mn, mx = w.min(), w.max()
    return 1.0 + 4.0 * (w - mn) / (mx - mn) if mx > mn else w

weights = make_weights(dates_tr)

# =============================================================================
# 5. OPTUNA TUNING
# =============================================================================
tuning_label = f"Optuna {N_OPTUNA} trials" if HAS_OPTUNA else "default params"
print(f"\nStep 4: Tuning ({tuning_label})...")

BASE_P = dict(
    objective="regression", metric="mae",
    n_estimators=2000, learning_rate=0.03,
    num_leaves=48, max_depth=5,
    min_child_samples=25, feature_fraction=0.9,
    bagging_fraction=0.85, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbose=-1,
)

if HAS_OPTUNA:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=30)

    def objective(trial):
        p = dict(
            objective="regression", metric="mae",
            n_estimators=trial.suggest_int("n_estimators", 800, 3500, step=200),
            learning_rate=trial.suggest_float("lr", 0.008, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 80),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_samples=trial.suggest_int("mcs", 10, 60),
            feature_fraction=trial.suggest_float("ff", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bf", 0.5, 1.0),
            bagging_freq=5,
            reg_alpha=trial.suggest_float("alpha", 1e-4, 2.0, log=True),
            reg_lambda=trial.suggest_float("lam", 1e-4, 2.0, log=True),
            random_state=42, n_jobs=-1, verbose=-1,
        )
        maes = []
        for tri, vai in tscv.split(X_tr):
            m = lgb.LGBMRegressor(**p)
            m.fit(X_tr.iloc[tri], y_tr[tri], sample_weight=weights[tri],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)],
                  eval_set=[(X_tr.iloc[vai], y_tr[vai])])
            maes.append(mean_absolute_error(
                np.expm1(y_tr[vai]), np.expm1(m.predict(X_tr.iloc[vai]))))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    bp = study.best_params
    FINAL_P = {**BASE_P, "n_estimators": bp["n_estimators"],
               "learning_rate": bp["lr"], "num_leaves": bp["num_leaves"],
               "max_depth": bp["max_depth"], "min_child_samples": bp["mcs"],
               "feature_fraction": bp["ff"], "bagging_fraction": bp["bf"],
               "reg_alpha": bp["alpha"], "reg_lambda": bp["lam"]}
    print(f"  Best CV MAE: {study.best_value:,.0f}")
    print(f"  Params    : {bp}")
else:
    FINAL_P = BASE_P
    print("  Using base params.")

# =============================================================================
# 6. TRAIN FINAL MODEL
# =============================================================================
print("\nStep 5: Training model...")
model = lgb.LGBMRegressor(**FINAL_P)
model.fit(X_tr, y_tr, sample_weight=weights)

imp = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
print("  Feature importances:")
print(imp.to_string())

# =============================================================================
# 7. VALIDATE ON 2022
# =============================================================================
print("\nStep 6: Validation 2022...")
vm          = dates_tr.dt.year == 2022
yt          = np.expm1(y_tr[vm.values])
yp_raw      = np.expm1(model.predict(X_tr[vm.values]))
dates_val22 = dates_tr[vm.values].reset_index(drop=True)
yp = np.array([
    min(yp_raw[i] * decay_factor(dates_val22[i]), CLIP_HIGH)
    for i in range(len(yp_raw))
])
mae_v  = mean_absolute_error(yt, yp)
mape_v = np.mean(np.abs((yp - yt) / (yt + 1))) * 100
print(f"  MAE  (2022): {mae_v:,.0f}")
print(f"  MAPE (2022): {mape_v:.2f}%")
print(f"  [v1=105,795 | v2+Optuna=86,251 | v3=49,706]")

# =============================================================================
# 8. RECURSIVE FORECASTING
# =============================================================================
print("\nStep 7: Recursive forecasting...")
sub    = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"])
sub    = sub.sort_values("Date").reset_index(drop=True)
fdates = sub["Date"]
print(f"  {fdates.min().date()} -> {fdates.max().date()} ({len(fdates)} days)")

rolling_map = dict(rmap_train)
preds       = {}

for i, fdate in enumerate(fdates):
    if (i + 1) % 100 == 0 or i == 0:
        df_fac = decay_factor(fdate)
        print(f"  Day {i+1:3d}/{len(fdates)}: {fdate.date()}  decay={df_fac:.4f}")

    row    = build_features(pd.Series([fdate]), rolling_map)
    lgbm_p = float(np.expm1(model.predict(row[FEAT_COLS])[0]))

    # YoY blend anchor
    yoy_base = rolling_yoy_lag(fdate, 364, rolling_map)
    if np.isnan(yoy_base):
        yoy_base = lgbm_p
    yoy_ratio_val = float(row["yoy_ratio"].values[0])
    yoy_p = yoy_base * yoy_ratio_val

    blended = YOY_ALPHA * yoy_p + (1.0 - YOY_ALPHA) * lgbm_p

    # Apply trend decay (key addition in v4)
    decayed = blended * decay_factor(fdate)

    # Clip spike
    final = float(np.clip(decayed, 0.0, CLIP_HIGH))

    preds[fdate]       = final
    rolling_map[fdate] = final   # feed back

# =============================================================================
# 9. WRITE SUBMISSION
# =============================================================================
pred_s   = pd.Series(preds)
last_avg = df[df["Date"].dt.year == 2022]["Revenue"].mean()

print(f"\nStep 8: Forecast stats:")
print(f"  Min={pred_s.min():,.0f}  Max={pred_s.max():,.0f}")
print(f"  Mean={pred_s.mean():,.0f}  Median={pred_s.median():,.0f}")
print(f"  2022 train avg : {last_avg:,.0f}")
print(f"  Forecast/2022  : {pred_s.mean()/last_avg:.3f}")

sub["Revenue"] = sub["Date"].map(preds)
if sub["Revenue"].isna().any():
    sub["Revenue"] = sub["Revenue"].interpolate("linear")
sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub[["Date", "Revenue", "COGS"]].to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved -> {OUTPUT_PATH}")
print(sub[["Date", "Revenue", "COGS"]].head(15).to_string(index=False))

print("\n" + "=" * 60)
print("DONE! v4 complete.")
print("Key differences vs v3:")
print("  - Removed lag_7/14/28 (recursive noise over 1.5yr horizon)")
print("  - Added Trend Decay post-processing from log-linear fit")
print(f"    Annual decay: {annual_decay:.4f} ({(annual_decay-1)*100:.1f}%/yr)")
print(f"  - Spike clip at P95: {CLIP_HIGH:,.0f}")
print(f"  - YoY blend alpha  : {YOY_ALPHA}")
print("=" * 60)
