# -*- coding: utf-8 -*-
"""
VinUni Datathon 2026 - Recursive Revenue Forecasting v3
Strategy: Lean model - chi 11 features co anh huong nhat (tu v2 importance)
  1. lag_7, lag_14, lag_28       -- short-term pattern (top 3 importance)
  2. yoy_ratio                   -- xu huong tang truong YoY
  3. dom_ratio                   -- vi tri ngay trong thang
  4. trend_days_from_peak        -- xu huong dai han
  5. lag_364_roll                -- rolling YoY lag (+-7 ngay)
  6. lag_728_roll                -- rolling 2YoY lag
  7. days_to_tet                 -- khoang cach Tet chinh xac
  8. sin_year_1, cos_year_1      -- 1 cap Fourier nam (nen tang mua vu)
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
    print("  [WARN] optuna not found. pip install optuna")

# =============================================================================
# CONFIG
# =============================================================================
BASE        = Path(r"d:\vinuni_datathon2026\vinuni_datathon2026\model")
DATA_PATH   = BASE / "processed_data.csv"
SAMPLE_PATH = BASE / "sample_submission.csv"
OUTPUT_PATH = BASE / "submission_v3.csv"

PEAK_DATE   = pd.Timestamp("2017-01-01")
YEAR_PERIOD = 365.25
ROLLING_WIN = 7     # window +-7 ngay cho rolling YoY lag
YOY_ALPHA   = 0.15  # blend: 85% lgbm + 15% yoy_scaled
N_OPTUNA    = 80    # Optuna trials
N_SPLITS    = 5     # TimeSeriesSplit folds

# 11 features duoc chon (thu tu theo importance tu v2)
SELECTED_FEATURES = [
    "lag_7",               # 1. short-term lag 7 ngay
    "lag_14",              # 2. short-term lag 14 ngay
    "lag_28",              # 3. short-term lag 28 ngay
    "yoy_ratio",           # 4. ti le tang truong YoY
    "dom_ratio",           # 5. vi tri ngay trong thang [0,1]
    "trend_days_from_peak",# 6. so ngay tu dinh 2017
    "lag_364_roll",        # 7. rolling YoY lag (+-7 ngay)
    "lag_728_roll",        # 8. rolling 2-nam lag
    "days_to_tet",         # 9. khoang cach den Tet am lich
    "sin_year_1",          # 10. Fourier nam (sin)
    "cos_year_1",          # 11. Fourier nam (cos)
]

# Lich Tet am lich chinh xac tung nam
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
# HELPER: lag lookup
# =============================================================================
def rolling_yoy_lag(date: pd.Timestamp, center: int,
                    rmap: dict, win: int = ROLLING_WIN) -> float:
    vals = [rmap[date - pd.Timedelta(days=center + d)]
            for d in range(-win, win + 1)
            if (date - pd.Timedelta(days=center + d)) in rmap]
    return float(np.mean(vals)) if vals else np.nan

def exact_lag_fallback(date: pd.Timestamp, lag: int, rmap: dict) -> float:
    v = rmap.get(date - pd.Timedelta(days=lag), np.nan)
    if np.isnan(v):
        for delta in range(1, 8):
            for sign in (1, -1):
                v = rmap.get(date - pd.Timedelta(days=lag + sign * delta), np.nan)
                if not np.isnan(v):
                    return float(v)
    return float(v)

# =============================================================================
# FEATURE BUILDER (chi 11 features)
# =============================================================================
def build_features(dates: pd.Series, rmap: dict) -> pd.DataFrame:
    feats = pd.DataFrame(index=dates.index)
    d = pd.DatetimeIndex(dates)

    # Trend
    feats["trend_days_from_peak"] = (dates - PEAK_DATE).dt.days

    # DOM ratio [0, 1]
    feats["dom_ratio"] = (d.day - 1) / (d.days_in_month - 1)

    # Fourier yearly - 1 cap sin/cos
    t = d.dayofyear / YEAR_PERIOD * 2 * np.pi
    feats["sin_year_1"] = np.sin(t)
    feats["cos_year_1"] = np.cos(t)

    # Tet distance
    feats["days_to_tet"] = dates.apply(days_to_nearest_tet)

    # Rolling YoY lags (log1p)
    roll364 = dates.apply(lambda x: rolling_yoy_lag(x, 364, rmap))
    roll728 = dates.apply(lambda x: rolling_yoy_lag(x, 728, rmap))
    ex364   = dates.apply(lambda x: exact_lag_fallback(x, 364, rmap))
    ex728   = dates.apply(lambda x: exact_lag_fallback(x, 728, rmap))

    lag364 = roll364.fillna(ex364)
    lag728 = roll728.fillna(ex728).fillna(lag364)

    feats["lag_364_roll"] = np.log1p(lag364.clip(lower=0))
    feats["lag_728_roll"] = np.log1p(lag728.clip(lower=0))

    # YoY ratio (tang truong nam nay vs nam truoc)
    safe728 = lag728.replace(0, np.nan)
    feats["yoy_ratio"] = (lag364 / safe728).clip(0.5, 2.0).fillna(1.0)

    # Short-term lags (log1p, fallback = lag364)
    for short in [7, 14, 28]:
        vals = dates.apply(lambda x: exact_lag_fallback(x, short, rmap))
        feats[f"lag_{short}"] = np.log1p(vals.clip(lower=0).fillna(lag364.clip(lower=0)))

    return feats[SELECTED_FEATURES]   # chi giu 11 features

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("Step 1: Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.rename(columns={"date": "Date"})
df = df[["Date", "Revenue"]].dropna().sort_values("Date").reset_index(drop=True)
rmap_train = dict(zip(df["Date"], df["Revenue"]))
print(f"  Rows: {len(df)}  {df['Date'].min().date()} -> {df['Date'].max().date()}")
print(f"  Features: {len(SELECTED_FEATURES)} -> {SELECTED_FEATURES}")

# =============================================================================
# 2. BUILD TRAINING MATRIX
# =============================================================================
print("\nStep 2: Building training matrix...")
X_all = build_features(df["Date"], rmap_train)
y_all = np.log1p(df["Revenue"].values)

valid_mask  = X_all["lag_364_roll"].notna() & (X_all["lag_364_roll"] > 0)
X_train     = X_all[valid_mask].copy()
y_train     = y_all[valid_mask]
train_dates = df.loc[valid_mask, "Date"].reset_index(drop=True)
print(f"  Training samples: {len(X_train)}")

# =============================================================================
# 3. SAMPLE WEIGHTS
# =============================================================================
def make_weights(dates: pd.Series) -> np.ndarray:
    w = np.ones(len(dates))
    for yr, wt in {2019: 2.0, 2020: 3.0, 2021: 4.0, 2022: 5.0}.items():
        w[dates.dt.year.values == yr] = wt
    mn, mx = w.min(), w.max()
    return 1.0 + 4.0 * (w - mn) / (mx - mn) if mx > mn else w

weights = make_weights(train_dates)

# =============================================================================
# 4. OPTUNA TUNING
# =============================================================================
print(f"\nStep 3: Tuning ({'Optuna ' + str(N_OPTUNA) + ' trials' if HAS_OPTUNA else 'default'})...")

BASE_PARAMS = dict(
    objective="regression", metric="mae",
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=6,
    min_child_samples=25, feature_fraction=0.9,
    bagging_fraction=0.85, bagging_freq=5,
    reg_alpha=0.05, reg_lambda=0.05,
    random_state=42, n_jobs=-1, verbose=-1,
)

if HAS_OPTUNA:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=30)

    def objective(trial):
        p = dict(
            objective="regression", metric="mae",
            n_estimators=trial.suggest_int("n_estimators", 800, 3500, step=200),
            learning_rate=trial.suggest_float("lr", 0.008, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 80),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_samples=trial.suggest_int("min_child", 10, 60),
            feature_fraction=trial.suggest_float("ff", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bf", 0.5, 1.0),
            bagging_freq=5,
            reg_alpha=trial.suggest_float("alpha", 1e-4, 2.0, log=True),
            reg_lambda=trial.suggest_float("lambda", 1e-4, 2.0, log=True),
            random_state=42, n_jobs=-1, verbose=-1,
        )
        maes = []
        for tr_i, va_i in tscv.split(X_train):
            m = lgb.LGBMRegressor(**p)
            m.fit(X_train.iloc[tr_i], y_train[tr_i],
                  sample_weight=weights[tr_i],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)],
                  eval_set=[(X_train.iloc[va_i], y_train[va_i])])
            maes.append(mean_absolute_error(
                np.expm1(y_train[va_i]),
                np.expm1(m.predict(X_train.iloc[va_i]))
            ))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    best = study.best_params
    FINAL_PARAMS = {**BASE_PARAMS,
                    "n_estimators": best["n_estimators"],
                    "learning_rate": best["lr"],
                    "num_leaves": best["num_leaves"],
                    "max_depth": best["max_depth"],
                    "min_child_samples": best["min_child"],
                    "feature_fraction": best["ff"],
                    "bagging_fraction": best["bf"],
                    "reg_alpha": best["alpha"],
                    "reg_lambda": best["lambda"]}
    print(f"  Best CV MAE : {study.best_value:,.0f}")
    print(f"  Best params : {best}")
else:
    FINAL_PARAMS = BASE_PARAMS
    print("  Using base params.")

# =============================================================================
# 5. TRAIN FINAL MODEL
# =============================================================================
print("\nStep 4: Training final model...")
model = lgb.LGBMRegressor(**FINAL_PARAMS)
model.fit(X_train, y_train, sample_weight=weights)

imp = pd.Series(model.feature_importances_, index=SELECTED_FEATURES).sort_values(ascending=False)
print("  Feature importances:")
print(imp.to_string())

# =============================================================================
# 6. VALIDATE ON 2022
# =============================================================================
print("\nStep 5: Validation on 2022...")
vm         = train_dates.dt.year == 2022
y_true     = np.expm1(y_train[vm.values])
y_pred     = np.expm1(model.predict(X_train[vm.values]))
mae_val    = mean_absolute_error(y_true, y_pred)
mape_val   = np.mean(np.abs((y_pred - y_true) / (y_true + 1))) * 100
print(f"  MAE  (2022): {mae_val:,.0f}")
print(f"  MAPE (2022): {mape_val:.2f}%")
print(f"  [v1 baseline: MAE=105,795 | v2+Optuna: MAE=86,251]")

# =============================================================================
# 7. RECURSIVE FORECASTING
# =============================================================================
print("\nStep 6: Recursive forecasting...")
sub  = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"])
sub  = sub.sort_values("Date").reset_index(drop=True)
fdates = sub["Date"]
print(f"  {fdates.min().date()} -> {fdates.max().date()} ({len(fdates)} days)")

rolling_map = dict(rmap_train)
preds       = {}

for i, fdate in enumerate(fdates):
    if (i + 1) % 100 == 0 or i == 0:
        print(f"  Day {i+1:3d}/{len(fdates)}: {fdate.date()}")

    row      = build_features(pd.Series([fdate]), rolling_map)
    lgbm_p   = float(np.expm1(model.predict(row)[0]))

    # YoY blend anchor
    yoy_base = rolling_yoy_lag(fdate, 364, rolling_map)
    if np.isnan(yoy_base):
        yoy_base = lgbm_p
    yoy_ratio = float(row["yoy_ratio"].values[0])
    yoy_p     = yoy_base * yoy_ratio

    final = max(YOY_ALPHA * yoy_p + (1.0 - YOY_ALPHA) * lgbm_p, 0.0)
    preds[fdate]       = final
    rolling_map[fdate] = final

# =============================================================================
# 8. WRITE SUBMISSION
# =============================================================================
pred_s = pd.Series(preds)
last_avg = df[df["Date"].dt.year == 2022]["Revenue"].mean()
print(f"\nStep 7: Stats:")
print(f"  Min={pred_s.min():,.0f}  Max={pred_s.max():,.0f}")
print(f"  Mean={pred_s.mean():,.0f}  Median={pred_s.median():,.0f}")
print(f"  Ratio vs 2022 avg: {pred_s.mean()/last_avg:.3f}")

sub["Revenue"] = sub["Date"].map(preds)
if sub["Revenue"].isna().any():
    sub["Revenue"] = sub["Revenue"].interpolate("linear")
sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub[["Date", "Revenue", "COGS"]].to_csv(OUTPUT_PATH, index=False)

print(f"\nStep 8: Saved -> {OUTPUT_PATH}")
print(sub[["Date", "Revenue", "COGS"]].head(10).to_string(index=False))
print("\n" + "=" * 60)
print("DONE! v3 (11 features) complete.")
print("=" * 60)
