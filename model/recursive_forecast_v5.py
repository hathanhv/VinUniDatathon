# -*- coding: utf-8 -*-
"""
VinUni Datathon 2026 - Recursive Revenue Forecasting v5
=========================================================
Changes vs v2:
  1. Bỏ lag_7, lag_14, lag_28 (recursive noise tich luy 1.5 nam)
  2. Chi giu lag dài hạn: lag_364_roll + lag_728_roll
  3. Tang Fourier len k=1..8 (yearly) + k=1..3 (weekly)
  4. Them Vietnamese public holidays chinh xac:
       - Fixed: 1/1, 30/4, 1/5, 2/9 (+/-2 days window)
       - Variable: Hung Kings Day (10/3 am lich), Tet (da co)
       - days_to_nearest_vn_holiday (continuous)
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
    print("[WARN] pip install optuna")

# =============================================================================
# CONFIG
# =============================================================================
BASE        = Path(r"d:\vinuni_datathon2026\vinuni_datathon2026\model")
DATA_PATH   = BASE / "processed_data.csv"
SAMPLE_PATH = BASE / "sample_submission.csv"
OUTPUT_PATH = BASE / "submission_v5.csv"

PEAK_DATE    = pd.Timestamp("2017-01-01")
YEAR_PERIOD  = 365.25
ROLLING_WIN  = 7
YOY_ALPHA    = 0.20    # 80% lgbm + 20% yoy blend
FOURIER_K_Y  = 8       # Fourier yearly terms k=1..8
FOURIER_K_W  = 3       # Fourier weekly terms k=1..3
N_OPTUNA     = 80
N_SPLITS     = 5

# =============================================================================
# VIETNAMESE HOLIDAY CALENDAR (deterministic — known in advance)
# =============================================================================
# Fixed Gregorian holidays (each year)
VN_FIXED_HOLIDAYS = [
    (1, 1),    # New Year's Day / Tet Duong Lich
    (4, 30),   # Reunification Day / Giai phong mien Nam
    (5, 1),    # International Labor Day / Quoc te Lao dong
    (9, 2),    # National Day / Quoc khanh
    (12, 25),  # Christmas (commercial peak)
]

# Hung Kings Commemoration Day (Gio To Hung Vuong) - 10th of 3rd lunar month
# Pre-computed Gregorian dates for 2012-2024
HUNG_KINGS_DATES = {
    2012: pd.Timestamp("2012-03-31"),
    2013: pd.Timestamp("2013-04-19"),
    2014: pd.Timestamp("2014-04-09"),
    2015: pd.Timestamp("2015-03-29"),
    2016: pd.Timestamp("2016-04-16"),
    2017: pd.Timestamp("2017-04-05"),
    2018: pd.Timestamp("2018-03-25"),
    2019: pd.Timestamp("2019-04-14"),
    2020: pd.Timestamp("2020-04-02"),
    2021: pd.Timestamp("2021-04-21"),
    2022: pd.Timestamp("2022-04-10"),
    2023: pd.Timestamp("2023-04-29"),
    2024: pd.Timestamp("2024-04-18"),
}

# Tet Nguyen Dan (Lunar New Year) dates
TET_DATES = {
    2012: pd.Timestamp("2012-01-23"), 2013: pd.Timestamp("2013-02-10"),
    2014: pd.Timestamp("2014-01-31"), 2015: pd.Timestamp("2015-02-19"),
    2016: pd.Timestamp("2016-02-08"), 2017: pd.Timestamp("2017-01-28"),
    2018: pd.Timestamp("2018-02-16"), 2019: pd.Timestamp("2019-02-05"),
    2020: pd.Timestamp("2020-01-25"), 2021: pd.Timestamp("2021-02-12"),
    2022: pd.Timestamp("2022-02-01"), 2023: pd.Timestamp("2023-01-22"),
    2024: pd.Timestamp("2024-02-10"),
}

def get_all_holiday_dates(years=range(2012, 2025)):
    """Build a set of all Vietnamese holiday dates."""
    holidays = []
    for yr in years:
        # Fixed holidays
        for m, d in VN_FIXED_HOLIDAYS:
            try:
                holidays.append(pd.Timestamp(f"{yr}-{m:02d}-{d:02d}"))
            except Exception:
                pass
        # Hung Kings
        if yr in HUNG_KINGS_DATES:
            holidays.append(HUNG_KINGS_DATES[yr])
        # Tet (3 core days)
        if yr in TET_DATES:
            tet = TET_DATES[yr]
            for delta in range(-1, 4):  # Tet Eve + 3 Tet days
                holidays.append(tet + pd.Timedelta(days=delta))
    return sorted(set(holidays))

ALL_HOLIDAY_DATES = get_all_holiday_dates()

def days_to_nearest_holiday(date: pd.Timestamp) -> float:
    """Distance (in days) to the nearest Vietnamese public holiday."""
    dists = [abs((date - h).days) for h in ALL_HOLIDAY_DATES
             if abs(date.year - h.year) <= 1]
    return float(min(dists)) if dists else 30.0

def days_to_nearest_tet(date: pd.Timestamp) -> float:
    cands = [abs((date - t).days) for yr, t in TET_DATES.items()
             if abs(date.year - yr) <= 1]
    return float(min(cands)) if cands else 60.0

def is_holiday_window(date: pd.Timestamp, window: int = 2) -> int:
    """1 if within `window` days of any VN public holiday."""
    for h in ALL_HOLIDAY_DATES:
        if abs((date - h).days) <= window:
            return 1
    return 0

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
    return float(v) if not np.isnan(v) else np.nan

# =============================================================================
# FEATURE ENGINEERING (v5)
# =============================================================================
def build_features(dates: pd.Series, rmap: dict) -> pd.DataFrame:
    feats = pd.DataFrame(index=dates.index)
    d = pd.DatetimeIndex(dates)

    # -- Trend
    feats["trend_days_from_peak"] = (dates - PEAK_DATE).dt.days

    # -- DOM ratio [0, 1]
    feats["dom_ratio"] = (d.day - 1) / (d.days_in_month - 1)

    # -- Calendar (discrete — keep minimal)
    feats["month"]       = d.month
    feats["day_of_week"] = d.dayofweek
    feats["quarter"]     = d.quarter
    feats["is_weekend"]  = (d.dayofweek >= 5).astype(int)

    # -- Fourier YEARLY k=1..FOURIER_K_Y (smooth seasonality)
    t_y = d.dayofyear / YEAR_PERIOD * 2 * np.pi
    for k in range(1, FOURIER_K_Y + 1):
        feats[f"sin_year_{k}"] = np.sin(k * t_y)
        feats[f"cos_year_{k}"] = np.cos(k * t_y)

    # -- Fourier WEEKLY k=1..FOURIER_K_W
    t_w = d.dayofweek / 7 * 2 * np.pi
    for k in range(1, FOURIER_K_W + 1):
        feats[f"sin_week_{k}"] = np.sin(k * t_w)
        feats[f"cos_week_{k}"] = np.cos(k * t_w)

    # -- Vietnamese holidays (deterministic — key improvement in v5)
    feats["days_to_tet"]      = dates.apply(days_to_nearest_tet)
    feats["days_to_vn_hol"]   = dates.apply(days_to_nearest_holiday)
    feats["is_holiday_window"] = dates.apply(is_holiday_window).astype(int)

    # Pre/post holiday effect (asymmetric: pre-holiday buy-up vs post-holiday dip)
    feats["pre_holiday"]  = dates.apply(
        lambda x: 1 if any(0 < (h - x).days <= 5 for h in ALL_HOLIDAY_DATES) else 0
    ).astype(int)
    feats["post_holiday"] = dates.apply(
        lambda x: 1 if any(0 < (x - h).days <= 3 for h in ALL_HOLIDAY_DATES) else 0
    ).astype(int)

    # Hung Kings window (pre-holiday buying peak ~April)
    feats["near_hung_kings"] = dates.apply(
        lambda x: 1 if any(
            abs((x - h).days) <= 5
            for h in HUNG_KINGS_DATES.values()
        ) else 0
    ).astype(int)

    # -- Rolling YoY Lags ONLY (no lag_7/14/28)
    r364 = dates.apply(lambda x: rolling_yoy_lag(x, 364, rmap))
    r728 = dates.apply(lambda x: rolling_yoy_lag(x, 728, rmap))
    e364 = dates.apply(lambda x: exact_lag(x, 364, rmap))
    e728 = dates.apply(lambda x: exact_lag(x, 728, rmap))

    lag364 = r364.fillna(e364)
    lag728 = r728.fillna(e728).fillna(lag364)

    feats["lag_364_roll"] = np.log1p(lag364.clip(lower=0))
    feats["lag_728_roll"] = np.log1p(lag728.clip(lower=0))

    # YoY growth ratio
    safe728 = lag728.replace(0, np.nan)
    feats["yoy_ratio"] = (lag364 / safe728).clip(0.3, 2.5).fillna(1.0)

    return feats


def get_feat_cols(rmap_sample):
    tmp = build_features(
        pd.Series([pd.Timestamp("2022-06-01")]),
        rmap_sample
    )
    return tmp.columns.tolist()

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

FEAT_COLS = get_feat_cols(rmap_train)
print(f"  Features ({len(FEAT_COLS)}): {FEAT_COLS}")
print(f"  Fourier: yearly k=1..{FOURIER_K_Y}, weekly k=1..{FOURIER_K_W}")
print(f"  VN Holidays: {len(ALL_HOLIDAY_DATES)} holiday dates encoded")

# =============================================================================
# 2. BUILD TRAINING MATRIX
# =============================================================================
print("\nStep 2: Building training matrix...")
X_all = build_features(df["Date"], rmap_train)
y_all = np.log1p(df["Revenue"].values)

valid    = X_all["lag_364_roll"].notna() & (X_all["lag_364_roll"] > 0)
X_tr     = X_all[valid][FEAT_COLS].copy()
y_tr     = y_all[valid]
dates_tr = df.loc[valid, "Date"].reset_index(drop=True)
print(f"  Training samples: {len(X_tr)}")

# =============================================================================
# 3. SAMPLE WEIGHTS
# =============================================================================
def make_weights(dates):
    w = np.ones(len(dates))
    for yr, wt in {2019: 2.0, 2020: 3.0, 2021: 4.0, 2022: 5.0}.items():
        w[dates.dt.year.values == yr] = wt
    mn, mx = w.min(), w.max()
    return 1.0 + 4.0 * (w - mn) / (mx - mn) if mx > mn else w

weights = make_weights(dates_tr)

# =============================================================================
# 4. OPTUNA TUNING
# =============================================================================
tuning_label = f"Optuna {N_OPTUNA} trials" if HAS_OPTUNA else "default params"
print(f"\nStep 3: Hyperparameter tuning ({tuning_label})...")

BASE_PARAMS = dict(
    objective="regression", metric="mae",
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=6,
    min_child_samples=25, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=5,
    reg_alpha=0.08, reg_lambda=0.08,
    random_state=42, n_jobs=-1, verbose=-1,
)

if HAS_OPTUNA:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=30)

    def objective(trial):
        p = dict(
            objective="regression", metric="mae",
            n_estimators=trial.suggest_int("n_estimators", 800, 3500, step=200),
            learning_rate=trial.suggest_float("lr", 0.008, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 96),
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
            m.fit(X_tr.iloc[tri], y_tr[tri],
                  sample_weight=weights[tri],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)],
                  eval_set=[(X_tr.iloc[vai], y_tr[vai])])
            maes.append(mean_absolute_error(
                np.expm1(y_tr[vai]),
                np.expm1(m.predict(X_tr.iloc[vai]))
            ))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    bp = study.best_params
    FINAL_PARAMS = {**BASE_PARAMS,
                    "n_estimators":     bp["n_estimators"],
                    "learning_rate":    bp["lr"],
                    "num_leaves":       bp["num_leaves"],
                    "max_depth":        bp["max_depth"],
                    "min_child_samples":bp["mcs"],
                    "feature_fraction": bp["ff"],
                    "bagging_fraction": bp["bf"],
                    "reg_alpha":        bp["alpha"],
                    "reg_lambda":       bp["lam"]}
    print(f"  Best CV MAE : {study.best_value:,.0f}")
    print(f"  Best params : {bp}")
else:
    FINAL_PARAMS = BASE_PARAMS
    print("  Using base params.")

# =============================================================================
# 5. TRAIN FINAL MODEL
# =============================================================================
print("\nStep 4: Training final model...")
model = lgb.LGBMRegressor(**FINAL_PARAMS)
model.fit(X_tr, y_tr, sample_weight=weights)

imp = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
print("  Top 15 Feature Importances:")
print(imp.head(15).to_string())

# =============================================================================
# 6. VALIDATE ON 2022
# =============================================================================
print("\nStep 5: Validation on 2022...")
vm      = dates_tr.dt.year == 2022
y_true  = np.expm1(y_tr[vm.values])
y_pred  = np.expm1(model.predict(X_tr[vm.values]))
mae_v   = mean_absolute_error(y_true, y_pred)
mape_v  = np.mean(np.abs((y_pred - y_true) / (y_true + 1))) * 100
print(f"  MAE  (2022): {mae_v:,.0f}")
print(f"  MAPE (2022): {mape_v:.2f}%")
print(f"  [v2 default=51,077 | v2+Optuna=86,251 | v3=49,706]")

# =============================================================================
# 7. RECURSIVE FORECASTING
# =============================================================================
print("\nStep 6: Recursive forecasting 2023-01-01 -> 2024-07-01...")
sub    = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"])
sub    = sub.sort_values("Date").reset_index(drop=True)
fdates = sub["Date"]
print(f"  {fdates.min().date()} -> {fdates.max().date()} ({len(fdates)} days)")

rolling_map = dict(rmap_train)
preds       = {}

for i, fdate in enumerate(fdates):
    if (i + 1) % 100 == 0 or i == 0:
        print(f"  Day {i+1:3d}/{len(fdates)}: {fdate.date()}")

    row      = build_features(pd.Series([fdate]), rolling_map)
    lgbm_p   = float(np.expm1(model.predict(row[FEAT_COLS])[0]))

    # YoY blend (20%)
    yoy_base = rolling_yoy_lag(fdate, 364, rolling_map)
    if not np.isnan(yoy_base):
        yoy_ratio_val = float(row["yoy_ratio"].values[0])
        lgbm_p = YOY_ALPHA * (yoy_base * yoy_ratio_val) + (1.0 - YOY_ALPHA) * lgbm_p

    final = max(lgbm_p, 0.0)
    preds[fdate]        = final
    rolling_map[fdate]  = final

# =============================================================================
# 8. WRITE SUBMISSION
# =============================================================================
pred_s   = pd.Series(preds)
last_avg = df[df["Date"].dt.year == 2022]["Revenue"].mean()

print(f"\nStep 7: Forecast stats:")
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

print(f"\nSaved -> {OUTPUT_PATH}")
print(sub[["Date", "Revenue", "COGS"]].head(10).to_string(index=False))

print("\nHoliday window check (30/4 - 2/5/2023):")
mask = (sub["Date"] >= "2023-04-28") & (sub["Date"] <= "2023-05-03")
print(sub[mask][["Date", "Revenue", "COGS"]].to_string(index=False))

print("\nHung Kings window (Apr 29/2023):")
mask2 = (sub["Date"] >= "2023-04-25") & (sub["Date"] <= "2023-05-01")
print(sub[mask2][["Date", "Revenue", "COGS"]].to_string(index=False))

print("\n" + "=" * 60)
print("DONE! v5 complete.")
print(f"  NO lag_7/14/28 (removed recursive noise)")
print(f"  Fourier: yearly k=1..{FOURIER_K_Y}  weekly k=1..{FOURIER_K_W}")
print(f"  VN Holidays: Tet + Hung Kings + 30/4 + 1/5 + 2/9 + 1/1 + 25/12")
print("=" * 60)
