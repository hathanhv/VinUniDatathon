"""
decomposed_forecast.py – Decomposed Temporal Fusion
======================================================
Architecture:
  1. Winsorize → STL decomposition (Trend + Seasonal + Residual)
  2. Trend    → Damped Holt-Winters (ETS)
  3. Seasonal → Seasonal Naive (lag-365 from actual data)
  4. Residual → LightGBM with calendar + exogenous features
  5. Recompose on log scale → exponentiate
  6. Walk-forward validation (5 cutoffs: 2018-2021)
  7. Clip to [0, 1.5 * max_historical]
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mstats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
DATA_PATH   = BASE / "processed_data.csv"
SAMPLE_PATH = BASE / "sample_submission.csv"
OUT_PATH    = BASE / "submission.csv"

SEED = 42
np.random.seed(SEED)

# ── Exogenous columns available in processed_data.csv ────────────────────────
EXOG_COLS = [
    "sessions", "unique_visitors", "page_views", "bounce_rate",
    "avg_session_duration_sec",
    "total_stock_on_hand", "days_since_snapshot",
    "total_stockout_flags", "total_overstock_flags", "avg_sell_through_rate",
    "avg_fill_rate", "total_shipping_fee",
]

TET = {y: pd.Timestamp(d) for y, d in {
    2012:"2012-01-23",2013:"2013-02-10",2014:"2014-01-31",2015:"2015-02-19",
    2016:"2016-02-08",2017:"2017-01-28",2018:"2018-02-16",2019:"2019-02-05",
    2020:"2020-01-25",2021:"2021-02-12",2022:"2022-02-01",2023:"2023-01-22",
    2024:"2024-02-10",2025:"2025-01-29"}.items()}
VN_HOLIDAYS = {"01-01","04-30","05-01","09-02","12-25"}

def days_to_tet(dt):
    cands = [TET[y] for y in [dt.year-1, dt.year, dt.year+1] if y in TET]
    return int(min([(dt-t).days for t in cands], key=abs)) if cands else 0

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df = df[df["date"] <= "2022-12-31"].copy()
    for c in EXOG_COLS:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(method="ffill").fillna(0)
    print(f"[LOAD] {len(df)} rows  {df['date'].min().date()} → {df['date'].max().date()}")
    return df

# ── Winsorize + log transform ─────────────────────────────────────────────────
def prepare_series(series, limits=(0.01, 0.01)):
    """Winsorize outliers then log1p transform."""
    arr = series.values.astype(float)
    arr_w = mstats.winsorize(arr, limits=limits)
    return np.log1p(arr_w)

# ── STL decomposition ─────────────────────────────────────────────────────────
def stl_decompose(log_series, period=365):
    """
    STL decomposition on log-transformed series.
    Returns (trend, seasonal, residual) as numpy arrays.
    """
    stl = STL(log_series, period=period, robust=True)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid

# ── Trend forecasting: Damped Holt-Winters (ETS) ─────────────────────────────
def forecast_trend(trend_arr, n_forecast):
    """
    Fit damped additive Holt-Winters on the trend component.
    Damped trend prevents extrapolation blowup.
    """
    try:
        model = ExponentialSmoothing(
            trend_arr,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)  # no 'disp' param in newer statsmodels
        fcast = model.forecast(n_forecast)
        # Clip trend: do not deviate more than ±20% from last observed trend
        last_trend = float(trend_arr[-1])
        fcast = np.clip(np.array(fcast), last_trend * 0.8, last_trend * 1.2)
        return fcast
    except Exception as e:
        print(f"  [WARN] Trend ETS failed: {e}. Using flat (last 90-day mean).")
        # Conservative fallback: flat at the mean of last 90 days of trend
        flat_val = float(np.mean(trend_arr[-90:]))
        return np.full(n_forecast, flat_val)

# ── Seasonal forecasting: Seasonal Naive (lag-365) ───────────────────────────
def forecast_seasonal(seasonal_arr, n_forecast, period=365):
    """
    Repeat seasonal pattern from the last full period.
    Simply wraps the seasonal component cyclically.
    """
    n = len(seasonal_arr)
    seasonal_block = seasonal_arr[-period:]  # last full year of seasonal
    repeats = (n_forecast // period) + 2
    extended = np.tile(seasonal_block, repeats)
    return extended[:n_forecast]

# ── Residual model: LightGBM with exogenous features ─────────────────────────
def build_residual_features(dates, residuals, exog_df, is_future=False):
    """
    Build feature matrix for the residual model.
    Features: calendar, residual lags (7,14,30,60), rolling stats, exogenous.
    Residual lags are SAFE because residuals are small & stationary.
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates), "residual": residuals})
    df = df.set_index("date")

    # Calendar features
    d = df.index
    df["dow"]           = d.dayofweek
    df["month"]         = d.month
    df["quarter"]       = d.quarter
    df["doy"]           = d.dayofyear
    df["is_weekend"]    = (d.dayofweek >= 5).astype(int)
    df["is_month_end"]  = d.is_month_end.astype(int)
    df["is_month_start"]= d.is_month_start.astype(int)
    df["is_vn_holiday"] = pd.Series(
        [int(f"{x.month:02d}-{x.day:02d}" in VN_HOLIDAYS) for x in d], index=d)
    dt2t = pd.Series([days_to_tet(x) for x in d], index=d)
    df["days_to_tet"]   = dt2t
    df["is_tet_week"]   = (dt2t.abs() <= 7).astype(int)
    df["is_pre_tet2w"]  = ((dt2t >= -14) & (dt2t < 0)).astype(int)
    df["tet_proximity"] = np.exp(-0.5 * (dt2t / 7.0) ** 2)
    for k in [1, 2, 3, 4]:
        df[f"fsin{k}"] = np.sin(2*np.pi*k*df["doy"]/365.25)
        df[f"fcos{k}"] = np.cos(2*np.pi*k*df["doy"]/365.25)
    df["dow_sin"]   = np.sin(2*np.pi*d.dayofweek/7)
    df["dow_cos"]   = np.cos(2*np.pi*d.dayofweek/7)
    df["month_sin"] = np.sin(2*np.pi*d.month/12)
    df["month_cos"] = np.cos(2*np.pi*d.month/12)

    # Residual lags: safe because residuals are small & stationary
    r = df["residual"]
    for lag in [1, 7, 14, 30, 60]:
        df[f"resid_lag{lag}"] = r.shift(lag)
    for w in [7, 30]:
        df[f"resid_rmean{w}"] = r.shift(1).rolling(w).mean()
        df[f"resid_rstd{w}"]  = r.shift(1).rolling(w).std()
    df["resid_yoy"] = r / (r.shift(365).abs() + 1e-9)

    # Exogenous features (shifted 1 day to avoid future leakage)
    if exog_df is not None:
        exog_aligned = exog_df.reindex(df.index, method="ffill")
        for col in EXOG_COLS:
            if col in exog_aligned.columns:
                df[f"exog_{col}"] = exog_aligned[col].shift(1)

    feat_cols = [c for c in df.columns if c != "residual"
                 and df[c].dtype in (np.float64, np.int64, np.float32, np.int32,
                                     bool, np.bool_)]
    return df, feat_cols

def train_residual_model(train_dates, train_resid, exog_df, val_dates=None, val_resid=None):
    """Train LightGBM on residuals with optional early stopping on val."""
    df_tr, feat_cols = build_residual_features(train_dates, train_resid, exog_df)
    df_tr = df_tr.dropna()

    X_tr = df_tr[feat_cols].values.astype(np.float32)
    y_tr = df_tr["residual"].values

    # StandardScaler on target (residuals) for better LGB convergence
    scaler = StandardScaler()
    y_tr_s = scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

    params = dict(n_estimators=1000, learning_rate=0.03, num_leaves=31,
                  max_depth=5, min_child_samples=30,
                  subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=0.1, reg_lambda=1.0,
                  random_state=SEED, n_jobs=-1, verbose=-1)

    if val_dates is not None and val_resid is not None:
        df_va, _ = build_residual_features(
            list(val_dates) + list(train_dates),  # need history for lags
            list(val_resid) + list(train_resid), exog_df)
        # Take only val rows
        df_va = df_va.loc[pd.to_datetime(val_dates)].dropna()
        X_va = df_va[feat_cols].values.astype(np.float32)
        y_va_s = scaler.transform(df_va["residual"].values.reshape(-1,1)).ravel()
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr_s, eval_set=[(X_va, y_va_s)],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(-1)])
    else:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr_s, callbacks=[lgb.log_evaluation(-1)])

    return model, scaler, feat_cols, df_tr

def predict_residual(model, scaler, feat_cols, future_dates, resid_history_dates,
                     resid_history, exog_df):
    """
    Predict future residuals using iterative 1-step-ahead for lag_1,
    but lag_7+ use previously predicted values (small errors, bounded).
    """
    all_dates = list(resid_history_dates) + list(pd.to_datetime(future_dates))
    all_resid = list(resid_history) + [0.0] * len(future_dates)
    preds = []

    for i, fdate in enumerate(pd.to_datetime(future_dates)):
        idx = len(resid_history) + i
        df_tmp, _ = build_residual_features(all_dates[:idx+1], all_resid[:idx+1], exog_df)
        row = df_tmp.iloc[[-1]]
        row = row.reindex(columns=feat_cols, fill_value=0.0)
        x = row.values.astype(np.float32)
        pred_s = model.predict(x)[0]
        pred = float(scaler.inverse_transform([[pred_s]])[0, 0])
        # Clip residual: should be within [-1.5, 1.5] on log scale
        pred = np.clip(pred, -1.5, 1.5)
        preds.append(pred)
        all_resid[idx] = pred

    return np.array(preds)

# ── Walk-forward validation ───────────────────────────────────────────────────
def walk_forward_validate(df, target, n_forecast=365):
    """
    Walk-forward validation with 4 cutoffs (2018, 2019, 2020, 2021).
    Gap of 30 days between train end and val start.
    """
    cutoffs = [
        ("2018-12-31", "2019-01-31", "2019-12-31"),
        ("2019-12-31", "2020-01-31", "2020-12-31"),
        ("2020-12-31", "2021-01-31", "2021-12-31"),
        ("2021-12-31", "2022-01-31", "2022-12-31"),
    ]
    metrics_list = []

    for train_end, val_start, val_end in cutoffs:
        tr = df[df["date"] <= train_end].copy()
        va = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
        if len(tr) < 365 or len(va) < 30:
            continue

        y_hat = _run_pipeline(tr, pd.to_datetime(va["date"].values), target,
                              exog_df=df.set_index("date"))
        y_true = va[target].values[:len(y_hat)]

        mae  = mean_absolute_error(y_true, y_hat)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        r2   = r2_score(y_true, y_hat)
        print(f"  [{train_end[:4]}→{val_end[:4]}] R²={r2:.4f}  RMSE={rmse:,.0f}  MAE={mae:,.0f}")
        metrics_list.append({"cutoff": train_end, "r2": r2, "rmse": rmse, "mae": mae})

    return metrics_list

# ── Core pipeline ─────────────────────────────────────────────────────────────
def _run_pipeline(train_df, future_dates, target, exog_df=None):
    """
    Full DTF pipeline for one target variable.
    train_df: historical DataFrame with 'date' and target column.
    future_dates: DatetimeIndex of dates to forecast.
    """
    n_forecast = len(future_dates)
    dates = pd.to_datetime(train_df["date"].values)

    # Step 1: Winsorize + log1p
    log_series = prepare_series(train_df[target], limits=(0.02, 0.02))

    # Step 2: STL decomposition
    trend, seasonal, resid = stl_decompose(log_series, period=365)

    # Step 3a: Trend forecast (Damped Holt-Winters)
    trend_fc = forecast_trend(trend, n_forecast)

    # Step 3b: Seasonal forecast (Seasonal Naive)
    seasonal_fc = forecast_seasonal(seasonal, n_forecast, period=365)

    # Step 3c: Residual model (LightGBM)
    model, scaler, feat_cols, _ = train_residual_model(
        dates, resid, exog_df)
    resid_fc = predict_residual(model, scaler, feat_cols,
                                future_dates, dates, resid, exog_df)

    # Step 4: Recompose on log scale
    log_fc = trend_fc + seasonal_fc + resid_fc

    # Step 5: Back-transform
    raw_fc = np.expm1(log_fc)

    # Step 6: Safety clipping
    # Clip at 1.2x the 95th percentile to prevent blowup
    hist_p95 = train_df[target].quantile(0.95)
    raw_fc = np.clip(raw_fc, 0, 1.2 * hist_p95)

    return raw_fc

# ── Evaluation helper ─────────────────────────────────────────────────────────
def eval_metrics(y_true, y_pred, label):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {label:<20} R²={r2:.4f}  RMSE={rmse:,.0f}  MAE={mae:,.0f}")
    return r2, rmse, mae

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n"+"="*70)
    print("  DECOMPOSED TEMPORAL FUSION — Revenue & COGS Forecasting")
    print("="*70)

    raw = load_data()
    exog_df = raw.set_index("date")

    sample = pd.read_csv(SAMPLE_PATH); sample.columns = sample.columns.str.lower()
    future_dates = pd.to_datetime(sample["date"].values)
    print(f"[TARGET] {future_dates[0].date()} → {future_dates[-1].date()} ({len(future_dates)} days)")

    results = {}

    for target in ["Revenue", "COGS"]:
        print(f"\n{'#'*70}\n  TARGET: {target}\n{'#'*70}")

        # Walk-forward validation
        print("\n[VAL] Walk-forward validation:")
        metrics = walk_forward_validate(raw, target)
        if metrics:
            avg_r2   = np.mean([m["r2"]   for m in metrics])
            avg_mae  = np.mean([m["mae"]  for m in metrics])
            avg_rmse = np.mean([m["rmse"] for m in metrics])
            print(f"  [AVG] R²={avg_r2:.4f}  RMSE={avg_rmse:,.0f}  MAE={avg_mae:,.0f}")

        # Full pipeline on all 2012-2022 data
        print("\n[FORECAST] Running full pipeline...")
        preds = _run_pipeline(raw, future_dates, target, exog_df=exog_df)
        results[target] = preds
        print(f"  {target} stats: min={preds.min():,.0f}  mean={preds.mean():,.0f}"
              f"  max={preds.max():,.0f}")
        print(f"  Sample: {preds[:7].round(0)}")

    # Save submission
    out = pd.DataFrame({
        "date":    future_dates,
        "revenue": np.round(results["Revenue"], 2),
        "cogs":    np.round(results["COGS"],    2),
    })
    out.to_csv(OUT_PATH, index=False)
    print(f"\n[DONE] Saved {OUT_PATH}  ({len(out)} rows)")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
