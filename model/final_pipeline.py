"""
final_pipeline.py — VinUni Datathon 2026
=========================================
Two-Stage Cascade Stacking for Revenue & COGS forecasting (2023-2024).
Target MAE: ~600,000.

STAGE LAYOUT
------------
  Stage 1 — Setup & Feature Engineering
    - load_data()            : Load, clean, compute YoY growth anchor
    - engineer_features()    : All feature engineering (noise drop, trend,
                               seasonality, Fourier, lags, Tet, flags)
    - get_sample_weights()   : Exponential-decay weights, 2021-22 x5 vs pre-2018

  Stage 2 — Revenue Stacking (this commit)
    - _make_lgb()            : LightGBM base model factory
    - _make_xgb()            : XGBoost base model factory
    - _make_cat()            : CatBoost base model factory
    - RevenueStackingModel   : Container for base + meta models
    - train_revenue_stage()  : TimeSeriesSplit OOF + final fit, returns
                               (model_bundle, oof_preds_original_scale)

  Stage 3 — COGS Cascade (this commit)
    - COGSCascadeModel   : Container for CatBoost + ElasticNet base + RidgeCV meta
    - train_cogs_stage() : Uses OOF Revenue as causal driver + all COGS features,
                           TimeSeriesSplit OOF, RidgeCV meta in log-space

  Stage 4 (next commit): Inference & Submission
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ── ML / Gradient Boosting ────────────────────────────────────────────────────
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import HuberRegressor, ElasticNet, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_PATH   = ROOT / "processed_data.csv"
SUBMIT_IN   = ROOT / "sample_submission.csv"
SUBMIT_OUT  = ROOT / "submission.csv"

# ── Global constants ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Trend peak identified by STL decomposition (Section 3.2 of EDA report)
TREND_PEAK_DATE = pd.Timestamp("2017-01-01")
TREND_NORM_DAYS = 1826  # ≈ 5 years, used to normalise _trend_days_from_peak

# Fourier harmonics (N=10 captures weekly + annual seasonality down to ~5-day cycles)
N_FOURIER = 10

# ── Features to drop (EDA Section 5.1) ────────────────────────────────────────
# MI < 0.01 in both full-era and real-data-era analyses
DROP_NOISE: set[str] = {
    # Traffic channel breakdown — info already captured by `sessions` & order_source_*
    "traffic_direct",
    "traffic_email_campaign",
    "traffic_referral",
    "traffic_paid_search",
    "traffic_organic_search",
    "traffic_social_media",
    # Session behaviour — uncorrelated with revenue (MI ≈ 0)
    "avg_session_duration_sec",
    # Duplicate temporal columns (engineered equivalents kept)
    "day_of_week",   # duplicate of _dow  (dayofweek int)
    "month",         # duplicate of _month (month int)
}

# ── Leaky columns (must never appear as features) ─────────────────────────────
LEAKY_COLS: set[str] = {
    "payment_value", "total_refund_amount",
    "order_reviews", "customer_reviews", "product_reviews",
    "rating", "Revenue", "COGS",
}

# ── Lunar New Year (Tết) dates ────────────────────────────────────────────────
# Covers training period (2012-2022) + forecast horizon (2023-2024)
TET_DATES: dict[int, pd.Timestamp] = {
    y: pd.Timestamp(d)
    for y, d in {
        2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31",
        2015: "2015-02-19", 2016: "2016-02-08", 2017: "2017-01-28",
        2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25",
        2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
        2024: "2024-02-10", 2025: "2025-01-29",
    }.items()
}

# ── Vietnamese public holidays (MM-DD) ───────────────────────────────────────
VN_PUBLIC_HOLIDAYS: set[str] = {
    "01-01",  # New Year's Day
    "04-30",  # Reunification Day
    "05-01",  # International Labour Day
    "09-02",  # National Day
    "12-25",  # Christmas (retail spike)
    # Tết cluster (approximate fixed MM-DD for surrounding days)
    "01-25", "01-26", "01-27", "01-28", "01-29", "01-30",
}

# =============================================================================
# HELPER: Tết proximity
# =============================================================================

def _days_to_tet(dt: pd.Timestamp) -> int:
    """
    Return signed distance (days) from `dt` to the nearest Tết date.
    Negative  → dt is BEFORE Tết (pre-Tết shopping surge).
    Positive  → dt is AFTER  Tết (post-holiday slowdown).
    """
    candidates = [
        TET_DATES[y]
        for y in (dt.year - 1, dt.year, dt.year + 1)
        if y in TET_DATES
    ]
    # Signed: (dt - tet).days  → negative before Tết, positive after
    signed = [(dt - t).days for t in candidates]
    return int(min(signed, key=abs))


# =============================================================================
# STAGE 1-A: load_data()
# =============================================================================

def load_data() -> tuple[pd.DataFrame, float]:
    """
    Load and minimally clean the processed dataset.

    Returns
    -------
    df : pd.DataFrame
        Sorted, deduplicated training frame (2012-07-04 → 2022-12-31).
    last_yoy_growth : float
        Most recent annual YoY revenue growth rate (used as anchor for
        future-row construction in the forecast stage).

    Notes
    -----
    - `_annual_yoy_growth` is attached here because it requires the full
      time-series to compute, and engineer_features() works row-wise.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = (
        df.sort_values("date")
          .drop_duplicates("date")
          .reset_index(drop=True)
    )
    # Restrict to confirmed training window
    df = df[df["date"] <= "2022-12-31"].copy()

    # Annual mean Revenue → YoY ratio (trend proxy for each calendar year)
    annual_rev = df.groupby(df["date"].dt.year)["Revenue"].mean()
    yoy_ratio  = (annual_rev / annual_rev.shift(1)).fillna(1.0)
    df["_annual_yoy_growth"] = df["date"].dt.year.map(yoy_ratio.to_dict())

    last_yoy_growth = float(yoy_ratio.iloc[-1])

    print(
        f"[load_data] {len(df):,} rows | "
        f"{df['date'].min().date()} -> {df['date'].max().date()} | "
        f"Last YoY growth: {last_yoy_growth:.4f}"
    )
    return df, last_yoy_growth


# =============================================================================
# STAGE 1-B: engineer_features()
# =============================================================================

def engineer_features(
    df: pd.DataFrame,
    target: str,
    rev_pred_series: pd.Series | None = None,
) -> tuple[pd.DataFrame, str, list[str], list[str], list[str]]:
    """
    Full feature engineering pipeline for one target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training frame from load_data() (must contain `date` column).
    target : str
        Either "Revenue" or "COGS".
    rev_pred_series : pd.Series or None
        OOF Revenue predictions (index-aligned with df) used only when
        target == "COGS" (cascade stage).  Pass None for Revenue.

    Returns
    -------
    df_feat      : pd.DataFrame  — enriched frame (NaN rows dropped)
    log_col      : str           — name of the log1p-transformed target column
    all_features : list[str]     — all usable numeric feature columns
    trend_feats  : list[str]     — subset for Ridge trend specialist
    seas_feats   : list[str]     — subset for ElasticNet seasonal specialist
    """
    df = df.copy()
    log_col = f"_log_{target}"
    df[log_col] = np.log1p(df[target])

    d   = df["date"]
    doy = d.dt.dayofyear

    # ── 1. CALENDAR BASICS ────────────────────────────────────────────────────
    df["_dow"]       = d.dt.dayofweek          # 0=Mon … 6=Sun
    df["_dom"]       = d.dt.day
    df["_month"]     = d.dt.month
    df["_quarter"]   = d.dt.quarter
    df["_year"]      = d.dt.year
    df["_woy"]       = d.dt.isocalendar().week.astype(int)
    df["_is_weekend"]    = (d.dt.dayofweek >= 5).astype(int)
    df["_is_month_end"]  = d.dt.is_month_end.astype(int)
    df["_is_month_start"]= d.dt.is_month_start.astype(int)

    # Simple cyclical encodings for month and day-of-week
    df["_month_sin"] = np.sin(2 * np.pi * d.dt.month / 12)
    df["_month_cos"] = np.cos(2 * np.pi * d.dt.month / 12)
    df["_dow_sin"]   = np.sin(2 * np.pi * d.dt.dayofweek / 7)
    df["_dow_cos"]   = np.cos(2 * np.pi * d.dt.dayofweek / 7)

    # ── 2. MONTH-END DISTANCE (continuous + binary flags) ─────────────────────
    days_to_me = (d + pd.offsets.MonthEnd(0) - d).dt.days
    days_to_qe = (d + pd.offsets.QuarterEnd(0) - d).dt.days
    df["_days_to_month_end"] = days_to_me
    df["_is_last3_days"]     = (days_to_me <= 2).astype(int)
    df["_is_last7_days"]     = (days_to_me <= 6).astype(int)
    df["_days_to_qtr_end"]   = days_to_qe
    df["_is_last3_qtr"]      = (days_to_qe <= 2).astype(int)

    # EDA Section 5.2-B: continuous day-in-month position (0 → 1)
    # Better than binary flag — revenue ramps up gradually toward month-end
    df["_dom_ratio"] = d.dt.day / d.dt.days_in_month

    # ── 3. SEASONAL FLAGS (EDA Section 5.2-D) ─────────────────────────────────
    # Peak season: April–June (T4-T6) — confirmed as highest revenue months
    df["_is_peak_season"]   = d.dt.month.isin([4, 5, 6]).astype(int)
    # Low season: November–January (T11-T1) — lowest revenue cluster
    df["_is_low_season"]    = d.dt.month.isin([11, 12, 1]).astype(int)
    # Quarter-end months: activity spike from B2B / promotion flush
    df["_is_qtr_end_month"] = d.dt.month.isin([3, 6, 9, 12]).astype(int)

    # ── 4. TREND FEATURE (EDA Section 5.2-A & 3.2) ───────────────────────────
    # STL decomposition shows Revenue peaked in 2017 then declined 45% by 2022.
    # This feature allows tree/linear models to extrapolate the decline into 2023-24.
    #   Negative values → before peak (growth phase)
    #   Positive values → after  peak (decline phase)
    df["_trend_days_from_peak"] = (d - TREND_PEAK_DATE).dt.days
    df["_trend_norm"]           = df["_trend_days_from_peak"] / TREND_NORM_DAYS
    # _annual_yoy_growth is already attached by load_data()

    # ── 5. FOURIER TERMS (N=10) ────────────────────────────────────────────────
    # N=10 harmonics → captures seasonality at ~37-day resolution and shorter.
    # Combined with lag-364/365 these are the primary seasonal signal carriers
    # (EDA Section 3.3: seasonal strength = 0.86).
    for k in range(1, N_FOURIER + 1):
        df[f"_fs{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f"_fc{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

    # ── 6. TARGET LAGS ────────────────────────────────────────────────────────
    # lag-364 and lag-365 are the primary YoY anchors.
    # lag-7/14/30/60 capture short/medium-term momentum.
    for lag in [7, 14, 30, 60, 364, 365]:
        df[f"{log_col}_lag{lag}"] = df[log_col].shift(lag)

    for win in [7, 14, 30]:
        shifted = df[log_col].shift(1)          # avoid leakage
        df[f"{log_col}_rmean{win}"] = shifted.rolling(win).mean()
        df[f"{log_col}_rstd{win}"]  = shifted.rolling(win).std()

    # Smoothed lag-364: ±3-day centred rolling mean reduces noise from
    # one-off anomalous days in the reference year
    df[f"{log_col}_lag364_sm"] = (
        df[log_col].shift(364).rolling(7, center=True, min_periods=1).mean()
    )

    # ── 7. CAUSAL DRIVER LAGS (operational features) ──────────────────────────
    causal_cols = [
        "sessions", "unique_visitors",
        "avg_fill_rate", "total_stockout_flags", "total_stock_on_hand",
        "order_id", "customer_id",
    ]
    for col in causal_cols:
        if col in df.columns:
            for lag in [7, 14, 30]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Sessions & unique_visitors YoY ratio (EDA Section 4.2: MI gains most in real era)
    if "sessions" in df.columns:
        log_sess = np.log1p(df["sessions"])
        df["_sessions_yoy"] = log_sess - log_sess.shift(364)

    if "unique_visitors" in df.columns:
        log_uv = np.log1p(df["unique_visitors"])
        df["_uv_yoy"] = log_uv - log_uv.shift(364)

    # ── 8. TET (LUNAR NEW YEAR) PROXIMITY ─────────────────────────────────────
    df["_is_vn_holiday"] = d.apply(
        lambda x: int(f"{x.month:02d}-{x.day:02d}" in VN_PUBLIC_HOLIDAYS)
    )
    days_to_tet_series = d.apply(_days_to_tet)
    df["_days_to_tet"]    = days_to_tet_series
    df["_is_tet_week"]    = (days_to_tet_series.abs() <= 7).astype(int)
    # Pre-Tết window: strong demand pull (gift buying, stocking up)
    df["_is_pre_tet2w"]   = (
        (days_to_tet_series >= -14) & (days_to_tet_series < 0)
    ).astype(int)
    # Post-Tết: brief recovery after holiday lull
    df["_is_post_tet1w"]  = (
        (days_to_tet_series > 0) & (days_to_tet_series <= 7)
    ).astype(int)
    # Gaussian proximity kernel: smooth bell-curve centred on Tết day (σ = 7 days)
    df["_tet_proximity"]  = np.exp(-0.5 * (days_to_tet_series / 7.0) ** 2)

    # ── 9. CASCADE FEATURE (COGS only) ────────────────────────────────────────
    # Revenue prediction (log-scale) passed from Stage-2 cascade.
    # Pearson r(Revenue, COGS) = 0.976 → powerful causal signal.
    if rev_pred_series is not None:
        df["_rev_pred_log"] = np.log1p(rev_pred_series.values)

    # ── 10. DROP NaN rows introduced by lags ──────────────────────────────────
    df = df.dropna().reset_index(drop=True)

    # ── 11. FEATURE COLUMN SELECTION ──────────────────────────────────────────
    # Exclude: leaky targets, raw DROP_NOISE cols, meta columns, log target itself
    exclude = (
        LEAKY_COLS
        | DROP_NOISE
        | {"date", "day_name", log_col, "month_x", "day_of_week"}
        | {"Revenue", "COGS"}
    )
    valid_dtypes = (np.float64, np.int64, np.float32, np.int32, bool, np.bool_)
    all_features = [
        c for c in df.columns
        if c not in exclude and df[c].dtype.type in valid_dtypes
    ]

    # ── 12. SPECIALIST SUBSETS ────────────────────────────────────────────────
    # Trend specialist (Ridge): captures long-run direction
    trend_keywords = ["_trend", "_year", "_annual_yoy", "_woy"]
    trend_feats = [c for c in all_features if any(k in c for k in trend_keywords)]

    # Seasonal specialist (ElasticNet): captures periodic patterns
    seas_keywords = [
        "_fs", "_fc",                        # Fourier sin/cos
        "_lag364", "_lag365",                # YoY lags
        "_tet", "_month", "_quarter",        # calendar seasonality
        "_dom_ratio", "_peak", "_low",       # within-month + season flags
        "_qtr_end", "_dow_sin", "_dow_cos",  # other cyclical
        "_is_vn", "_is_month",
    ]
    seas_feats = [c for c in all_features if any(k in c for k in seas_keywords)]

    print(
        f"[engineer_features | {target}] "
        f"rows={len(df):,} | "
        f"all={len(all_features)} | "
        f"trend={len(trend_feats)} | "
        f"seasonal={len(seas_feats)}"
    )
    return df, log_col, all_features, trend_feats, seas_feats


# =============================================================================
# STAGE 1-C: get_sample_weights()
# =============================================================================

def get_sample_weights(df: pd.DataFrame, target_col: str = "Revenue") -> np.ndarray:
    """
    Compute per-row sample weights using exponential decay in time, with an
    additional multiplier applied to the most reliable data era.

    Rationale (EDA Section 1.1)
    ---------------------------
    - Pre-2018 data: web-traffic columns are zero/simulated (is_web_data_simulated=1).
      Revenue patterns in this era may not reflect true business dynamics.
    - 2021-2022 data: most recent, highest quality; model should trust it ~5× more
      than the simulated-data era.

    Weight construction
    -------------------
    Base weight  = exp(λ · t_norm)   where t_norm ∈ [0, 1] maps oldest→newest row
                                     and λ controls the decay steepness.

    Era multipliers applied on top of the base weight:
      - pre-2018  :  × 1.0   (baseline — oldest, partially simulated)
      - 2018-2020 :  × 2.0   (transition era — real but early web data)
      - 2021-2022 :  × 3.0   (high-quality, recent era)

    Parameters
    ----------
    df : pd.DataFrame
        Feature frame from engineer_features() (must contain a `date` column
        and have NaN rows already removed).
    target_col : str
        Unused in the computation; kept for API consistency.

    Returns
    -------
    weights : np.ndarray, shape (n_samples,)
        Unnormalised per-row weights (positive floats, suitable for passing
        directly to LightGBM's `sample_weight` / sklearn's `sample_weight`).
    """
    years = df["date"].dt.year.values

    # ── Base exponential decay ────────────────────────────────────────────────
    # Normalise date to [0, 1]: 0 = first training row, 1 = last training row
    dates_ord   = df["date"].map(pd.Timestamp.toordinal).values.astype(float)
    t_norm      = (dates_ord - dates_ord.min()) / (dates_ord.max() - dates_ord.min() + 1e-9)
    lambda_     = 3.0                         # decay steepness
    base_weight = np.exp(lambda_ * t_norm)    # shape: (n,)

    # ── Era multipliers ─────────────────────────────────────────────────────
    era_mult = np.ones(len(df), dtype=np.float64)
    era_mult[years < 2018]                    = 1.0   # simulated web-data era
    era_mult[(years >= 2018) & (years < 2021)] = 2.0  # transition era
    era_mult[years >= 2021]                   = 3.0   # high-quality, recent era

    raw_weights = base_weight * era_mult

    # FIX 1: Normalize to [1, 5] so max/min <= 5x.
    # Original scheme produced ~86x ratio which caused models to nearly ignore
    # pre-2019 seasonality patterns, leading to severe inference underprediction.
    w_min = raw_weights.min()
    w_max = raw_weights.max()
    weights = 1.0 + 4.0 * (raw_weights - w_min) / (w_max - w_min + 1e-9)

    # Sanity report
    df_report = df.copy()
    df_report["_w"] = weights
    summary = (
        df_report.groupby(df_report["date"].dt.year)["_w"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "mean_w", "sum": "sum_w", "count": "n_rows"})
    )
    print("\n[get_sample_weights] Era weight summary:")
    print(summary.to_string())
    print()

    return weights.astype(np.float32)


# =============================================================================
# STAGE 2-A: Base Model Factories
# =============================================================================

N_SPLITS = 5   # TimeSeriesSplit folds


def _make_lgb() -> lgb.LGBMRegressor:
    """
    LightGBM regressor tuned for MAE optimisation on daily revenue data.

    Key choices
    -----------
    - objective='huber' / metric='mae': robust to the extreme month-end spikes
      (max revenue = 5.7x median per EDA).
    - num_leaves=63, max_depth=7: enough capacity for 162 features without
      overfitting on ~2,800-row folds.
    - subsample + colsample_bytree = 0.8: row/column bagging reduces variance.
    - min_child_samples=20: prevents leaf nodes with too few samples.
    """
    return lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=7,
        min_child_samples=20,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="huber",       # robust to spikes
        metric="mae",
        huber_delta=1.35,        # standard Huber threshold
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def _make_xgb() -> xgb.XGBRegressor:
    """
    XGBoost regressor trained in log-space with squared error objective.

    Key choices
    -----------
    - reg:squarederror: stable and sufficient because log1p transform already
      compresses the skewed Revenue distribution. No overflow risk unlike
      reg:pseudohubererror which can be numerically unstable on log-scale data.
    - eval_metric='mae': monitor with MAE for consistency with competition metric.
    - early_stopping_rounds=80: set in constructor (required by modern XGBoost API).
    - tree_method='hist': fast histogram-based splits (same speed as LGBM).
    - subsample/colsample_bytree=0.8: consistent with LightGBM for model diversity.
    """
    return xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=10,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",   # stable on log-scale; skew handled by log1p
        eval_metric="mae",
        tree_method="hist",
        early_stopping_rounds=80,       # must be in constructor for modern XGB API
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def _make_cat() -> CatBoostRegressor:
    """
    CatBoost regressor with MAE loss.

    Key choices
    -----------
    - loss_function='MAE': directly optimises the competition metric.
    - depth=6, l2_leaf_reg=3.0: moderate regularisation.
    - subsample=0.8: row bagging (requires bootstrap_type='Bernoulli').
    - od_type='Iter', od_wait=80: Catboost-native early stopping.
    - verbose=0: suppress training log (CatBoost is noisy by default).
    """
    return CatBoostRegressor(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3.0,
        subsample=0.80,
        bootstrap_type="Bernoulli",
        loss_function="MAE",
        eval_metric="MAE",
        od_type="Iter",
        od_wait=80,
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=0,
    )


# =============================================================================
# STAGE 2-B: RevenueStackingModel — container
# =============================================================================

@dataclass
class RevenueStackingModel:
    """
    Container for a trained Revenue stacking ensemble.

    Attributes
    ----------
    lgb_models  : list of fitted LGBMRegressor (one per fold)
    xgb_models  : list of fitted XGBRegressor  (one per fold)
    cat_models  : list of fitted CatBoostRegressor (one per fold)
    meta        : fitted HuberRegressor (trained in log-space on OOF stack)
    meta_scaler : StandardScaler fitted on OOF stack columns
    feature_cols: ordered list of feature column names used by base models
    log_col     : name of the log1p-transformed target column
    fold_val_ranges : list of (start_idx, end_idx) for each validation fold
    """
    lgb_models: list       = field(default_factory=list)
    xgb_models: list       = field(default_factory=list)
    cat_models: list       = field(default_factory=list)
    meta: Optional[HuberRegressor]   = None
    meta_scaler: Optional[StandardScaler] = None
    feature_cols: list     = field(default_factory=list)
    log_col: str           = "_log_Revenue"
    fold_val_ranges: list  = field(default_factory=list)

    def predict_base(self, X: np.ndarray) -> np.ndarray:
        """
        Average base-model predictions across all stored fold-models.
        Returns shape (n_samples, 3): columns = [LGB, XGB, CAT] in log-space.
        """
        lgb_preds = np.mean(
            [m.predict(X) for m in self.lgb_models], axis=0
        )
        xgb_preds = np.mean(
            [m.predict(X) for m in self.xgb_models], axis=0
        )
        cat_preds = np.mean(
            [m.predict(X) for m in self.cat_models], axis=0
        )
        return np.column_stack([lgb_preds, xgb_preds, cat_preds])   # (n, 3)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Full stacking predict: base models -> meta -> original scale.
        Returns 1-D array in original Revenue scale (NOT log-space).
        """
        stack = self.predict_base(X)               # (n, 3) log-space
        stack_scaled = self.meta_scaler.transform(stack)
        log_pred = self.meta.predict(stack_scaled) # (n,)  log-space
        return np.expm1(log_pred)                  # (n,)  original scale


# =============================================================================
# STAGE 2-C: train_revenue_stage()
# =============================================================================

def train_revenue_stage(
    df_feat: pd.DataFrame,
    log_col: str,
    all_features: list[str],
    sample_weights: np.ndarray,
    n_splits: int = N_SPLITS,
    early_stopping_rounds: int = 80,
    verbose: bool = True,
) -> tuple[RevenueStackingModel, np.ndarray]:
    """
    Train the Revenue stacking ensemble using TimeSeriesSplit OOF strategy.

    Architecture
    ------------
    Level-0 (base models — trained in log-space)
      - LightGBM  : Huber objective, subsample + colsample bagging
      - XGBoost   : Pseudo-Huber objective
      - CatBoost  : MAE objective

    Level-1 (meta-learner — trained in log-space)
      - HuberRegressor: robust linear combiner.  Uses log-scale predictions
        from base models as inputs, therefore operates entirely in log-space
        and maps to log(Revenue). Final expm1() restores original scale.

    OOF strategy
    ------------
    TimeSeriesSplit guarantees NO look-ahead: validation always comes AFTER
    training. Each fold's validation predictions are collected into `oof_stack`
    (shape n_train x 3). The meta-learner is fitted on `oof_stack` vs.
    `y_log` (the ground truth in log-space).

    Sample weights
    --------------
    Passed to base models during training to up-weight 2021-22 data.
    NOT applied to meta-learner (it sees only well-calibrated OOF preds).

    Parameters
    ----------
    df_feat        : Feature frame from engineer_features()
    log_col        : Name of the log1p-transformed target column (e.g. '_log_Revenue')
    all_features   : Ordered list of feature column names
    sample_weights : Per-row weights from get_sample_weights()
    n_splits       : Number of TimeSeriesSplit folds (default 5)
    early_stopping_rounds : ES patience for LGB / XGB (default 80)
    verbose        : Print fold-level MAE and final OOF summary

    Returns
    -------
    model_bundle : RevenueStackingModel
        Fully trained bundle — base models (all folds) + fitted meta + scaler.
        Use model_bundle.predict(X) for inference.
    oof_log_preds : np.ndarray, shape (n_train,)
        OOF predictions in ORIGINAL Revenue scale (expm1 applied).
        Used as the `rev_pred_series` input to Stage 3 (COGS cascade).
    """
    X = df_feat[all_features].values.astype(np.float32)
    y_log = df_feat[log_col].values.astype(np.float64)    # log-space ground truth
    n = len(X)

    # OOF stacking matrix: columns = [LGB_log, XGB_log, CAT_log]
    oof_stack = np.zeros((n, 3), dtype=np.float64)

    bundle = RevenueStackingModel(feature_cols=all_features, log_col=log_col)
    tscv   = TimeSeriesSplit(n_splits=n_splits)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  STAGE 2 — Revenue Stacking  ({n_splits} folds, {n:,} rows)")
        print(f"{'='*65}")

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        bundle.fold_val_ranges.append((int(val_idx[0]), int(val_idx[-1])))

        X_tr,  X_val  = X[tr_idx],        X[val_idx]
        y_tr,  y_val  = y_log[tr_idx],    y_log[val_idx]
        w_tr          = sample_weights[tr_idx]

        if verbose:
            n_tr, n_val = len(tr_idx), len(val_idx)
            print(f"\n  [Fold {fold_idx+1}/{n_splits}] "
                  f"train={n_tr:,}  val={n_val:,}  "
                  f"val_date={df_feat['date'].iloc[val_idx[0]].date()}"
                  f" -> {df_feat['date'].iloc[val_idx[-1]].date()}")

        # ── LightGBM ──────────────────────────────────────────────────────────
        lgb_m = _make_lgb()
        lgb_m.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        lgb_val_pred = lgb_m.predict(X_val)
        oof_stack[val_idx, 0] = lgb_val_pred
        bundle.lgb_models.append(lgb_m)

        # ── XGBoost ───────────────────────────────────────────────────────────
        xgb_m = _make_xgb()
        xgb_m.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        xgb_val_pred = xgb_m.predict(X_val)
        oof_stack[val_idx, 1] = xgb_val_pred
        bundle.xgb_models.append(xgb_m)

        # ── CatBoost ──────────────────────────────────────────────────────────
        cat_m = _make_cat()
        cat_m.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )
        cat_val_pred = cat_m.predict(X_val)
        oof_stack[val_idx, 2] = cat_val_pred
        bundle.cat_models.append(cat_m)

        # ── Per-fold diagnostics ───────────────────────────────────────────────
        if verbose:
            y_val_orig = np.expm1(y_val)
            for name, preds_log in [
                ("LGB", lgb_val_pred),
                ("XGB", xgb_val_pred),
                ("CAT", cat_val_pred),
            ]:
                # Clip to safe log-space range before expm1 to avoid overflow
                preds_clipped = np.clip(preds_log, -10, 25)
                mae = mean_absolute_error(y_val_orig, np.expm1(preds_clipped))
                print(f"    {name:<6} val MAE = {mae:>12,.0f}")


    # ── Determine which rows have OOF coverage ────────────────────────────────
    # TimeSeriesSplit leaves the first ~n//(n_splits+1) rows unscored.
    # Find first row with any non-zero OOF prediction.
    first_oof = int(np.min(np.where(oof_stack.any(axis=1))[0]))
    meta_idx  = np.arange(first_oof, n)

    # ── Meta-learner: HuberRegressor in log-space ─────────────────────────────
    # Scale the 3-column OOF stack before feeding to linear meta-learner.
    scaler = StandardScaler()
    stack_meta_X = scaler.fit_transform(oof_stack[meta_idx])   # (m, 3)
    stack_meta_y = y_log[meta_idx]                             # (m,) log-space

    meta = HuberRegressor(
        epsilon=1.35,   # Huber threshold (same as delta in LGB) — robust to spikes
        alpha=0.001,    # L2 regularisation; small because input is already 3-d
        max_iter=500,
    )
    meta.fit(stack_meta_X, stack_meta_y)

    bundle.meta        = meta
    bundle.meta_scaler = scaler

    # ── OOF summary ────────────────────────────────────────────────────────────
    # Compute full OOF predictions in original scale for return value
    stack_all_scaled = scaler.transform(oof_stack[meta_idx])
    oof_log_meta     = meta.predict(stack_all_scaled)          # log-space
    oof_orig_meta    = np.expm1(oof_log_meta)                  # original scale
    y_val_orig_all   = np.expm1(y_log[meta_idx])

    if verbose:
        print(f"\n{'='*65}")
        print(f"  OOF Summary (rows {first_oof} -> {n-1}, n={len(meta_idx):,})")
        print(f"{'='*65}")
        for name, col_idx in [("LGB",0),("XGB",1),("CAT",2)]:
            preds_clipped = np.clip(oof_stack[meta_idx, col_idx], -10, 25)
            mae = mean_absolute_error(y_val_orig_all, np.expm1(preds_clipped))
            print(f"  Base {name:<6}  OOF MAE = {mae:>12,.0f}")
        mae_meta = mean_absolute_error(y_val_orig_all, oof_orig_meta)
        print(f"  Meta (Huber)  OOF MAE = {mae_meta:>12,.0f}")
        print(f"  Meta coefs  : LGB={meta.coef_[0]:.4f}  "
              f"XGB={meta.coef_[1]:.4f}  CAT={meta.coef_[2]:.4f}  "
              f"intercept={meta.intercept_:.4f}")
        print(f"{'='*65}\n")

    # Build full-length OOF array (NaN for rows without coverage)
    oof_full = np.full(n, np.nan)
    oof_full[meta_idx] = oof_orig_meta

    return bundle, oof_full


# =============================================================================
# STAGE 3-A: COGS Model Factories
# =============================================================================

def _make_cat_cogs() -> CatBoostRegressor:
    """
    CatBoost for COGS cascade.

    Identical structure to _make_cat() but kept separate so hyperparameters
    can be tuned independently for the COGS target.
    COGS has slightly lower skew than Revenue (1.625 vs 1.670) so the same
    MAE loss and depth settings apply directly.
    """
    return CatBoostRegressor(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3.0,
        subsample=0.80,
        bootstrap_type="Bernoulli",
        loss_function="MAE",
        eval_metric="MAE",
        od_type="Iter",
        od_wait=80,
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=0,
    )


def _make_en_cogs() -> ElasticNet:
    """
    ElasticNet for COGS cascade — captures the near-linear Revenue/COGS relationship.

    Rationale
    ---------
    EDA shows Pearson r(Revenue, COGS) = 0.976 — the relationship is almost
    perfectly linear in log-space. ElasticNet (L1+L2) can efficiently learn
    this proportionality while simultaneously applying feature selection across
    the 163 columns, reducing noise from low-MI features.

    Key choices
    -----------
    - alpha=0.01: mild regularisation — COGS signal is strong.
    - l1_ratio=0.5: balanced Lasso/Ridge mix.
    - max_iter=5000: enough iterations for convergence on 3k+ rows.
    - fit_intercept=True: necessary because log(COGS) ≠ log(Revenue) exactly.
    """
    return ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=5000,
        fit_intercept=True,
        random_state=RANDOM_STATE,
    )


# =============================================================================
# STAGE 3-B: COGSCascadeModel — container
# =============================================================================

@dataclass
class COGSCascadeModel:
    """
    Container for a trained COGS cascade ensemble.

    Attributes
    ----------
    cat_models  : list of fitted CatBoostRegressor (one per fold)
    en_models   : list of fitted ElasticNet  (one per fold)
    en_scalers  : list of fitted StandardScaler (one per fold, for ElasticNet input)
    meta        : fitted RidgeCV (trained in log-space on OOF stack)
    meta_scaler : StandardScaler fitted on OOF stack columns
    feature_cols: ordered list of feature column names (includes _rev_pred_log)
    log_col     : name of the log1p-transformed target column (_log_COGS)
    fold_val_ranges : list of (start_idx, end_idx) for each validation fold
    """
    cat_models:  list = field(default_factory=list)
    en_models:   list = field(default_factory=list)
    en_scalers:  list = field(default_factory=list)
    meta: Optional[RidgeCV]          = None
    meta_scaler: Optional[StandardScaler] = None
    feature_cols: list               = field(default_factory=list)
    log_col: str                     = "_log_COGS"
    fold_val_ranges: list            = field(default_factory=list)

    def predict_base(self, X: np.ndarray) -> np.ndarray:
        """
        Average base-model predictions across all stored fold-models.
        Returns shape (n_samples, 2): columns = [CAT, EN] in log-space.
        """
        cat_preds = np.mean(
            [m.predict(X) for m in self.cat_models], axis=0
        )
        en_preds = np.mean(
            [m.predict(sc.transform(X)) for m, sc in
             zip(self.en_models, self.en_scalers)], axis=0
        )
        return np.column_stack([cat_preds, en_preds])   # (n, 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Full cascade predict: base models -> meta -> original COGS scale.
        Returns 1-D array in original COGS scale (NOT log-space).
        """
        stack = self.predict_base(X)               # (n, 2) log-space
        stack_scaled = self.meta_scaler.transform(stack)
        log_pred = self.meta.predict(stack_scaled) # (n,)  log-space
        return np.expm1(log_pred)                  # (n,)  original scale


# =============================================================================
# STAGE 3-C: train_cogs_stage()
# =============================================================================

def train_cogs_stage(
    df_feat: pd.DataFrame,
    log_col: str,
    all_features: list[str],
    sample_weights: np.ndarray,
    rev_oof_series: np.ndarray,
    n_splits: int = N_SPLITS,
    verbose: bool = True,
) -> tuple[COGSCascadeModel, np.ndarray]:
    """
    Train the COGS cascade ensemble using TimeSeriesSplit OOF strategy.

    Architecture
    ------------
    Level-0 (base models — trained in log-space on COGS features + Revenue OOF)
      - CatBoost  : MAE objective — captures non-linear COGS drivers
      - ElasticNet: L1+L2 — efficiently captures the near-linear Rev/COGS
                    relationship (r = 0.976, dominant in log-space)

    Level-1 (meta-learner — trained in log-space)
      - RidgeCV   : Stable linear combination of the two base models.
                    Uses built-in CV to select the best alpha automatically.
                    Chosen over HuberRegressor here because the ElasticNet
                    base already handles outliers via regularisation, and
                    RidgeCV is simpler / less prone to overfitting with only
                    2 input columns.

    Cascade feature: `_rev_pred_log`
    ---------------------------------
    `rev_oof_series` (Revenue OOF predictions in original scale) is converted
    to log-scale and injected as the column `_rev_pred_log` into df_feat.
    This allows base models to directly exploit the 0.976 correlation without
    any look-ahead — OOF Revenue predictions are themselves leak-free.

    For rows without OOF Revenue coverage (NaN), the column is filled with
    the rolling 30-day mean of known Revenue in log-space as a fallback.

    Parameters
    ----------
    df_feat        : Feature frame from engineer_features(df, 'COGS')
    log_col        : Name of the log1p-transformed COGS column ('_log_COGS')
    all_features   : Ordered list of feature column names (may or may not
                     include '_rev_pred_log'; this function ensures it's added)
    sample_weights : Per-row weights from get_sample_weights()
    rev_oof_series : 1-D array (length = len(df_feat)) of OOF Revenue predictions
                     in ORIGINAL scale (expm1 applied), as returned by
                     train_revenue_stage(). NaN entries are handled gracefully.
    n_splits       : Number of TimeSeriesSplit folds (default 5)
    verbose        : Print fold-level MAE and final OOF summary

    Returns
    -------
    model_bundle : COGSCascadeModel
        Fully trained bundle — base models (all folds) + fitted meta + scaler.
        Use model_bundle.predict(X) for inference.
    oof_preds    : np.ndarray, shape (n_train,)
        OOF predictions in ORIGINAL COGS scale (expm1 applied).
    """
    df_feat = df_feat.copy()

    # ── Inject Revenue OOF as causal driver ───────────────────────────────────
    # Convert to log-scale; fill NaN slots with rolling mean fallback
    rev_log = np.log1p(np.where(
        np.isnan(rev_oof_series), 0.0, rev_oof_series
    ))
    rev_log_series = pd.Series(rev_log, index=df_feat.index)

    # Smooth fallback for NaN positions (rolling 30-day mean)
    rev_log_filled = rev_log_series.fillna(
        rev_log_series.rolling(30, min_periods=1).mean()
    ).fillna(rev_log_series.mean())

    df_feat["_rev_pred_log"] = rev_log_filled.values

    # Ensure _rev_pred_log is in feature list
    if "_rev_pred_log" not in all_features:
        all_features = all_features + ["_rev_pred_log"]

    X = df_feat[all_features].values.astype(np.float32)
    y_log = df_feat[log_col].values.astype(np.float64)   # log-space COGS
    n = len(X)

    # OOF stacking matrix: columns = [CAT_log, EN_log]
    oof_stack = np.zeros((n, 2), dtype=np.float64)

    bundle = COGSCascadeModel(feature_cols=all_features, log_col=log_col)
    tscv   = TimeSeriesSplit(n_splits=n_splits)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  STAGE 3 — COGS Cascade  ({n_splits} folds, {n:,} rows)")
        print(f"  Causal driver: _rev_pred_log (r=0.976 with COGS)")
        print(f"{'='*65}")

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        bundle.fold_val_ranges.append((int(val_idx[0]), int(val_idx[-1])))

        X_tr,  X_val  = X[tr_idx],      X[val_idx]
        y_tr,  y_val  = y_log[tr_idx],  y_log[val_idx]
        w_tr          = sample_weights[tr_idx]

        if verbose:
            n_tr, n_val = len(tr_idx), len(val_idx)
            print(f"\n  [Fold {fold_idx+1}/{n_splits}] "
                  f"train={n_tr:,}  val={n_val:,}  "
                  f"val_date={df_feat['date'].iloc[val_idx[0]].date()}"
                  f" -> {df_feat['date'].iloc[val_idx[-1]].date()}")

        # ── CatBoost ──────────────────────────────────────────────────────────
        cat_m = _make_cat_cogs()
        cat_m.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )
        cat_val_pred = cat_m.predict(X_val)
        oof_stack[val_idx, 0] = cat_val_pred
        bundle.cat_models.append(cat_m)

        # ── ElasticNet (scaled) ────────────────────────────────────────────────
        # ElasticNet requires feature scaling to work properly (unlike trees).
        # Scaler fitted only on training fold to avoid leakage.
        en_sc = StandardScaler()
        X_tr_sc  = en_sc.fit_transform(X_tr)
        X_val_sc = en_sc.transform(X_val)

        en_m = _make_en_cogs()
        en_m.fit(X_tr_sc, y_tr, sample_weight=w_tr)
        en_val_pred = en_m.predict(X_val_sc)
        oof_stack[val_idx, 1] = en_val_pred
        bundle.en_models.append(en_m)
        bundle.en_scalers.append(en_sc)

        # ── Per-fold diagnostics ──────────────────────────────────────────────
        if verbose:
            y_val_orig = np.expm1(y_val)
            for name, preds_log in [
                ("CAT", cat_val_pred),
                ("EN",  en_val_pred),
            ]:
                preds_clipped = np.clip(preds_log, -10, 25)
                mae = mean_absolute_error(y_val_orig, np.expm1(preds_clipped))
                print(f"    {name:<6} val MAE = {mae:>12,.0f}")

    # ── OOF coverage boundary (same logic as Revenue stage) ───────────────────
    first_oof = int(np.min(np.where(oof_stack.any(axis=1))[0]))
    meta_idx  = np.arange(first_oof, n)

    # ── Meta-learner: RidgeCV in log-space ────────────────────────────────────
    # Scale the 2-column OOF stack before feeding to RidgeCV.
    # RidgeCV selects best alpha via Leave-One-Out CV internally.
    scaler = StandardScaler()
    stack_meta_X = scaler.fit_transform(oof_stack[meta_idx])   # (m, 2)
    stack_meta_y = y_log[meta_idx]                             # (m,) log-space

    meta = RidgeCV(
        alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        fit_intercept=True,
        scoring="neg_mean_absolute_error",
        cv=5,
    )
    meta.fit(stack_meta_X, stack_meta_y)

    bundle.meta        = meta
    bundle.meta_scaler = scaler

    # ── OOF summary ───────────────────────────────────────────────────────────
    stack_all_scaled = scaler.transform(oof_stack[meta_idx])
    oof_log_meta     = meta.predict(stack_all_scaled)     # log-space
    oof_orig_meta    = np.expm1(oof_log_meta)             # original scale
    y_val_orig_all   = np.expm1(y_log[meta_idx])

    if verbose:
        print(f"\n{'='*65}")
        print(f"  OOF Summary (rows {first_oof} -> {n-1}, n={len(meta_idx):,})")
        print(f"{'='*65}")
        for name, col_idx in [("CAT", 0), ("EN", 1)]:
            preds_clipped = np.clip(oof_stack[meta_idx, col_idx], -10, 25)
            mae = mean_absolute_error(y_val_orig_all, np.expm1(preds_clipped))
            print(f"  Base {name:<6}  OOF MAE = {mae:>12,.0f}")
        mae_meta = mean_absolute_error(y_val_orig_all, oof_orig_meta)
        print(f"  Meta (RidgeCV) OOF MAE = {mae_meta:>12,.0f}")
        print(f"  RidgeCV best alpha : {meta.alpha_:.4f}")
        print(f"  Meta coefs : CAT={meta.coef_[0]:.4f}  EN={meta.coef_[1]:.4f}  "
              f"intercept={meta.intercept_:.4f}")
        print(f"{'='*65}\n")

    # Build full-length OOF array (NaN for rows without coverage)
    oof_full = np.full(n, np.nan)
    oof_full[meta_idx] = oof_orig_meta
    return bundle, oof_full


# =============================================================================
# STAGE 4-A: predict_future()
# =============================================================================

# Business ratio bounds (EDA Section 2.3)
COGS_RATIO_MEAN = 0.8746    # historical mean COGS/Revenue
COGS_RATIO_STD  = 0.1274    # historical std of ratio
COGS_RATIO_LO   = COGS_RATIO_MEAN - 2 * COGS_RATIO_STD   # ~0.62
COGS_RATIO_HI   = COGS_RATIO_MEAN + 2 * COGS_RATIO_STD   # ~1.13

# Freeze inference trend features at last-training-day (2022-12-31).
# Tree models cannot extrapolate beyond training range for monotonic features
# like _trend_days_from_peak; extrapolation caused ~2.4x Revenue underprediction.
_INFERENCE_FREEZE_DATE = pd.Timestamp("2022-12-31")
_FROZEN_TREND_DAYS     = int((_INFERENCE_FREEZE_DATE - TREND_PEAK_DATE).days)  # 2190
_FROZEN_TREND_NORM     = _FROZEN_TREND_DAYS / TREND_NORM_DAYS                  # ~1.20
_FROZEN_YEAR           = 2022.0   # 2023/2024 are out-of-training-range for _year

# Causal lags used during autoregressive future row construction
_CAUSAL_COLS = [
    "sessions", "unique_visitors",
    "avg_fill_rate", "total_stockout_flags", "total_stock_on_hand",
    "order_id", "customer_id",
]


def _build_future_row(
    fdate: pd.Timestamp,
    hist_log: pd.Series,        # log-scale target history (Revenue or COGS)
    hist_raw: pd.DataFrame,     # full raw history df (for causal cols)
    log_col: str,
    all_features: list[str],
    last_yoy_growth: float,
    rev_pred_value: float | None = None,   # only for COGS rows
) -> np.ndarray:
    """
    Build a single future feature row for autoregressive inference.

    Strategy
    --------
    - Calendar/Fourier/Trend features: computed deterministically from date.
    - Target lags (lag-7…lag-365): pulled from the growing history buffer.
    - Causal lags (sessions, etc.): pulled from the last known row of raw history
      (we assume operational data stays at its 2022-12-31 level; a simple but
      reasonable assumption since we have no future operational data).
    - _rev_pred_log: log(Revenue prediction) for COGS rows (cascade input).

    Returns
    -------
    np.ndarray, shape (1, n_features), dtype float32
    """
    n   = len(hist_log)
    doy = fdate.dayofyear

    days_to_me = (fdate + pd.offsets.MonthEnd(0) - fdate).days
    days_to_qe = (fdate + pd.offsets.QuarterEnd(0) - fdate).days
    dt_tet     = _days_to_tet(fdate)

    row: dict[str, float] = {
        # ── Calendar ──────────────────────────────────────────────────────
        "_dow":              float(fdate.dayofweek),
        "_dom":              float(fdate.day),
        "_month":            float(fdate.month),
        "_quarter":          float((fdate.month - 1) // 3 + 1),
        "_year":             float(fdate.year),
        "_woy":              float(fdate.isocalendar()[1]),
        "_is_weekend":       float(fdate.dayofweek >= 5),
        "_is_month_end":     float(fdate.day == pd.Timestamp(fdate).days_in_month),
        "_is_month_start":   float(fdate.day == 1),
        "_month_sin":        float(np.sin(2 * np.pi * fdate.month / 12)),
        "_month_cos":        float(np.cos(2 * np.pi * fdate.month / 12)),
        "_dow_sin":          float(np.sin(2 * np.pi * fdate.dayofweek / 7)),
        "_dow_cos":          float(np.cos(2 * np.pi * fdate.dayofweek / 7)),
        # ── Month / quarter end ────────────────────────────────────────────
        "_days_to_month_end": float(days_to_me),
        "_is_last3_days":    float(days_to_me <= 2),
        "_is_last7_days":    float(days_to_me <= 6),
        "_days_to_qtr_end":  float(days_to_qe),
        "_is_last3_qtr":     float(days_to_qe <= 2),
        "_dom_ratio":        float(fdate.day / fdate.days_in_month),
        # ── Season flags ──────────────────────────────────────────────────
        "_is_peak_season":   float(fdate.month in [4, 5, 6]),
        "_is_low_season":    float(fdate.month in [11, 12, 1]),
        "_is_qtr_end_month": float(fdate.month in [3, 6, 9, 12]),
        # ── Trend ─────────────────────────────────────────────────────────
        "_trend_days_from_peak": float(_FROZEN_TREND_DAYS),  # frozen @ 2022-12-31
        "_annual_yoy_growth":    float(last_yoy_growth),
        "_year":                 _FROZEN_YEAR,  # guard OOD: 2023/2024 outside training
        # ── Tết ───────────────────────────────────────────────────────────
        "_is_vn_holiday":    float(f"{fdate.month:02d}-{fdate.day:02d}" in VN_PUBLIC_HOLIDAYS),
        "_days_to_tet":      float(dt_tet),
        "_is_tet_week":      float(abs(dt_tet) <= 7),
        "_is_pre_tet2w":     float(-14 <= dt_tet < 0),
        "_is_post_tet1w":    float(0 < dt_tet <= 7),
        "_tet_proximity":    float(np.exp(-0.5 * (dt_tet / 7.0) ** 2)),
    }

    # Derived from frozen trend
    row["_trend_norm"] = _FROZEN_TREND_NORM

    # ── Fourier ───────────────────────────────────────────────────────────────
    for k in range(1, N_FOURIER + 1):
        row[f"_fs{k}"] = float(np.sin(2 * np.pi * k * doy / 365.25))
        row[f"_fc{k}"] = float(np.cos(2 * np.pi * k * doy / 365.25))

    # ── Target lags ───────────────────────────────────────────────────────────
    for lag in [7, 14, 30, 60, 364, 365]:
        idx = n - lag
        row[f"{log_col}_lag{lag}"] = float(hist_log.iloc[idx]) if idx >= 0 else 0.0

    for win in [7, 14, 30]:
        row[f"{log_col}_rmean{win}"] = float(hist_log.iloc[-win:].mean())
        row[f"{log_col}_rstd{win}"]  = float(hist_log.iloc[-win:].std()) if win > 1 else 0.0

    # Smoothed lag-364 (±3-day window)
    idx364 = n - 364
    if idx364 >= 3:
        row[f"{log_col}_lag364_sm"] = float(
            hist_log.iloc[max(0, idx364 - 3): idx364 + 4].mean()
        )
    else:
        row[f"{log_col}_lag364_sm"] = row.get(f"{log_col}_lag364", 0.0)

    # ── Causal driver lags ────────────────────────────────────────────────────
    # Use last known value from raw history (2022-12-31 level held constant)
    last_raw = hist_raw.iloc[-1]
    for col in _CAUSAL_COLS:
        if col in hist_raw.columns:
            for lag in [7, 14, 30]:
                row[f"{col}_lag{lag}"] = float(last_raw[col])

    # Sessions / UV YoY (log-delta)
    # Note: hist_raw is fixed at training length (2022-12-31 frozen).
    # For future dates we have no real sessions data, so YoY delta = 0.0
    # (flat assumption; real sessions/UV are unavailable for 2023-2024).
    n_raw = len(hist_raw)
    if "sessions" in hist_raw.columns:
        ls = np.log1p(hist_raw["sessions"])
        idx_yoy = n_raw - 364
        row["_sessions_yoy"] = (
            float(ls.iloc[-1] - ls.iloc[idx_yoy]) if idx_yoy >= 0 else 0.0
        )
    if "unique_visitors" in hist_raw.columns:
        lv = np.log1p(hist_raw["unique_visitors"])
        idx_yoy = n_raw - 364
        row["_uv_yoy"] = (
            float(lv.iloc[-1] - lv.iloc[idx_yoy]) if idx_yoy >= 0 else 0.0
        )

    # ── COGS cascade feature ──────────────────────────────────────────────────
    if rev_pred_value is not None:
        row["_rev_pred_log"] = float(np.log1p(max(rev_pred_value, 0.0)))

    # ── Assemble feature vector ───────────────────────────────────────────────
    vec = np.array(
        [row.get(c, 0.0) for c in all_features], dtype=np.float32
    ).reshape(1, -1)
    return vec


def predict_future(
    raw_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    rev_bundle: RevenueStackingModel,
    cogs_bundle: COGSCascadeModel,
    log_rev: str,
    log_cogs: str,
    rev_features: list[str],
    cogs_features: list[str],
    last_yoy_growth: float,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Autoregressive future inference for Revenue then COGS.

    Strategy
    --------
    Revenue first
    ~~~~~~~~~~~~~
    Iterate over future dates one-by-one. At each step:
      1. Build a feature row from the growing history buffer.
      2. Predict Revenue via rev_bundle.predict().
      3. Append the prediction (in log-space) to the history buffer.
         This makes lag-7/14/30 features of the *next* future row causal.

    COGS cascade
    ~~~~~~~~~~~~
    After all Revenue predictions are computed:
      1. Iterate over future dates again.
      2. Build a COGS feature row, injecting the already-computed Revenue
         prediction for that date as `_rev_pred_log`.
      3. Predict COGS via cogs_bundle.predict().

    Parameters
    ----------
    raw_df         : Training data frame (features + Revenue + COGS columns)
    future_dates   : pd.DatetimeIndex of dates to predict (2023-01-01 onwards)
    rev_bundle     : Trained RevenueStackingModel
    cogs_bundle    : Trained COGSCascadeModel
    log_rev        : Name of log1p Revenue column (e.g. '_log_Revenue')
    log_cogs       : Name of log1p COGS column
    rev_features   : Ordered feature list for Revenue model
    cogs_features  : Ordered feature list for COGS model (includes _rev_pred_log)
    last_yoy_growth: Last observed YoY growth ratio (anchor for future trend)
    verbose        : Print progress every 90 days

    Returns
    -------
    rev_preds  : np.ndarray, shape (n_future,) — Revenue in original scale
    cogs_preds : np.ndarray, shape (n_future,) — COGS in original scale
    """
    n_future = len(future_dates)

    # ── Initialise history buffers ────────────────────────────────────────────
    # We maintain rolling log-history for lag construction.
    # Start from the full training series so lag-364/365 are available immediately.
    rev_hist_log  = np.log1p(raw_df["Revenue"].values).tolist()
    cogs_hist_log = np.log1p(raw_df["COGS"].values).tolist()
    hist_raw      = raw_df.copy().reset_index(drop=True)

    rev_preds  = np.zeros(n_future)
    cogs_preds = np.zeros(n_future)

    # ── Phase 1: Revenue ─────────────────────────────────────────────────────
    if verbose:
        print(f"\n[predict_future] Predicting Revenue for {n_future} days...")

    for i, fdate in enumerate(future_dates):
        hist_log_series = pd.Series(rev_hist_log)

        vec = _build_future_row(
            fdate=fdate,
            hist_log=hist_log_series,
            hist_raw=hist_raw,
            log_col=log_rev,
            all_features=rev_features,
            last_yoy_growth=last_yoy_growth,
            rev_pred_value=None,
        )
        pred_rev = float(np.maximum(rev_bundle.predict(vec)[0], 0.0))
        rev_preds[i] = pred_rev

        # Append to history so next iteration can use lag-1..lag-30
        rev_hist_log.append(float(np.log1p(pred_rev)))

        if verbose and (i % 90 == 0 or i == n_future - 1):
            print(f"  Rev [{i+1:4d}/{n_future}]  {fdate.date()}  pred={pred_rev:>12,.0f}")

    # ── Phase 2: COGS (cascade) ───────────────────────────────────────────────
    if verbose:
        print(f"\n[predict_future] Predicting COGS for {n_future} days (cascade)...")

    for i, fdate in enumerate(future_dates):
        hist_log_series = pd.Series(cogs_hist_log)

        # Inject corresponding Revenue prediction as causal driver
        vec = _build_future_row(
            fdate=fdate,
            hist_log=hist_log_series,
            hist_raw=hist_raw,
            log_col=log_cogs,
            all_features=cogs_features,
            last_yoy_growth=last_yoy_growth,
            rev_pred_value=rev_preds[i],
        )
        pred_cogs = float(np.maximum(cogs_bundle.predict(vec)[0], 0.0))
        cogs_preds[i] = pred_cogs

        # Append to COGS history buffer
        cogs_hist_log.append(float(np.log1p(pred_cogs)))

        if verbose and (i % 90 == 0 or i == n_future - 1):
            ratio = pred_cogs / pred_rev if pred_rev > 0 else 0
            print(f"  COGS[{i+1:4d}/{n_future}]  {fdate.date()}  "
                  f"pred={pred_cogs:>12,.0f}  ratio={ratio:.3f}")

    return rev_preds, cogs_preds


# =============================================================================
# STAGE 4-B: post_process()
# =============================================================================

def post_process(
    rev_preds: np.ndarray,
    cogs_preds: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforce COGS/Revenue business ratio constraint derived from EDA Section 2.3.

    EDA findings
    ------------
      - COGS/Revenue ratio mean = 0.8746, std = 0.1274
      - Allowed band: mean ± 2*std  =>  [0.62, 1.13]
        (user spec: 0.87 +/- 0.25 => [0.62, 1.12], almost identical)

    Correction logic
    ----------------
    For each row where ratio = COGS/Revenue falls outside [LO, HI]:
      - If ratio < LO: set COGS = Revenue * LO  (floor clip)
      - If ratio > HI: set COGS = Revenue * HI  (ceiling clip)
    Revenue is never modified — only COGS is adjusted.

    Parameters
    ----------
    rev_preds  : 1-D array of Revenue predictions (original scale)
    cogs_preds : 1-D array of COGS predictions    (original scale)
    verbose    : Print summary of corrections applied

    Returns
    -------
    rev_preds  : unchanged
    cogs_preds : corrected COGS predictions
    """
    cogs_out = cogs_preds.copy()
    ratio    = np.where(rev_preds > 0, cogs_out / rev_preds, COGS_RATIO_MEAN)

    # FIX 3: Hard cap at 1.0 (COGS can never exceed Revenue — negative margin
    # is not observed in EDA and is economically implausible for this business).
    # Also apply a soft 40% blend toward historical mean (0.8746) for rows
    # where ratio > 0.95 or < 0.70, reducing systematic bias without hard-clipping.
    HARD_HI = 1.0          # COGS <= Revenue always
    HARD_LO = COGS_RATIO_LO  # 0.62
    BLEND   = 0.40         # pull-toward-mean strength for soft-outlier zone

    mask_lo = ratio < HARD_LO
    mask_hi = ratio > HARD_HI
    cogs_out[mask_lo] = rev_preds[mask_lo] * HARD_LO
    cogs_out[mask_hi] = rev_preds[mask_hi] * HARD_HI

    # Soft blend for mild outliers: ratio in (0.95, 1.0] or [0.62, 0.70)
    ratio2 = np.where(rev_preds > 0, cogs_out / rev_preds, COGS_RATIO_MEAN)
    soft_hi = (ratio2 > 0.95) & ~mask_hi
    soft_lo = (ratio2 < 0.70) & ~mask_lo
    mean_target = rev_preds * COGS_RATIO_MEAN
    cogs_out[soft_hi] = (1 - BLEND) * cogs_out[soft_hi] + BLEND * mean_target[soft_hi]
    cogs_out[soft_lo] = (1 - BLEND) * cogs_out[soft_lo] + BLEND * mean_target[soft_lo]

    if verbose:
        n_hard    = int(mask_lo.sum() + mask_hi.sum())
        n_soft    = int(soft_hi.sum() + soft_lo.sum())
        n_total   = len(cogs_out)
        ratio_after = np.where(rev_preds > 0, cogs_out / rev_preds, COGS_RATIO_MEAN)
        print(f"\n[post_process] Hard clip [{HARD_LO:.3f}, {HARD_HI:.3f}]  "
              f"Soft-blend ({BLEND:.0%}) for ratio outside [0.70, 0.95]")
        print(f"  Hard-clipped : {n_hard:,} / {n_total:,}  "
              f"(lo={mask_lo.sum()}, hi={mask_hi.sum()})")
        print(f"  Soft-blended : {n_soft:,} / {n_total:,}")
        print(f"  Ratio after  : mean={ratio_after.mean():.4f}  "
              f"std={ratio_after.std():.4f}  "
              f"min={ratio_after.min():.4f}  max={ratio_after.max():.4f}")

    return rev_preds, cogs_out


# =============================================================================
# STAGE 4-C: run_pipeline()   — full production orchestrator
# =============================================================================

def run_pipeline(n_splits: int = N_SPLITS, verbose: bool = True) -> None:
    """
    End-to-end production pipeline.

    Steps
    -----
    1. Load data + compute YoY anchor
    2. Engineer Revenue features + sample weights
    3. Train Revenue stacking ensemble (LGB + XGB + CAT + HuberRegressor meta)
    4. Engineer COGS features + sample weights
    5. Train COGS cascade ensemble (CAT + ElasticNet + RidgeCV meta)
       using Revenue OOF predictions as causal driver
    6. Load future dates from sample_submission.csv
    7. Autoregressive forecast: Revenue first, then COGS (cascade)
    8. Post-process: enforce COGS/Revenue ratio in [0.62, 1.13]
    9. Write submission.csv

    Parameters
    ----------
    n_splits : Number of TimeSeriesSplit folds (5 for production, 2 for debug)
    verbose  : Verbosity flag passed through to all sub-functions
    """
    sep = "=" * 65

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 1 — Load Data\n{sep}")
    raw_df, last_yoy = load_data()

    # ── Step 2: Revenue feature engineering + weights ─────────────────────────
    print(f"\n{sep}\n  STEP 2 — Revenue Feature Engineering\n{sep}")
    df_rev, log_rev, rev_feats, _, _ = engineer_features(raw_df, "Revenue")
    w_rev = get_sample_weights(df_rev)

    # ── Step 3: Train Revenue stacking ────────────────────────────────────────
    print(f"\n{sep}\n  STEP 3 — Train Revenue Stacking ({n_splits} folds)\n{sep}")
    rev_bundle, rev_oof = train_revenue_stage(
        df_feat=df_rev,
        log_col=log_rev,
        all_features=rev_feats,
        sample_weights=w_rev,
        n_splits=n_splits,
        verbose=verbose,
    )

    # ── Step 4: COGS feature engineering + weights ────────────────────────────
    print(f"\n{sep}\n  STEP 4 — COGS Feature Engineering\n{sep}")
    df_cogs, log_cogs, cogs_feats, _, _ = engineer_features(raw_df, "COGS")
    w_cogs = get_sample_weights(df_cogs)

    # Align OOF Revenue to COGS frame length (same dropna → same length, but guard)
    min_len = min(len(df_cogs), len(rev_oof))
    rev_oof_aligned = np.full(len(df_cogs), np.nan)
    rev_oof_aligned[:min_len] = rev_oof[:min_len]

    # ── Step 5: Train COGS cascade ────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 5 — Train COGS Cascade ({n_splits} folds)\n{sep}")
    cogs_bundle, _ = train_cogs_stage(
        df_feat=df_cogs,
        log_col=log_cogs,
        all_features=cogs_feats,
        sample_weights=w_cogs,
        rev_oof_series=rev_oof_aligned,
        n_splits=n_splits,
        verbose=verbose,
    )

    # ── Step 6: Load future dates ─────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 6 — Load Submission Template\n{sep}")
    sub_template = pd.read_csv(SUBMIT_IN, parse_dates=["Date"])
    sub_template.columns = sub_template.columns.str.strip()
    future_dates = pd.DatetimeIndex(sub_template["Date"].values)
    print(f"  Future horizon: {future_dates[0].date()} -> {future_dates[-1].date()}"
          f"  ({len(future_dates):,} days)")

    # Retrieve COGS feature list with _rev_pred_log guaranteed present
    cogs_feats_with_cascade = cogs_bundle.feature_cols   # set by train_cogs_stage

    # ── Step 7: Autoregressive forecast ───────────────────────────────────────
    print(f"\n{sep}\n  STEP 7 — Autoregressive Forecast\n{sep}")
    rev_preds, cogs_preds = predict_future(
        raw_df=raw_df,
        future_dates=future_dates,
        rev_bundle=rev_bundle,
        cogs_bundle=cogs_bundle,
        log_rev=log_rev,
        log_cogs=log_cogs,
        rev_features=rev_feats,
        cogs_features=cogs_feats_with_cascade,
        last_yoy_growth=last_yoy,
        verbose=verbose,
    )

    # ── Step 8: Post-processing (ratio constraint) ─────────────────────────────
    print(f"\n{sep}\n  STEP 8 — Post-Processing (COGS/Revenue ratio)\n{sep}")
    rev_preds, cogs_preds = post_process(rev_preds, cogs_preds, verbose=verbose)

    # ── Step 9: Write submission ──────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 9 — Write submission.csv\n{sep}")
    out = pd.DataFrame({
        "Date":    sub_template["Date"].dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(rev_preds,  2),
        "COGS":    np.round(cogs_preds, 2),
    })
    out.to_csv(SUBMIT_OUT, index=False)

    # Sanity print
    print(f"  Saved -> {SUBMIT_OUT}  ({len(out):,} rows)")
    print(out.head(10).to_string(index=False))
    rev_mae_proxy  = out["Revenue"].mean()
    ratio_col      = out["COGS"] / out["Revenue"]
    print(f"\n  Revenue  mean={out['Revenue'].mean():>12,.0f}  "
          f"min={out['Revenue'].min():>12,.0f}  max={out['Revenue'].max():>12,.0f}")
    print(f"  COGS     mean={out['COGS'].mean():>12,.0f}  "
          f"min={out['COGS'].min():>12,.0f}  max={out['COGS'].max():>12,.0f}")
    print(f"  Ratio    mean={ratio_col.mean():.4f}  "
          f"std={ratio_col.std():.4f}  "
          f"min={ratio_col.min():.4f}  max={ratio_col.max():.4f}")
    print(f"\n{sep}\n  Pipeline complete.\n{sep}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    # ── Production run (default) ──────────────────────────────────────────────
    # python final_pipeline.py              -> full 5-fold run, writes submission.csv
    # python final_pipeline.py --fast       -> 2-fold quick run (debug)
    # python final_pipeline.py --smoke      -> Stage 1-3 unit smoke tests (no submission)

    if "--smoke" in sys.argv:
        # ── Smoke test (legacy: stages 1-3 validation) ───────────────────────
        run_s2 = "--stage2" in sys.argv or "--stage3" in sys.argv
        run_s3 = "--stage3" in sys.argv

        print("=" * 65)
        print("  STAGE 1 SMOKE TEST")
        print("=" * 65)
        raw_df, last_yoy = load_data()
        df_rev, log_rev, all_f, trend_f, seas_f = engineer_features(raw_df, "Revenue")
        w = get_sample_weights(df_rev)
        print(f"\n[smoke] Feature frame : {df_rev.shape}")
        print(f"[smoke] Weights        : min={w.min():.3f} max={w.max():.3f} mean={w.mean():.3f}")
        print("[OK] Stage 1 smoke test passed.")

        if run_s2 or run_s3:
            print("\n" + "=" * 65)
            print("  STAGE 2 SMOKE TEST  (2 folds)")
            print("=" * 65)
            rev_bundle, rev_oof = train_revenue_stage(
                df_feat=df_rev, log_col=log_rev, all_features=all_f,
                sample_weights=w, n_splits=2, verbose=True,
            )
            print(f"[OK] Stage 2 OOF non-NaN: {(~np.isnan(rev_oof)).sum():,}")
            print("[OK] Stage 2 smoke test passed.")

        if run_s3:
            print("\n" + "=" * 65)
            print("  STAGE 3 SMOKE TEST  (2 folds)")
            print("=" * 65)
            df_cogs, log_cogs, all_f_cogs, _, _ = engineer_features(raw_df, "COGS")
            min_len = min(len(df_cogs), len(rev_oof))
            rev_oof_al = np.full(len(df_cogs), np.nan)
            rev_oof_al[:min_len] = rev_oof[:min_len]
            w_cogs = get_sample_weights(df_cogs)
            cogs_bundle, cogs_oof = train_cogs_stage(
                df_feat=df_cogs, log_col=log_cogs, all_features=all_f_cogs,
                sample_weights=w_cogs, rev_oof_series=rev_oof_al,
                n_splits=2, verbose=True,
            )
            print(f"[OK] Stage 3 OOF non-NaN: {(~np.isnan(cogs_oof)).sum():,}")
            print("[OK] Stage 3 smoke test passed.")

    else:
        # ── Production pipeline ───────────────────────────────────────────────
        folds = 2 if "--fast" in sys.argv else N_SPLITS
        if "--fast" in sys.argv:
            print("[INFO] --fast mode: using 2 folds (debug). Use without flag for production.")
        run_pipeline(n_splits=folds, verbose=True)


