"""
hybrid.py v4 — ONE-SHOT future prediction using ONLY actual historical lags.
No iterative update. All lag features for future dates use actual 2022 data.
"""
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

BASE        = Path(__file__).parent
DATA_PATH   = BASE / "processed_data.csv"
SAMPLE_PATH = BASE / "sample_submission.csv"
OUT_PATH    = BASE / "submission.csv"

SEED    = 42
np.random.seed(SEED)
# Training lags (short lags help train but we proxy them during future prediction)
TRAIN_LAGS = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365]
WINDOWS    = [7, 30, 90]
WEIGHT_HALFLIFE_YR = 2.5

EXOG_COLS = [
    "sessions", "unique_visitors", "page_views", "bounce_rate",
    "avg_session_duration_sec", "total_stock_on_hand", "avg_fill_rate",
    "total_stockout_flags", "total_overstock_flags", "avg_sell_through_rate",
    "order_id", "customer_id", "total_shipping_fee", "days_since_snapshot",
]

TET = {y: pd.Timestamp(d) for y, d in {
    2012:"2012-01-23", 2013:"2013-02-10", 2014:"2014-01-31", 2015:"2015-02-19",
    2016:"2016-02-08", 2017:"2017-01-28", 2018:"2018-02-16", 2019:"2019-02-05",
    2020:"2020-01-25", 2021:"2021-02-12", 2022:"2022-02-01", 2023:"2023-01-22",
    2024:"2024-02-10", 2025:"2025-01-29"}.items()}
VN_HOLIDAYS = {"01-01", "04-30", "05-01", "09-02", "12-25"}

def days_to_tet(dt):
    cands = [TET[y] for y in [dt.year-1, dt.year, dt.year+1] if y in TET]
    return int(min([(dt-t).days for t in cands], key=abs)) if cands else 0

def make_sw(dates, hl_yr=WEIGHT_HALFLIFE_YR):
    days_ago = (dates.max() - dates).dt.days.values.astype(float)
    return np.exp(-days_ago * np.log(2) / (hl_yr * 365.25))

# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df = df[df["date"] <= "2022-12-31"].copy()
    for c in EXOG_COLS:
        if c not in df.columns: df[c] = 0.0
    for c in ["total_stock_on_hand","avg_fill_rate","total_stockout_flags",
              "total_overstock_flags","avg_sell_through_rate","days_since_snapshot"]:
        if c in df.columns: df[c] = df[c].fillna(method="ffill").fillna(0)
    for c in ["sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec"]:
        if c in df.columns: df[c] = df[c].fillna(0)
    print(f"[LOAD] {len(df)} rows  {df['date'].min().date()} → {df['date'].max().date()}")
    return df

# ── Calendar features helper ──────────────────────────────────────────────────
def _cal(df):
    d = df["date"]; doy = d.dt.dayofyear
    df["dow"]  = d.dt.dayofweek; df["dom"] = d.dt.day
    df["month"]= d.dt.month;     df["quarter"] = d.dt.quarter
    df["year"] = d.dt.year;      df["woy"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"]    = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_end"]  = d.dt.is_month_end.astype(int)
    df["is_month_start"]= d.dt.is_month_start.astype(int)
    df["is_vn_holiday"] = d.apply(lambda x: int(f"{x.month:02d}-{x.day:02d}" in VN_HOLIDAYS))
    me = (d + pd.offsets.MonthEnd(0) - d).dt.days
    df["days_to_month_end"] = me
    df["days_to_qtr_end"]   = (d + pd.offsets.QuarterEnd(0) - d).dt.days
    df["is_last7_days"]     = (me <= 6).astype(int)
    dt2t = d.apply(days_to_tet)
    df["days_to_tet"]   = dt2t
    df["is_tet_week"]   = (dt2t.abs() <= 7).astype(int)
    df["is_pre_tet2w"]  = ((dt2t >= -14) & (dt2t < 0)).astype(int)
    df["tet_proximity"] = np.exp(-0.5 * (dt2t / 7.0) ** 2)
    for k in [1, 2, 3, 4, 6]:
        df[f"fsin{k}"] = np.sin(2*np.pi*k*doy/365.25)
        df[f"fcos{k}"] = np.cos(2*np.pi*k*doy/365.25)
    df["dow_sin"]   = np.sin(2*np.pi*d.dt.dayofweek/7)
    df["dow_cos"]   = np.cos(2*np.pi*d.dt.dayofweek/7)
    df["month_sin"] = np.sin(2*np.pi*d.dt.month/12)
    df["month_cos"] = np.cos(2*np.pi*d.dt.month/12)
    return df

# ── Build training features ───────────────────────────────────────────────────
def build_features(df, target):
    df = df.copy()
    lc = f"log_{target}"; df[lc] = np.log1p(df[target])
    # Revenue lags for COGS
    if target == "COGS" and "Revenue" in df.columns:
        lr = np.log1p(df["Revenue"])
        for lag in [7,14,30,90,365]: df[f"log_Revenue_lag{lag}"] = lr.shift(lag)
        for w in WINDOWS: df[f"log_Revenue_rmean{w}"] = lr.shift(1).rolling(w).mean()
    df = _cal(df)
    s = df[lc]
    for lag in TRAIN_LAGS: df[f"{lc}_lag{lag}"] = s.shift(lag)
    for w in WINDOWS:
        df[f"{lc}_rmean{w}"] = s.shift(1).rolling(w).mean()
        df[f"{lc}_rstd{w}"]  = s.shift(1).rolling(w).std()
        df[f"{lc}_rmin{w}"]  = s.shift(1).rolling(w).min()
        df[f"{lc}_rmax{w}"]  = s.shift(1).rolling(w).max()
    df[f"{lc}_yoy"] = s / (s.shift(365) + 1e-9)
    for col in EXOG_COLS:
        if col in df.columns:
            for lag in [7,14,30]: df[f"{col}_lag{lag}"] = df[col].shift(lag)
    if "days_since_snapshot" in df.columns:
        df["snapshot_weight"] = np.exp(-df["days_since_snapshot"] / 30.0)
    else:
        df["snapshot_weight"] = 1.0
    df[f"{target}_positive"] = (df[target] > 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    LEAKY = {"Revenue","COGS","payment_value","total_refund_amount",
             "order_reviews","customer_reviews","product_reviews","rating",
             "day_name","date", lc, f"{target}_positive"}
    feat_cols = [c for c in df.columns if c not in LEAKY
                 and df[c].dtype in (np.float64,np.int64,np.float32,np.int32,bool,np.bool_)]
    return df, lc, feat_cols

# ── ONE-SHOT future feature builder ──────────────────────────────────────────
def build_future_features(raw, future_dates, target, feat_cols, rev_preds=None):
    """
    Build feature matrix for ALL future dates AT ONCE using ONLY actual history.
    For lag_k where source date is in future:
        → use value from source_date - 365 (same day last year) instead.
    This eliminates cascading errors completely.
    """
    lc = f"log_{target}"
    # Index history by date for O(1) lookups
    hist = raw.copy()
    hist[lc] = np.log1p(hist[target])
    if target == "COGS":
        hist["log_Revenue"] = np.log1p(hist["Revenue"])
    hist = hist.set_index("date")
    lc_idx = hist[lc]               # pd.Series indexed by date

    future_dates = pd.to_datetime(future_dates)
    min_future   = future_dates[0]

    def get_val(src_date, series, fallback_days=365):
        """Get series value. If src_date is in future, use -365 fallback."""
        if src_date in series.index:
            return float(series[src_date])
        if src_date >= min_future:
            alt = src_date - pd.Timedelta(days=fallback_days)
            if alt in series.index: return float(series[alt])
        return 0.0

    rows = []
    for fdate in future_dates:
        r = {}
        # Calendar
        doy = fdate.dayofyear
        r["dow"]  = fdate.dayofweek; r["dom"] = fdate.day
        r["month"]= fdate.month;     r["quarter"] = (fdate.month-1)//3+1
        r["year"] = fdate.year;      r["woy"] = fdate.isocalendar()[1]
        r["is_weekend"]    = int(fdate.dayofweek >= 5)
        r["is_month_end"]  = int(fdate.day == pd.Timestamp(fdate).days_in_month)
        r["is_month_start"]= int(fdate.day == 1)
        mdt = f"{fdate.month:02d}-{fdate.day:02d}"
        r["is_vn_holiday"] = int(mdt in VN_HOLIDAYS)
        me = (fdate + pd.offsets.MonthEnd(0) - fdate).days
        r["days_to_month_end"] = me
        r["days_to_qtr_end"]   = (fdate + pd.offsets.QuarterEnd(0) - fdate).days
        r["is_last7_days"]     = int(me <= 6)
        dt2t = days_to_tet(fdate)
        r["days_to_tet"]   = dt2t; r["is_tet_week"]   = int(abs(dt2t) <= 7)
        r["is_pre_tet2w"]  = int(-14 <= dt2t < 0)
        r["tet_proximity"] = float(np.exp(-0.5*(dt2t/7)**2))
        for k in [1,2,3,4,6]:
            r[f"fsin{k}"] = np.sin(2*np.pi*k*doy/365.25)
            r[f"fcos{k}"] = np.cos(2*np.pi*k*doy/365.25)
        r["dow_sin"]   = np.sin(2*np.pi*fdate.dayofweek/7)
        r["dow_cos"]   = np.cos(2*np.pi*fdate.dayofweek/7)
        r["month_sin"] = np.sin(2*np.pi*fdate.month/12)
        r["month_cos"] = np.cos(2*np.pi*fdate.month/12)

        # Lag features — always from actual history (fallback to -365 days)
        for lag in TRAIN_LAGS:
            col = f"{lc}_lag{lag}"
            if col in feat_cols:
                r[col] = get_val(fdate - pd.Timedelta(days=lag), lc_idx)

        # Rolling window: use actual history window, fallback to -365 for future slots
        for w in WINDOWS:
            vals = []
            for k in range(1, w+1):
                src = fdate - pd.Timedelta(days=k)
                vals.append(get_val(src, lc_idx))
            r[f"{lc}_rmean{w}"] = float(np.mean(vals))
            r[f"{lc}_rstd{w}"]  = float(np.std(vals))
            r[f"{lc}_rmin{w}"]  = float(np.min(vals))
            r[f"{lc}_rmax{w}"]  = float(np.max(vals))

        r[f"{lc}_yoy"] = get_val(fdate-pd.Timedelta(days=365), lc_idx) / \
                         (get_val(fdate-pd.Timedelta(days=730), lc_idx) + 1e-9)

        # Revenue lags for COGS
        if target == "COGS":
            rev_lc = hist["log_Revenue"] if "log_Revenue" in hist.columns else lc_idx*0
            for lag in [7,14,30,90,365]:
                col = f"log_Revenue_lag{lag}"
                if col in feat_cols:
                    r[col] = get_val(fdate - pd.Timedelta(days=lag), rev_lc)
            for w in WINDOWS:
                col = f"log_Revenue_rmean{w}"
                if col in feat_cols:
                    vals = [get_val(fdate-pd.Timedelta(days=k), rev_lc) for k in range(1,w+1)]
                    r[col] = float(np.mean(vals))

        # Exog lags
        for col in EXOG_COLS:
            if col in hist.columns:
                for lag in [7,14,30]:
                    cname = f"{col}_lag{lag}"
                    if cname in feat_cols:
                        r[cname] = get_val(fdate-pd.Timedelta(days=lag), hist[col])

        r["snapshot_weight"] = 1.0
        # Fill remaining from feature list
        for c in feat_cols:
            if c not in r: r[c] = 0.0
        rows.append(r)

    return np.array([[row.get(c, 0.0) for c in feat_cols] for row in rows], dtype=np.float32)

# ── Model params ──────────────────────────────────────────────────────────────
def lgb_p():
    return dict(n_estimators=3000,learning_rate=0.01,num_leaves=63,max_depth=7,
                min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
                reg_alpha=0.05,reg_lambda=1.0,random_state=SEED,n_jobs=-1,verbose=-1)
def xgb_p():
    return dict(n_estimators=2000,learning_rate=0.015,max_depth=6,min_child_weight=20,
                subsample=0.8,colsample_bytree=0.8,reg_alpha=0.05,reg_lambda=1.0,
                random_state=SEED,tree_method="hist",n_jobs=-1,early_stopping_rounds=150)
def rf_p():
    return dict(n_estimators=300,max_depth=12,min_samples_leaf=20,
                max_features=0.5,random_state=SEED,n_jobs=-1)

# ── LGB Meta ──────────────────────────────────────────────────────────────────
def fit_meta_lgb(M_log, y_orig):
    tscv = TimeSeriesSplit(n_splits=3)
    meta_oof = np.zeros(len(y_orig))
    for tr, val in tscv.split(M_log):
        mm = lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,
                               num_leaves=15,random_state=SEED,verbose=-1)
        mm.fit(M_log[tr], np.log1p(y_orig[tr]))
        meta_oof[val] = np.expm1(mm.predict(M_log[val]))
    meta_final = lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,
                                    num_leaves=15,random_state=SEED,verbose=-1)
    meta_final.fit(M_log, np.log1p(y_orig))
    return meta_final, np.maximum(meta_oof, 0)

# ── OOF generation ────────────────────────────────────────────────────────────
def generate_oof(df_feat, feat_cols, lc, target):
    X     = df_feat[feat_cols].values.astype(np.float32)
    y_log = df_feat[lc].values
    y_orig= np.expm1(y_log)
    sw    = make_sw(df_feat["date"]).astype(np.float32)
    n     = len(X)
    tscv  = TimeSeriesSplit(n_splits=5)
    ol=np.zeros(n); ox=np.zeros(n); orf=np.zeros(n); osn=np.zeros(n)
    fitted = {"lgb":[],"xgb":[],"rf":[]}
    for fold,(tr,val) in enumerate(tscv.split(X)):
        print(f"  [Fold {fold+1}/5] train={len(tr)} val={len(val)}")
        Xtr,Xv = X[tr],X[val]; ytr,yv = y_log[tr],y_log[val]; sw_tr=sw[tr]
        lm = lgb.LGBMRegressor(**lgb_p())
        lm.fit(Xtr,ytr,sample_weight=sw_tr,eval_set=[(Xv,yv)],
               callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(-1)])
        ol[val] = np.expm1(lm.predict(Xv)); fitted["lgb"].append(lm)
        xp = {k:v for k,v in xgb_p().items() if k!="early_stopping_rounds"}
        xm = xgb.XGBRegressor(**xp, early_stopping_rounds=150)
        xm.fit(Xtr,ytr,sample_weight=sw_tr,eval_set=[(Xv,yv)],verbose=False)
        ox[val] = np.expm1(xm.predict(Xv)); fitted["xgb"].append(xm)
        rm = RandomForestRegressor(**rf_p())
        rm.fit(Xtr,ytr,sample_weight=sw_tr)
        orf[val] = np.expm1(rm.predict(Xv)); fitted["rf"].append(rm)
        osn[val] = y_orig[np.maximum(val-365, 0)]
    M = np.column_stack([ol,ox,orf,osn])
    return M, y_orig, fitted

# ── Final models ──────────────────────────────────────────────────────────────
def fit_final(df_feat, feat_cols, lc, target):
    X  = df_feat[feat_cols].values.astype(np.float32)
    y  = df_feat[lc].values
    sw = make_sw(df_feat["date"]).astype(np.float32)
    lf = lgb.LGBMRegressor(**lgb_p())
    lf.fit(X,y,sample_weight=sw,callbacks=[lgb.log_evaluation(-1)])
    xp = {k:v for k,v in xgb_p().items() if k!="early_stopping_rounds"}
    xf = xgb.XGBRegressor(**xp); xf.fit(X,y,sample_weight=sw,verbose=False)
    rf = RandomForestRegressor(**rf_p()); rf.fit(X,y,sample_weight=sw)
    return lf, xf, rf

# ── Eval ──────────────────────────────────────────────────────────────────────
def eval_report(y_true, M, meta_oof, target):
    names = ["LightGBM","XGBoost","RandomForest","SeasonalNaive","META(LGB)"]
    print(f"\n{'='*70}\n  OOF Evaluation — {target}\n{'='*70}")
    for nm,p in zip(names, list(M.T)+[meta_oof]):
        p = np.maximum(p,0)
        print(f"  {nm:<18} MAE={mean_absolute_error(y_true,p):>14,.0f}"
              f"  RMSE={np.sqrt(mean_squared_error(y_true,p)):>14,.0f}"
              f"  R²={r2_score(y_true,p):.4f}")
    print("="*70)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n"+"="*70+"\n  HYBRID ENSEMBLE v4 — ONE-SHOT future prediction\n"+"="*70)
    raw = load_data()
    sample = pd.read_csv(SAMPLE_PATH); sample.columns = sample.columns.str.lower()
    future_dates = pd.to_datetime(sample["date"].values)
    print(f"[TARGET] {future_dates[0].date()} → {future_dates[-1].date()}  ({len(future_dates)} days)")

    results = {}; feat_cols_store = {}; final_store = {}

    for target in ["Revenue","COGS"]:
        print(f"\n{'#'*70}\n  TARGET: {target}\n{'#'*70}")
        df_feat, lc, feat_cols = build_features(raw, target)
        feat_cols_store[target] = feat_cols
        print(f"  Features: {len(feat_cols)}")

        val_mask = df_feat["date"].dt.year == 2022
        df_tr = df_feat[~val_mask].reset_index(drop=True)
        df_va = df_feat[val_mask].reset_index(drop=True)
        print(f"  Train: {len(df_tr)}  Val(2022): {len(df_va)}")

        print("[4] OOF...")
        M_oof, y_oof, fitted = generate_oof(df_tr, feat_cols, lc, target)
        M_log = np.log1p(np.maximum(M_oof, 0))

        print("[5] LGB Meta...")
        meta_model, meta_oof = fit_meta_lgb(M_log, y_oof)
        eval_report(y_oof, M_oof, meta_oof, target)

        # Val 2022 (fold-5 models)
        Xv = df_va[feat_cols].fillna(0).values.astype(np.float32)
        lm_l = fitted["lgb"][-1]; xm_l = fitted["xgb"][-1]; rm_l = fitted["rf"][-1]
        pl_v = np.expm1(lm_l.predict(Xv)); px_v = np.expm1(xm_l.predict(Xv))
        pr_v = np.expm1(rm_l.predict(Xv))
        lc_l = f"log_{target}_lag365"
        sn_v = df_va[lc_l].apply(np.expm1).values if lc_l in df_va.columns else pl_v
        stk_v = np.log1p(np.maximum(np.column_stack([pl_v,px_v,pr_v,sn_v]),0))
        meta_v = np.maximum(np.expm1(meta_model.predict(stk_v)),0)
        y_val  = np.expm1(df_va[lc].values)
        print(f"\n  [VAL 2022] R²={r2_score(y_val,meta_v):.4f}"
              f"  RMSE={np.sqrt(mean_squared_error(y_val,meta_v)):,.0f}"
              f"  MAE={mean_absolute_error(y_val,meta_v):,.0f}")

        print("[8] Final fit on full data...")
        lf, xf, rf_f = fit_final(df_feat, feat_cols, lc, target)
        final_store[target] = (lf, xf, rf_f, meta_model)

    # ONE-SHOT future prediction
    print(f"\n[9] ONE-SHOT future prediction ({len(future_dates)} days)...")
    for target in ["Revenue","COGS"]:
        lf, xf, rf_f, meta_model = final_store[target]
        feat_cols = feat_cols_store[target]
        print(f"  Building features for {target}...")
        X_fut = build_future_features(raw, future_dates, target, feat_cols,
                                      rev_preds=results.get("Revenue"))
        pl = np.expm1(lf.predict(X_fut))
        px = np.expm1(xf.predict(X_fut))
        pr = np.expm1(rf_f.predict(X_fut))
        # Seasonal naive from actual history (lag_365)
        raw_idx = raw.set_index("date")[target]
        ps = np.array([
            float(raw_idx.get(fd - pd.Timedelta(days=365), pl[i]))
            for i, fd in enumerate(future_dates)
        ])
        stk = np.log1p(np.maximum(np.column_stack([pl, px, pr, ps]), 0))
        preds = np.maximum(np.expm1(meta_model.predict(stk)), 0)
        results[target] = preds
        print(f"  {target}: {preds[:7].round(0)}")

    out = pd.DataFrame({
        "date":    future_dates,
        "revenue": np.round(results["Revenue"], 2),
        "cogs":    np.round(results["COGS"],    2),
    })
    out.to_csv(OUT_PATH, index=False)
    print(f"\n[DONE] {OUT_PATH}  ({len(out)} rows)")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
