"""
Stacking Ensemble v5 — EDA-Driven Architecture
Level 0: Ridge(trend) + ElasticNet(seasonal) + LightGBM(all) + HistGBRT(all)
Level 1: NNLS meta → Revenue
Level 2: COGS cascade (all + Revenue_pred)
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import nnls
import lightgbm as lgb

DATA_PATH  = Path(__file__).parent / "processed_data.csv"
SUBMIT_IN  = Path(__file__).parent / "sample_submission.csv"
SUBMIT_OUT = Path(__file__).parent / "submission.csv"
RS, N_SPLITS = 42, 5
np.random.seed(RS)

LEAKY = {"payment_value","total_refund_amount","order_reviews",
         "customer_reviews","product_reviews","rating","Revenue","COGS"}
DROP  = {"traffic_direct","traffic_email_campaign","traffic_referral",
         "traffic_paid_search","traffic_organic_search","traffic_social_media",
         "avg_session_duration_sec","day_of_week","month"}

TET = {y:pd.Timestamp(d) for y,d in {
    2012:"2012-01-23",2013:"2013-02-10",2014:"2014-01-31",2015:"2015-02-19",
    2016:"2016-02-08",2017:"2017-01-28",2018:"2018-02-16",2019:"2019-02-05",
    2020:"2020-01-25",2021:"2021-02-12",2022:"2022-02-01",2023:"2023-01-22",
    2024:"2024-02-10",2025:"2025-01-29"}.items()}

def d2tet(dt):
    c=[TET[y] for y in [dt.year-1,dt.year,dt.year+1] if y in TET]
    return int(min([(dt-t).days for t in c],key=abs))

def load_data():
    df=pd.read_csv(DATA_PATH,parse_dates=["date"])
    df=df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df=df[df["date"]<="2022-12-31"].copy()
    # Annual YoY growth rate (trend proxy)
    ann=df.groupby(df["date"].dt.year)["Revenue"].mean()
    yoy=(ann/ann.shift(1)).fillna(1.0)
    df["_annual_yoy_growth"]=df["date"].dt.year.map(yoy.to_dict())
    print(f"[LOAD] {len(df)} rows {df['date'].min().date()}→{df['date'].max().date()}")
    return df, float(yoy.iloc[-1])  # return last known growth rate for future

def build_features(df, target, rev_pred_series=None):
    df=df.copy()
    lc=f"_log_{target}"
    df[lc]=np.log1p(df[target])
    d=df["date"]; doy=d.dt.dayofyear
    # ── Calendar ──
    df["_dow"]=d.dt.dayofweek; df["_dom"]=d.dt.day
    df["_month"]=d.dt.month;   df["_quarter"]=d.dt.quarter
    df["_year"]=d.dt.year;     df["_woy"]=d.dt.isocalendar().week.astype(int)
    df["_is_weekend"]=(d.dt.dayofweek>=5).astype(int)
    df["_is_month_end"]=d.dt.is_month_end.astype(int)
    df["_is_month_start"]=d.dt.is_month_start.astype(int)
    df["_month_sin"]=np.sin(2*np.pi*d.dt.month/12)
    df["_month_cos"]=np.cos(2*np.pi*d.dt.month/12)
    df["_dow_sin"]=np.sin(2*np.pi*d.dt.dayofweek/7)
    df["_dow_cos"]=np.cos(2*np.pi*d.dt.dayofweek/7)
    me=(d+pd.offsets.MonthEnd(0)-d).dt.days
    qe=(d+pd.offsets.QuarterEnd(0)-d).dt.days
    df["_days_to_month_end"]=me
    df["_is_last3_days"]=(me<=2).astype(int)
    df["_is_last7_days"]=(me<=6).astype(int)
    df["_days_to_qtr_end"]=qe
    df["_is_last3_qtr"]=(qe<=2).astype(int)
    df["_dom_ratio"]=d.dt.day/d.dt.days_in_month        # NEW: continuous month-end
    df["_is_peak_season"]=d.dt.month.isin([4,5,6]).astype(int)  # NEW: EDA T4-T6
    df["_is_low_season"]=d.dt.month.isin([11,12,1]).astype(int)  # NEW: EDA T11-T1
    df["_is_qtr_end_month"]=d.dt.month.isin([3,6,9,12]).astype(int)
    vn={"01-01","04-30","05-01","09-02","12-25",
        "01-25","01-26","01-27","01-28","01-29","01-30"}
    df["_is_vn_holiday"]=d.apply(lambda x:int(f"{x.month:02d}-{x.day:02d}" in vn))
    dt2t=d.apply(d2tet)
    df["_days_to_tet"]=dt2t
    df["_is_tet_week"]=(dt2t.abs()<=7).astype(int)
    df["_is_pre_tet2w"]=((dt2t>=-14)&(dt2t<0)).astype(int)
    df["_is_post_tet1w"]=((dt2t>0)&(dt2t<=7)).astype(int)
    df["_tet_proximity"]=np.exp(-0.5*(dt2t/7.0)**2)
    # ── Fourier ──
    for k in range(1,11):
        df[f"_fs{k}"]=np.sin(2*np.pi*k*doy/365.25)
        df[f"_fc{k}"]=np.cos(2*np.pi*k*doy/365.25)
    # ── Trend features (NEW from EDA) ──
    df["_trend_days"]=(d-pd.Timestamp("2017-01-01")).dt.days
    df["_trend_norm"]=df["_trend_days"]/1826
    # _annual_yoy_growth already in df from load_data
    # ── Target lags ──
    for lag in [7,14,30,60,364,365]:
        df[f"{lc}_lag{lag}"]=df[lc].shift(lag)
    for win in [7,14,30]:
        s=df[lc].shift(1)
        df[f"{lc}_rmean{win}"]=s.rolling(win).mean()
        df[f"{lc}_rstd{win}"]=s.rolling(win).std()
    # Smoothed lag-364 (±3 days average, reduces noise)
    df[f"{lc}_lag364_sm"]=df[lc].shift(364).rolling(7,center=True,min_periods=1).mean()
    # ── sessions YoY (NEW from EDA) ──
    if "sessions" in df.columns:
        ls=np.log1p(df["sessions"])
        df["_sessions_yoy"]=ls-ls.shift(364)
    if "unique_visitors" in df.columns:
        lv=np.log1p(df["unique_visitors"])
        df["_uv_yoy"]=lv-lv.shift(364)
    # ── Causal lags ──
    for col in ["sessions","unique_visitors","avg_fill_rate",
                "total_stockout_flags","total_stock_on_hand","order_id","customer_id"]:
        if col in df.columns:
            for lag in [7,14,30]:
                df[f"{col}_lag{lag}"]=df[col].shift(lag)
    # ── COGS cascade ──
    if rev_pred_series is not None:
        df["_rev_pred_log"]=np.log1p(rev_pred_series.values)
    df=df.dropna().reset_index(drop=True)
    excl=LEAKY|DROP|{"date","day_name",lc}|set(["Revenue","COGS"])|{"month_x","day_of_week"}
    num_types=(np.float64,np.int64,np.float32,np.int32,bool,np.bool_)
    all_f=[c for c in df.columns if c not in excl and df[c].dtype in num_types]
    # Feature subsets for specialists
    trend_f=[c for c in all_f if any(k in c for k in
              ["_trend","_year","_annual_yoy","_woy"])]
    seas_f =[c for c in all_f if any(k in c for k in
              ["_fs","_fc","_lag364","_lag365","_tet","_month","_quarter",
               "_dom_ratio","_peak","_low","_qtr_end","_dow_sin","_dow_cos",
               "_is_vn","_is_month"])]
    return df, lc, all_f, trend_f, seas_f

# ── Model helpers ──
def make_ridge():
    return Ridge(alpha=10.0,fit_intercept=True,max_iter=3000)

def make_en():
    return ElasticNetCV(alphas=[0.001,0.01,0.1,1,10],
                        l1_ratio=[0.1,0.3,0.5,0.7,0.9],
                        cv=3,max_iter=5000,random_state=RS,n_jobs=-1)

def make_lgb():
    return lgb.LGBMRegressor(n_estimators=1500,learning_rate=0.02,num_leaves=63,
        max_depth=7,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
        reg_alpha=0.05,reg_lambda=1.0,random_state=RS,n_jobs=-1,verbose=-1)

def make_hgbt():
    return HistGradientBoostingRegressor(max_iter=1000,learning_rate=0.04,
        max_leaf_nodes=63,max_depth=8,min_samples_leaf=20,l2_regularization=1.0,
        max_bins=255,early_stopping=True,validation_fraction=0.1,
        n_iter_no_change=50,random_state=RS)

class ScaledModel:
    """Wrapper: scale X, fit linear model."""
    def __init__(self,model):
        self.m=model; self.sc=StandardScaler()
    def fit(self,X,y):
        self.m.fit(self.sc.fit_transform(X),y); return self
    def predict(self,X):
        return self.m.predict(self.sc.transform(X))

class NNLSMeta:
    def __init__(self): self.w=self.b=None
    def fit(self,M,y):
        sol,_=nnls(np.column_stack([M,np.ones(len(y))]),y)
        self.w,self.b=sol[:-1],sol[-1]
        names=["Ridge","ElasticNet","LightGBM","HistGBRT"]
        tw=self.w.sum()+1e-9
        print("  NNLS weights:")
        for n,v in zip(names,self.w):
            print(f"    {n:<12}= {v:.4f} ({100*v/tw:.1f}%)")
        print(f"    intercept   = {self.b:,.0f}")
        return self
    def predict(self,M): return M@self.w+self.b

# ── OOF ──
def generate_oof(df_feat, lc, all_f, trend_f, seas_f):
    X_all=df_feat[all_f].values.astype(np.float32)
    X_tr =df_feat[trend_f].values.astype(np.float32)
    X_se =df_feat[[c for c in seas_f if c in df_feat.columns]].values.astype(np.float32)
    y_log=df_feat[lc].values; y_orig=np.expm1(y_log)
    tscv=TimeSeriesSplit(n_splits=N_SPLITS)
    or_=np.zeros(len(X_all)); oe=np.zeros(len(X_all))
    ol=np.zeros(len(X_all)); oh=np.zeros(len(X_all))

    for fold,(tr,val) in enumerate(tscv.split(X_all)):
        print(f"  [Fold {fold+1}/{N_SPLITS}] train={len(tr)} val={len(val)}")
        ytr,yv=y_log[tr],y_log[val]
        # Ridge on trend features
        rm=ScaledModel(make_ridge())
        rm.fit(X_tr[tr],ytr); or_[val]=np.expm1(rm.predict(X_tr[val]))
        # ElasticNet on seasonal features
        em=ScaledModel(make_en())
        em.fit(X_se[tr],ytr); oe[val]=np.expm1(em.predict(X_se[val]))
        # LightGBM on all features
        lm=make_lgb()
        lm.fit(X_all[tr],ytr,eval_set=[(X_all[val],yv)],
               callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(-1)])
        ol[val]=np.expm1(lm.predict(X_all[val]))
        # HistGBRT on all features
        hm=make_hgbt(); hm.fit(X_all[tr],ytr)
        oh[val]=np.expm1(hm.predict(X_all[val]))

    return np.column_stack([or_,oe,ol,oh]),y_orig

def fit_final(df_feat, lc, all_f, trend_f, seas_f):
    X_all=df_feat[all_f].values.astype(np.float32)
    X_tr =df_feat[trend_f].values.astype(np.float32)
    sea_cols=[c for c in seas_f if c in df_feat.columns]
    X_se =df_feat[sea_cols].values.astype(np.float32)
    y=df_feat[lc].values
    rf=ScaledModel(make_ridge()); rf.fit(X_tr,y)
    ef=ScaledModel(make_en());   ef.fit(X_se,y)
    lf=make_lgb(); lf.fit(X_all,y,callbacks=[lgb.log_evaluation(-1)])
    hf=make_hgbt(); hf.fit(X_all,y)
    return rf,ef,ef.m,lf,hf,sea_cols

def eval_report(yo,M,ym,target):
    print(f"\n{'='*65}\n  OOF — {target}\n{'='*65}")
    for nm,p in zip(["Ridge","ElasticNet","LightGBM","HistGBRT","META"],
                    list(M.T)+[ym]):
        p=np.maximum(p,0)
        print(f"  {nm:<12} MAE={mean_absolute_error(yo,p):>13,.0f} "
              f"RMSE={np.sqrt(mean_squared_error(yo,p)):>13,.0f} "
              f"R²={r2_score(yo,p):.4f}")
    print(f"{'='*65}\n")

# ── Future Builder ──
def build_future(raw, future_dates, target, lc, rf, ef, lf, hf,
                 all_f, trend_f, sea_cols, meta,
                 last_yoy_growth, rev_pred_arr=None):
    hist=raw.copy(); hist[lc]=np.log1p(hist[target])
    vn={"01-01","04-30","05-01","09-02","12-25",
        "01-25","01-26","01-27","01-28","01-29","01-30"}
    preds=[]
    for i,fdate in enumerate(pd.to_datetime(future_dates)):
        n=len(hist); lh=hist[lc]; row={"date":fdate}
        doy=fdate.dayofyear
        me=(fdate+pd.offsets.MonthEnd(0)-fdate).days
        qe=(fdate+pd.offsets.QuarterEnd(0)-fdate).days
        dt=d2tet(fdate)
        row.update({
            "_dow":fdate.dayofweek,"_dom":fdate.day,"_month":fdate.month,
            "_quarter":(fdate.month-1)//3+1,"_year":fdate.year,
            "_woy":fdate.isocalendar()[1],
            "_is_weekend":int(fdate.dayofweek>=5),
            "_is_month_end":int(fdate.day==pd.Timestamp(fdate).days_in_month),
            "_is_month_start":int(fdate.day==1),
            "_month_sin":np.sin(2*np.pi*fdate.month/12),
            "_month_cos":np.cos(2*np.pi*fdate.month/12),
            "_dow_sin":np.sin(2*np.pi*fdate.dayofweek/7),
            "_dow_cos":np.cos(2*np.pi*fdate.dayofweek/7),
            "_days_to_month_end":me,"_is_last3_days":int(me<=2),
            "_is_last7_days":int(me<=6),"_days_to_qtr_end":qe,
            "_is_last3_qtr":int(qe<=2),
            "_dom_ratio":fdate.day/pd.Timestamp(fdate).days_in_month,
            "_is_peak_season":int(fdate.month in [4,5,6]),
            "_is_low_season":int(fdate.month in [11,12,1]),
            "_is_qtr_end_month":int(fdate.month in [3,6,9,12]),
            "_is_vn_holiday":int(f"{fdate.month:02d}-{fdate.day:02d}" in vn),
            "_days_to_tet":dt,"_is_tet_week":int(abs(dt)<=7),
            "_is_pre_tet2w":int(-14<=dt<0),"_is_post_tet1w":int(0<dt<=7),
            "_tet_proximity":float(np.exp(-0.5*(dt/7)**2)),
            "_trend_days":(fdate-pd.Timestamp("2017-01-01")).days,
            "_annual_yoy_growth":last_yoy_growth,
        })
        row["_trend_norm"]=row["_trend_days"]/1826
        for k in range(1,11):
            row[f"_fs{k}"]=np.sin(2*np.pi*k*doy/365.25)
            row[f"_fc{k}"]=np.cos(2*np.pi*k*doy/365.25)
        for lag in [7,14,30,60,364,365]:
            idx=n-lag
            row[f"{lc}_lag{lag}"]=(float(lh.iloc[idx]) if idx>=0 else 0.0)
        for win in [7,14,30]:
            row[f"{lc}_rmean{win}"]=float(lh.iloc[-win:].mean())
            row[f"{lc}_rstd{win}"]=float(lh.iloc[-win:].std())
        # Smoothed lag-364
        idx364=n-364
        if idx364>=3:
            row[f"{lc}_lag364_sm"]=float(lh.iloc[max(0,idx364-3):idx364+4].mean())
        else:
            row[f"{lc}_lag364_sm"]=row.get(f"{lc}_lag364",0.0)
        # sessions YoY
        if "sessions" in hist.columns:
            ls=np.log1p(hist["sessions"])
            row["_sessions_yoy"]=(float(ls.iloc[-1]-ls.iloc[n-364]) if n>364 else 0.0)
        if "unique_visitors" in hist.columns:
            lv=np.log1p(hist["unique_visitors"])
            row["_uv_yoy"]=(float(lv.iloc[-1]-lv.iloc[n-364]) if n>364 else 0.0)
        # Causal lags
        for col in ["sessions","unique_visitors","avg_fill_rate",
                    "total_stockout_flags","total_stock_on_hand","order_id","customer_id"]:
            if col in hist.columns:
                for lag in [7,14,30]:
                    idx=n-lag
                    row[f"{col}_lag{lag}"]=(float(hist[col].iloc[idx]) if idx>=0 else 0.0)
        # COGS cascade
        if rev_pred_arr is not None:
            row["_rev_pred_log"]=float(np.log1p(rev_pred_arr[i]))
        # Fill missing features
        for c in all_f:
            if c not in row:
                row[c]=(float(hist[c].iloc[-1]) if c in hist.columns else 0.0)

        # Predict
        def get(cols):
            return np.array([[row.get(c,0.0) for c in cols]],dtype=np.float32)
        tr_arr=get(trend_f); se_arr=get(sea_cols); all_arr=get(all_f)
        p_r=np.expm1(float(rf.predict(tr_arr)[0]))
        p_e=np.expm1(float(ef.predict(se_arr)[0]))
        p_l=np.expm1(float(lf.predict(all_arr)[0]))
        p_h=np.expm1(float(hf.predict(all_arr)[0]))
        pred=float(np.maximum(meta.predict(np.array([[p_r,p_e,p_l,p_h]])),0))
        preds.append(pred)

        nr={c:(hist[c].iloc[-1] if c in hist.columns else 0) for c in hist.columns}
        nr["date"]=fdate; nr[target]=pred; nr[lc]=float(np.log1p(pred))
        hist=pd.concat([hist,pd.DataFrame([nr])],ignore_index=True)

    return np.array(preds)

# ── Pipeline ──
def run_pipeline():
    print("\n[1] Loading data...")
    raw, last_yoy = load_data()
    sub=pd.read_csv(SUBMIT_IN,parse_dates=["Date"])
    sub.columns=sub.columns.str.lower()
    future_dates=pd.to_datetime(sub["date"].values)
    results={}; rev_pred_future=None

    for target in ["Revenue","COGS"]:
        print(f"\n{'#'*60}\n  TARGET: {target}\n{'#'*60}")

        # Rev cascade: pass Revenue predictions as feature for COGS
        rev_series=None
        if target=="COGS" and "Revenue" in results:
            rev_series=pd.Series(
                np.log1p(raw["Revenue"])-np.log1p(raw["Revenue"]).shift(364),
                index=raw.index)

        print("[2] Building features...")
        df_feat,lc,all_f,trend_f,seas_f=build_features(raw,target,rev_series)
        sea_cols_=[c for c in seas_f if c in df_feat.columns]
        print(f"  All={len(all_f)} Trend={len(trend_f)} Seasonal={len(sea_cols_)}")

        print("[3] OOF predictions...")
        M,yo=generate_oof(df_feat,lc,all_f,trend_f,sea_cols_)

        print("[4] Meta-learner (NNLS)...")
        meta=NNLSMeta(); meta.fit(M,yo)
        ym=np.maximum(meta.predict(M),0)

        print("[5] OOF evaluation..."); eval_report(yo,M,ym,target)

        print("[6] Final training...")
        rf,ef,_,lf,hf,sea_cols_fit=fit_final(df_feat,lc,all_f,trend_f,sea_cols_)

        print(f"[7] Building future ({len(future_dates)} days)...")
        rev_future_for_cogs=rev_pred_future if target=="COGS" else None
        pred=build_future(raw,future_dates,target,lc,rf,ef,lf,hf,
                          all_f,trend_f,sea_cols_fit,meta,
                          last_yoy,rev_pred_arr=rev_future_for_cogs)
        results[target]=pred
        if target=="Revenue": rev_pred_future=pred
        print(f"  {target} first 5: {pred[:5].round(0)}")

    print("\n[8] Saving submission.csv...")
    out=pd.DataFrame({"Date":sub["date"].values,
                      "Revenue":np.round(results["Revenue"],2),
                      "COGS":np.round(results["COGS"],2)})
    out.to_csv(SUBMIT_OUT,index=False)
    print(f"  Saved → {SUBMIT_OUT}  ({len(out)} rows)")
    print(out.head(10).to_string(index=False))

if __name__=="__main__":
    run_pipeline()
