# ============================================================
# EDA & FEATURE SELECTION — VinUni Datathon 2026
# Vai trò: Senior Data Scientist & Data Architect
# Mục tiêu: Phân tích chất lượng dữ liệu, chọn lọc đặc trưng
#           để cải thiện dự báo Revenue & COGS (2023-2024)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Cấu hình đồ thị
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
sns.set_theme(style="darkgrid", palette="muted")

DATA_PATH = "processed_data.csv"

# ─────────────────────────────────────────────────────────────
# TẢI DỮ LIỆU
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("BƯỚC 0: TẢI DỮ LIỆU")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"  Số dòng      : {len(df):,}")
print(f"  Số cột       : {len(df.columns)}")
print(f"  Thời gian    : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Targets      : Revenue min={df['Revenue'].min():,.0f}  max={df['Revenue'].max():,.0f}")

# ─────────────────────────────────────────────────────────────
# BƯỚC 1A: KIỂM TRA CHẤT LƯỢNG DỮ LIỆU
# Tính % giá trị zero/NaN theo từng nhóm đặc trưng
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 1A: KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU")
print("=" * 60)

# Định nghĩa các nhóm đặc trưng theo nguồn gốc dữ liệu
FEATURE_GROUPS = {
    "Web Traffic"  : [c for c in df.columns if any(k in c for k in
                       ["sessions","visitors","page_views","bounce",
                        "session_duration","traffic_","is_web"])],
    "Inventory"    : [c for c in df.columns if any(k in c for k in
                       ["stock","fill_rate","stockout","overstock",
                        "sell_through","units_","days_of_supply",
                        "snapshot","_gap"])],
    "Promotions"   : [c for c in df.columns if any(k in c for k in
                       ["promo","discount","installments"])],
    "Orders/Trans" : [c for c in df.columns if any(k in c for k in
                       ["order_","payment_","order_source","device_",
                        "order_status","shipped","delivered","quantity",
                        "shipping_fee","refund"])],
    "Reviews"      : [c for c in df.columns if any(k in c for k in
                       ["review","rating"])],
    "Customer"     : [c for c in df.columns if any(k in c for k in
                       ["customer_id","customer_id"])],
    "Category/Seg" : [c for c in df.columns if any(k in c for k in
                       ["category_","segment_"])],
}

# Tính % zero + NaN cho từng nhóm
print(f"\n{'Nhóm':<20} {'Số cột':>7} {'% Zero':>9} {'% NaN':>9} {'% Vô ích':>10}")
print("-" * 60)
group_stats = {}
for grp, cols in FEATURE_GROUPS.items():
    cols = [c for c in cols if c in df.columns]
    if not cols:
        continue
    pct_zero = (df[cols] == 0).mean().mean() * 100
    pct_nan  = df[cols].isna().mean().mean() * 100
    group_stats[grp] = {"cols": cols, "pct_zero": pct_zero, "pct_nan": pct_nan}
    print(f"  {grp:<18} {len(cols):>7} {pct_zero:>8.1f}% {pct_nan:>8.1f}% "
          f"{pct_zero+pct_nan:>9.1f}%")

# ─────────────────────────────────────────────────────────────
# BƯỚC 1B: HEATMAP CHẤT LƯỢNG DỮ LIỆU THEO THỜI GIAN
# Trực quan hóa khi nào từng nhóm bắt đầu có tín hiệu thật
# ─────────────────────────────────────────────────────────────
print("\n[Vẽ] Data Quality Heatmap theo năm-tháng...")

# Resample theo tháng, tính tỉ lệ NON-zero (tín hiệu thật)
df_monthly = df.set_index("date").resample("ME")

heatmap_data = {}
for grp, info in group_stats.items():
    cols = [c for c in info["cols"] if c in df.columns]
    if not cols:
        continue
    # % dòng có ít nhất 1 giá trị khác 0 trong nhóm
    nonzero = (df[cols].abs().sum(axis=1) > 0).astype(float)
    nonzero.index = df["date"]
    monthly = nonzero.resample("ME").mean() * 100
    heatmap_data[grp] = monthly

heatmap_df = pd.DataFrame(heatmap_data).T
heatmap_df.columns = [str(c)[:7] for c in heatmap_df.columns]

fig, ax = plt.subplots(figsize=(18, 4))
sns.heatmap(heatmap_df, ax=ax, cmap="YlOrRd",
            vmin=0, vmax=100, linewidths=0.3,
            cbar_kws={"label": "% Ngày có dữ liệu thật (non-zero)"})
ax.set_title("DATA QUALITY HEATMAP — % Ngày có tín hiệu thật theo tháng",
             fontweight="bold", pad=12)
ax.set_xlabel("Năm-Tháng")
ax.set_ylabel("Nhóm đặc trưng")
# Chỉ hiện nhãn mỗi 6 tháng
n_cols = len(heatmap_df.columns)
step = max(1, n_cols // 22)
ax.set_xticks(range(0, n_cols, step))
ax.set_xticklabels([heatmap_df.columns[i] for i in range(0, n_cols, step)],
                   rotation=45, ha="right", fontsize=7)
plt.tight_layout()
plt.savefig("plot_1b_quality_heatmap.png", bbox_inches="tight")
plt.show()
print("  → Lưu: plot_1b_quality_heatmap.png")

# ─────────────────────────────────────────────────────────────
# BƯỚC 2A: PHÂN PHỐI REVENUE & COGS
# Kiểm tra outlier và độ lệch (skewness)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 2A: PHÂN TÍCH PHÂN PHỐI TARGET")
print("=" * 60)

for t in ["Revenue", "COGS"]:
    s = df[t]
    print(f"\n  {t}:")
    print(f"    Mean   = {s.mean():>14,.0f}")
    print(f"    Median = {s.median():>14,.0f}")
    print(f"    Std    = {s.std():>14,.0f}")
    print(f"    Skew   = {s.skew():>14.3f}  "
          f"({'lệch phải → cần log transform' if s.skew()>1 else 'tương đối đối xứng'})")
    print(f"    P95    = {s.quantile(0.95):>14,.0f}")
    print(f"    P99    = {s.quantile(0.99):>14,.0f}")
    print(f"    Max    = {s.max():>14,.0f}  "
          f"(gấp {s.max()/s.median():.1f}x median → spike cuối tháng)")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("PHÂN PHỐI REVENUE & COGS", fontweight="bold", fontsize=13)

for i, t in enumerate(["Revenue", "COGS"]):
    s = df[t]
    # Histogram gốc
    axes[i, 0].hist(s / 1e6, bins=60, color="steelblue" if i==0 else "coral",
                    edgecolor="white", alpha=0.8)
    axes[i, 0].set_title(f"{t} — Phân phối gốc (triệu VND)")
    axes[i, 0].set_xlabel("Giá trị (triệu VND)")
    axes[i, 0].axvline(s.median()/1e6, color="red", linestyle="--",
                       label=f"Median={s.median()/1e6:.1f}M")
    axes[i, 0].legend()

    # Histogram log
    axes[i, 1].hist(np.log1p(s), bins=60,
                    color="steelblue" if i==0 else "coral",
                    edgecolor="white", alpha=0.8)
    axes[i, 1].set_title(f"{t} — Sau log1p transform")
    axes[i, 1].set_xlabel("log1p(Giá trị)")

    # Time series
    axes[i, 2].plot(df["date"], s / 1e6, linewidth=0.5, alpha=0.7,
                    color="steelblue" if i==0 else "coral")
    axes[i, 2].set_title(f"{t} — Chuỗi thời gian")
    axes[i, 2].set_ylabel("Triệu VND")
    axes[i, 2].xaxis.set_major_locator(mdates.YearLocator())
    axes[i, 2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("plot_2a_target_distribution.png", bbox_inches="tight")
plt.show()
print("  → Lưu: plot_2a_target_distribution.png")

# ─────────────────────────────────────────────────────────────
# BƯỚC 2B: TƯƠNG QUAN REVENUE vs COGS
# Kiểm tra tính khả thi của Cross-Target Forecasting
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 2B: TƯƠNG QUAN REVENUE vs COGS")
print("=" * 60)

corr_pearson = df["Revenue"].corr(df["COGS"])
corr_spearman = df["Revenue"].corr(df["COGS"], method="spearman")
ratio = df["COGS"] / df["Revenue"]

print(f"  Pearson correlation  : {corr_pearson:.4f}")
print(f"  Spearman correlation : {corr_spearman:.4f}")
print(f"  COGS/Revenue ratio   : mean={ratio.mean():.4f}  std={ratio.std():.4f}  "
      f"→ {'Rất ổn định, cascade hợp lệ' if ratio.std()<0.05 else 'Biến động, cần cẩn thận'}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("TƯƠNG QUAN REVENUE & COGS", fontweight="bold")

axes[0].scatter(df["Revenue"]/1e6, df["COGS"]/1e6, alpha=0.2, s=5, color="teal")
axes[0].set_xlabel("Revenue (triệu VND)")
axes[0].set_ylabel("COGS (triệu VND)")
axes[0].set_title(f"Scatter — Pearson r={corr_pearson:.3f}")
z = np.polyfit(df["Revenue"], df["COGS"], 1)
x_line = np.linspace(df["Revenue"].min(), df["Revenue"].max(), 100)
axes[0].plot(x_line/1e6, np.polyval(z, x_line)/1e6, "r--", linewidth=1.5,
             label=f"y = {z[0]:.3f}x + {z[1]/1e6:.2f}M")
axes[0].legend()

axes[1].plot(df["date"], ratio, linewidth=0.6, alpha=0.8, color="purple")
axes[1].axhline(ratio.mean(), color="red", linestyle="--",
                label=f"Mean ratio = {ratio.mean():.4f}")
axes[1].fill_between(df["date"],
                     ratio.mean() - 2*ratio.std(),
                     ratio.mean() + 2*ratio.std(),
                     alpha=0.15, color="purple", label="±2σ band")
axes[1].set_title("COGS/Revenue ratio theo thời gian")
axes[1].set_ylabel("Tỉ lệ COGS/Revenue")
axes[1].legend()
axes[1].xaxis.set_major_locator(mdates.YearLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("plot_2b_revenue_cogs_corr.png", bbox_inches="tight")
plt.show()
print("  → Lưu: plot_2b_revenue_cogs_corr.png")

# ─────────────────────────────────────────────────────────────
# BƯỚC 3: PHÂN TÍCH THỜI GIAN
# 3A: STL Decomposition — Xu hướng, Mùa vụ, Phần dư
# 3B: Tác động Ngày trong tuần và Tháng trong năm
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 3: PHÂN TÍCH THÀNH PHẦN THỜI GIAN")
print("=" * 60)

# 3A — STL Decomposition
print("\n[3A] STL Decomposition trên Revenue...")

# Cần chuỗi không có giá trị thiếu và có tần suất đều
rev_daily = df.set_index("date")["Revenue"].asfreq("D", fill_value=np.nan)
rev_daily = rev_daily.interpolate(method="time")  # nội suy tuyến tính nếu có lỗ hổng

stl = STL(rev_daily, period=365, robust=True)
stl_result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
fig.suptitle("STL DECOMPOSITION — Revenue hàng ngày", fontweight="bold")

components = {
    "Observed (Quan sát)": rev_daily / 1e6,
    "Trend (Xu hướng)":    stl_result.trend / 1e6,
    "Seasonal (Mùa vụ)":   stl_result.seasonal / 1e6,
    "Residual (Phần dư)":  stl_result.resid / 1e6,
}
colors = ["steelblue", "green", "orange", "red"]
for ax, (title, data), color in zip(axes, components.items(), colors):
    ax.plot(rev_daily.index, data, linewidth=0.6, color=color, alpha=0.85)
    ax.set_ylabel("Triệu VND", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

print(f"  Sức mạnh Trend   : {stl_result.trend.std()/rev_daily.std():.3f}")
print(f"  Sức mạnh Seasonal: {stl_result.seasonal.std()/rev_daily.std():.3f}")
print(f"  Sức mạnh Residual: {stl_result.resid.std()/rev_daily.std():.3f}")

plt.tight_layout()
plt.savefig("plot_3a_stl_decomposition.png", bbox_inches="tight")
plt.show()
print("  → Lưu: plot_3a_stl_decomposition.png")

# 3B — Phân tích theo Ngày tuần và Tháng
print("\n[3B] Phân tích tác động Ngày/Tháng lên Revenue...")

df["_dow"]   = df["date"].dt.dayofweek
df["_month"] = df["date"].dt.month
DOW_LABELS   = ["T2","T3","T4","T5","T6","T7","CN"]
MONTH_LABELS = ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("TÁC ĐỘNG NGÀY TRONG TUẦN & THÁNG LÊN REVENUE", fontweight="bold")

# Boxplot theo ngày trong tuần
dow_data = [df.loc[df["_dow"]==d, "Revenue"].values / 1e6 for d in range(7)]
axes[0].boxplot(dow_data, labels=DOW_LABELS, patch_artist=True,
                medianprops=dict(color="red", linewidth=2),
                boxprops=dict(facecolor="steelblue", alpha=0.6))
axes[0].set_title("Revenue theo Ngày trong tuần")
axes[0].set_ylabel("Triệu VND")
axes[0].set_xlabel("Ngày")

# Boxplot theo tháng
month_data = [df.loc[df["_month"]==m, "Revenue"].values / 1e6 for m in range(1,13)]
axes[1].boxplot(month_data, labels=MONTH_LABELS, patch_artist=True,
                medianprops=dict(color="red", linewidth=2),
                boxprops=dict(facecolor="coral", alpha=0.6))
axes[1].set_title("Revenue theo Tháng trong năm")
axes[1].set_ylabel("Triệu VND")
axes[1].set_xlabel("Tháng")

# In thống kê
print("\n  Revenue trung bình theo ngày trong tuần:")
for d, lbl in enumerate(DOW_LABELS):
    m = df.loc[df["_dow"]==d, "Revenue"].median()
    print(f"    {lbl}: {m:>12,.0f}")

print("\n  Revenue trung bình theo tháng:")
for m, lbl in enumerate(MONTH_LABELS, 1):
    v = df.loc[df["_month"]==m, "Revenue"].median()
    print(f"    {lbl:>3}: {v:>12,.0f}")

plt.tight_layout()
plt.savefig("plot_3b_dow_month_boxplot.png", bbox_inches="tight")
plt.show()
print("  → Lưu: plot_3b_dow_month_boxplot.png")

# ─────────────────────────────────────────────────────────────
# BƯỚC 4: XẾP HẠNG ĐẶC TRƯNG — MUTUAL INFORMATION
# Chạy 2 lần: toàn bộ dataset vs chỉ 2019-2022 (era dữ liệu thật)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 4: MUTUAL INFORMATION — XẾP HẠNG ĐẶC TRƯNG")
print("=" * 60)

TARGET = "Revenue"

# Cột loại trừ (target leakage hoặc không phải feature)
EXCLUDE = {"date","Revenue","COGS","day_name",
           "payment_value","total_refund_amount",
           "order_reviews","customer_reviews","product_reviews","rating"}

feat_cols = [c for c in df.columns
             if c not in EXCLUDE
             and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]

def compute_mi(subset_df, feat_cols, target, label):
    """Tính Mutual Information score cho từng feature."""
    X = subset_df[feat_cols].fillna(0).values
    y = np.log1p(subset_df[target].values)   # MI trên log-scale ổn định hơn
    scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    result = pd.Series(scores, index=feat_cols, name=label).sort_values(ascending=False)
    return result

print(f"\n  Số features phân tích: {len(feat_cols)}")
print("  Đang tính MI trên toàn bộ dataset (2012-2022)...")
mi_full = compute_mi(df, feat_cols, TARGET, "MI_Full_2012-2022")

mask_real = df["date"] >= "2019-01-01"
print(f"  Đang tính MI trên Real-Data era (2019-2022, {mask_real.sum()} dòng)...")
mi_real = compute_mi(df[mask_real], feat_cols, TARGET, "MI_RealEra_2019-2022")

# Kết hợp kết quả
mi_compare = pd.DataFrame({"MI_Full": mi_full, "MI_Real": mi_real})
mi_compare["Delta"] = mi_compare["MI_Real"] - mi_compare["MI_Full"]
mi_compare["Rank_Full"] = mi_compare["MI_Full"].rank(ascending=False).astype(int)
mi_compare["Rank_Real"] = mi_compare["MI_Real"].rank(ascending=False).astype(int)
mi_compare = mi_compare.sort_values("MI_Real", ascending=False)

print(f"\n  TOP 30 Features theo MI Real-Data Era (2019-2022):\n")
print(f"  {'Feature':<45} {'MI_Full':>8} {'MI_Real':>8} {'Delta':>8}")
print("  " + "-" * 72)
for feat, row in mi_compare.head(30).iterrows():
    delta_str = f"+{row['Delta']:.3f}" if row['Delta'] >= 0 else f"{row['Delta']:.3f}"
    print(f"  {feat:<45} {row['MI_Full']:>8.3f} {row['MI_Real']:>8.3f} {delta_str:>8}")

# Features có MI ~ 0 trong cả 2 kỳ → ứng cử viên loại bỏ
useless = mi_compare[(mi_compare["MI_Full"] < 0.01) & (mi_compare["MI_Real"] < 0.01)]
print(f"\n  ⚠ Features vô nghĩa (MI < 0.01 ở cả 2 kỳ): {len(useless)}")
print("   ", list(useless.index[:15]))

# Vẽ biểu đồ so sánh MI
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("MUTUAL INFORMATION — Xếp hạng đặc trưng", fontweight="bold")

top30 = mi_compare.head(30)

# MI Full dataset
axes[0].barh(range(30), top30["MI_Full"].values[::-1], color="steelblue", alpha=0.8)
axes[0].set_yticks(range(30))
axes[0].set_yticklabels(top30.index[::-1], fontsize=7)
axes[0].set_title("Top 30 — TOÀN BỘ dữ liệu (2012-2022)")
axes[0].set_xlabel("Mutual Information Score")
axes[0].invert_yaxis()

# MI Real era
colors_real = ["#e74c3c" if d > 0.02 else "#3498db"
               for d in top30["Delta"].values]
axes[1].barh(range(30), top30["MI_Real"].values[::-1],
             color=list(reversed(colors_real)), alpha=0.8)
axes[1].set_yticks(range(30))
axes[1].set_yticklabels(top30.index[::-1], fontsize=7)
axes[1].set_title("Top 30 — ERA DỮ LIỆU THẬT (2019-2022)\n"
                  "  🔴 Tăng mạnh so với full  |  🔵 Tương đương")
axes[1].set_xlabel("Mutual Information Score")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("plot_4_mutual_information.png", bbox_inches="tight")
plt.show()
print("  → Lưu: plot_4_mutual_information.png")

# ─────────────────────────────────────────────────────────────
# TỔNG HỢP KẾT QUẢ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TỔNG HỢP — CÁC FILE ĐÃ TẠO")
print("=" * 60)
outputs = [
    ("plot_1b_quality_heatmap.png",    "Heatmap chất lượng dữ liệu theo thời gian"),
    ("plot_2a_target_distribution.png","Phân phối Revenue & COGS"),
    ("plot_2b_revenue_cogs_corr.png",  "Tương quan Revenue vs COGS"),
    ("plot_3a_stl_decomposition.png",  "STL Decomposition của Revenue"),
    ("plot_3b_dow_month_boxplot.png",  "Tác động ngày/tháng lên Revenue"),
    ("plot_4_mutual_information.png",  "Mutual Information ranking"),
]
for fname, desc in outputs:
    print(f"  ✓ {fname:<40} — {desc}")

print("\n→ Hãy gửi kết quả (charts + số liệu) để tiến hành BƯỚC 2:")
print("  Xác định features nên loại bỏ và features mới cần tạo thêm.")
