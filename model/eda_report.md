# EDA Report — VinUni Datathon 2026
**Mục tiêu:** Phân tích chất lượng dữ liệu và chọn lọc đặc trưng để cải thiện dự báo Revenue & COGS (2023-2024)
**Dataset:** `processed_data.csv` — 3,833 dòng | 94 cột | 2012-07-04 → 2022-12-31

---

## 1. Kiểm định chất lượng dữ liệu

### 1.1 Tỉ lệ Zero/NaN theo nhóm feature

| Nhóm | Số cột | % Zero | % NaN | Đánh giá |
|---|---|---|---|---|
| Web Traffic | 12 | 52.0% | 0.0% | ⚠ Dữ liệu thật chỉ từ ~2019 |
| Inventory | 31 | 0.2% | 0.0% | ✅ Tốt (do forward-fill) |
| Promotions | 1 | 0.0% | 0.0% | ✅ Tốt |
| Orders/Trans | 28 | 1.7% | 0.0% | ✅ Tốt |
| Reviews | 4 | 0.2% | 0.0% | ✅ Tốt |
| Customer | 1 | 0.0% | 0.0% | ✅ Tốt |
| Category/Seg | 24 | 0.4% | 0.0% | ✅ Tốt |

> **Lưu ý:** Web Traffic có `is_web_data_simulated=1` cho các dòng trước khi có dữ liệu thật.
> Các cột `sessions`, `unique_visitors` = 0 trước ~2019.

### 1.2 Heatmap chất lượng dữ liệu

Từ heatmap quan sát thấy:
- Toàn bộ dữ liệu là **100% non-zero** ngay từ 2012 cho hầu hết nhóm
- Điều này do cơ chế **forward-fill** và **zero-fill** được áp dụng khi tạo `processed_data.csv`
- Nhóm **Reviews** có một khoảng trống nhỏ ở tháng 7/2012

---

## 2. Phân tích phân phối target

### 2.1 Thống kê Revenue & COGS

| Chỉ số | Revenue | COGS |
|---|---|---|
| Mean | 4,286,584 | 3,695,134 |
| Median | 3,647,304 | 3,161,113 |
| Std | 2,624,840 | 2,219,789 |
| **Skewness** | **1.670 (lệch phải)** | **1.625 (lệch phải)** |
| P95 | 9,398,760 | 8,090,776 |
| P99 | 13,801,990 | 11,574,112 |
| Max | 20,905,271 (5.7× median) | 16,535,858 (5.2× median) |

### 2.2 Nhận xét

- Cả hai target đều **lệch phải mạnh (skew > 1.5)** → **bắt buộc phải dùng `log1p` transform** trước khi train
- Giá trị max gấp 5-6 lần median → **spike cuối tháng rất cực đoan**
- Sau `log1p` transform: phân phối gần đối xứng hơn nhưng vẫn hơi có 2 đỉnh (bimodal)
- Chuỗi thời gian cho thấy **volatility tăng mạnh từ 2014-2019**, sau đó giảm

### 2.3 Tương quan Revenue vs COGS

| Chỉ số | Giá trị |
|---|---|
| Pearson correlation | **0.9760** |
| Spearman correlation | **0.9719** |
| COGS/Revenue ratio mean | 0.8746 |
| COGS/Revenue ratio std | 0.1274 |

- Tương quan cực cao (r=0.976) → **Revenue và COGS gần như đồng biến**
- Tuy nhiên **std(ratio) = 0.127 → 14.5% biến động** → ratio không cố định, cần dự báo COGS độc lập
- Cross-target cascade (dùng Revenue pred làm feature cho COGS) có thể hỗ trợ nhưng không thay thế hoàn toàn

---

## 3. Phân tích thành phần thời gian (STL)

### 3.1 Kết quả STL Decomposition

| Thành phần | Sức mạnh (std / std_observed) | Ý nghĩa |
|---|---|---|
| **Trend** | 0.349 | Xu hướng trung bình |
| **Seasonal** | **0.859** | 🔴 Rất mạnh — dominant signal |
| **Residual** | 0.431 | Còn noise đáng kể |

### 3.2 Phát hiện QUAN TRỌNG: Trend đảo chiều từ 2017

```
Revenue theo trend:
  2012-2016:  ~4.8M → 5.5M/ngày  (TĂNG)
  2017:       5.5M/ngày           (ĐỈNH)
  2017-2022:  5.5M → 3.0M/ngày   (GIẢM 45%)
```

> ⚠️ **Hệ quả:** Nếu model dùng `lag_364` từ 2022 (~3.0M) làm anchor để forecast 2023-2024,
> và trend tiếp tục giảm → dự báo có thể vẫn overestimate.
> Model cần thêm **trend feature** để học được sự sụt giảm này.

### 3.3 Thành phần Seasonal cực mạnh (0.86)

Seasonal chiếm 86% variance sau khi loại trend → **Fourier features + lag-364 là bắt buộc**.
Seasonal pattern lặp lại đều đặn hàng năm với amplitude ±5-10M.

### 3.4 Tác động của tháng trong năm

| Tháng | Median Revenue | So với T12 |
|---|---|---|
| T1 (Tết) | 2,416,733 | +10% |
| T2 | 3,242,763 | +47% |
| T3 | 4,606,235 | +109% |
| **T4** | **5,795,217** | **+163%** |
| **T5** | **6,067,769** | **+176% (đỉnh)** |
| **T6** | **5,826,463** | **+164%** |
| T7 | 4,379,419 | +99% |
| T8 | 3,646,382 | +66% |
| T9 | 3,604,251 | +64% |
| T10 | 3,172,297 | +44% |
| T11 | 2,419,588 | +10% |
| **T12** | **2,201,375** | **baseline (thấp nhất)** |

> ⚠️ **Bất thường so với retail thông thường:** Peak ở T4-T6 (mùa hè),
> thay vì T11-T12 (Giáng sinh, Black Friday). Đây là đặc thù ngành thời trang outdoor/streetwear.

### 3.5 Tác động của ngày trong tuần

| Ngày | Median | Nhận xét |
|---|---|---|
| T2 | 3,848,593 | |
| T3 | 3,925,730 | Cao nhất tuần |
| T4 | 3,929,024 | Cao nhất tuần |
| T5 | 3,629,497 | |
| T6 | 3,359,440 | Bắt đầu giảm |
| T7 | 3,384,605 | |
| CN | 3,603,536 | |

> Tác động DOW nhỏ (~16% khác biệt T4 vs T6), ít quan trọng hơn Month.

---

## 4. Mutual Information — Xếp hạng đặc trưng

### 4.1 Top features theo ERA DỮ LIỆU THẬT (2019-2022)

| Rank | Feature | MI_Full | MI_Real | Nhận xét |
|---|---|---|---|---|
| 1 | `order_id` | 1.165 | 1.121 | ⚠️ Proxy của order count |
| 2 | `customer_id` | 1.162 | 1.119 | ⚠️ Proxy của customer count |
| 3 | `order_status_delivered` | 1.100 | 1.010 | Số đơn giao thành công |
| 4 | `payment_method_credit_card` | 1.081 | 0.963 | |
| 5 | `device_type_mobile` | 1.014 | 0.912 | |
| 6 | `device_type_desktop` | 1.011 | 0.895 | |
| 7 | `order_source_organic_search` | 0.926 | 0.777 | |
| 8 | `order_source_paid_search` | 0.886 | 0.740 | |
| 22 | `unique_visitors` | 0.140 | **0.270** | 🔴 Tăng +0.13 trong real era |
| 23 | `sessions` | 0.142 | **0.261** | 🔴 Tăng +0.12 trong real era |
| 20 | `_month` | 0.238 | **0.364** | 🔴 Tăng +0.13 trong real era |

### 4.2 Features tăng mạnh trong Real-Data Era (Delta > 0)

Các feature có `MI_Real > MI_Full` (trở nên quan trọng hơn trong giai đoạn dữ liệu thật):

| Feature | Delta | Lý do |
|---|---|---|
| `_month` / `month` | +0.12 to +0.13 | Mùa vụ rõ hơn khi data đầy đủ |
| `unique_visitors` | +0.13 | Web data thật từ 2019 |
| `sessions` | +0.12 | Web data thật từ 2019 |
| `segment_Performance` | +0.04 | Segment ngày càng quan trọng hơn |

### 4.3 Features vô nghĩa — MI < 0.01 ở CẢ HAI era

```
_dow, day_of_week,
avg_session_duration_sec,
traffic_direct, traffic_email_campaign, traffic_referral,
traffic_paid_search, traffic_organic_search, traffic_social_media
```

**Nguyên nhân:**
- `traffic_*`: Là breakdown của `sessions` theo kênh, nhưng tổng (`sessions`) đã có — breakdown không thêm thông tin
- `avg_session_duration_sec`: Hành vi session duration không liên quan đến doanh thu
- `_dow` / `day_of_week`: Ngày trong tuần tác động rất nhỏ (chỉ ~16% khác biệt)

---

## 5. Kết luận: Feature Engineering Plan

### 5.1 Features cần LOẠI BỎ khỏi model

```python
DROP_FEATURES = [
    # Traffic breakdown — MI=0, thông tin đã có trong sessions/order_source_*
    "traffic_direct", "traffic_email_campaign", "traffic_referral",
    "traffic_paid_search", "traffic_organic_search", "traffic_social_media",
    # Session behavior — MI=0
    "avg_session_duration_sec",
    # Duplicate columns
    "day_of_week",   # trùng với _dow
    "month",         # trùng với _month (engineered)
]
```

### 5.2 Features mới cần THÊM VÀO

#### A. Trend Feature — Bắt xu hướng đảo chiều từ 2017

```python
# Số ngày kể từ đỉnh trend (2017-01-01)
# Âm = trước đỉnh (giai đoạn tăng), Dương = sau đỉnh (giai đoạn giảm)
df["_trend_days_from_peak"] = (df["date"] - pd.Timestamp("2017-01-01")).dt.days

# Có thể normalize về [-1, 1]
df["_trend_normalized"] = df["_trend_days_from_peak"] / 1826  # /5 năm
```

**Lý do:** STL cho thấy Revenue giảm 45% từ 2017 → 2022. Model cần biết "đang ở giai đoạn nào của chu kỳ" để không overestimate cho 2023-2024.

#### B. Continuous Month-End Feature — Thay thế binary flag

```python
# Giá trị liên tục: 0.0 (đầu tháng) → 1.0 (cuối tháng)
# Tốt hơn _is_last3_days (binary) vì ramp-up revenue dần dần
df["_dom_ratio"] = df["date"].dt.day / df["date"].dt.days_in_month
```

#### C. Quarter-End Month Flag

```python
# Tháng kết thúc quý: T3, T6, T9, T12 thường có hoạt động đặc biệt
df["_is_qtr_end_month"] = df["date"].dt.month.isin([3, 6, 9, 12]).astype(int)
```

#### D. Peak Season Flag

```python
# Dựa trên phát hiện từ EDA: T4-T6 là peak season
df["_is_peak_season"] = df["date"].dt.month.isin([4, 5, 6]).astype(int)
# Low season: T11-T1
df["_is_low_season"]  = df["date"].dt.month.isin([11, 12, 1]).astype(int)
```

#### E. Lag features nâng cao

```python
# YoY ratio của sessions (causal driver khi có real web data)
log_sess = np.log1p(df["sessions"])
df["_sessions_yoy_ratio"] = log_sess - log_sess.shift(364)

# Rolling mean của cùng kỳ năm ngoái (trung bình 7 ngày quanh ngày lag-364)
# Giảm noise do ngày bất thường
for win in [3, 7]:
    df[f"_log_target_lag364_rmean{win}"] = (
        df["_log_target"].shift(364).rolling(win, center=True).mean()
    )
```

#### F. Trend-Adjusted Lag

```python
# Nếu Revenue 2022 thấp hơn 2021 một lượng nhất định,
# điều chỉnh lag-364 theo tốc độ giảm đó
annual_rev = df.groupby(df["date"].dt.year)["Revenue"].mean()
yoy_growth = annual_rev / annual_rev.shift(1)  # tỉ lệ YoY theo năm

# Ánh xạ vào từng dòng
df["_annual_yoy_growth"] = df["date"].dt.year.map(yoy_growth.to_dict())
```

---

## 6. Tóm tắt thứ tự ưu tiên triển khai

| Ưu tiên | Feature | Tác động dự kiến | Độ phức tạp |
|---|---|---|---|
| 🔴 Cao | `_trend_days_from_peak` | Giảm bias do declining trend | Thấp |
| 🔴 Cao | Drop 9 useless features | Giảm noise, tăng R² | Thấp |
| 🟡 Trung | `_dom_ratio` | Cải thiện spike prediction | Thấp |
| 🟡 Trung | `_is_peak_season`, `_is_low_season` | Tăng signal mùa vụ | Thấp |
| 🟡 Trung | `_sessions_yoy_ratio` | Causal driver cho 2019+ | Trung |
| 🟢 Thấp | `_log_target_lag364_rmean` | Giảm noise lag-364 | Trung |
| 🟢 Thấp | `_annual_yoy_growth` | Trend correction | Cao |
