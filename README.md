# VinUniversity Datathon 2026 - Sales Forecasting Project

## Tổng Quan Dự Án

Dự án này là một bài tập trong cuộc thi **VinUniversity Datathon 2026**, tập trung vào:
- **Dự báo doanh thu (Revenue) và Giá vốn hàng bán (COGS)** từ 2023-01-01 đến 2024-07-01
- **Phân tích hành vi khách hàng** và các yếu tố ảnh hưởng đến kinh doanh
- **Làm sạch và xử lý dữ liệu** từ nhiều nguồn khác nhau
- **Xây dựng mô hình dự báo** sử dụng các phương pháp thống kê và machine learning

---

## Cấu Trúc Thư Mục

```
vinuni_datathon2026/
├── raw_datasets/                  # Dữ liệu thô từ nguồn gốc
│   ├── customers.csv              # Thông tin khách hàng
│   ├── geography.csv              # Dữ liệu địa lý (zip code, city)
│   ├── orders.csv                 # Thông tin đơn đặt hàng
│   ├── order_items.csv            # Chi tiết items trong đơn hàng
│   ├── products.csv               # Thông tin sản phẩm
│   ├── payments.csv               # Thông tin thanh toán
│   ├── sales.csv                  # Doanh thu hàng ngày (Target)
│   ├── reviews.csv                # Đánh giá sản phẩm
│   ├── returns.csv                # Thông tin trả hàng
│   ├── shipments.csv              # Thông tin vận chuyển
│   ├── inventory.csv              # Tồn kho
│   ├── promotions.csv             # Các chương trình khuyến mãi
│   ├── web_traffic.csv            # Lưu lượng truy cập website
│   └── sample_submission.csv      # Mẫu submission cho cuộc thi
│
├── cleaned_datasets/              # Dữ liệu đã được làm sạch
│   ├── customers_cleaned.csv
│   ├── geography_cleaned.csv
│   ├── products_cleaned.csv
│   ├── promotions_cleaned.csv
│   ├── sales_cleaned.csv
│   ├── web_traffic_cleaned.csv
│   ├── merge.csv                  # Dữ liệu merged từ nhiều bảng
│   ├── processed_data.csv         # Dữ liệu xử lý cuối cùng
│   └── ... (các file khác)
│
├── eda_preprocessing/             # Preprocessing theo domain
│   ├── master/
│   │   ├── customers.ipynb        # Làm sạch và xử lý dữ liệu customers
│   │   ├── geography.ipynb        # Làm sạch và xử lý dữ liệu geography
│   │   └── products.ipynb         # Làm sạch và xử lý dữ liệu products
│   ├── operational/
│   │   ├── inventory.ipynb        # Làm sạch và xử lý dữ liệu inventory
│   │   └── web_traffic.ipynb      # Làm sạch và xử lý dữ liệu web_traffic
│   ├── analytical/
│   │   └── sales.ipynb            # Phân tích doanh thu
│   └── transaction/
│       └── TRANSACTION.ipynb      # Làm sạch và xử lý dữ liệu trong lớp TRANSACTION
│
├── process_data/                  # Tổng hợp dữ liệu
│   ├── processed_data.ipynb       # Notebook để merge dữ liệu theo thời gian chạy mô hình baseline ban đầu
│
├── union_datasets/                # Gộp các dataset
│   └── Unioned_dataset.ipynb
│
├── model/                         # Xây dựng và chạy mô hình
│   ├── run_model.ipynb            # Notebook để Feature Engineering và chạy mô hình thử ban đầu
│   ├── processed_data.csv         # Dữ liệu dùng để chạy mô hình baseline
|   ├── workflow.png               # Sơ đồ quy trình dự báo đệ quy (recursive forecasting workflow)
|   ├── recursive_forecast.py      # Script dự báo đệ quy chính
|   ├── eda_report.md              # Báo cáo phân tích khám phá dữ liệu (EDA)
|   ├── eda_feature_selection.py   # Script lựa chọn đặc trưng dựa trên Mutual Information & EDA
|   └── technical.doc              # Cách merge dữ liệu cho mô hình
│
├── insight_by_class/              # Phân tích insights
│   ├── customer_behavior.ipynb    # Phân tích hành vi khách hàng
│   ├── product_return_analysis.ipynb
│   ├── promotion_effectiveness.ipynb
│   └── supplychain_inventory.ipynb
│
├── dashboard/                     # Trực quan hóa dữ liệu
│   ├── customer_behavior.pbix     # Dashboard hành vi khách hàng
│   ├── eda_return_analysis.pbix   # Dashboard phân tích trả hàng
│   ├── inventory_supplychain.pbix # Dashboard tồn kho & chuỗi cung ứng
│   └── promotion_effectiveness.pbix # Dashboard hiệu quả khuyến mãi
│
├── baseline.ipynb                 # Mô hình baseline (seasonal avg + trend)
├── part1_mcq/                     # Bài tập Multiple Choice Questions
├── database_schema/               # Database design (MySQL Workbench)
│   └── luocdocsdl.mwb
│
└── README.md                      # File này
```

---

## Quy Trình Xử Lý Dữ Liệu (Data Pipeline)

### Bước 1: Làm sạch dữ liệu (Data Cleaning)
Các notebook trong `eda_preprocessing/` xử lý dữ liệu từ `raw_datasets/`:

```
raw_datasets/{file}.csv
    ↓
eda_preprocessing/{domain}/{file}.ipynb
    ↓
cleaned_datasets/{file}_cleaned.csv
```

**Các notebook chính:**
- `master/customers.ipynb` - Xử lý NULL, kiểu dữ liệu, ngoại lệ
- `master/products.ipynb` - Làm sạch thông tin sản phẩm
- `analytical/sales.ipynb` - Phân tích doanh thu thô

### Bước 2: Tổng hợp dữ liệu (Data Merging)
```
process_data/processed_data.ipynb
    ↓
- Load tất cả cleaned_datasets/
- Merge customers + geography + products + ...
- Tạo feature mới (ngày, thời gian, mùa vụ, v.v.)
    ↓
cleaned_datasets/processed_data.csv
```

### Bước 3: Phân tích Insights
Các notebook trong `insight_by_class/` phân tích:
- Customer behavior analysis
- Product return patterns
- Promotion effectiveness
- Supply chain & inventory

### Bước 4: Xây dựng mô hình (Model Building)
```
model/run_model.ipynb
    ↓
Sử dụng processed_data.csv
    ↓
- Feature engineering (trend, seasonality, lag features, holidays)
- Train/Test split
- Xây dựng mô hình (Linear Regression, Tree-based models, v.v.)
- Dự báo Revenue & COGS
```

---

## Cách Chạy Lại Toàn Bộ Pipeline

### **Yêu cầu Môi Trường**
```
Python >= 3.8
Jupyter Notebook hoặc Google Colab
```

### **Thư viện cần thiết**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy holidays xgboost lightgbm
```

### **Thứ tự chạy:**

#### **Bước 1: Data Cleaning (Optional - nếu muốn làm lại từ đầu)**
```
Chạy lần lượt:
1. eda_preprocessing/master/customers.ipynb
2. eda_preprocessing/master/geography.ipynb
3. eda_preprocessing/master/products.ipynb
4. eda_preprocessing/master/promotions.ipynb
5. eda_preprocessing/operational/inventory.ipynb
6. eda_preprocessing/operational/web_traffic.ipynb
7. eda_preprocessing/analytical/sales.ipynb
8. eda_preprocessing/transaction/TRANSACTION.ipynb
```
Kết quả: Tạo ra các file trong `cleaned_datasets/`

---

#### **Bước 2: Data Integration & Feature Engineering** - QUAN TRỌNG
```
process_data/Master.ipynb
```
**Điều chỉnh cần thiết:**
- Nếu chạy trên máy tính cá nhân: Thay đổi đường dẫn từ Google Drive sang đường dẫn local
  ```python
  # Thay vì:
  file_path = '/content/drive/MyDrive/vinuni_datathon2026/cleaned_datasets/products_cleaned.csv'

  # Thành:
  file_path = './cleaned_datasets/products_cleaned.csv'
  ```

Kết quả:
- `cleaned_datasets/processed_data.csv` (dữ liệu đã được merge và xử lý)

---

#### **Bước 3: Insights Analysis (Optional - cho hiểu rõ dữ liệu)**
```
Chạy một hoặc nhiều notebook trong insight_by_class/:
- customer_behavior.ipynb
- product_return_analysis.ipynb
- promotion_effectiveness.ipynb
- supplychain_inventory.ipynb
```

Kết quả: Biểu đồ, insights và hiểu rõ về dữ liệu

---

#### **Bước 4: Model Building & Forecasting** - CHÍNH
```
model/run_model.ipynb
```

**Điều chỉnh cần thiết:**
```python
# Thay đường dẫn từ Google Drive thành local:
file_path = './model/processed_data.csv'  # hoặc từ cleaned_datasets/processed_data.csv
```

**Các bước trong notebook:**
1. Load `processed_data.csv`
2. Feature engineering:
   - Trend component (Linear Regression)
   - Seasonality features
   - Lag features (1, 7, 30 ngày)
   - Holiday flags (sử dụng library `holidays`)
   - Weekend/Quarter indicators
3. Train/Test split: Dữ liệu trước 2021 để train
4. Xây dựng mô hình
5. Dự báo cho 2023-2024

Kết quả: Dự báo Revenue & COGS, file submission

---

#### **(Optional) Baseline Model**
```
baseline.ipynb
```

**Phương pháp đơn giản:**
- Tính average YoY growth rate (2013-2022)
- Xây dựng seasonal profile (average cho mỗi ngày trong năm)
- Scale profile theo trend dự báo

Kết quả: Baseline predictions để so sánh

---

## Các Bảng Dữ Liệu Chính

| Bảng | Số hàng | Mô tả |
|------|--------|-------|
| **sales.csv** | 4,018 | Doanh thu hàng ngày (Revenue, COGS) - TARGET |
| **customers.csv** | 35,000+ | Khách hàng, địa lý, giới tính, nhóm tuổi, kênh tiếp thị |
| **orders.csv** | 500,000+ | Chi tiết đơn đặt hàng |
| **order_items.csv** | 1,000,000+ | Items trong từng đơn hàng |
| **products.csv** | 1,000+ | Sản phẩm, category, giá, margin |
| **returns.csv** | 50,000+ | Trả hàng |
| **reviews.csv** | 100,000+ | Đánh giá sản phẩm |
| **shipments.csv** | 500,000+ | Vận chuyển |
| **payments.csv** | 500,000+ | Thanh toán |
| **inventory.csv** | 365+ | Tồn kho theo ngày |
| **promotions.csv** | 100+ | Chương trình khuyến mãi |
| **web_traffic.csv** | 4,018 | Sessions, page views, hàng ngày |
| **geography.csv** | 500+ | Zip code, city |

---

## Mục Tiêu Chính

**Dự báo:**
- `Revenue` hàng ngày từ 2023-01-01 → 2024-07-01
- `COGS` hàng ngày từ 2023-01-01 → 2024-07-01
- **Dữ liệu huấn luyện:** 2012-07-04 → 2022-12-31

**Evaluation Metric:** MAE, RMSE, MAPE (tuỳ theo yêu cầu cuộc thi)

---

## Gợi ý Cải thiện Mô hình

1. **Feature Engineering:**
   - Thêm COVID-19 impact flag
   - Thêm competitor pricing data
   - Interaction features (promo × season)
   - Moving averages, exponential smoothing

2. **Model Ensemble:**
   - XGBoost, LightGBM
   - Prophet (Facebook time series)
   - Hybrid models (trend + seasonality + residuals)

3. **Cross-validation:**
   - Time series cross-validation
   - Nested validation cho hyperparameter tuning

4. **Post-processing:**
   - Smooth predictions
   - Reconcile với inventory constraints
   - Expert adjustment dựa trên business rules

---

## Lưu ý Quan Trọng

### Đường dẫn tệp
Các notebook được viết cho **Google Colab** (với Google Drive). Nếu chạy **local**:

**Tìm và thay thế:**
```python
# TRƯỚC:
file_path = '/content/drive/MyDrive/vinuni_datathon2026/...'

# SAU:
file_path = './...'
```

### Data Leakage
- **Training set:** Data trước 2021
- **Test set:** Data từ 2021 onwards (để giả lập mô hình dự báo thực tế)
- **Submission:** 2023-2024

Không sử dụng thông tin từ future trong feature engineering cho training!

### Thứ tự chạy
Phải chạy `process_data/Master.ipynb` trước `model/run_model.ipynb` để tạo `processed_data.csv`

---

## Key Insights (từ EDA)

- Revenue có tính **mùa vụ mạnh** (peaks quanh holiday seasons)
- **YoY growth rate** ~10-15% (phụ thuộc năm)
- Khách hàng chủ yếu từ **TP. HCM và Hà Nội**
- **Promotional campaigns** có impact tích cực
- **Web traffic** correlate với orders và revenue

---

## Hỗ Trợ & Liên Hệ

Nếu có câu hỏi:
1. Kiểm tra các notebook `eda_preprocessing/` để hiểu data cleaning logic
2. Xem `insight_by_class/` để hiểu data patterns
3. Tham khảo docstrings trong `model/run_model.ipynb`

---

## Tài Liệu Bổ Sung

- `De_thi_Vong_1.pdf` - Đề bài cuộc thi
- `database_schema/luocdocsdl.mwb` - Database schema (MySQL Workbench)

---

**Cập nhật lần cuối:** Tháng 5 năm 2026  
**Trạng thái:** Ready for Datathon Submission
