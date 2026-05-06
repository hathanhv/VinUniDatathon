# OVERVIEW

**Mục tiêu:**
Tạo ra một tập dữ liệu (dataset) theo dòng thời gian (Time-series) để dự báo hai chỉ số tài chính quan trọng:
- **REVENUE** (Doanh thu): Tổng giá trị thanh toán thực tế nhận được mỗi ngày.
- **COGS** (Giá vốn hàng bán): Dựa trên giá trị đơn hàng và các chi phí vận hành đi kèm.

**Thông số dataset mục tiêu:**
- Số lượng dòng: 3.833 dòng (liên tục, không ngắt quãng).
- Phạm vi thời gian: Từ 04/07/2012 đến 31/12/2022.
- Đặc điểm nổi bật: Cơ chế tự động bù đắp dữ liệu cho những ngày thiếu hụt giao dịch để duy trì tính toàn vẹn của chuỗi thời gian.

---

# TRANSACTION

## returns

- **Chỉ số:** Tính tổng `quantity` và `refund_amount` (số lượng hàng trả về) trong 1 ngày.
- **Cơ chế bù ngày:** Sử dụng `pd.date_range` để tạo danh sách tất cả các ngày từ 2012–2022. Những ngày không có khách trả hàng sẽ được hệ thống tự động chèn dòng với giá trị bằng 0.

## orders

- Không sử dụng cột `zip`.

**Xử lý kỹ thuật:**
- Sử dụng One-hot Encoding để chuyển đổi các trạng thái đơn hàng (`delivered`, `cancelled`, `returned`...) thành các cột định lượng.
- Tính toán số lượng khách hàng độc lập (`customer_id`) và mã đơn hàng (`order_id`) mỗi ngày.

## shipments

- **Chỉ số:** Thống kê dựa trên ngày xuất kho (`ship_date`) và ngày giao hàng (`delivery_date`).
- Số đơn được gửi đi trong ngày hôm đó.
- Số đơn hàng đã được giao tới khách trong ngày hôm đó.
- Toán dòng tiền chi phí.

## payments

- **Nguồn:** `cleaned_payments.csv`
- **Xử lý:**
  - Merge với bảng Orders để trích xuất `date` và gán giá trị thanh toán (`payment_value`) vào đúng ngày đặt hàng.
  - Tính toán mức chi tiêu trung bình (`payment_value`) và số kỳ trả góp trung bình (`installments`).
  - Sử dụng One-hot Encoding cột `payment_method` sau đó đếm số lượng từng method được sử dụng theo ngày.

## order_items

- Vì bảng này ở level product nên trước hết sẽ merge theo `key = 'order_id'` để chuyển sang level order.
- Với từng order:
  - `total_quantity`: Tổng số lượng tất cả các sản phẩm có trong đơn hàng đó.
  - `unique_products_count`: Đếm số lượng loại sản phẩm khác nhau có trong đơn.
  - `avg_discount_amount`: Trung bình giá trị giảm giá trên từng đơn hàng.
  - `has_promo_id`: Đơn hàng có khuyến mãi hay không.
- Sau khi bảng `order_items` đã ở level đơn hàng, nó sẽ được merge với bảng `orders_payments` (`key: order_id`) để gắn thêm thông tin về ngày đặt hàng và giá trị tiền tệ.

## Bước tổng hợp

- Tất cả các cột mốc thời gian từ các bảng thành phần đều được đổi tên thành `date`.
- Tạo 1 bảng `daily_transaction_df` gộp tất cả các bảng bằng **outer merge** tuần tự: Orders + Payments + Returns + Shipments.
- Sau outer join, những ngày không có hoạt động sẽ xuất hiện `NaN` → dùng `fillna(0)`.

## reviews

- Bỏ cột `review_id`.
- Dựa vào `review_date`, đếm: số đơn được review, số khách hàng unique review, số sản phẩm được review, trung bình `rating`.

---

# MASTER

## products

Không xử lý.

## customers

### 1. Mục tiêu
Chuyển đổi dữ liệu thô thành dữ liệu biến động tích lũy theo từng ngày.

### 2. Quy trình xử lý
- **Thiết lập Baseline:** Tính tổng số khách hàng và các đặc tính từ dữ liệu lịch sử (trước 2012-07-04).
- **Nguyên tắc tích lũy:** Giá trị tại ngày hiện tại = `Baseline + Cumsum(Dữ liệu tích lũy tới ngày hôm đó)`.

### 3. Output

| Nhóm | Cột |
|---|---|
| Định danh | `date` |
| Số lượng | `customer_cumulative`, `customer_daily_signup` |
| Nhân khẩu học | Các cột `_cumulative` cho Gender, Age Group, Channel |
| Địa lý | `unique_cities_cumulative` |

## geography

*(Không có xử lý đặc biệt được ghi nhận.)*

## promotions

**Xử lý kỹ thuật:**
- Chuyển đổi ngày tháng về định dạng `datetime`.
- Tạo biến flag `flag_min_order_value` (có yêu cầu giá tối thiểu hay không).
- Sử dụng One-hot Encoding cho: loại khuyến mãi, trạng thái và đối tượng mục tiêu.
- Biến đổi mỗi dòng khuyến mãi thành nhiều dòng, mỗi dòng tương ứng với một ngày hoạt động duy nhất.

**Output:**

| Cột | Ý nghĩa |
|---|---|
| `promo_flag` | Cờ hiệu (0/1) ngày đó có khuyến mãi hay không |
| `avg_discount_value` | Mức giảm giá trung bình trong ngày |
| `flag_min_order_value` | Có chương trình yêu cầu giá trị đơn tối thiểu không |
| `promo_type_percentage` | Khuyến mãi theo % |
| `promo_type_fixed` | Khuyến mãi cố định |

---

# OPERATIONAL

## inventory

**Xử lý kỹ thuật:**
- Loại bỏ các cột: `year`, `month`, `product_name`, `reorder_flag`.
- `category` và `segment` → One-hot encoding (ví dụ: `category_Casual`, `segment_Activewear`).
- Tổng hợp dữ liệu tồn kho theo `snapshot_date` → `daily_inventory_summary` (chỉ có ngày cuối tháng).

**Xử lý độ trễ (lag):**
- Dữ liệu tồn kho chốt cuối tháng T chỉ khả dụng từ đầu tháng T+1.
- Toàn bộ dữ liệu Snapshot được **shift +1 ngày** để đảm bảo không dùng thông tin tương lai.

**Forward Fill:**
- Sau merge vào bảng giao dịch hàng ngày, áp dụng `ffill` để duy trì giá trị tồn kho gần nhất cho đến snapshot tiếp theo.

**Đặc trưng chất lượng dữ liệu:**
- `days_since_snapshot`: Số ngày kể từ lần kiểm kho gần nhất.
- Các cột `_gap`: Cung cấp cho model thông tin về "độ tươi" của dữ liệu tồn kho.

## web_traffic

- One-Hot Encoding cho nguồn traffic.
- `is_web_data_simulated = 1` đánh dấu các dòng dữ liệu trước khi có dữ liệu web thật.
- Điền 0 cho các cột web bị thiếu.
- Các cột rate giữ nguyên theo ngày.

## sales

- Không xử lý, merge vào các bảng đã có.

---

# OUTPUT

| Thuộc tính | Giá trị |
|---|---|
| File | `processed_data.csv` |
| Số dòng | 3,833 |
| Phạm vi | 2012-07-04 → 2022-12-31 |
| Target 1 | `Revenue` — Doanh thu ngày |
| Target 2 | `COGS` — Giá vốn hàng bán ngày |
| Flag | `is_web_data_simulated` — 1 nếu dữ liệu web là mô phỏng |