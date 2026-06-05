# Synthetic Data Generation System

Hệ thống Generative AI sinh dữ liệu bảng nhân tạo bảo toàn đặc tính thống kê và quyền riêng tư.

## Mục tiêu

Dự án tập trung vào việc xây dựng hệ thống sinh dữ liệu bảng nhân tạo (Synthetic Tabular Data) bằng các mô hình Generative AI như Gaussian Copula, CTGAN và TVAE nhằm:

* Bảo toàn các đặc tính thống kê của dữ liệu gốc.
* Hạn chế rò rỉ thông tin nhạy cảm.
* Hỗ trợ chia sẻ dữ liệu an toàn cho nghiên cứu và phát triển mô hình Machine Learning.
* Đánh giá chất lượng dữ liệu sinh thông qua các chỉ số thống kê, quyền riêng tư và hiệu quả mô hình học máy.

---

## Cấu trúc dự án

```text
synthetic-data-project/
├── data/
│   ├── raw/              # Dataset gốc (KHÔNG commit lên Git)
│   ├── processed/        # Dữ liệu sau tiền xử lý
│   └── synthetic/        # Dữ liệu nhân tạo đã sinh
│
├── notebooks/            # EDA & experiments (Jupyter/Colab)
│
├── models/               # Các mô hình đã train (.pkl)
│
├── src/
│   ├── preprocessing.py  # Data cleaning + sensitive detection
│   ├── train_ctgan.py    # Train GaussianCopula / CTGAN / TVAE
│   ├── generate_data.py  # Sinh dữ liệu nhân tạo
│   └── audit.py          # Audit Framework
│
├── app/
│   └── streamlit_app.py  # Giao diện demo
│
├── reports/              # Báo cáo, biểu đồ, hình ảnh kết quả
│
├── requirements.txt
└── README.md
```

---

## Công nghệ sử dụng

* Python
* Pandas
* NumPy
* SDV (Synthetic Data Vault)
* CTGAN
* Gaussian Copula
* TVAE
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn

---

## Datasets

| Dataset           |    Rows | Imbalance | Vai trò                  |
| ----------------- | ------: | --------: | ------------------------- |
| Credit Card Fraud | 284,807 |     0.17% | Dataset chính            |
| Credit Default    |  30,000 |     28.4% | Kiểm chứng              |
| Telco Churn       |   7,043 |     36.1% | Kiểm chứng (mixed-type) |

---

## Audit Targets

Hệ thống được đánh giá trên 3 tiêu chí chính:

| Tiêu chí                        | Mục tiêu                 |
| --------------------------------- | -------------------------- |
| Statistical Similarity (CS Score) | > 85%                      |
| DCR Privacy                       | Synthetic ≥ Real Baseline |
| ML Efficacy (A_syn / A_real)      | > 95%                      |

---

## Quy trình thực hiện

### 1. Data Preprocessing

* Làm sạch dữ liệu.
* Xử lý giá trị thiếu.
* Chuẩn hóa dữ liệu.
* Phát hiện thuộc tính nhạy cảm.

```bash
python src/preprocessing.py
```

---

### 2. Train Generative Model

Huấn luyện các mô hình:

* Gaussian Copula
* CTGAN
* TVAE

Khuyến nghị chạy trên Google Colab để tận dụng GPU.

```bash
python src/train_ctgan.py
```

---

### 3. Generate Synthetic Data

Sinh dữ liệu nhân tạo từ mô hình đã huấn luyện.

```bash
python src/generate_data.py
```

Kết quả được lưu tại:

```text
data/synthetic/
```

---

### 4. Audit Framework

Đánh giá chất lượng dữ liệu sinh:

* Statistical Similarity
* Privacy Preservation
* Machine Learning Utility

```bash
python src/audit.py
```

---

### 5. Demo UI

Khởi chạy giao diện Streamlit:

```bash
streamlit run app/streamlit_app.py
```

---

## Chạy nhanh

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Quy trình đầy đủ

```bash
# 1. Cleaning
python src/preprocessing.py

# 2. Train (khuyến nghị chạy trên Colab)
python src/train_ctgan.py

# 3. Audit
python src/audit.py

# 4. Demo UI
streamlit run app/streamlit_app.py
```

---

## Kết quả mong đợi

* Sinh dữ liệu nhân tạo có phân phối gần với dữ liệu gốc.
* Giảm nguy cơ rò rỉ thông tin cá nhân.
* Duy trì hiệu suất mô hình học máy trên dữ liệu nhân tạo.
* Hỗ trợ chia sẻ dữ liệu an toàn trong môi trường doanh nghiệp và nghiên cứu.

---

## Tác giả

Sinh viên ngành Khoa học Dữ liệu

Đề tài: **Hệ thống Generative AI sinh dữ liệu bảng nhân tạo bảo toàn đặc tính thống kê và quyền riêng tư**
