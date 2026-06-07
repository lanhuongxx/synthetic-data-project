"""
preprocessing.py
----------------
Làm sạch dữ liệu thật trước khi đưa vào CTGAN.
Bước quan trọng nhất: loại bỏ cột ID/sensitive trước khi train.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Cột nhạy cảm cần loại bỏ TRƯỚC KHI train (tránh Privacy leakage)
SENSITIVE_COLS = [
    "customerID", "id", "ID", "email", "phone", "name",
    "address", "ssn", "passport",
    "Time",         # Credit Fraud: cột Time không có ý nghĩa thống kê
]


def detect_and_drop_sensitive(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Bước Sensitive Attribute Detection.
    Tự động phát hiện và loại bỏ cột ID/nhận dạng cá nhân.

    Lý do: CTGAN sẽ bị lỗi nếu cột categorical có quá nhiều giá trị unique.
    Ví dụ: cột 'customerID' có 7043 giá trị khác nhau = CTGAN không học được gì,
    chỉ tốn memory và thời gian.
    """
    cols_to_drop = []

    for col in df.columns:
        # Loại bỏ cột nằm trong danh sách sensitive
        if col in SENSITIVE_COLS or col.lower() in [s.lower() for s in SENSITIVE_COLS]:
            cols_to_drop.append(col)
            continue

        # Loại bỏ cột categorical có > 50% giá trị unique (rất có thể là ID ẩn)
        if df[col].dtype == object:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                cols_to_drop.append(col)

    if verbose and cols_to_drop:
        print(f"[Sensitive Detection] Loại bỏ {len(cols_to_drop)} cột: {cols_to_drop}")
    elif verbose:
        print("[Sensitive Detection] Không tìm thấy cột nhạy cảm.")

    return df.drop(columns=cols_to_drop, errors="ignore")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý missing values theo kiểu dữ liệu.
    - Số (float/int): điền bằng median (ít bị ảnh hưởng bởi outlier hơn mean)
    - Phân loại (object): điền bằng mode (giá trị xuất hiện nhiều nhất)
    """
    n_missing = df.isnull().sum().sum()
    if n_missing == 0:
        print("[Missing Values] Không có missing values.")
        return df

    print(f"[Missing Values] Tìm thấy {n_missing} missing. Đang xử lý...")
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ép kiểu dữ liệu đúng cho SDV Metadata:
    - Cột object → category
    - Cột int có ≤ 20 giá trị unique → category
      (Ví dụ: cột SEX = {1,2}, EDUCATION = {1,2,3,4} trong Credit Default)
    """
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category")
        elif df[col].dtype in ["int64", "int32"]:
            if df[col].nunique() <= 20:
                df[col] = df[col].astype("category")
    return df


def clean_dataset(df: pd.DataFrame, dataset_name: str = "") -> pd.DataFrame:
    """
    Pipeline làm sạch đầy đủ. Gọi hàm này trước khi đưa vào CTGAN.

    Thứ tự quan trọng — KHÔNG được đổi:
      1. Drop sensitive cols  →  tránh CTGAN học thông tin nhận dạng
      2. Handle missing       →  CTGAN không chấp nhận NaN
      3. Fix dtypes           →  SDV Metadata cần kiểu dữ liệu chính xác
    """
    print(f"\n{'='*50}")
    print(f"Cleaning: {dataset_name} | Shape ban đầu: {df.shape}")

    df = detect_and_drop_sensitive(df.copy())
    df = handle_missing_values(df)
    df = fix_dtypes(df)

    print(f"Shape sau cleaning: {df.shape}")
    print(f"Dtypes: {df.dtypes.value_counts().to_dict()}")
    return df


# ── Chạy thử trực tiếp ──
if __name__ == "__main__":
    import sys
    datasets = {
        "Credit Fraud":        ("../data/raw/credit-card-fraud-detection.csv",    "Class"),
        "Credit Default":      ("../data/raw/default-of-credit-card-clients.xls", "default payment next month"),
        "Telco Churn":         ("../data/raw/Telco-Customer-Churn.csv",            "Churn"),
        "Online Shoppers":     ("../data/raw/online_shoppers_intention.csv",       "Revenue"),
    }

    for name, (path, target) in datasets.items():
        try:
            df = pd.read_csv(path)
            cleaned = clean_dataset(df, name)
            out = f"../data/processed/{name.lower().replace(' ', '_')}_cleaned.csv"
            cleaned.to_csv(out, index=False)
            print(f"Đã lưu: {out}\n")
        except FileNotFoundError:
            print(f"[SKIP] Không tìm thấy file: {path}")