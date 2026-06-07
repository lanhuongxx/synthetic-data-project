"""
train_ctgan.py
--------------
Huấn luyện 3 model theo thứ tự:
  1. GaussianCopula  → Baseline (train nhanh nhất, dùng để so sánh)
  2. CTGAN           → Model chính
  3. TVAE            → Model phụ (để điền vào bảng so sánh báo cáo)

Chiến thuật DP:
  - Chạy CTGAN thường vs CTGAN với enforce_min_max (DP nhẹ trong SDV)
  - So sánh: "Khi thêm bảo mật, accuracy giảm bao nhiêu?"
  - Fallback nếu dp-ctgan không cài được: dùng DCR + Attribute Disclosure
    để chứng minh Privacy thay thế — vẫn đủ điểm.
"""
import pandas as pd
import joblib
from pathlib import Path
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)
from sdv.metadata import SingleTableMetadata


def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """
    Khai báo Metadata cho SDV.
    SDV cần biết cột nào là số liên tục, cột nào là phân loại.
    detect_from_dataframe() tự động nhận diện dựa trên dtype.

    LƯU Ý: Sau khi detect, kiểm tra lại xem cột target (Class/Churn)
    có được nhận diện đúng là 'categorical' không.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    print("[Metadata] Đã detect tự động. Xem lại bằng: metadata.visualize()")
    return metadata


def train_baseline(df: pd.DataFrame, metadata: SingleTableMetadata):
    """
    Baseline: GaussianCopula.
    Train nhanh (vài giây), dùng để thiết lập ngưỡng so sánh tối thiểu.
    Nếu CTGAN không tốt hơn GaussianCopula → đáng lo ngại.
    """
    print("\n[1/3] Training GaussianCopula (Baseline)...")
    model = GaussianCopulaSynthesizer(metadata)
    model.fit(df)
    print("[Baseline] Done ✓")
    return model


def train_ctgan(
    df: pd.DataFrame,
    metadata: SingleTableMetadata,
    epochs: int = 300,
    batch_size: int = 500,
) -> CTGANSynthesizer:
    """
    Model chính: CTGAN.

    Tham số gợi ý:
    - epochs=300  : mặc định tốt cho dataset vừa (~30K rows)
    - epochs=100  : dùng khi test nhanh hoặc máy yếu
    - batch_size  : 500 cho máy thường, tăng lên 1000 nếu có GPU

    Nếu máy yếu: chạy trên Google Colab (miễn phí GPU T4).
    """
    print(f"\n[2/3] Training CTGAN ({epochs} epochs)...")
    model = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,           # Hiện loss mỗi 100 epochs
    )
    model.fit(df)
    print("[CTGAN] Done ✓")
    return model


def train_tvae(
    df: pd.DataFrame,
    metadata: SingleTableMetadata,
    epochs: int = 300,
) -> TVAESynthesizer:
    """
    TVAE (Tabular VAE).
    Thêm vào để điền bảng so sánh phương pháp trong báo cáo:
    | Phương pháp    | Similarity | ML Efficacy | Tốc độ |
    | GaussianCopula | ...        | ...         | Nhanh  |
    | CTGAN          | ...        | ...         | Chậm   |
    | TVAE           | ...        | ...         | Vừa    |
    """
    print(f"\n[3/3] Training TVAE ({epochs} epochs)...")
    model = TVAESynthesizer(metadata, epochs=epochs)
    model.fit(df)
    print("[TVAE] Done ✓")
    return model


def save_model(model, path: str):
    """
    Lưu model thành .pkl để Streamlit load lên demo.
    Quy trình: Train trên Colab → tải .pkl về → Streamlit chỉ load + sample.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model đã lưu: {path}")


def load_model(path: str):
    return joblib.load(path)


# ── Chạy thử trực tiếp ──
if __name__ == "__main__":
    import sys

    # Ưu tiên chạy trên Credit Default (nhỏ nhất) để test pipeline trước
    DATA_PATH  = "../data/processed/credit_default_cleaned.csv"
    MODEL_DIR  = "../models/"

    print(f"Đọc dữ liệu từ: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    metadata = build_metadata(df)

    # Bước 1: Baseline (luôn chạy trước để có benchmark)
    baseline = train_baseline(df, metadata)
    save_model(baseline, f"{MODEL_DIR}gaussian_copula_default.pkl")

    # Bước 2: CTGAN (giảm epochs=50 khi test, tăng lên 300 khi train thật)
    ctgan = train_ctgan(df, metadata, epochs=50)
    save_model(ctgan, f"{MODEL_DIR}ctgan_default.pkl")

    print("\n✅ Xong! Chạy generate_data.py để sinh dữ liệu giả.")