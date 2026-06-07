"""
generate_data.py
----------------
Sinh dữ liệu giả từ model đã train.

Có 2 chế độ:
  1. generate_synthetic()        → Sinh toàn bộ bảng mới
  2. generate_augmented_fraud()  → Ghép thêm mẫu fraud vào tập thật
                                   (kịch bản B1/B2/B3 cho Credit Fraud)
"""
import pandas as pd
from pathlib import Path


def generate_synthetic(model, num_rows: int = 10000) -> pd.DataFrame:
    """
    Sinh dữ liệu giả toàn bộ bảng.
    Dùng cho: Credit Default, Telco Churn, và Credit Fraud (đánh giá chung).
    """
    print(f"Đang sinh {num_rows:,} dòng dữ liệu giả...")
    synthetic_df = model.sample(num_rows=num_rows)
    print(f"Done. Shape: {synthetic_df.shape}")
    return synthetic_df


def generate_augmented_fraud(
    fraud_model,
    real_df: pd.DataFrame,
    target_col: str = "Class",
    target_fraud_ratio: float = 0.05,
) -> pd.DataFrame:
    """
    Kịch bản Augmentation cho Credit Fraud.

    Chiến thuật đã chốt:
    - Sinh RIÊNG class=1 (fraud) → ghép vào tập thật
    - KHÔNG sinh toàn bộ bảng mới (sẽ làm loãng mẫu thật và
      không phản ánh đúng scenario thực tế)

    Kịch bản:
      B1: target_fraud_ratio=None    → dùng real_df gốc (0.17%)
      B2: target_fraud_ratio=0.05    → 5% fraud
      B3: target_fraud_ratio=0.10    → 10% fraud

    So sánh kết quả:
      "Model AI được train trên tập Giả+Thật có Recall tốt hơn
       model chỉ train trên tập Thật ban đầu không?"
    """
    n_normal        = (real_df[target_col] == 0).sum()
    n_fraud_needed  = int(n_normal * target_fraud_ratio / (1 - target_fraud_ratio))
    n_fraud_exist   = (real_df[target_col] == 1).sum()
    n_to_generate   = max(0, n_fraud_needed - n_fraud_exist)

    print(f"[Augmentation] Fraud hiện có: {n_fraud_exist:,}")
    print(f"[Augmentation] Cần đạt:       {n_fraud_needed:,} ({target_fraud_ratio*100:.0f}%)")
    print(f"[Augmentation] Cần sinh thêm: {n_to_generate:,}")

    if n_to_generate == 0:
        print("[Augmentation] Không cần sinh thêm.")
        return real_df

    synthetic_fraud = fraud_model.sample(num_rows=n_to_generate)
    augmented_df    = pd.concat([real_df, synthetic_fraud], ignore_index=True)

    # Shuffle để tránh model học theo thứ tự dòng
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

    new_ratio = (augmented_df[target_col] == 1).sum() / len(augmented_df)
    print(f"[Augmentation] Tỉ lệ fraud sau ghép: {new_ratio*100:.2f}%")
    return augmented_df


# ── Chạy thử trực tiếp ──
if __name__ == "__main__":
    import joblib

    MODEL_PATH = "../models/ctgan_default.pkl"
    OUT_PATH   = "../data/synthetic/synthetic_default_10k.csv"

    print(f"Load model từ: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    synthetic_df = generate_synthetic(model, num_rows=10000)
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(OUT_PATH, index=False)
    print(f"Đã lưu dữ liệu giả: {OUT_PATH}")