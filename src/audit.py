"""
audit.py
--------
Hệ thống đo lường 3 tiêu chí — ĐÂY LÀ PHẦN ĂN ĐIỂM NHẤT.

Tiêu chí 1: Statistical Similarity  → CS Score > 85%
Tiêu chí 2: Privacy (DCR)           → DCR_syn ≥ DCR_real_baseline × 0.8
Tiêu chí 3: ML Efficacy             → Asyn/Areal ≥ 95%
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, average_precision_score
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import pairwise_distances
from xgboost import XGBClassifier
from sdmetrics.reports.single_table import QualityReport


# ═══════════════════════════════════════════════════
# TIÊU CHÍ 1: STATISTICAL SIMILARITY
# ═══════════════════════════════════════════════════

def compute_statistical_similarity(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata,
) -> dict:
    """
    Dùng sdmetrics.QualityReport để tính:
    - Column Shapes (CS): phân phối từng cột khớp nhau không?
    - Column Pair Trends: mối quan hệ giữa các cặp cột có giống nhau không?

    Điểm tổng hợp: 0.0 → 1.0 (mục tiêu > 0.85)
    """
    report = QualityReport()
    report.generate(real_df, synthetic_df, metadata.to_dict())
    score = report.get_score()
    details = report.get_details("Column Shapes")

    print(f"\n[Similarity] Overall: {score:.4f} ({score*100:.1f}%) | Target: >85%")
    print(f"  → {'PASS ✓' if score >= 0.85 else 'FAIL ✗ — cần train thêm epochs'}")

    return {
        "overall_score": score,
        "pass": score >= 0.85,
        "column_details": details,
    }


def plot_distributions(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    columns: list,
    save_path: str = None,
):
    """
    Vẽ KDE Plot so sánh Real vs Synthetic cho từng cột số.
    Nếu 2 đường KDE đè lên nhau → dữ liệu giả tốt.
    """
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        if real_df[col].dtype in ["float64", "int64", "float32"]:
            real_df[col].plot.kde(ax=ax, label="Real", color="steelblue", linewidth=2)
            synthetic_df[col].plot.kde(ax=ax, label="Synthetic", color="tomato",
                                       linewidth=2, linestyle="--")
        else:
            # Categorical: vẽ bar chart tần suất
            r = real_df[col].value_counts(normalize=True)
            s = synthetic_df[col].value_counts(normalize=True)
            pd.DataFrame({"Real": r, "Synthetic": s}).plot.bar(ax=ax, color=["steelblue", "tomato"])

        ax.set_title(col, fontsize=11)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Distribution: Real vs Synthetic", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu: {save_path}")
    plt.show()


def plot_correlation_heatmap(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    save_path: str = None,
) -> float:
    """
    So sánh Correlation Matrix Real vs Synthetic.
    Mục tiêu: mean absolute difference < 0.1
    """
    num_cols = real_df.select_dtypes(include="number").columns.tolist()
    corr_real = real_df[num_cols].corr()
    corr_syn  = synthetic_df[num_cols].corr()
    diff      = (corr_real - corr_syn).abs()

    # Chỉ lấy tam giác trên (tránh tính 2 lần)
    upper_idx  = np.triu_indices_from(diff.values, k=1)
    mean_diff  = diff.values[upper_idx].mean()

    print(f"\n[Correlation] Mean absolute diff: {mean_diff:.4f} | Target: <0.10")
    print(f"  → {'PASS ✓' if mean_diff < 0.1 else 'Cần cải thiện'}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    kw = dict(cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.3)
    sns.heatmap(corr_real, ax=axes[0], **kw)
    axes[0].set_title("Real Data")
    sns.heatmap(corr_syn, ax=axes[1], **kw)
    axes[1].set_title("Synthetic Data")
    sns.heatmap(diff, ax=axes[2], cmap="Reds", square=True, linewidths=0.3)
    axes[2].set_title(f"Difference (mean={mean_diff:.3f})")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return mean_diff


# ═══════════════════════════════════════════════════
# TIÊU CHÍ 2: PRIVACY METRICS
# ═══════════════════════════════════════════════════

def compute_dcr(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    n_sample: int = 2000,
) -> dict:
    """
    Distance to Closest Record (DCR).

    Ý nghĩa:
    - DCR = 0: AI đã copy nguyên xi một người thật → FAIL bảo mật
    - DCR_syn ≈ DCR_real_baseline: dữ liệu giả "xa" dữ liệu thật như
      các bản ghi thật cũng xa nhau → PASS

    Lấy n_sample=2000 để tính nhanh (tránh O(n²) với 284K dòng).
    """
    num_cols = real_df.select_dtypes(include="number").columns.tolist()

    real_s = real_df[num_cols].sample(min(n_sample, len(real_df)), random_state=42).values
    syn_s  = synthetic_df[num_cols].sample(min(n_sample, len(synthetic_df)), random_state=42).values

    scaler     = MinMaxScaler()
    real_scaled = scaler.fit_transform(real_s)
    syn_scaled  = scaler.transform(syn_s)

    # DCR synthetic → real (khoảng cách từ mỗi dòng giả đến dòng thật gần nhất)
    dist_syn_real = pairwise_distances(syn_scaled, real_scaled, metric="euclidean")
    dcr_syn       = dist_syn_real.min(axis=1)

    # Baseline: DCR trong chính tập thật (khoảng cách gần nhất giữa 2 bản ghi thật)
    dist_real_real = pairwise_distances(real_scaled, metric="euclidean")
    np.fill_diagonal(dist_real_real, np.inf)
    dcr_real_base  = dist_real_real.min(axis=1)

    privacy_pass = dcr_syn.mean() >= dcr_real_base.mean() * 0.8

    result = {
        "dcr_synthetic_mean":      float(dcr_syn.mean()),
        "dcr_real_baseline_mean":  float(dcr_real_base.mean()),
        "dcr_min":                 float(dcr_syn.min()),
        "has_exact_copy":          bool(dcr_syn.min() < 1e-6),
        "privacy_pass":            privacy_pass,
    }

    print(f"\n[Privacy - DCR]")
    print(f"  DCR Synthetic mean:      {result['dcr_synthetic_mean']:.4f}")
    print(f"  DCR Real baseline mean:  {result['dcr_real_baseline_mean']:.4f}")
    print(f"  Có bản copy exact:       {'CÓ ✗' if result['has_exact_copy'] else 'Không ✓'}")
    print(f"  → Privacy {'PASS ✓' if privacy_pass else 'FAIL ✗'}")

    return result


# ═══════════════════════════════════════════════════
# TIÊU CHÍ 3: ML EFFICACY
# ═══════════════════════════════════════════════════

def _encode_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode tất cả cột categorical để XGBoost / LR có thể xử lý."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def compute_ml_efficacy(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    target_col: str,
    dataset_type: str = "standard",   # "fraud" hoặc "standard"
) -> dict:
    """
    Đo ML Efficacy theo phương pháp Train-on-Synthetic / Test-on-Real (TSTR).

    dataset_type="fraud"    → Credit Fraud  → dùng Recall, F1, PR-AUC
    dataset_type="standard" → Default/Telco → dùng Accuracy, F1 (weighted)

    Mục tiêu: Asyn/Areal ≥ 95% (accuracy loss < 5%)
    """
    real_enc = _encode_for_ml(real_df)
    syn_enc  = _encode_for_ml(synthetic_df)

    X_real = real_enc.drop(columns=[target_col])
    y_real = real_enc[target_col]
    X_syn  = syn_enc.drop(columns=[target_col])
    y_syn  = syn_enc[target_col]

    # Dùng 80% real để train (real baseline), 20% để test cả 2
    X_train_r, X_test, y_train_r, y_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )

    # Đảm bảo cột X_syn khớp với X_test
    X_syn = X_syn.reindex(columns=X_test.columns, fill_value=0)

    classifiers = {
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    }

    results = {}
    print(f"\n[ML Efficacy] dataset_type='{dataset_type}'")

    for clf_name, clf in classifiers.items():
        # Train trên REAL
        clf_r = clf.__class__(**clf.get_params())
        clf_r.fit(X_train_r, y_train_r)
        y_pred_r = clf_r.predict(X_test)

        # Train trên SYNTHETIC, test trên REAL (TSTR)
        clf_s = clf.__class__(**clf.get_params())
        clf_s.fit(X_syn, y_syn)
        y_pred_s = clf_s.predict(X_test)

        if dataset_type == "fraud":
            m = {
                "real_recall":  recall_score(y_test, y_pred_r, zero_division=0),
                "syn_recall":   recall_score(y_test, y_pred_s, zero_division=0),
                "real_f1":      f1_score(y_test, y_pred_r, zero_division=0),
                "syn_f1":       f1_score(y_test, y_pred_s, zero_division=0),
            }
            ratio = m["syn_recall"] / m["real_recall"] if m["real_recall"] > 0 else 0
            m["efficacy_ratio"] = ratio
        else:
            m = {
                "real_accuracy": accuracy_score(y_test, y_pred_r),
                "syn_accuracy":  accuracy_score(y_test, y_pred_s),
                "real_f1":       f1_score(y_test, y_pred_r, average="weighted", zero_division=0),
                "syn_f1":        f1_score(y_test, y_pred_s, average="weighted", zero_division=0),
            }
            ratio = m["syn_accuracy"] / m["real_accuracy"] if m["real_accuracy"] > 0 else 0
            m["efficacy_ratio"] = ratio

        results[clf_name] = m
        print(f"\n  [{clf_name}]")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")
        print(f"    → Efficacy ratio: {ratio*100:.1f}% | {'PASS ✓' if ratio >= 0.95 else 'FAIL ✗'}")

    return results


def plot_ml_comparison(results: dict, save_path: str = None):
    """
    Vẽ bar chart so sánh Real vs Synthetic cho XGBoost và LR.
    Dùng trong Streamlit Dashboard.
    """
    models, real_scores, syn_scores = [], [], []
    for model_name, metrics in results.items():
        models.append(model_name)
        key = "real_recall" if "real_recall" in metrics else "real_accuracy"
        real_scores.append(metrics[key])
        syn_key = key.replace("real_", "syn_")
        syn_scores.append(metrics[syn_key])

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, real_scores, width, label="Real Data", color="steelblue")
    ax.bar(x + width/2, syn_scores,  width, label="Synthetic Data", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("ML Efficacy: Real vs Synthetic")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════
# CHẠY TOÀN BỘ AUDIT
# ═══════════════════════════════════════════════════

def run_full_audit(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata,
    target_col: str,
    dataset_type: str = "standard",
    report_dir: str = "../reports/",
) -> dict:
    """Pipeline chạy đầy đủ 3 tiêu chí và in kết quả tổng hợp."""
    print("\n" + "=" * 60)
    print("  BẮT ĐẦU AUDIT FRAMEWORK")
    print("=" * 60)

    sim = compute_statistical_similarity(real_df, synthetic_df, metadata)
    dcr = compute_dcr(real_df, synthetic_df)
    ml  = compute_ml_efficacy(real_df, synthetic_df, target_col, dataset_type)

    # Vẽ và lưu biểu đồ
    num_cols = real_df.select_dtypes(include="number").columns[:4].tolist()
    plot_distributions(real_df, synthetic_df, num_cols,
                       save_path=f"{report_dir}distribution_plot.png")
    plot_correlation_heatmap(real_df, synthetic_df,
                             save_path=f"{report_dir}correlation_heatmap.png")
    plot_ml_comparison(ml, save_path=f"{report_dir}ml_efficacy_chart.png")

    print("\n" + "=" * 60)
    print("  KẾT QUẢ TỔNG HỢP")
    print(f"  Statistical Similarity : {sim['overall_score']*100:.1f}% | {'✓ PASS' if sim['pass'] else '✗ FAIL'}")
    print(f"  Privacy (DCR)          : {'✓ PASS' if dcr['privacy_pass'] else '✗ FAIL'}")
    xgb_ratio = ml.get("XGBoost", {}).get("efficacy_ratio", 0)
    print(f"  ML Efficacy (XGBoost)  : {xgb_ratio*100:.1f}% | {'✓ PASS' if xgb_ratio >= 0.95 else '✗ FAIL'}")
    print("=" * 60)

    return {"similarity": sim, "privacy": dcr, "ml_efficacy": ml}


if __name__ == "__main__":
    import joblib
    # Test nhanh với Credit Default (nhỏ nhất)
    real_df = pd.read_csv("../data/processed/credit_default_cleaned.csv")
    syn_df  = pd.read_csv("../data/synthetic/synthetic_default_10k.csv")
    model   = joblib.load("../models/ctgan_default.pkl")

    from sdv.metadata import SingleTableMetadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_df)

    run_full_audit(real_df, syn_df, metadata,
                   target_col="default payment next month",
                   dataset_type="standard")