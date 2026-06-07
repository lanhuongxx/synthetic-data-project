"""
streamlit_app.py
----------------
Demo UI cho đề tài.

QUAN TRỌNG — Quy trình đúng:
  1. Train model trên Colab → lưu .pkl
  2. Tải .pkl về máy
  3. Chạy: streamlit run app/streamlit_app.py
  4. Upload CSV thật + model .pkl → bấm Generate → xem Dashboard

KHÔNG train trực tiếp ở đây nếu máy yếu.
"""
import streamlit as st
import pandas as pd
import joblib
import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

# ── Cấu hình trang ──
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🔒",
    layout="wide",
)

st.title("🔒 Synthetic Tabular Data Generator")
st.caption("Sinh dữ liệu giả bảo toàn đặc tính thống kê & quyền riêng tư | CTGAN + SDV")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Cài đặt")
    num_rows    = st.slider("Số dòng cần sinh", 1_000, 50_000, 10_000, step=1_000)
    target_col  = st.text_input("Cột target (label)", value="Class")
    dataset_type = st.selectbox("Loại dataset", ["fraud", "standard"])
    st.divider()
    st.markdown("**Hướng dẫn nhanh**")
    st.markdown("1. Upload CSV thật\n2. Upload model .pkl\n3. Bấm Generate\n4. Xem Dashboard")

# ── Session state ──
if "synthetic_df" not in st.session_state:
    st.session_state.synthetic_df = None

# ── Bước 1: Upload ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Dữ liệu thật")
    uploaded_csv = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_csv:
        real_df = pd.read_csv(uploaded_csv)
        st.dataframe(real_df.head(5), use_container_width=True)
        st.caption(f"📊 {real_df.shape[0]:,} dòng × {real_df.shape[1]} cột")

with col2:
    st.subheader("2️⃣ Model đã train")
    uploaded_pkl = st.file_uploader("Upload model (.pkl)", type=["pkl"])
    if uploaded_pkl:
        st.success("Model đã load thành công!")

# ── Bước 2: Generate ──
st.divider()
st.subheader("3️⃣ Sinh dữ liệu giả")

if uploaded_csv and uploaded_pkl:
    if st.button("🚀 Generate Synthetic Data", type="primary", use_container_width=True):
        with st.spinner(f"Đang sinh {num_rows:,} dòng... (thường mất 5-30 giây)"):
            model = joblib.load(uploaded_pkl)
            st.session_state.synthetic_df = model.sample(num_rows=num_rows)
        st.success(f"✅ Đã sinh {num_rows:,} dòng dữ liệu giả!")
else:
    st.info("← Upload CSV và model .pkl ở trên để bắt đầu.")

# ── Bước 3: Kết quả + Audit Dashboard ──
if st.session_state.synthetic_df is not None:
    synthetic_df = st.session_state.synthetic_df

    # Preview + Download
    col3, col4 = st.columns([2, 1])
    with col3:
        st.dataframe(synthetic_df.head(10), use_container_width=True)
    with col4:
        csv_buf = io.StringIO()
        synthetic_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Tải Synthetic CSV",
            csv_buf.getvalue(),
            file_name="synthetic_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.metric("Số dòng đã sinh", f"{len(synthetic_df):,}")
        st.metric("Số cột", synthetic_df.shape[1])

    # Audit Dashboard
    st.divider()
    st.subheader("📊 Audit Dashboard")
    tab1, tab2, tab3 = st.tabs(
        ["📈 Statistical Similarity", "🔐 Privacy (DCR)", "🤖 ML Efficacy"]
    )

    with tab1:
        st.markdown("**Phân phối Real vs Synthetic**")
        if uploaded_csv:
            real_df = pd.read_csv(uploaded_csv)
            num_cols = real_df.select_dtypes(include="number").columns[:3].tolist()

            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4))
            if len(num_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, num_cols):
                real_df[col].plot.kde(ax=ax, label="Real", color="steelblue")
                synthetic_df[col].plot.kde(ax=ax, label="Synthetic", color="tomato", linestyle="--")
                ax.set_title(col)
                ax.legend()
                ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    with tab2:
        st.markdown("**Distance to Closest Record (DCR)**")
        if uploaded_csv:
            from audit import compute_dcr
            with st.spinner("Đang tính DCR..."):
                real_df = pd.read_csv(uploaded_csv)
                dcr_result = compute_dcr(real_df, synthetic_df)

            c1, c2, c3 = st.columns(3)
            c1.metric("DCR Synthetic (mean)", f"{dcr_result['dcr_synthetic_mean']:.4f}")
            c2.metric("DCR Real Baseline", f"{dcr_result['dcr_real_baseline_mean']:.4f}")
            c3.metric(
                "Privacy Status",
                "✅ PASS" if dcr_result["privacy_pass"] else "❌ FAIL",
            )
            if dcr_result["has_exact_copy"]:
                st.error("⚠️ Phát hiện bản copy chính xác! DCR_min ≈ 0")
            else:
                st.success("Không có bản copy chính xác ✓")

    with tab3:
        st.markdown("**Train-on-Synthetic / Test-on-Real (TSTR)**")
        st.info(
            "Chạy `python src/audit.py` để xem kết quả ML Efficacy đầy đủ.\n\n"
            "Hoặc mở notebook `notebooks/04_Audit.ipynb` trên Google Colab."
        )
        st.markdown("""
        **Hướng dẫn đọc kết quả:**
        - `efficacy_ratio ≥ 0.95` → ✅ Dữ liệu giả đủ tốt để thay thế dữ liệu thật
        - `efficacy_ratio < 0.90` → ❌ Cần train CTGAN thêm epochs
        """)