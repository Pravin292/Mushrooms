import streamlit as st
import pandas as pd
import numpy as np
from utils import load_model_and_metrics, generate_feature_importance_plot

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)

# ---------------- Custom Styling "Midnight Glass" ----------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a, #1e293b); color: #e2e8f0; font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3 { color: #38bdf8 !important; font-weight: 700; }
.glass-panel { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); margin-bottom: 20px; }
.stNumberInput input, .stSelectbox select, .stTextInput input { background-color: rgba(15, 23, 42, 0.6) !important; color: #f8fafc !important; border: 1px solid rgba(56, 189, 248, 0.3) !important; border-radius: 8px !important; }
.stNumberInput input:focus, .stSelectbox select:focus { border: 1px solid #38bdf8 !important; box-shadow: 0 0 10px rgba(56, 189, 248, 0.5) !important; }
.stButton>button { background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: white; border-radius: 8px; font-weight: 600; padding: 0.6em 1.5em; border: none; transition: all 0.3s ease; width: 100%; margin-top: 15px; }
.stButton>button:hover { background: linear-gradient(90deg, #60a5fa, #a78bfa); box-shadow: 0 0 15px rgba(139, 92, 246, 0.6); transform: translateY(-2px); }
.result-card-success { background: rgba(16, 185, 129, 0.15); border-left: 5px solid #10b981; padding: 20px; border-radius: 10px; margin-top: 20px; }
.result-card-danger { background: rgba(239, 68, 68, 0.15); border-left: 5px solid #ef4444; padding: 20px; border-radius: 10px; margin-top: 20px; }
.metric-text { font-size: 24px; font-weight: bold; color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
try:
    model, feature_columns, metrics = load_model_and_metrics()
except Exception as e:
    st.error(f"Fungi Engine Crash: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

if model is None:
    st.warning("🍄 Mushroom models not identified. Please run `python model_training.py` first.")
    st.stop()

# ---------------- Sidebar UX ----------------
st.sidebar.markdown("### 🧬 Fungal Biomarkers")
st.sidebar.caption("Input LabelEncoded indices to process biological traits.")

input_data = {}
for i, feature in enumerate(feature_columns):
    # Truncating display names slightly to fit sidebars cleanly
    display_name = feature.replace('-', ' ').title()
    input_data[feature] = st.sidebar.number_input(f"{display_name}", value=0)

analyze_btn = st.sidebar.button("Run Diagnostics")

# ---------------- Header ----------------
st.markdown("<h1>🍄 Fungi Classification Module</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Gradient Boosting Edibility Detector</p>", unsafe_allow_html=True)
st.markdown("---")


# ---------------- UI TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔮 Inference Engine", "📊 Architecture & Metrics", "📁 Source Informatics"])

with tab1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Toxicological Analysis Target")
    
    if analyze_btn:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        if str(prediction).lower() in ["e", "edible", "0"]:
            st.markdown(f"""
                <div class="result-card-success">
                    <div class="metric-text">✅ Classification: Edible (Class: {prediction})</div>
                    <p style="color: #cbd5e1; margin-top: 5px;">This fungus exhibits standard characteristics historically associated with non-toxic species.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card-danger">
                    <div class="metric-text">☠️ Classification: Poisonous (Class: {prediction})</div>
                    <p style="color: #cbd5e1; margin-top: 5px;">Highly active toxic characteristics detected. This specimen must not be consumed.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
         st.info("👈 Please map biomarker vectors via the sidebar controls to initiate inference.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 Gradient Boosting Output Weights")
    
    col1, col2 = st.columns(2)
    col1.metric("Validation Accuracy", f"{metrics['Accuracy'] * 100:.2f}%")
    col2.metric("Ensemble Engine", metrics["Model Type"])
    
    st.markdown("---")
    fig = generate_feature_importance_plot(metrics)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📥 Label Encoded Analytics Dataset")
    try:
        df = pd.read_csv("data/mushrooms.csv")
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Excerpt: Top 100 biological records (UCI Machine Learning Repository).")
    except Exception as e:
        st.error(f"Failed to mount local filesystem records: {e}")
    st.markdown('</div>', unsafe_allow_html=True)