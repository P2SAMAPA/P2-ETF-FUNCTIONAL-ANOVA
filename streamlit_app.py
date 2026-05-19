import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Functional ANOVA Engine", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Functional ANOVA Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Decompose ETF return curves | Spline norm + regime effect | Multi‑window selection</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 📊 Functional ANOVA")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Method:** B‑spline smoothing + regime effect")
st.sidebar.markdown("**Windows evaluated:** 63, 252, 504, 1008, 2016 days (best per ETF)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'functional_anova_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Functional ANOVA Score (Spline Norm + Regime Effect)")

with st.expander("📖 Interpretation", expanded=True):
    st.markdown("""
    - Each ETF's return curve over the rolling window is smoothed using B‑splines.
    - The **spline norm** measures the curvature / amplitude of the ETF's smooth path.
    - The **regime effect** is the absolute difference in mean returns between high‑volatility and low‑volatility periods (VIX threshold).
    - The **score** = spline norm + regime effect – higher means more dynamic return shape and stronger regime sensitivity.
    - For each ETF, the rolling window that gives the highest score is selected.
    """)

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">ANOVA score = {etf['anova_score']:.4f}</div>
                <div class="etf-score">best window = {etf.get('best_window', 'N/A')}d</div>
            </div>
            """, unsafe_allow_html=True)
    with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
        full = uni_data.get("full_scores", {})
        if full:
            rows = []
            for ticker, info in full.items():
                if isinstance(info, dict):
                    score = info.get("score", 0.0)
                    win = info.get("best_window", "N/A")
                else:
                    score = info
                    win = "N/A"
                rows.append({"ETF": ticker, "ANOVA Score": score, "Best Window": win})
            df = pd.DataFrame(rows)
            df["ANOVA Score"] = pd.to_numeric(df["ANOVA Score"], errors='coerce')
            df = df.dropna(subset=["ANOVA Score"]).sort_values("ANOVA Score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("Functional ANOVA decomposes return curves into basis functions. Higher score = more dynamic curve shape and stronger regime sensitivity → overweight signal.")
