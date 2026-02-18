"""
F1 Predictor - Predictions Dashboard
Clean glassmorphic design
"""

import streamlit as st
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent

if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from config import COLORS, F1_CALENDAR_2024, get_team_color, get_team_name, get_driver_full_name
from backend_bridge import generate_prediction_workflow, get_race_predictions, check_race_exists

# Header
st.markdown(f"""
<div style='margin-bottom: 2rem;'>
    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 5px;'>
        <div style='width: 40px; height: 2px; background-color: #FF1801;'></div>
        <span style='color: #FF1801; font-size: 0.85rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;'>Advanced Analysis</span>
    </div>
    <div style='display: flex; align-items: center; gap: 15px;'>
        <h1 style='color: #FFFFFF; font-size: 3.5rem; font-weight: 700; margin: 0;'>Prediction Lab</h1>
        <div style='background: rgba(255, 24, 1, 0.1); padding: 12px; border-radius: 12px; display: flex; align-items: center; justify-content: center;'>
            <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#FF1801" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M7 2h10" />
                <path d="M10 2v7.5L4.7 20.3a2 2 0 0 0 1.8 2.7h11a2 2 0 0 0 1.8-2.7L14 9.5V2" />
                <path d="M8.5 13h7" />
            </svg>
        </div>
    </div>
    <p style='color: #888888; font-size: 1.1rem; margin-top: 10px;'>Forecasting race results by analyzing historical data.</p>
    <div style='width: 100%; height: 1px; background: linear-gradient(90deg, #FF1801 0%, transparent 100%); margin-top: 20px;'></div>
</div>
""", unsafe_allow_html=True)

# Selectors
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown(f"<p style='color: {COLORS['text_muted']}; font-size: 0.85rem; margin-bottom: 0.3rem; text-transform: uppercase;'>YEAR</p>", unsafe_allow_html=True)
    year = st.selectbox("Year", [2024, 2025, 2026], index=0, key='pred_year', label_visibility="collapsed")

with col2:
    st.markdown(f"<p style='color: {COLORS['text_muted']}; font-size: 0.85rem; margin-bottom: 0.3rem; text-transform: uppercase;'>GRAND PRIX</p>", unsafe_allow_html=True)
    race_name = st.selectbox("GP", F1_CALENDAR_2024, index=0, key='pred_race', label_visibility="collapsed")

with col3:
    st.markdown(f"<p style='color: {COLORS['text_muted']}; font-size: 0.85rem; margin-bottom: 0.3rem; text-transform: uppercase;'>&nbsp;</p>", unsafe_allow_html=True)
    generate_btn = st.button("GENERATE PREDICTION", key="gen_pred", use_container_width=True)

# Generate predictions
if generate_btn:
    with st.spinner("🔄 Analyzing..."):
        success, msg, race_id = generate_prediction_workflow(race_name, year)
        if success:
            st.success(f"✅ {msg}")
            st.session_state.pred_race_id = race_id
            st.rerun()
        else:
            st.error(f"❌ {msg}")
            st.stop()

# Check for predictions
if 'pred_race_id' in st.session_state:
    race_id = st.session_state.pred_race_id
else:
    exists, race_id, status = check_race_exists(race_name, year)
    if exists and status in ['predicted', 'completed']:
        st.session_state.pred_race_id = race_id
    else:
        st.info("👆 Select a race and click **GENERATE PREDICTION**")
        st.stop()

predictions_df = get_race_predictions(race_id)
if predictions_df.empty:
    st.warning("⚠️ No predictions found")
    st.stop()

# Display results
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
    <h3 style='color: #FF1801 !important; font-size: 1.5rem; font-weight: 700; '>
        Predicted Results
    </h3>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

for idx, (_, row) in enumerate(predictions_df.iterrows(), 1):
    driver_code = row['driver']
    driver = get_driver_full_name(driver_code)
    team = get_team_name(driver_code)
    color = get_team_color(driver_code)
    grid_pos = int(row['grid_position'])
    
    # Fallback color if none found
    if not color or color == '':
        color = '#B6BABD'  # Default gray
    
    st.markdown(f"""
    <div style="
        background: rgba(26,26,26,0.8);
        backdrop-filter: blur(10px);
        border-left: 4px solid {color};
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.6rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    " onmouseover="this.style.background='rgba(36,36,36,0.9)'; this.style.transform='translateX(5px)';" 
       onmouseout="this.style.background='rgba(26,26,26,0.8)'; this.style.transform='translateX(0)';">
        <div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">{driver}</div>
            <div style="font-size: 0.9rem; color: {color}; margin-top: 0.3rem;">{team}</div>
        </div>
        <div style="
            font-size: 0.85rem;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            background: rgba(255,255,255,0.05);
            color: rgba(255,255,255,0.6);
        ">
            Grid: P{grid_pos}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"""
    <div style="background: rgba(26,26,26,0.8); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
        <h4 style="color: {COLORS['primary']}; margin-bottom: 1rem;">📌 How to Read</h4>
        <p style="color: rgba(255,255,255,0.6); font-size: 0.95rem; line-height: 1.8; margin: 0;">
            <strong>Order:</strong> Top to bottom = 1st to last<br>
            <strong>68% CI:</strong> Confidence interval (±2 positions)<br>
            <strong>Team Colors:</strong> Left border shows team
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div style="background: rgba(26,26,26,0.8); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
        <h4 style="color: {COLORS['primary']}; margin-bottom: 1rem;">⚙️ Model Info</h4>
        <p style="color: rgba(255,255,255,0.6); font-size: 0.95rem; line-height: 1.8; margin: 0;">
            <strong>Algorithm:</strong> XGBoost Regression<br>
            <strong>Features:</strong> 25 engineered metrics<br>
            <strong>Training Data:</strong> 6 historical races
        </p>
    </div>
    """, unsafe_allow_html=True)