"""
F1 Predictor - Sandbagging Detective
Advanced Telemetry Analysis & Visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# ============================================================================
# SAFETY CHECK
# ============================================================================
if 'current_page' not in st.session_state:
    st.warning("Please run the main app instead: streamlit run app.py")
    st.stop()

# Add project root to path
root_path = Path(__file__).resolve().parent.parent

if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from config import COLORS, F1_CALENDAR_2024, get_driver_full_name, get_team_color
from backend_bridge import check_race_exists, get_sandbagging_analysis

# ============================================================================
# SMART INSIGHTS ENGINE (NO EMOJIS)
# ============================================================================
def generate_smart_insights(suspects_df, field_avg):
    """Generates professional analysis text"""
    if suspects_df.empty:
        return "The data indicates a clean session. Most drivers ran representative race-pace programs in FP2, with performance gains in Qualifying matching the expected track evolution."
    
    top_suspect = suspects_df.iloc[0]
    driver = get_driver_full_name(top_suspect['driver'])
    gain = top_suspect['speed_gain_pct']
    extra_gain = gain - field_avg
    
    # Logic for text generation
    if extra_gain > 1.5:
        implication = "suggesting a heavy fuel load or significantly detuned engine in practice."
    elif extra_gain > 0.8:
        implication = "likely hiding true aerodynamic performance."
    else:
        implication = "indicating a conservative approach to practice sessions."
        
    return f"**{driver}** was the biggest anomaly, finding a **{gain:.1f}%** pace improvement (vs field avg {field_avg:.1f}%). This {extra_gain:.1f}% deviation above the norm is {implication}"

# ============================================================================
# CUSTOM CSS (Reverted Radar Style)
# ============================================================================

st.markdown("""
<style>
/* RADAR CONTAINER */
.radar-container {
    position: relative;
    width: 340px;
    height: 340px;
    margin: 0 auto;
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 50%;
    background: radial-gradient(circle at center, rgba(10, 20, 30, 0.5), transparent 70%);
}

/* GRID RINGS */
.radar-grid-outer {
    position: absolute; width: 100%; height: 100%;
    border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 50%;
}
.radar-grid-mid {
    position: absolute; width: 66%; height: 66%;
    border: 1px solid rgba(0, 212, 255, 0.1); border-radius: 50%;
}
.radar-grid-inner {
    position: absolute; width: 33%; height: 33%;
    border: 1px solid rgba(0, 212, 255, 0.1); border-radius: 50%;
}

/* RADIAL SWEEP ANIMATION */
.radar-sweep {
    position: absolute; width: 100%; height: 100%; border-radius: 50%;
    background: conic-gradient(from 0deg, transparent 0deg, transparent 260deg, rgba(0, 212, 255, 0.1) 300deg, rgba(0, 212, 255, 0.6) 360deg);
    animation: radar-spin 3s linear infinite; z-index: 2;
}
@keyframes radar-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

/* CENTER EYE */
.radar-eye {
    position: absolute; width: 44px; height: 44px; background: #0b1116;
    border: 2px solid #00d4ff; border-radius: 50%; z-index: 10;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
}

/* DOTS */
.radar-dot {
    position: absolute; width: 14px; height: 14px;
    background-color: #ff3b3b; border-radius: 50%;
    box-shadow: 0 0 12px #ff3b3b; z-index: 5;
    transform: translate(-50%, -50%);
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
    50% { transform: translate(-50%, -50%) scale(1.2); opacity: 0.8; }
    100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
}

/* BADGE */
.detected-badge {
    position: absolute; bottom: -40px; left: 50%; transform: translateX(-50%);
    background-color: rgba(40, 10, 10, 0.95); border: 1px solid #ff3b3b;
    color: #ff3b3b; padding: 6px 24px; border-radius: 16px;
    font-size: 0.85rem; font-weight: 800; letter-spacing: 1.5px; white-space: nowrap;
}

/* METRIC CARDS */
.metric-box {
    background: rgba(26, 26, 26, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-box:hover {
    transform: translateY(-2px);
    border-color: #E10600;
}
.metric-title {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-val {
    color: #FFFFFF;
    font-size: 1.4rem;
    font-weight: 900;
}
.metric-sub {
    color: #E10600;
    font-size: 0.8rem;
    margin-top: 5px;
}

/* DETECTION CARD */
.detection-card {
    background-color: #0d1117; border: 1px solid #21262d;
    border-radius: 12px; padding: 24px; height: 100%;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.suspect-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 0; border-bottom: 1px solid #21262d;
}
.gain-text { color: #ff3b3b; font-weight: 800; font-size: 0.9rem; }
</style>""", unsafe_allow_html=True)

# ============================================================================
# PAGE HEADER
# ============================================================================

st.markdown("""
<div style='text-align: center; margin-top: -20px; margin-bottom: 40px;'>
<div style='display: inline-block; padding: 4px 12px; background: rgba(225, 6, 0, 0.1); border: 1px solid rgba(225, 6, 0, 0.5); border-radius: 20px; margin-bottom: 15px;'>
<span style='color: #E10600; font-size: 0.7rem; font-weight: 700; letter-spacing: 1px;'>PERFORMANCE ANALYSIS</span>
</div>
<h1 style='font-size: 3rem; font-weight: 900; margin: 0; line-height: 1;'>
Sandbagging <span style='color: #E10600;'>Detective</span>
</h1>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# CONTROLS
# ============================================================================

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    col_year, col_race = st.columns([1, 2])
    with col_year:
        selected_year = st.selectbox("Year", [2024, 2025], index=0)
    with col_race:
        selected_race = st.selectbox("Race Session", F1_CALENDAR_2024, index=0)

# Check Data
exists, race_id, status = check_race_exists(selected_race, selected_year)

if not exists:
    st.info(f"Predictions not found for {selected_race}. Please run the prediction model first.")
    st.stop()

# Fetch Analysis
df_analysis = get_sandbagging_analysis(race_id)

if df_analysis.empty:
    st.warning("Insufficient data.")
    st.stop()

suspects = df_analysis[df_analysis['is_suspect'] == True]
suspect_count = len(suspects)

# ============================================================================
# RADAR & ALERT
# ============================================================================

col_radar, col_card = st.columns([1, 1.3], gap="large")

with col_radar:
    dots_html = ""
    for _, row in df_analysis.iterrows():
        if row['is_suspect']:
            dots_html += f"""<div class="radar-dot" style="top: {row['radar_top']}; left: {row['radar_left']};" title="{row['driver']}"></div>"""
    
    badge_html = f"""<div class="detected-badge">{suspect_count} DETECTED</div>""" if suspect_count > 0 else \
                 f"""<div class="detected-badge" style="border-color:#00d4ff; color:#00d4ff; background:rgba(0,212,255,0.1);">ALL CLEAR</div>"""

    st.markdown(f"""
<div style="padding: 20px;">
<div class="radar-container">
<div class="radar-grid-outer"></div>
<div class="radar-grid-mid"></div>
<div class="radar-grid-inner"></div>
<div class="radar-sweep"></div>
<div class="radar-eye">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#E10600" stroke-width="2">
<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
<circle cx="12" cy="12" r="3"></circle>
</svg>
</div>
{dots_html}
{badge_html}
</div>
</div>
""", unsafe_allow_html=True)

with col_card:
    if suspect_count > 0:
        header_title = "Detection Alert"
        alert_msg = f"<span style='color:#E10600; font-weight:700;'>{suspect_count} teams were</span> hiding their true speed."
    else:
        header_title = "Clean Session"
        alert_msg = "No significant sandbagging detected."

    rows_html = ""
    for _, row in suspects.head(5).iterrows():
        d_name = get_driver_full_name(row['driver'])
        rows_html += f"""
<div class="suspect-row">
<div style="display: flex; align-items: center; gap: 10px;">
<span style="color: {COLORS['primary']}; font-size: 1.2rem;">●</span>
<span style="color: #e6edf3; font-size: 0.9rem;">{d_name}</span>
</div>
<span class="gain-text">+{row['speed_gain_pct']:.1f}% gain</span>
</div>
"""

    st.markdown(f"""
<div class="detection-card">
<h3 style="margin: 0 0 10px 0; color: #fff; font-size: 1.1rem; font-weight: 700;">{header_title}</h3>
<p style="color: #8b949e; font-size: 0.95rem; line-height: 1.6; margin-bottom: 25px;">
{alert_msg}
</p>
<div style="margin-top: 10px;">
{rows_html}
</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# METRICS
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)

if not suspects.empty:
    biggest_gain = suspects.iloc[0]
    vmax_king = df_analysis.sort_values('relative_vmax_gain', ascending=False).iloc[0] if 'relative_vmax_gain' in df_analysis.columns else df_analysis.iloc[0]
    vmax_val = f"+{vmax_king['relative_vmax_gain']:.1f}%" if 'relative_vmax_gain' in df_analysis.columns else "N/A"
else:
    biggest_gain = df_analysis.iloc[0]
    vmax_king = df_analysis.iloc[0]
    vmax_val = "N/A"

field_avg = df_analysis['speed_gain_pct'].mean()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">BIGGEST SANDBAG</div>
        <div class="metric-val">{biggest_gain['driver']}</div>
        <div class="metric-sub">+{biggest_gain['speed_gain_pct']:.1f}% Pace</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">ENGINE JUMP</div>
        <div class="metric-val">{vmax_king['driver']}</div>
        <div class="metric-sub">{vmax_val} vs Avg</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">FIELD AVERAGE</div>
        <div class="metric-val">+{field_avg:.1f}%</div>
        <div class="metric-sub">Track Evolution</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">SUSPECTS</div>
        <div class="metric-val">{suspect_count}</div>
        <div class="metric-sub">Above Threshold</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# READING DATA & CHART
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)



# 2. The Chart
st.markdown("### FP2 vs. Qualifying: The Truth")

st.markdown("""
<div class="explainer-block">
    <div class="explainer-item">
        <p style="font-size: 1.2rem; line-height: 1.6; color: #ddd; margin: 0;">
            <strong>Steep Ascent:</strong> A sharp line going up means massive time found between practice and quali (Sandbagging).
        </p>        
        </div>
    </div>
    <div class="explainer-item">
        <p style="font-size: 1.2rem; line-height: 1.6; color: #ddd; margin: 0;">
            <strong>Flat Line:</strong> Means the driver was pushing near limits in practice (Honest). 
        </p>    
        </div>
    </div>
    <div class="explainer-item">
        <p style="font-size: 1.2rem; line-height: 1.6; color: #ddd; margin: 0;">
            <strong>Engine Jump:</strong> High Vmax delta indicates turning up the engine mode.
        </p>    
    </div>
</div>
""", unsafe_allow_html=True)


if not df_analysis.empty:
    chart_data = df_analysis.sort_values('speed_gain_pct', ascending=False).head(5)
    
    fig = go.Figure()

    for idx, row in chart_data.iterrows():
        driver = row['driver']
        color = get_team_color(driver)
        
        fig.add_trace(go.Scatter(
            x=['FP2 Pace', 'Qualifying Pace'],
            y=[row['avg_speed_fp2'], row['avg_speed_q']],
            mode='lines+markers',
            name=driver,
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color),
            hovertemplate=f"<b>{driver}</b><br>Speed: %{{y:.1f}} km/h<br>Gain: +{row['speed_gain_pct']:.1f}%<extra></extra>"
        ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.7)', family="Inter"),
        margin=dict(t=30, b=30, l=40, r=40),
        height=350,
        xaxis=dict(showgrid=False, gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        yaxis=dict(title="Avg Speed (km/h)", showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# AI ANALYSIS (NOW BELOW CHART)
# ============================================================================

insight_text = generate_smart_insights(suspects, field_avg)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### AI Analysis")
st.markdown(f"""
<div class="glass-card" style="border-left: 4px solid #E10600;">
    <p style="font-size: 1rem; line-height: 1.6; color: #ddd; margin: 0;">
        {insight_text}
    </p>
</div>
""", unsafe_allow_html=True)