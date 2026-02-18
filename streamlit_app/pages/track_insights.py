import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
import re
from pathlib import Path

# --- PATH SETUP (Fail-Safe) ---
# We look for the 'assets' folder in multiple common locations to avoid errors
current_file = Path(__file__).resolve()
possible_asset_dirs = [
    current_file.parent.parent / "assets",       # ../assets/
    Path.cwd() / "streamlit_app" / "assets",     # Relative to root execution
    Path("assets")                               # Absolute fallback
]

ASSETS_DIR = None
for p in possible_asset_dirs:
    if p.exists():
        ASSETS_DIR = p
        break

# Fallback if not found
if ASSETS_DIR is None:
    ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
# Import backend bridge
import sys
root_path = Path(__file__).resolve().parent.parent

if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
try:
    # Import the NEW function from backend_bridge
    from backend_bridge import get_track_radar_metrics, get_historical_podiums, get_track_list, get_track_text_insights
except ImportError:
    st.error("⚠️ Backend Bridge not found. Please check your directory structure.")
    st.stop()




# --- PAGE CONFIG ---
st.set_page_config(page_title="Track Insights", page_icon="🏎️", layout="wide")

# --- CUSTOM CSS (Glassmorphism & Expanders) ---
st.markdown("""
<style>
    /* Global Settings */
    .stApp { background-color: #050505; color: white; }
    
    /* REMOVE DEFAULT STREAMLIT PADDING/MARGINS FOR HTML BLOCKS */
    div.stMarkdown { margin-bottom: 0px; }

    /* GLASS EXPANDER (The Container) */
    details.glass-expander {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 12px;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    details.glass-expander:hover {
        background: rgba(255, 255, 255, 0.05);
    }

    details.glass-expander[open] {
        background: rgba(0, 0, 0, 0.4);
        border-color: #00F0FF; /* Highlight border when open */
    }

    /* SUMMARY (The Header / Clickable Area) */
    summary.glass-header {
        display: flex;
        align-items: center;
        padding: 24px 30px;
        cursor: pointer;
        list-style: none; /* Hide default triangle */
        position: relative;
    }

    /* Hide default triangle in Webkit */
    summary.glass-header::-webkit-details-marker {
        display: none;
    }

    /* ICON CONTAINER */
    .icon-box {
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0, 240, 255, 0.1);
        border-radius: 8px; /* Slightly rounded square as per screenshot */
        margin-right: 16px;
        flex-shrink: 0;
    }
    
    .icon-box svg {
        stroke: #00F0FF; /* Cyan Neon */
        width: 20px;
        height: 20px;
    }

    /* TEXT STYLING */
    .header-text {
        flex-grow: 1;
    }
    
    .red-subhead { 
        color: #FF1801; 
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 1rem; 
        letter-spacing: 2px; 
        margin-bottom: 5px; 
        text-transform: uppercase;
    }      
            
    .header-title {
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    

    .header-subtitle {
        color: #888; font-size: 0.95rem; margin-top: 0.5px;
    }

    /* ARROW ICON (Right Side) */
    .arrow-icon {
        color: #555;
        font-size: 12px;
        transition: transform 0.3s ease;
        margin-left: auto;
    }

    details[open] summary .arrow-icon {
        transform: rotate(90deg);
        color: #00F0FF;
    }

    /* CONTENT (The Expanded Part) */
    .glass-content {
        padding: 0 24px 24px 80px; /* Indented to align with text */
        color: #cccccc;
        font-size: 1.2rem;
        line-height: 1.7;
        border-top: 1px solid rgba(255,255,255,0.05);
        padding-top: 10px;
        margin-top: 0;
    }
    
    /* ANIMATION */
    details[open] .glass-content {
        animation: fadeIn 0.4s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-5px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)



# --- STATE MANAGEMENT ---
if 'selected_track' not in st.session_state:
    st.session_state.selected_track = "Bahrain Grand Prix"

tracks = get_track_list()
if not tracks:
    st.error("No tracks found in database.")
    st.stop()

# --- HEADER SECTION ---
c1, c2 = st.columns([3, 1])

with c2:
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    selected = st.selectbox(
        "Select Track", 
        tracks, 
        index=tracks.index(st.session_state.selected_track) if st.session_state.selected_track in tracks else 0,
        label_visibility="collapsed"
    )
    if selected != st.session_state.selected_track:
        st.session_state.selected_track = selected
        st.rerun()

with c1:
    # 1. Red Subheader
    st.markdown(f"""
<div style='margin-bottom: 2rem;'>
    <div style='display: flex; align-items: center;'>
        <div style='width: 40px; height: 2px; background-color: #FF1801;'></div>
        <span style='color: #FF1801; font-size: 1rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;'>Track Insights</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Main Title
    display_name = st.session_state.selected_track.replace(" Grand Prix", "").upper()
    st.markdown(f"<h1>{display_name}<span class='track-subtitle'>, GP</span></h1>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- MAIN CONTENT ---
col_L, col_R = st.columns([1.3, 1], gap="large")

with col_L:
    # 1. MAP VISUALIZATION (Robust Path Finding)
    st.markdown("<h5>CIRCUIT LAYOUT</h5>", unsafe_allow_html=True)
    
    clean_name = re.sub(r'[^a-z0-9]', '', st.session_state.selected_track.replace(" Grand Prix", "").lower())
    
    # Try to find the map in multiple places
    map_found = False
    map_path = ASSETS_DIR / "track_maps" / f"{clean_name}.svg"
    
    with st.container(border=True):
        if map_path.exists():
            with open(map_path, "r") as f:
                svg_content = f.read()
            # Force width to 100%
            svg_content = svg_content.replace('<svg ', '<svg style="width: 100%; height: auto; max-height: 380px;" ')
            st.markdown(f"<div style='text-align:center; padding: 20px;'>{svg_content}</div>", unsafe_allow_html=True)
            map_found = True
        else:
            st.warning(f"Map file missing: {clean_name}.svg")
            st.caption(f"Checked: {map_path}")

    # 2. RADAR CHART
    st.markdown("<h5 style='margin-top: 20px;'>TRACK DEMANDS</h5>", unsafe_allow_html=True)
    metrics = get_track_radar_metrics(st.session_state.selected_track)
    
    if metrics:
        with st.container(border=True):
            fig = go.Figure(data=go.Scatterpolar(
                r=list(metrics.values()),
                theta=list(metrics.keys()),
                fill='toself',
                line_color='#00F0FF', # Cyan
                fillcolor='rgba(0, 240, 255, 0.1)',
                marker=dict(size=0)
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, linecolor='#333', gridcolor='#222'),
                    angularaxis=dict(tickfont=dict(color='#888', size=10), linecolor='#333'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=20, b=20),
                height=300,
                showlegend=False,
                dragmode=False
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col_R:
    st.markdown("<h5>TECHNICAL ANALYSIS</h5>", unsafe_allow_html=True)

    # 1. Fetch Data
    track_info = get_track_text_insights(st.session_state.selected_track)
    
    # Fallback Data
    if not track_info:
        track_info = {
            "length_speed": {"tagline": "Data Pending", "description": "Update track_insights.json"},
            "corner_types": {"tagline": "Data Pending", "description": "Update track_insights.json"},
            "evolution_grip": {"tagline": "Data Pending", "description": "Update track_insights.json"},
            "drs_zones": {"tagline": "Data Pending", "description": "Update track_insights.json"}
        }

    # 2. DEFINE THE SVG ICONS (Exact match to screenshot)
    SVGS = {
        "bolt":  """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path></svg>""",
        "curve": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/><path d="M15 12H3"/></svg>""",
        "drop":  """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path></svg>""",
        "wind":  """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/></svg>"""
    }

    # 3. HELPER FUNCTION TO RENDER HTML CARD
    def render_html_card(icon_svg, title, category_key, default_open=False):
        data = track_info.get(category_key, {})
        tagline = data.get('tagline', 'No Data')
        desc = data.get('description', 'No Data')
        open_attr = "open" if default_open else ""
        
        html = f"""
        <details class="glass-expander" {open_attr}>
            <summary class="glass-header">
                <div class="icon-box">{icon_svg}</div>
                <div class="text-container">
                    <div class="header-title">{title}</div>
                    <div class="header-subtitle">{tagline}</div>
                </div>
                <div class="arrow-icon">➜</div>
            </summary>
            <div class="glass-content">
                {desc}
            </div>
        </details>
        """
        st.markdown(html, unsafe_allow_html=True)

    # 4. RENDER THE CARDS
    render_html_card(SVGS['bolt'], "LENGTH & SPEED", "length_speed", default_open=True)
    render_html_card(SVGS['curve'], "CORNER TYPES", "corner_types")
    render_html_card(SVGS['drop'], "TRACK EVOLUTION & GRIP", "evolution_grip")
    render_html_card(SVGS['wind'], "DRS ZONES", "drs_zones")

    # 5. HISTORICAL PODIUMS (Keep existing Plotly logic below this)
    st.markdown("<h5 style='margin-top: 30px;'>CAREER PODIUMS</h5>", unsafe_allow_html=True)
    
    # ... [Keep your existing Hist DF code here] ...
    hist_df = get_historical_podiums(st.session_state.selected_track)
    if not hist_df.empty:
        # Sort and Color
        hist_df = hist_df.head(5).sort_values('Podiums', ascending=True)
        # Using simple color map logic
        driver_colors = {"VER": "#0600EF", "HAM": "#00D2BE", "NOR": "#FF8000", "LEC": "#DC0000", "SAI": "#DC0000", "RUS": "#00D2BE", "PER": "#0600EF", "PIA": "#FF8000", "ALO": "#006F62"}
        colors = [driver_colors.get(d, "#444") for d in hist_df['Driver']]
        
        fig_bar = go.Figure(go.Bar(
            x=hist_df['Podiums'],
            y=hist_df['Driver'],
            orientation='h',
            marker=dict(color=colors, line=dict(width=0)),
            text=hist_df['Podiums'],
            textposition='inside',
            textfont=dict(color='white', weight='bold')
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(color='white', size=12)),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

        
# --- WINNING EDGE (BOTTOM) ---
metrics = get_track_radar_metrics(st.session_state.selected_track)
edge_text = "Ideally, a car with high efficiency. The straights reward low drag, but the heavy braking zones demand a stable platform."

if metrics:
    if metrics.get('Downforce', 5) > 8:
        edge_text = "Maximum downforce is non-negotiable here. Teams will sacrifice top speed for cornering grip. Qualifying position is critical as overtaking is difficult."
    elif metrics.get('Top Speed', 5) > 8:
        edge_text = "This track favors cars with **high aerodynamic efficiency** and strong traction out of slow corners. Low-drag configurations are essential for competitive straight-line speed."

