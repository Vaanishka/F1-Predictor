"""
F1 Predictor - Main App
Working Navigation with Radio Buttons
"""

import streamlit as st
from pathlib import Path
import sys
# Define the Root and the App folder
ROOT_PATH = Path(__file__).resolve().parent
APP_PATH = ROOT_PATH / 'streamlit_app'

# ADD BOTH TO PATH: 
# This allows 'import config' (from Root) and 'import backend_bridge' (from streamlit_app)
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))
if str(APP_PATH) not in sys.path:
    sys.path.insert(0, str(APP_PATH))


from config import PAGE_CONFIG, CUSTOM_CSS, COLORS, get_model_metrics

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'


# ============================================================================
# SIDEBAR
# ============================================================================
SIDEBAR_BTN_CSS = f"""
<style>
    /* Unhide the header transparently so we can see the button */
    header {{ 
        visibility: visible !important; 
        background: transparent !important; 
    }}
    
    /* Hide the decoration line and hamburger menu, leaving only the arrow */
    header > div:first-child {{
        background: transparent !important; 
    }}
    #MainMenu {{ 
        visibility: hidden !important; 
    }}
    
    /* STYLE THE ARROW BUTTON */
    [data-testid="collapsedControl"] {{
        display: flex !important;
        align-items: center;
        justify-content: center;
        background-color: {COLORS['primary']} !important; /* F1 Red */
        color: white !important;
        border-radius: 0 0 12px 0 !important; /* Cool corner effect */
        width: 45px !important;
        height: 45px !important;
        top: 0 !important;
        left: 0 !important;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5) !important;
        transition: all 0.3s ease !important;
        z-index: 999999 !important;
    }}
    
    /* Hover Effect */
    [data-testid="collapsedControl"]:hover {{
        background-color: #ff1801 !important;
        width: 50px !important; /* Grow animation */
        box-shadow: 4px 4px 15px rgba(225, 6, 0, 0.4) !important;
    }}
    
    /* Icon Size */
    [data-testid="collapsedControl"] svg {{
        height: 24px !important;
        width: 24px !important;
        stroke-width: 3px !important;
    }}
</style>
"""

# 2. Inject the Button Styles
st.markdown(SIDEBAR_BTN_CSS, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION

    # Logo/Branding
    
with st.sidebar:
# Logo/Branding
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 1.6rem; margin-bottom: 2rem;'>🏎️</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align: center; margin: 1rem 0 2rem 0;'>
        <h1 style='color: {COLORS["primary"]}; font-size: 1.6rem; font-weight: 900; margin: 0;'>
            F1 PREDICTOR
        </h1>
        <p style='color: {COLORS["text_muted"]}; font-size: 0.85rem; margin: 0.5rem 0 0 0;'>
            AI-Powered Forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Header - Updated to COLORS["primary"]
    st.markdown(f"""
    <h3 style='color: {COLORS["primary"]}; font-size: 1.1rem; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 1px;'>
        Navigation
    </h3>
    """, unsafe_allow_html=True)
    
    # Radio Button Navigation
    page = st.radio(
        "Select Mode",
        ["Home", "Predictions", "Sandbagging Lab", "Track Insights"],
        index=["Home", "Predictions", "Sandbagging Lab", "Track Insights"].index(st.session_state.current_page),
        label_visibility="collapsed"
    )
    
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.rerun()

# ============================================================================
# PAGE ROUTING
# ============================================================================

if st.session_state.current_page == "Home":
    st.markdown("""
        <div style='text-align: center; padding-top: 1rem;'>
            <h1 class='hero-title'>F1 PREDICTOR</h1>
            <p class='hero-subtitle'>AI-Powered Race Analytics & Podium Forecasting</p>
        </div>
        """, unsafe_allow_html=True)
    

# --- VIDEO SECTION (Cinematic Loop - No Controls) ---
    
    # 1. Force Video Height & Hide Controls via CSS
    st.markdown("""
    <style>
        /* Target the video player */
        [data-testid="stVideo"] {
            width: 100%;
        }
        
        video {
            height: 720px !important;       /* Adjust height as needed */
            width: 100% !important;         /* Force full width */
            object-fit: cover !important;   /* Crops top/bottom to fit */
            border-radius: 15px !important; /* Rounded corners */
            pointer-events: none;           /* Disables clicking/pausing */
        }
        
        /* HIDE ALL VIDEO CONTROLS (Chrome/Safari/Edge) */
        video::-webkit-media-controls {
            display: none !important;
        }
        video::-webkit-media-controls-enclosure {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # 2. Render Video
    video_path = Path(__file__).parent / 'streamlit_app' / 'assets' / 'animation.mp4'
    if video_path.exists():
        # Render directly (no columns) for full width
        st.video(str(video_path), autoplay=True, loop=True, muted=True)
    else:
        st.info("🎬 Add animation.mp4 to streamlit_app/assets/")


    # Get REAL metrics from database
    metrics = get_model_metrics()
    
    # Metrics Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; padding: 6px;">
            <p style="color: {COLORS["text_muted"]}; font-size: 0.85rem; margin-bottom: 8px; margin-top: 2px; text-transform: uppercase; letter-spacing: 1px;">
                ACCURACY
            </p>
            <h3 style="color: #FFFFFF; font-size: 2.5rem; margin: 0; font-weight: 900;">
                {metrics['accuracy']}
            </h3>
            <p style="color: {COLORS["text_muted"]}; font-size: 0.8rem; margin-top: 8px;">
                Predictions within ±2 positions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; padding: 6px;">
            <p style="color: {COLORS["text_muted"]}; font-size: 0.85rem; margin-bottom: 8px; margin-top: 2px; text-transform: uppercase; letter-spacing: 1px;">
                DATA SOURCES
            </p>
            <h3 style="color: #FFFFFF; font-size: 2.5rem; margin: 0; font-weight: 900;">
                {metrics['data_points']}
            </h3>
            <p style="color: {COLORS["text_muted"]}; font-size: 0.8rem; margin-top: 8px;">
                Historical race sessions analyzed
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; padding: 6px;">
            <p style="color: {COLORS["text_muted"]}; font-size: 0.85rem; margin-bottom: 8px; margin-top: 2px; text-transform: uppercase; letter-spacing: 1px;">
                LAST UPDATED
            </p>
            <h3 style="color: #FFFFFF; font-size: 2.5rem; margin: 0; font-weight: 900;">
                {metrics['last_updated']}
            </h3>
            <p style="color: {COLORS["text_muted"]}; font-size: 0.8rem; margin-top: 8px;">
                Predictions updated after qualifying
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; padding: 6px;">
            <p style="color: {COLORS["text_muted"]}; font-size: 0.85rem; margin-bottom: 8px; margin-top: 2px; text-transform: uppercase; letter-spacing: 1px;">
                CONFIDENCE
            </p>
            <h3 style="color: #FFFFFF; font-size: 2.5rem; margin: 0; font-weight: 900;">
                {metrics['confidence']}
            </h3>
            <p style="color: {COLORS["text_muted"]}; font-size: 0.8rem; margin-top: 8px;">
                Model status
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Methodology Section
    st.markdown(f"""
    <div class="glass-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
            <div style="color: {COLORS["primary"]}; font-size: 1.2rem; border: 2px solid {COLORS["primary"]}; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: 700;">
                i
            </div>
            <h3 style="margin: 0; font-size: 1.5rem; font-weight: 700;">
                Prediction Methodology
            </h3>
        </div>
        <p style="color: {COLORS["text_muted"]}; font-size: 1rem; line-height: 1.6; margin-bottom: 25px;">
            Our predictions are calculated by analyzing the delta between Qualifying performance and FP2 race simulations, weighted by historical track data.
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div style="background-color: {COLORS["card_bg"]}; border-radius: 12px; padding: 20px; border: 1px solid {COLORS["border"]};">
                <div style="background-color: rgba(225, 6, 0, 0.15); width: 48px; height: 48px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">
                    <span style="color: {COLORS["primary"]}; font-size: 1.5rem;">⏱</span>
                </div>
                <h4 style="color: #FFFFFF; margin: 0 0 8px 0; font-size: 1.1rem; font-weight: 700;">
                    Qualifying Pace
                </h4>
                <p style="color: {COLORS["text_muted"]}; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                    Raw single-lap speed under optimal conditions
                </p>      
            </div>
            <div style="background-color: {COLORS["card_bg"]}; border-radius: 12px; padding: 20px; border: 1px solid {COLORS["border"]};">
                <div style="background-color: rgba(225, 6, 0, 0.15); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">
                    <span style="color: {COLORS["primary"]}; font-size: 1rem;">▀▄▀▄</span>
                </div>
                <h4 style="color: #FFFFFF; margin: 0 0 8px 0; font-size: 1.1rem; font-weight: 700;">
                    FP2 Long Runs
                </h4>
                <p style="color: {COLORS["text_muted"]}; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                    Race simulation with high fuel loads
                </p>
            </div>
            <div style="background-color: {COLORS["card_bg"]}; border-radius: 12px; padding: 20px; border: 1px solid {COLORS["border"]};">
                <div style="background-color: rgba(225, 6, 0, 0.15); width: 48px; height: 48px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">
                    <span style="color: {COLORS["primary"]}; font-size: 1.5rem;">⚛</span>
                </div>
                <h4 style="color: #FFFFFF; margin: 0 0 8px 0; font-size: 1.1rem; font-weight: 700;">
                    ML Analysis
                </h4>
                <p style="color: {COLORS["text_muted"]}; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                    Pattern recognition using ML techniques
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-card" style="text-align: center; padding: 2.5rem;">
        <h3 style="color: #FFFFFF; font-size: 1.5rem; margin-bottom: 1rem;">
            Ready to predict the next race?
        </h3>
        <p style="color: {COLORS["text_muted"]}; font-size: 1rem;">
            Select <strong style="color: {COLORS["primary"]};">Predictions</strong> from the sidebar to get started!
        </p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == "Predictions":
    with open(APP_PATH / 'pages' / 'predictions.py', 'r', encoding='utf-8') as f:
        exec(f.read())

elif st.session_state.current_page == "Sandbagging Lab":
    with open(APP_PATH / 'pages' / 'sandbagging.py', 'r', encoding='utf-8') as f:
        exec(f.read())

elif st.session_state.current_page == "Track Insights":
    track_page_path = APP_PATH / 'pages' / 'track_insights.py'
    
    if track_page_path.exists():
        with open(track_page_path, 'r', encoding='utf-8') as f:
            exec(f.read())
    else:
        st.error(f"❌ File not found: {track_page_path}")