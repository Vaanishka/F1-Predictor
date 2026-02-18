"""
F1 Predictor - Backend Bridge
Interface to execute backend scripts and query database
"""

import json
import pandas as pd
import subprocess
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys
import os
import math
import random

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / 'script'
DATABASE_DIR = BASE_DIR / 'database'
DB_FILE = DATABASE_DIR / 'f1_predictions.db'

DNA_PATH = BASE_DIR / "coefficient" / "track_dna.json"
PODIUMS_PATH = BASE_DIR / "coefficient" / "historical_podiums.json"

# Add scripts directory to path
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(BASE_DIR / 'streamlit_app'))

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

def run_extract_race(race_name, year):
    """
    Execute extract_race.py script
    Returns: (success: bool, message: str)
    """
    try:
        script_path = SCRIPTS_DIR / 'extract_race.py'
        
        if not script_path.exists():
            return False, f"Script not found: {script_path}"
        
        # Set up environment for subprocess
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{BASE_DIR / 'streamlit_app'}{os.pathsep}{BASE_DIR / 'script'}{os.pathsep}{env.get('PYTHONPATH', '')}"
        
        # Run script with proper environment
        result = subprocess.run(
            ['python', str(script_path), '--race', race_name, '--year', str(year)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            env=env
        )
        
        if result.returncode == 0:
            return True, "Features extracted successfully"
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, f"Extraction failed: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, "Script timed out (5 minutes)"
    except Exception as e:
        return False, f"Error running extract_race: {str(e)}"


def run_predict_race(race_name, year):
    """
    Execute predict_race.py script
    Returns: (success: bool, message: str)
    """
    try:
        script_path = SCRIPTS_DIR / 'predict_race.py'
        
        if not script_path.exists():
            return False, f"Script not found: {script_path}"
        
        # Run script
        result = subprocess.run(
            ['python', str(script_path), '--race', race_name, '--year', str(year)],
            capture_output=True,
            text=True,
            timeout=180  # 3 min timeout
        )
        
        if result.returncode == 0:
            return True, "Predictions generated successfully"
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, f"Prediction failed: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, "Script timed out (3 minutes)"
    except Exception as e:
        return False, f"Error running predict_race: {str(e)}"


def run_persist_results(race_name, year):
    """
    Execute persist_results.py script
    Returns: (success: bool, message: str)
    """
    try:
        script_path = SCRIPTS_DIR / 'persist_results.py'
        
        if not script_path.exists():
            return False, f"Script not found: {script_path}"
        
        # Run script
        result = subprocess.run(
            ['python', str(script_path), '--race', race_name, '--year', str(year)],
            capture_output=True,
            text=True,
            timeout=120  # 2 min timeout
        )
        
        if result.returncode == 0:
            return True, "Results persisted successfully"
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, f"Persistence failed: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, "Script timed out (2 minutes)"
    except Exception as e:
        return False, f"Error running persist_results: {str(e)}"


# ============================================================================
# DATABASE QUERIES
# ============================================================================

def get_db_connection():
    """Get database connection"""
    if not DB_FILE.exists():
        raise FileNotFoundError(f"Database not found: {DB_FILE}")
    return sqlite3.connect(DB_FILE)


def check_race_exists(race_name, year):
    """
    Check if race exists in database
    Returns: (exists: bool, race_id: int, status: str)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT race_id, status 
            FROM races 
            WHERE year = ? AND race_name = ?
        """, (year, race_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return True, result[0], result[1]
        else:
            return False, None, None
            
    except Exception as e:
        print(f"Error checking race: {e}")
        return False, None, None


def get_race_predictions(race_id):
    """
    Get predictions for a race
    Returns: DataFrame with predictions
    """
    try:
        conn = get_db_connection()
        
        query = """
            SELECT 
                driver,
                grid_position,
                predicted_position,
                predicted_position_int,
                expected_change,
                confidence_68_lower,
                confidence_68_upper,
                model_version
            FROM predictions
            WHERE race_id = ?
            ORDER BY predicted_position ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(race_id,))
        conn.close()
        
        return df
        
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return pd.DataFrame()


def get_race_features(race_id):
    """
    Get race features from database
    Returns: DataFrame with features
    """
    try:
        conn = get_db_connection()
        
        query = """
            SELECT 
                driver,
                session_type,
                grid_position,
                avg_speed,
                features_json
            FROM race_features
            WHERE race_id = ?
            ORDER BY driver, session_type
        """
        
        df = pd.read_sql_query(query, conn, params=(race_id,))
        conn.close()
        
        # Parse JSON features
        if not df.empty:
            df['features'] = df['features_json'].apply(json.loads)
        
        return df
        
    except Exception as e:
        print(f"Error getting features: {e}")
        return pd.DataFrame()


def get_all_races(year=None):
    """
    Get all races from database
    Returns: DataFrame with race info
    """
    try:
        conn = get_db_connection()
        
        if year:
            query = """
                SELECT race_id, year, race_name, circuit, status, race_date
                FROM races
                WHERE year = ?
                ORDER BY race_date DESC
            """
            df = pd.read_sql_query(query, conn, params=(year,))
        else:
            query = """
                SELECT race_id, year, race_name, circuit, status, race_date
                FROM races
                ORDER BY year DESC, race_date DESC
            """
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"Error getting races: {e}")
        return pd.DataFrame()


def get_actual_results(race_id):
    """
    Get actual race results
    Returns: DataFrame with results
    """
    try:
        conn = get_db_connection()
        
        query = """
            SELECT 
                driver,
                final_position,
                grid_position,
                points_scored,
                status
            FROM actual_results
            WHERE race_id = ?
            ORDER BY final_position ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(race_id,))
        conn.close()
        
        return df
        
    except Exception as e:
        print(f"Error getting results: {e}")
        return pd.DataFrame()


"""
F1 Predictor - Backend Bridge
Interface to execute backend scripts and query database
"""

import sqlite3
import pandas as pd
import json
import sys
import random
import math
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / 'script'
DATABASE_DIR = BASE_DIR / 'database'
DB_FILE = DATABASE_DIR / 'f1_predictions.db'

# Add scripts directory to path
sys.path.insert(0, str(SCRIPTS_DIR))

# ============================================================================
# DATABASE HELPERS
# ============================================================================

def get_db_connection():
    """Get database connection"""
    if not DB_FILE.exists():
        return None
    return sqlite3.connect(DB_FILE)

def check_race_exists(race_name, year):
    """Check if race exists in database"""
    try:
        conn = get_db_connection()
        if not conn: return False, None, None
        
        cursor = conn.cursor()
        cursor.execute("SELECT race_id, status FROM races WHERE year = ? AND race_name = ?", (year, race_name))
        result = cursor.fetchone()
        conn.close()
        return (True, result[0], result[1]) if result else (False, None, None)
    except Exception as e:
        print(f"Error checking race: {e}")
        return False, None, None
"""
F1 Predictor - Backend Bridge
Interface to execute backend scripts and query database
"""

import sqlite3
import pandas as pd
import json
import sys
import random
import math
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / 'script'
DATABASE_DIR = BASE_DIR / 'database'
DB_FILE = DATABASE_DIR / 'f1_predictions.db'

# Add scripts directory to path
sys.path.insert(0, str(SCRIPTS_DIR))

# ============================================================================
# DATABASE HELPERS
# ============================================================================

def get_db_connection():
    """Get database connection"""
    if not DB_FILE.exists():
        return None
    return sqlite3.connect(DB_FILE)

def check_race_exists(race_name, year):
    """Check if race exists in database"""
    try:
        conn = get_db_connection()
        if not conn: return False, None, None
        
        cursor = conn.cursor()
        cursor.execute("SELECT race_id, status FROM races WHERE year = ? AND race_name = ?", (year, race_name))
        result = cursor.fetchone()
        conn.close()
        return (True, result[0], result[1]) if result else (False, None, None)
    except Exception as e:
        print(f"Error checking race: {e}")
        return False, None, None

# ============================================================================
# SANDBAGGING LOGIC
# ============================================================================

def get_sandbagging_data(race_id):
    """Get FP2 vs Qualifying data for sandbagging analysis"""
    try:
        conn = get_db_connection()
        if not conn: return pd.DataFrame()

        # We select specific columns but rely heavily on features_json 
        # as it contains the processed metrics like Max_Speed and Consistency
        query = """
            SELECT driver, session_type, avg_speed, features_json
            FROM race_features
            WHERE race_id = ? AND session_type IN ('FP2', 'Q')
            ORDER BY driver, session_type
        """
        df = pd.read_sql_query(query, conn, params=(race_id,))
        conn.close()
        
        if df.empty: return pd.DataFrame()

        # Parse JSON features safely
        df['features'] = df['features_json'].apply(lambda x: json.loads(x) if x and isinstance(x, str) else {})
        
        # Split into FP2 and Q datasets
        fp2 = df[df['session_type'] == 'FP2'][['driver', 'avg_speed', 'features']]
        q = df[df['session_type'] == 'Q'][['driver', 'avg_speed', 'features']]
        
        if fp2.empty or q.empty: return pd.DataFrame()
            
        # Merge dataframes on driver to compare sessions side-by-side
        comparison = pd.merge(fp2, q, on='driver', suffixes=('_fp2', '_q'))
        return comparison
    except Exception as e:
        print(f"Error getting sandbagging data: {e}")
        return pd.DataFrame()

def process_sandbagging_metrics(df):
    """
    Calculates Sandbagging Index based on:
    1. Lap Time/Speed Gain (Quali vs FP2) - Primary Indicator
    2. Top Speed (Vmax) Gain - Engine Mode Indicator
    3. FP2 Consistency - Supporting Indicator (Hiding true pace in race sims)
    """
    if df.empty: return df

    results = []
    for _, row in df.iterrows():
        # Extract features safely
        fp2_feat = row['features_fp2'] if isinstance(row['features_fp2'], dict) else json.loads(row['features_fp2'] or '{}')
        q_feat = row['features_q'] if isinstance(row['features_q'], dict) else json.loads(row['features_q'] or '{}')
        
        # 1. Base Speed Metrics (From SQL columns)
        avg_speed_fp2 = row['avg_speed_fp2']
        avg_speed_q = row['avg_speed_q']
        
        # 2. Advanced Feature Extraction (From JSON)
        # Max Speed is crucial for detecting engine mode changes
        vmax_fp2 = fp2_feat.get('Max_Speed', 0)
        vmax_q = q_feat.get('Max_Speed', 0)
        
        # Consistency helps identify if FP2 was a strict program (sandbagging) vs just messy driving
        consistency = fp2_feat.get('Driver_Consistency', 0)
        
        # True Pace (if available in your ML model features)
        true_pace_fp2 = fp2_feat.get('Avg_True_Pace', 0)
        true_pace_q = q_feat.get('Avg_True_Pace', 0)

        # --- CALCULATIONS ---

        # A. Speed Gain % (The "Jump")
        if avg_speed_fp2 > 0:
            speed_gain_pct = ((avg_speed_q - avg_speed_fp2) / avg_speed_fp2) * 100
        else:
            speed_gain_pct = 0
            
        # B. Vmax Gain % (Engine Mode Proxy)
        # If Vmax jumps significantly in Q, they were likely detuned in FP2
        if vmax_fp2 > 0:
            vmax_gain_pct = ((vmax_q - vmax_fp2) / vmax_fp2) * 100
        else:
            vmax_gain_pct = 0

        # C. True Pace Delta (Optional refinement if True Pace exists)
        true_pace_delta = true_pace_q - true_pace_fp2

        results.append({
            'driver': row['driver'],
            'speed_gain_pct': speed_gain_pct,
            'vmax_gain_pct': vmax_gain_pct,
            'consistency': consistency,
            'true_pace_delta': true_pace_delta,
            'avg_speed_fp2': avg_speed_fp2, # Kept for UI display
            'avg_speed_q': avg_speed_q      # Kept for UI display
        })
    
    result_df = pd.DataFrame(results)
    
    if result_df.empty: return result_df

    # --- NORMALIZATION & SCORING ---
    
    # Calculate Field Averages (To see who jumped *more* than the track evolution)
    mean_speed_gain = result_df['speed_gain_pct'].mean()
    mean_vmax_gain = result_df['vmax_gain_pct'].mean()
    
    # Calculate Relative Gains (Driver Gain - Field Average Gain)
    result_df['relative_speed_gain'] = result_df['speed_gain_pct'] - mean_speed_gain
    result_df['relative_vmax_gain'] = result_df['vmax_gain_pct'] - mean_vmax_gain
    
    # Sandbagging Index Formula
    # WEIGHTS ADJUSTED:
    # - Relative Speed Gain (1.0): The most direct evidence.
    # - Relative Vmax Gain (0.5): Evidence of engine mode masking.
    # - Consistency (0.5): Supporting evidence. Reduced from 2.0 to prevent false positives.
    
    result_df['sandbagging_index'] = (
        (result_df['relative_speed_gain'] * 1.0) + 
        (result_df['relative_vmax_gain'] * 0.5) + 
        (result_df['consistency'] * 0.5)
    )
    
    # Normalize Index for cleaner UI handling (Optional, but helps with thresholds)
    # This keeps values generally around 0-5 range
    
    return result_df.sort_values('sandbagging_index', ascending=False)

def get_radar_coordinates(df):
    """
    Generates X/Y coordinates for the CSS Radar.
    Top suspects get specific emphasis (inner/mid rings), others are random blips.
    """
    if df.empty: return df
    
    radar_data = []
    
    # STATISTICAL THRESHOLD FOR "RED DOTS"
    # We use quantile(0.85) to find the top 15% of suspicious activity.
    threshold = df['sandbagging_index'].quantile(0.85)
    
    for idx, row in df.iterrows():
        # LOGIC: Only flag as suspect if they meet the threshold AND the index is positive
        # This prevents forcing red dots if everyone is behaving normally (negative index).
        # We also ensure the relative speed gain is positive (they actually jumped in performance).
        is_suspect = (
            (row['sandbagging_index'] >= threshold) and 
            (row['sandbagging_index'] > 0) and
            (row['relative_speed_gain'] > 0)
        )
        
        # Generate coordinates (Random angle 0-360 deg)
        angle = random.uniform(0, 2 * math.pi)
        
        if is_suspect:
            # Suspects = Closer to center (bullseye) - Detection Zone
            distance = random.uniform(15, 35)
        else:
            # Honest = Further out (safe zone)
            distance = random.uniform(45, 80)
            
        # Convert to CSS 'top' and 'left' percentages
        # 50% is center.
        top = 50 + (distance * math.sin(angle) / 2)
        left = 50 + (distance * math.cos(angle) / 2)
        
        radar_data.append({
            'driver': row['driver'],
            'radar_top': f"{top:.1f}%",
            'radar_left': f"{left:.1f}%",
            'is_suspect': is_suspect,
            # Pass formatted gain for the UI tooltip/list
            'sandbagging_pct': row['speed_gain_pct'] 
        })
        
    # Merge back to original DF
    radar_df = pd.DataFrame(radar_data)
    return pd.merge(df, radar_df, on='driver')

def get_sandbagging_analysis(race_id):
    """
    Main entry point called by the UI.
    Orchestrates data fetch -> processing -> coordinate generation.
    """
    raw_data = get_sandbagging_data(race_id)
    if raw_data.empty: return pd.DataFrame()
    
    processed = process_sandbagging_metrics(raw_data)
    final_data = get_radar_coordinates(processed)
    return final_data

# ============================================================================
# TRACK ANALYSIS LOGIC
# ============================================================================

def scale_value(val, min_v, max_v, inverse=False):
    """
    Scales a value between min_v and max_v to a 1-10 score.
    Clamps values outside the range.
    """
    if val is None: return 5
    
    # Clamp inputs
    val = max(min_v, min(val, max_v))
    
    # Normalize 0-1
    norm = (val - min_v) / (max_v - min_v)
    
    if inverse:
        norm = 1 - norm
        
    return round(norm * 9 + 1, 1) # Scale to 1-10

# --- CORE FUNCTIONS ---

def get_track_list():
    """Returns a sorted list of track names from the DNA file."""
    try:
        with open(DNA_PATH, 'r') as f:
            data = json.load(f)
        return sorted([k for k in data.keys() if k != "_metadata"])
    except FileNotFoundError:
        return []

def get_track_radar_metrics(track_name):
    """
    Reads engineering DNA and returns 1-10 scores for UI Radar Chart.
    """
    try:
        with open(DNA_PATH, 'r') as f:
            full_data = json.load(f)
            
        track_data = full_data.get(track_name)
        if not track_data:
            return {}

        # 1. Top Speed (Max: Monza 264, Min: Monaco 165)
        speed_score = scale_value(track_data.get('avg_speed_kph'), 160, 265)

        # 2. Downforce (Based on Corner Ratio. High Ratio = High Downforce)
        downforce_score = scale_value(track_data.get('corner_to_straight_ratio'), 0.30, 0.90)

        # 3. Overtaking Difficulty (Based on Overtake Factor)
        overtake_diff = scale_value(track_data.get('overtake_factor'), 0.25, 0.95)

        # 4. Tyre Stress (Based on Tire Wear Rate)
        tyre_stress = scale_value(track_data.get('tire_wear_rate'), 0.70, 1.30)

        # 5. Braking (Inferred from Penalty per kg + Elevation)
        # Heavier penalty usually implies more stop-start nature
        braking = scale_value(track_data.get('penalty_per_kg'), 0.029, 0.040)

        return {
            "Top Speed": speed_score,
            "Downforce": downforce_score,
            "Overtaking Diff": overtake_diff,
            "Tyre Stress": tyre_stress,
            "Braking": braking
        }
    except Exception as e:
        print(f"Error calculating metrics for {track_name}: {e}")
        return {k: 5 for k in ["Top Speed", "Downforce", "Overtaking Diff", "Tyre Stress", "Braking"]}

def get_track_explanations(track_name):
    """
    Returns text summaries based on the metrics. 
    (In a real app, you might want to move these text strings to the JSON file)
    """
    metrics = get_track_radar_metrics(track_name)
    
    # Dynamic text generation based on the scores
    explanations = []
    
    # Speed
    if metrics['Top Speed'] > 8:
        explanations.append((" Length & Speed", "High-speed circuit. Requires low-drag setup and high engine power."))
    elif metrics['Top Speed'] < 4:
        explanations.append((" Length & Speed", "Tight and twisty. Top speed is less critical than agility."))
    else:
        explanations.append((" Length & Speed", "Balanced circuit requiring a mix of speed and cornering ability."))

    # Downforce
    if metrics['Downforce'] > 7:
        explanations.append(("Aerodynamics", "Maximum downforce required to stick the car to the road in corners."))
    else:
        explanations.append(("Aerodynamics", "Efficiency is key. Wings are trimmed to reduce drag on straights."))
        
    return explanations

def get_historical_podiums(track_name):
    """
    Reads the cached historical_podiums.json and returns a DataFrame.
    """
    try:
        with open(PODIUMS_PATH, 'r') as f:
            data = json.load(f)
            
        track_stats = data.get(track_name, {})
        
        # Convert dict {'VER': 5, 'HAM': 3} to DataFrame for Altair/Streamlit
        if not track_stats:
            return pd.DataFrame(columns=['Driver', 'Podiums'])
            
        df = pd.DataFrame(list(track_stats.items()), columns=['Driver', 'Podiums'])
        return df.sort_values('Podiums', ascending=False)
        
    except FileNotFoundError:
        return pd.DataFrame(columns=['Driver', 'Podiums'])


# --- NEW SECTION: TEXT INSIGHTS HANDLER ---
def get_text_insights_database():
    """Loads the qualitative data from assets/track_insights.json"""
    # Robust path finding relative to THIS file (backend_bridge.py)
    # Assumes structure: streamlit_app/backend_bridge.py  and  streamlit_app/assets/track_insights.json
    current_dir = Path(__file__).parent
    json_path = current_dir / "assets" / "track_insights.json"
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Error: track_insights.json is not valid JSON")
        return {}

# ==============================================================================
# NEW: TEXT INSIGHTS LOADER (Add to bottom of backend_bridge.py)
# ==============================================================================
import json
from pathlib import Path

def get_track_text_insights(track_name):
    """
    Robustly loads track text data from track_insights.json
    Handles missing 'Grand Prix' suffixes automatically.
    """
    # 1. Define Path to JSON (Relative to this file)
    current_dir = Path(__file__).parent
    json_path = current_dir / "assets" / "track_insights.json"
    
    # 2. Load Database
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None # Graceful fail if file missing

    # 3. Fuzzy Matching Logic
    # Try exact match
    if track_name in db:
        return db[track_name]
    
    # Try adding "Grand Prix"
    if f"{track_name} Grand Prix" in db:
        return db[f"{track_name} Grand Prix"]
        
    # Try removing "Grand Prix"
    simple_name = track_name.replace(" Grand Prix", "").strip()
    if simple_name in db:
        return db[simple_name]

    # No match found
    return None


def get_driver_podiums_at_track(circuit_name):
    """
    Get career podium counts at specific track for all drivers
    Returns: DataFrame with driver podium counts
    """
    try:
        conn = get_db_connection()
        
        query = """
            SELECT 
                ar.driver,
                COUNT(CASE WHEN ar.final_position <= 3 THEN 1 END) as podiums,
                COUNT(*) as races
            FROM actual_results ar
            JOIN races r ON ar.race_id = r.race_id
            WHERE r.circuit = ?
            GROUP BY ar.driver
            HAVING podiums > 0
            ORDER BY podiums DESC
            LIMIT 10
        """
        
        df = pd.read_sql_query(query, conn, params=(circuit_name,))
        conn.close()
        
        return df
        
    except Exception as e:
        print(f"Error getting podium data: {e}")
        return pd.DataFrame()


# ============================================================================
# WORKFLOW FUNCTIONS
# ============================================================================

def generate_prediction_workflow(race_name, year):
    """
    Complete workflow: Check DB → Extract → Predict
    Returns: (success: bool, message: str, race_id: int)
    """
    # Step 1: Check if race exists
    exists, race_id, status = check_race_exists(race_name, year)
    
    if exists:
        # Race exists - check status
        if status in ['predicted', 'completed']:
            return True, f"Predictions already exist (Status: {status})", race_id
        elif status == 'features_ready':
            # Features exist, just run prediction
            success, msg = run_predict_race(race_name, year)
            if success:
                # Re-fetch race_id after prediction
                _, race_id, _ = check_race_exists(race_name, year)
            return success, msg, race_id
    
    # Step 2: Extract features (race doesn't exist or status is 'upcoming')
    success, msg = run_extract_race(race_name, year)
    if not success:
        return False, f"Feature extraction failed: {msg}", None
    
    # Step 3: Run prediction
    success, msg = run_predict_race(race_name, year)
    if not success:
        return False, f"Prediction failed: {msg}", None
    
    # Step 4: Get race_id
    _, race_id, _ = check_race_exists(race_name, year)
    
    return True, "Predictions generated successfully!", race_id


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def clear_fastf1_cache():
    """Clear FastF1 cache"""
    try:
        cache_dir = Path(__file__).resolve().parent.parent / 'fastf1_cache'
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir()
            return True, "Cache cleared"
        return True, "No cache to clear"
    except Exception as e:
        return False, f"Error clearing cache: {e}"