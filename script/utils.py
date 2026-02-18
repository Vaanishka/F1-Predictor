"""
F1 Prediction Model - Shared Utility Functions

This module contains helper functions used by both training and testing scripts.
All functions read from rules.json and track_dna.json for flexibility.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import os
from datetime import datetime
import fastf1
import pandas as pd
import numpy as np
from collections import defaultdict
import gc



def cleanup_memory():
    """Aggressive garbage collection"""
    gc.collect()
    gc.collect()  # Call twice for better cleanup
    gc.collect()

def get_constructor_standings(year):
    """
    Get constructor standings with caching
    Returns: dict of {team_name: {points, position, gap_to_leader}}
    """
    cache_dir = get_base_dir() / 'cache' / 'standings'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{year}_constructor_standings.json'
    
    # Check cache first
    if cache_file.exists():
        cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if cache_age < 86400:  # 24 hours
            with open(cache_file, 'r') as f:
                return json.load(f)
    
    # Load from FastF1
    try:
        import fastf1
        schedule = fastf1.get_event_schedule(year)
        
        # Get last completed race
        completed = schedule[schedule['Session5DateUtc'] < pd.Timestamp.now(tz='UTC')]
        
        if len(completed) == 0:
            # No races yet, return defaults
            return get_default_constructor_standings(year)
        
        last_race = completed.iloc[-1]
        session = fastf1.get_session(year, last_race['EventName'], 'R')
        session.load(results=True)
        
        # Extract team points (simplified - you may want more detailed logic)
        teams = {}
        for idx, row in session.results.iterrows():
            team = row['TeamName']
            if team not in teams:
                teams[team] = {'points': 0, 'position': 0}
        
        # Sort by points and assign positions
        sorted_teams = sorted(teams.items(), key=lambda x: x[1]['points'], reverse=True)
        leader_points = sorted_teams[0][1]['points'] if sorted_teams else 0
        
        standings = {}
        for pos, (team, data) in enumerate(sorted_teams, 1):
            standings[team] = {
                'points': data['points'],
                'position': pos,
                'gap_to_leader': leader_points - data['points']
            }
        
        # Cache it
        with open(cache_file, 'w') as f:
            json.dump(standings, f, indent=2)
        
        return standings
        
    except Exception as e:
        print(f"⚠️  Failed to load standings from FastF1: {e}")
        return get_default_constructor_standings(year)

def get_default_constructor_standings(year):
    """Default standings when data unavailable"""
    # Top teams with estimated points
    default_teams = {
        'Red Bull Racing': {'points': 500, 'position': 1, 'gap_to_leader': 0},
        'Ferrari': {'points': 400, 'position': 2, 'gap_to_leader': 100},
        'Mercedes': {'points': 350, 'position': 3, 'gap_to_leader': 150},
        'McLaren': {'points': 300, 'position': 4, 'gap_to_leader': 200},
        'Aston Martin': {'points': 200, 'position': 5, 'gap_to_leader': 300},
        'Alpine': {'points': 100, 'position': 6, 'gap_to_leader': 400},
        'Williams': {'points': 50, 'position': 7, 'gap_to_leader': 450},
        'AlphaTauri': {'points': 30, 'position': 8, 'gap_to_leader': 470},
        'Alfa Romeo': {'points': 20, 'position': 9, 'gap_to_leader': 480},
        'Haas F1 Team': {'points': 10, 'position': 10, 'gap_to_leader': 490},
    }
    return default_teams

def get_base_dir():
    """Get the base directory (f1_pred/)"""
    return Path(__file__).resolve().parent.parent


def load_configs():
    """
    Load rules.json and track_dna.json
    
    Returns:
        tuple: (rules_dict, tracks_dict)
    """
    base_dir = get_base_dir()
    
    rules_path = base_dir / 'coefficient' / 'rules.json'
    dna_path = base_dir / 'coefficient' / 'track_dna.json'
    
    try:
        with open(rules_path, 'r') as f:
            rules = json.load(f)
        with open(dna_path, 'r') as f:
            tracks = json.load(f)
        
        print("✅ Loaded rules.json and track_dna.json")
        return rules, tracks
    
    except FileNotFoundError as e:
        print(f"❌ Config file not found: {e}")
        print(f"   Looking in: {base_dir / 'coefficient'}")
        return None, None


def get_year_coefficients(rules, year):
    """
    Get year-specific coefficients from rules.json
    
    Args:
        rules: Rules dictionary
        year: Session year (2024, 2025, 2026)
    
    Returns:
        dict: Year-specific coefficients
    """
    year_str = str(year)
    
    if year >= 2026:
        coeffs = rules.get('2026', rules['2024'])
    elif year >= 2025:
        coeffs = rules.get('2025', rules['2024'])
    else:
        coeffs = rules.get('2024')
    
    return coeffs


def get_track_params(tracks, track_name, year):
    """
    Get track-specific parameters for given year
    
    Args:
        tracks: Tracks dictionary
        track_name: Full race name (e.g., "Abu Dhabi Grand Prix")
        year: Session year
    
    Returns:
        dict: Track parameters for that year
    """
    track_data = tracks.get(track_name)
    
    if track_data is None:
        print(f"⚠️  Warning: Track '{track_name}' not found in track_dna.json")
        return None
    
    year_str = str(year)
    
    if year >= 2026:
        track_params = track_data.get('2026', track_data.get('2024'))
    elif year >= 2025:
        track_params = track_data.get('2025', track_data.get('2024'))
    else:
        track_params = track_data.get('2024')
    
    # Add base track characteristics
    track_params['overtake_factor'] = track_data.get('overtake_factor', 0.5)
    track_params['elevation_gradient'] = track_data.get('elevation_gradient', 0.01)
    track_params['penalty_per_kg'] = track_data.get('penalty_per_kg', 0.035)
    track_params['tire_wear_rate'] = track_data.get('tire_wear_rate', 1.0)
    
    return track_params


def calculate_tire_adjusted_pace(speed, compound, tire_age, rules):
    """
    Normalize speed by tire compound and age
    
    Args:
        speed: Raw speed (kph)
        compound: Tire compound (SOFT/MEDIUM/HARD/etc)
        tire_age: Laps on current tires
        rules: Rules dictionary for tire constants
    
    Returns:
        float: Tire-adjusted speed
    """
    # Get compound multipliers from rules
    tire_constants = rules.get('feature_engineering_constants', {})
    compound_multipliers = tire_constants.get('tire_compound_grip_multipliers', {
        'SOFT': 1.03,
        'MEDIUM': 1.00,
        'HARD': 0.97,
        'INTERMEDIATE': 0.90,
        'WET': 0.85
    })
    
    c_factor = compound_multipliers.get(compound, 1.0)
    
    # Age degradation
    age_deg_factor = tire_constants.get('tire_age_degradation_factor', 0.0005)
    age_factor = 1.0 - (tire_age * age_deg_factor)
    age_factor = max(age_factor, 0.92)
    
    adjusted_speed = speed / (c_factor * age_factor)
    return adjusted_speed


def calculate_fuel_adjusted_pace(speed, fuel_kg, fuel_confidence, coeffs):
    """
    Normalize speed by fuel load
    
    Args:
        speed: Raw speed (kph)
        fuel_kg: Current fuel load
        fuel_confidence: Confidence in fuel estimate (0-1)
        coeffs: Year-specific coefficients
    
    Returns:
        float: Fuel-adjusted speed
    """
    fuel_baseline = 10.0  # Qualifying fuel
    fuel_excess = fuel_kg - fuel_baseline
    
    # Get fuel penalty coefficient from year-specific rules
    fuel_penalty_coeff = coeffs.get('fuel_speed_penalty_coefficient', 0.30)
    
    fuel_penalty_kph = fuel_excess * fuel_penalty_coeff
    adjusted_speed = speed + (fuel_penalty_kph * fuel_confidence)
    
    return adjusted_speed


def estimate_fuel_load(lap, sess_type, coeffs):
    """
    Estimate fuel load based on session type and lap number
    
    Args:
        lap: Lap data (dict or FastF1 lap object)
        sess_type: Session type (Q/FP2/R)
        coeffs: Year-specific coefficients
    
    Returns:
        tuple: (fuel_kg, confidence)
    """
    # Get fuel constants
    fuel_race_start = coeffs.get('fuel_race_start_kg', 110.0)
    fuel_consumption = coeffs.get('fuel_consumption_per_lap_kg', 1.6)
    fuel_baseline = coeffs.get('fuel_baseline_kg', 10.0)
    
    if sess_type == 'Q':
        return fuel_baseline, 0.9
    
    elif sess_type in ['FP', 'P', 'FP2']:
        lap_num = lap.get('LapNumber', 1) if isinstance(lap, dict) else getattr(lap, 'LapNumber', 1)
        if lap_num < 5:
            return 80.0, 0.3
        else:
            return 40.0, 0.2
    
    elif sess_type == 'R':
        lap_num = lap.get('LapNumber', 1) if isinstance(lap, dict) else getattr(lap, 'LapNumber', 1)
        fuel_remaining = fuel_race_start - (lap_num * fuel_consumption)
        fuel_remaining = max(fuel_remaining, 5.0)
        return fuel_remaining, 0.7
    
    else:
        return 50.0, 0.1


def detect_sandbagging(driver_laps, sess_type, coeffs):
    """
    Detect sandbagging in practice sessions
    
    Args:
        driver_laps: DataFrame of laps for this driver
        sess_type: Session type
        coeffs: Year-specific coefficients
    
    Returns:
        float: Sandbagging score (0-100)
    """
    if sess_type not in ['FP', 'P', 'FP2']:
        return 0
    
    if len(driver_laps) < 5:
        return 0
    
    lap_times = driver_laps['LapTime'].dt.total_seconds()
    lap_times = lap_times[lap_times.notna()]
    
    if len(lap_times) < 5:
        return 0
    
    cv = (lap_times.std() / lap_times.mean()) * 100
    
    # Get multiplier from constants
    cv_multiplier = coeffs.get('consistency_cv_multiplier', 5.0)
    sandbagging_score = min(cv * cv_multiplier, 100)
    
    return sandbagging_score


def encode_categorical_features(df, driver_encoder=None, race_encoder=None):
    """
    Encode categorical features
    
    Args:
        df: DataFrame with categorical columns
        driver_encoder: Existing LabelEncoder for drivers (optional)
        race_encoder: Existing LabelEncoder for races (optional)
    
    Returns:
        tuple: (encoded_df, driver_encoder, race_encoder)
    """
    from sklearn.preprocessing import LabelEncoder
    
    df = df.copy()
    
    # Encode Driver
    if driver_encoder is None:
        driver_encoder = LabelEncoder()
        df['Driver_Encoded'] = driver_encoder.fit_transform(df['Driver'])
    else:
        # Handle unknown drivers
        known_drivers = set(driver_encoder.classes_)
        df['Driver_Encoded'] = df['Driver'].apply(
            lambda x: driver_encoder.transform([x])[0] if x in known_drivers else -1
        )
    
    # Encode Race
    if race_encoder is None:
        race_encoder = LabelEncoder()
        df['Race_Encoded'] = race_encoder.fit_transform(df['Race'])
    else:
        known_races = set(race_encoder.classes_)
        df['Race_Encoded'] = df['Race'].apply(
            lambda x: race_encoder.transform([x])[0] if x in known_races else -1
        )
    
    # Encode Tire Compound
    tire_map = {'SOFT': 3, 'MEDIUM': 2, 'HARD': 1, 'INTERMEDIATE': 0.5, 'WET': 0.2}
    df['Tyre_Compound_Encoded'] = df['Tyre_Compound'].map(tire_map).fillna(2)
    
    # Encode Session Type
    session_map = {'R': 2, 'Q': 1, 'FP2': 0, 'FP': 0, 'P': 0}
    df['Session_Type_Encoded'] = df['Session_Type'].map(session_map).fillna(0)
    
    return df, driver_encoder, race_encoder


def select_model_features(df):
    """
    Select features for modeling (exclude metadata and target)
    
    Args:
        df: Full DataFrame
    
    Returns:
        list: Feature column names
    """
    exclude = [
        'Driver',
        'Race',
        'Session_Type',
        'Tyre_Compound',
        'Final_Position',  # NEVER use as feature!
        'Target_Final_Position',  # Target variable
        'Position_Change',  # Derived from target
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude]
    
    available_features = []
    for col in feature_cols:
        if col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                available_features.append(col)
    
    return available_features


def map_race_name(race_name):
    """
    Map various race name formats to standard format
    
    Args:
        race_name: Race name (various formats)
    
    Returns:
        str: Standardized race name
    """
    mapping = {
        'Silverstone': 'British Grand Prix',
        'Monza': 'Italian Grand Prix',
        'Hungary': 'Hungarian Grand Prix',
        'Belgium': 'Belgian Grand Prix',
        'Singapore': 'Singapore Grand Prix',
        'Abu Dhabi': 'Abu Dhabi Grand Prix',
        'Bahrain': 'Bahrain Grand Prix',
        'Saudi Arabia': 'Saudi Arabian Grand Prix',
        'Australia': 'Australian Grand Prix',
        'Japan': 'Japanese Grand Prix',
        'China': 'Chinese Grand Prix',
        'Miami': 'Miami Grand Prix',
        'Imola': 'Emilia Romagna Grand Prix',
        'Monaco': 'Monaco Grand Prix',
        'Canada': 'Canadian Grand Prix',
        'Spain': 'Spanish Grand Prix',
        'Austria': 'Austrian Grand Prix',
        'Netherlands': 'Dutch Grand Prix',
        'Zandvoort': 'Dutch Grand Prix',
        'Azerbaijan': 'Azerbaijan Grand Prix',
        'Baku': 'Azerbaijan Grand Prix',
        'USA': 'United States Grand Prix',
        'COTA': 'United States Grand Prix',
        'Mexico': 'Mexican Grand Prix',
        'Brazil': 'São Paulo Grand Prix',
        'Las Vegas': 'Las Vegas Grand Prix',
        'Qatar': 'Qatar Grand Prix',
    }
    
    # Try exact match first
    if race_name in mapping:
        return mapping[race_name]
    
    # Try partial match
    for short_name, full_name in mapping.items():
        if short_name.lower() in race_name.lower():
            return full_name
    
    # Return as-is if no match
    return race_name


def calculate_confidence_intervals(predictions, std_dev=None):
    """
    Calculate confidence intervals for predictions
    
    Args:
        predictions: Array of predictions
        std_dev: Standard deviation (if known from CV)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if std_dev is None:
        # Use historical MAE as proxy (typically ~2.5 positions)
        std_dev = 2.5
    
    # 68% confidence (±1 std dev)
    lower_68 = predictions - std_dev
    upper_68 = predictions + std_dev
    
    # 95% confidence (±2 std dev)
    lower_95 = predictions - (2 * std_dev)
    upper_95 = predictions + (2 * std_dev)
    
    # Clip to valid range [1, 20]
    lower_68 = np.clip(lower_68, 1, 20)
    upper_68 = np.clip(upper_68, 1, 20)
    lower_95 = np.clip(lower_95, 1, 20)
    upper_95 = np.clip(upper_95, 1, 20)
    
    return {
        '68%': (lower_68, upper_68),
        '95%': (lower_95, upper_95)
    }


def format_prediction_output(predictions_df, confidence_intervals=None):
    """
    Format predictions for display/API output
    
    Args:
        predictions_df: DataFrame with predictions
        confidence_intervals: Optional confidence intervals
    
    Returns:
        dict: Formatted prediction data
    """
    output = {
        'predictions': [],
        'metadata': {
            'total_drivers': len(predictions_df),
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    for idx, row in predictions_df.iterrows():
        pred_dict = {
            'position': int(idx),
            'driver': row['Driver'],
            'grid_position': int(row['Grid_Position']),
            'predicted_position': int(row['Predicted_Position']),
            'expected_change': int(row['Expected_Change'])
        }
        
        if confidence_intervals is not None:
            pred_dict['confidence_68'] = {
                'lower': int(confidence_intervals['68%'][0][idx-1]),
                'upper': int(confidence_intervals['68%'][1][idx-1])
            }
        
        output['predictions'].append(pred_dict)
    
    return output


"""
Utils.py ADDITIONS - Add these functions to your existing utils.py

NEW FUNCTIONS:
1. calculate_driver_ratings_from_season() - Driver skill ratings
2. extract_session_weather() - Weather data extraction
"""

import fastf1
import pandas as pd
import numpy as np
from collections import defaultdict


def calculate_driver_ratings_from_season(year=2024):
    """
    Calculate driver skill ratings from completed races
    
    Returns dict: {driver_code: {skill_rating, avg_grid, avg_finish, ...}}
    
    Ratings calculated:
    - skill_rating: 0-100 overall score
    - avg_grid: Average qualifying position
    - avg_finish: Average race finish
    - consistency_rating: Based on finish variance
    - race_craft: Avg positions gained/lost
    - recent_form: Last 3 races performance
    """
    
    print(f"   Processing {year} races...")
    
    # Get completed races
    try:
        schedule = fastf1.get_event_schedule(year)
        now_utc = pd.Timestamp.now(tz='UTC')
        
        completed_mask = schedule['Session5DateUtc'].notna()
        schedule_completed = schedule[completed_mask].copy()
        
        if schedule_completed['Session5DateUtc'].dt.tz is None:
            schedule_completed['Session5DateUtc'] = pd.to_datetime(
                schedule_completed['Session5DateUtc']
            ).dt.tz_localize('UTC')
        
        completed = schedule_completed[schedule_completed['Session5DateUtc'] < now_utc]
        
    except Exception as e:
        print(f"   ⚠️  Schedule error: {e}")
        return get_default_driver_ratings()
    
    if len(completed) == 0:
        return get_default_driver_ratings()
    
    # Collect driver stats
    driver_stats = defaultdict(lambda: {
        'races': 0,
        'grid_positions': [],
        'finish_positions': [],
        'points': 0,
        'dnfs': 0
    })
    
    for idx, race_info in completed.iterrows():
        try:
            session = fastf1.get_session(year, race_info['EventName'], 'R')
            session.load(laps=False, telemetry=False, weather=False)
            
            if not hasattr(session, 'results') or session.results is None:
                continue
            
            for driver_id, row in session.results.iterrows():
                driver = row['Abbreviation'] if 'Abbreviation' in row else driver_id
                
                driver_stats[driver]['races'] += 1
                
                # Grid position
                grid = row['GridPosition']
                if pd.notna(grid) and grid > 0:
                    driver_stats[driver]['grid_positions'].append(grid)
                
                # Finish position
                finish = row['Position']
                if pd.notna(finish):
                    driver_stats[driver]['finish_positions'].append(finish)
                else:
                    driver_stats[driver]['dnfs'] += 1
                
                # Points
                points = row['Points']
                if pd.notna(points):
                    driver_stats[driver]['points'] += points
            
            # Memory cleanup
            del session
            import gc
            gc.collect()
            
        except Exception as e:
            continue
    
    # Calculate ratings
    driver_ratings = {}
    
    for driver, stats in driver_stats.items():
        if stats['races'] == 0:
            continue
        
        # Average grid & finish
        avg_grid = np.mean(stats['grid_positions']) if stats['grid_positions'] else 10.0
        avg_finish = np.mean(stats['finish_positions']) if stats['finish_positions'] else 10.0
        
        # Consistency (lower variance = better)
        if len(stats['finish_positions']) > 3:
            finish_std = np.std(stats['finish_positions'])
            consistency_rating = max(0, 100 - (finish_std * 5))
        else:
            consistency_rating = 85.0
        
        # Race craft (positions gained on average)
        if stats['grid_positions'] and stats['finish_positions']:
            min_len = min(len(stats['grid_positions']), len(stats['finish_positions']))
            race_craft = np.mean([
                stats['grid_positions'][i] - stats['finish_positions'][i]
                for i in range(min_len)
            ])
        else:
            race_craft = 0.0
        
        # Recent form (last 3 races)
        recent_finishes = stats['finish_positions'][-3:] if len(stats['finish_positions']) >= 3 else stats['finish_positions']
        recent_form = 100 - (np.mean(recent_finishes) * 5) if recent_finishes else 70.0
        recent_form = max(0, min(100, recent_form))
        
        # Overall skill rating
        # Based on: avg finish (inverse), consistency, race craft
        skill_rating = (
            (21 - min(avg_finish, 20)) * 3 +  # Better finish = higher score
            consistency_rating * 0.3 +
            (race_craft * 10) +  # Positions gained bonus
            (stats['points'] / max(stats['races'], 1))  # Points per race
        )
        skill_rating = max(0, min(100, skill_rating))
        
        driver_ratings[driver] = {
            'skill_rating': round(skill_rating, 1),
            'avg_grid': round(avg_grid, 1),
            'avg_finish': round(avg_finish, 1),
            'consistency_rating': round(consistency_rating, 1),
            'race_craft': round(race_craft, 2),
            'recent_form': round(recent_form, 1),
            'races_completed': stats['races'],
            'total_points': int(stats['points'])
        }
    
    # Show top 5
    sorted_drivers = sorted(driver_ratings.items(), key=lambda x: x[1]['skill_rating'], reverse=True)
    print(f"\n   Top 5 Driver Ratings:")
    for driver, rating in sorted_drivers[:5]:
        print(f"      {driver}: {rating['skill_rating']:.1f} (Grid: {rating['avg_grid']:.1f}, Finish: {rating['avg_finish']:.1f})")
    
    return driver_ratings


def get_default_driver_ratings():
    """Default driver ratings when data unavailable"""
    # Approximate 2024 driver ratings
    return {
        'VER': {'skill_rating': 95.0, 'avg_grid': 2.0, 'avg_finish': 2.0, 'consistency_rating': 95.0, 'race_craft': 1.5, 'recent_form': 95.0},
        'NOR': {'skill_rating': 85.0, 'avg_grid': 4.0, 'avg_finish': 4.0, 'consistency_rating': 90.0, 'race_craft': 1.0, 'recent_form': 85.0},
        'LEC': {'skill_rating': 85.0, 'avg_grid': 4.0, 'avg_finish': 4.5, 'consistency_rating': 85.0, 'race_craft': 0.5, 'recent_form': 85.0},
        'PIA': {'skill_rating': 82.0, 'avg_grid': 6.0, 'avg_finish': 5.5, 'consistency_rating': 88.0, 'race_craft': 0.5, 'recent_form': 82.0},
        'SAI': {'skill_rating': 80.0, 'avg_grid': 6.0, 'avg_finish': 6.0, 'consistency_rating': 87.0, 'race_craft': 0.0, 'recent_form': 80.0},
        'HAM': {'skill_rating': 80.0, 'avg_grid': 7.0, 'avg_finish': 7.0, 'consistency_rating': 85.0, 'race_craft': 0.0, 'recent_form': 75.0},
        'RUS': {'skill_rating': 78.0, 'avg_grid': 7.0, 'avg_finish': 7.5, 'consistency_rating': 86.0, 'race_craft': -0.5, 'recent_form': 78.0},
        'PER': {'skill_rating': 72.0, 'avg_grid': 8.0, 'avg_finish': 8.0, 'consistency_rating': 75.0, 'race_craft': 0.0, 'recent_form': 70.0},
    }


"""
Utility Functions - Charts, SVGs, Formatters
Midnight Carbon themed visualizations
"""

"""
Utility Functions - Charts, SVGs, Formatters
Version 2.1 - Midnight Carbon Theme (No Emojis)

CONTAINS:
- Plotly chart generators (with dark theme)
- SVG generators (confidence gauges, track outlines)
- Helper functions (colors, formats, tags)
"""

import sys
from pathlib import Path

# Fix config import
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / 'streamlit_app'))

import plotly.graph_objects as go
# 4. Now you can import safely

from config import COLORS, TEAM_COLORS, DRIVER_TEAMS_2024, DRIVER_TEAMS_2025
import math
import numpy as np


# ============================================================================
# TEAM & DRIVER HELPERS
# ============================================================================

def get_driver_team(driver_code, year=2024):
    """Get team name for a driver code."""
    teams = DRIVER_TEAMS_2025 if year >= 2025 else DRIVER_TEAMS_2024
    return teams.get(driver_code, 'Unknown')


def get_team_color(team_name):
    """Get hex color for a team."""
    return TEAM_COLORS.get(team_name, COLORS['f1_white'])


def get_driver_color(driver_code, year=2024):
    """Get color for a driver based on their team."""
    team = get_driver_team(driver_code, year)
    return get_team_color(team)


# ============================================================================
# PLOTLY CHART THEMES
# ============================================================================

def get_plotly_layout(title=""):
    """Base Plotly layout for Midnight Carbon theme."""
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(26,26,26,0.5)',
        'font': {
            'color': COLORS['f1_white'],
            'family': 'Inter, -apple-system, sans-serif',
            'size': 12
        },
        'title': {
            'text': title,
            'font': {'size': 18, 'color': COLORS['f1_red'], 'family': 'Inter'},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'zerolinecolor': 'rgba(255,255,255,0.2)',
            'color': COLORS['f1_white']
        },
        'yaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'zerolinecolor': 'rgba(255,255,255,0.2)',
            'color': COLORS['f1_white']
        },
        'hovermode': 'closest',
        'hoverlabel': {
            'bgcolor': COLORS['bg_secondary'],
            'bordercolor': COLORS['f1_red'],
            'font': {'color': COLORS['f1_white']}
        }
    }


# ============================================================================
# SANDBAGGING COMPARISON CHART
# ============================================================================

def create_sandbagging_chart(sandbagging_df, year=2024):
    """
    Create FP2 vs Qualifying comparison chart.
    
    Args:
        sandbagging_df: DataFrame from calculate_sandbagging()
        year: Season year for team colors
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for idx, row in sandbagging_df.iterrows():
        driver = row['driver']
        fp2 = row['fuel_adjusted_fp2']
        quali = row['q_pace']
        detected = row['detected']
        
        color = get_driver_color(driver, year)
        
        # FP2 bar
        fig.add_trace(go.Bar(
            name=f'{driver} FP2',
            x=[driver],
            y=[fp2],
            marker_color=color if not detected else 'rgba(255,255,255,0.3)',
            opacity=0.6,
            showlegend=False,
            hovertemplate=f'{driver} FP2: {fp2:.3f}s<extra></extra>'
        ))
        
        # Qualifying bar
        fig.add_trace(go.Bar(
            name=f'{driver} Q',
            x=[driver],
            y=[quali],
            marker_color=color if detected else 'rgba(255,255,255,0.5)',
            showlegend=False,
            hovertemplate=f'{driver} Qualifying: {quali:.3f}s<br>Sandbagging: {"YES" if detected else "NO"}<extra></extra>'
        ))
    
    layout = get_plotly_layout("FP2 vs Qualifying Pace (Fuel-Adjusted)")
    layout['xaxis']['title'] = 'Driver'
    layout['yaxis']['title'] = 'Lap Time (seconds)'
    layout['barmode'] = 'group'
    layout['height'] = 500
    
    fig.update_layout(**layout)
    
    return fig


# ============================================================================
# HISTORICAL PODIUM CHART
# ============================================================================

def create_historical_podium_chart(historical_df, year=2024):
    """
    Create historical podium bar chart.
    
    Args:
        historical_df: DataFrame with year, driver, final_position
        year: Season year for team colors
    
    Returns:
        Plotly figure or None if no data
    """
    if len(historical_df) == 0:
        return None
    
    podium_counts = historical_df['driver'].value_counts().head(10)
    
    fig = go.Figure()
    
    colors = [get_driver_color(driver, year) for driver in podium_counts.index]
    
    fig.add_trace(go.Bar(
        x=podium_counts.index,
        y=podium_counts.values,
        marker_color=colors,
        text=podium_counts.values,
        textposition='outside',
        hovertemplate='%{x}<br>Podiums: %{y}<extra></extra>'
    ))
    
    layout = get_plotly_layout("Historical Podium Appearances")
    layout['xaxis']['title'] = ''
    layout['yaxis']['title'] = 'Podium Count'
    layout['height'] = 400
    layout['showlegend'] = False
    
    fig.update_layout(**layout)
    
    return fig


# ============================================================================
# CONFIDENCE GAUGE (SVG-based)
# ============================================================================

def create_confidence_gauge_svg(confidence_pct, size=200):
    """
    Create circular confidence gauge as SVG.
    
    Args:
        confidence_pct: 0-100 confidence percentage
        size: SVG size in pixels
    
    Returns:
        SVG string (HTML-safe)
    """
    # Color based on confidence level
    if confidence_pct >= 85:
        color = COLORS['success']
    elif confidence_pct >= 70:
        color = COLORS['accent_gold']
    else:
        color = COLORS['error']
    
    # Calculate arc path
    radius = size / 2 - 20
    center = size / 2
    
    # Convert percentage to radians (270 degree total arc, starting at -135 degrees)
    start_angle = -135 * (math.pi / 180)
    end_angle = start_angle + (270 * (confidence_pct / 100) * (math.pi / 180))
    
    # Calculate arc endpoints
    x1 = center + radius * math.cos(start_angle)
    y1 = center + radius * math.sin(start_angle)
    x2 = center + radius * math.cos(end_angle)
    y2 = center + radius * math.sin(end_angle)
    
    large_arc = 1 if confidence_pct > 50 else 0
    
    svg = f'''
    <svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
        <circle cx="{center}" cy="{center}" r="{radius}" 
                fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="20"/>
        
        <path d="M {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2}"
              fill="none" stroke="{color}" stroke-width="20" stroke-linecap="round"/>
        
        <text x="{center}" y="{center - 10}" 
              text-anchor="middle" 
              font-size="48" 
              font-weight="bold" 
              fill="{color}">
            {int(confidence_pct)}%
        </text>
        
        <text x="{center}" y="{center + 25}" 
              text-anchor="middle" 
              font-size="14" 
              fill="rgba(255,255,255,0.6)">
            CONFIDENCE
        </text>
    </svg>
    '''
    
    return svg


# ============================================================================
# TRACK SVG GENERATOR (Simplified Geometric)
# ============================================================================

def generate_track_svg(track_name, width=400, height=300):
    """
    Generate simplified track outline as SVG.
    
    NOTE: This creates GEOMETRIC PLACEHOLDERS.
    For production, replace with real SVG circuits from:
    - https://github.com/bacinger/f1-circuits
    - https://www.racingcircuits.info/
    
    Args:
        track_name: Race name (e.g., "Monaco Grand Prix")
        width: SVG width
        height: SVG height
    
    Returns:
        SVG string (HTML-safe)
    """
    track_shapes = {
        'Monaco Grand Prix': {
            'path': 'M 50,150 Q 100,50 200,100 T 350,150 Q 300,250 150,200 Z',
            'color': COLORS['f1_red'],
            'type': 'Tight street circuit'
        },
        'Belgian Grand Prix': {
            'path': 'M 50,100 Q 150,50 250,100 L 350,200 Q 250,250 150,200 Z',
            'color': COLORS['accent_blue'],
            'type': 'High-speed, elevation'
        },
        'Italian Grand Prix': {
            'path': 'M 50,150 L 350,150 Q 380,180 350,210 L 50,210 Q 20,180 50,150 Z',
            'color': COLORS['accent_green'],
            'type': 'Temple of Speed'
        },
        'Singapore Grand Prix': {
            'path': 'M 100,50 L 300,80 L 350,200 L 200,250 L 50,180 Z',
            'color': COLORS['accent_purple'],
            'type': 'Night street circuit'
        },
    }
    
    default_shape = {
        'path': 'M 100,100 Q 200,50 300,100 Q 350,200 250,250 Q 150,250 50,200 Q 0,100 100,100 Z',
        'color': COLORS['glass_border'],
        'type': 'Standard layout'
    }
    
    track_data = track_shapes.get(track_name, default_shape)
    
    svg = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="trackGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:{track_data['color']};stop-opacity:0.6" />
                <stop offset="100%" style="stop-color:{COLORS['f1_red']};stop-opacity:0.3" />
            </linearGradient>
        </defs>
        
        <path d="{track_data['path']}" 
              fill="none" 
              stroke="url(#trackGradient)" 
              stroke-width="20"
              stroke-linecap="round"
              opacity="0.8"/>
        
        <rect x="45" y="140" width="10" height="30" fill="{COLORS['f1_white']}" opacity="0.8"/>
        
        <text x="{width/2}" y="{height - 20}" 
              text-anchor="middle" 
              font-size="12" 
              font-weight="600"
              fill="rgba(255,255,255,0.6)"
              text-transform="uppercase">
            {track_name[:20]}
        </text>
    </svg>
    '''
    
    return svg


# ============================================================================
# FORMAT HELPERS
# ============================================================================

def format_position(position):
    """Format position with suffix (1st, 2nd, 3rd, etc.)."""
    if position in [11, 12, 13]:
        return f"{position}th"
    
    last_digit = position % 10
    if last_digit == 1:
        return f"{position}st"
    elif last_digit == 2:
        return f"{position}nd"
    elif last_digit == 3:
        return f"{position}rd"
    else:
        return f"{position}th"


def format_change(change):
    """Format position change with arrow."""
    if change > 0:
        return f"UP {abs(int(change))}"
    elif change < 0:
        return f"DOWN {abs(int(change))}"
    else:
        return "NO CHANGE"


def get_position_emoji(position):
    """Get position label."""
    if position == 1:
        return "1ST"
    elif position == 2:
        return "2ND"
    elif position == 3:
        return "3RD"
    else:
        return f"P{position}"


# ============================================================================
# TRACK CHARACTERISTIC TAGS
# ============================================================================

def get_track_tags(track_name):
    """Get characteristic tags for a track."""
    track_tags = {
        'Monaco Grand Prix': ['Street Circuit', 'High Downforce', 'Low Speed', 'Iconic'],
        'Belgian Grand Prix': ['High Speed', 'Weather Variable', 'Spa-Francorchamps', 'Elevation'],
        'Italian Grand Prix': ['Temple of Speed', 'Low Downforce', 'Power Sensitive', 'Historic'],
        'Singapore Grand Prix': ['Night Race', 'Street Circuit', 'Humid', 'High Downforce'],
        'Japanese Grand Prix': ['Technical', 'Figure-8', 'Fast Corners', 'Challenging'],
        'British Grand Prix': ['High Speed', 'Fast Corners', 'British GP', 'Historic'],
        'Bahrain Grand Prix': ['Night Race', 'Desert', 'Overtaking', 'Season Opener'],
        'Abu Dhabi Grand Prix': ['Night Race', 'Modern', 'Season Finale', 'Yas Marina'],
    }
    
    return track_tags.get(track_name, ['High Speed', 'Technical', 'Challenging'])


def get_tag_color(tag):
    """Get color for a track tag."""
    tag_colors = {
        'Street Circuit': COLORS['accent_purple'],
        'Night Race': COLORS['accent_blue'],
        'High Speed': COLORS['f1_red'],
        'Low Speed': COLORS['accent_gold'],
        'High Downforce': COLORS['accent_green'],
        'Low Downforce': COLORS['warning'],
        'Weather Variable': COLORS['info'],
        'Power Sensitive': COLORS['error'],
        'Technical': COLORS['accent_blue'],
        'Iconic': COLORS['accent_gold'],
    }
    
    return tag_colors.get(tag, COLORS['glass_border'])



# ============================================================================
# WEATHER DATA EXTRACTION
# ============================================================================

def extract_session_weather(session):
    """
    Extract weather data from session
    
    Returns dict with:
    - air_temp: Air temperature (°C)
    - track_temp: Track temperature (°C)
    - humidity: Relative humidity (%)
    - pressure: Atmospheric pressure (mbar)
    - wind_speed: Wind speed (m/s)
    - rainfall: Rain flag (0=dry, 1=wet)
    """
    
    weather_data = {
        'air_temp': 25.0,
        'track_temp': 35.0,
        'humidity': 60.0,
        'pressure': 1013.0,
        'wind_speed': 2.0,
        'rainfall': 0
    }
    
    try:
        if not hasattr(session, 'weather_data') or session.weather_data is None or len(session.weather_data) == 0:
            return weather_data
        
        # Get average weather across session
        weather_df = session.weather_data
        
        if 'AirTemp' in weather_df.columns:
            air_temp = weather_df['AirTemp'].median()
            if pd.notna(air_temp):
                weather_data['air_temp'] = float(air_temp)
        
        if 'TrackTemp' in weather_df.columns:
            track_temp = weather_df['TrackTemp'].median()
            if pd.notna(track_temp):
                weather_data['track_temp'] = float(track_temp)
        
        if 'Humidity' in weather_df.columns:
            humidity = weather_df['Humidity'].median()
            if pd.notna(humidity):
                weather_data['humidity'] = float(humidity)
        
        if 'Pressure' in weather_df.columns:
            pressure = weather_df['Pressure'].median()
            if pd.notna(pressure):
                weather_data['pressure'] = float(pressure)
        
        if 'WindSpeed' in weather_df.columns:
            wind_speed = weather_df['WindSpeed'].median()
            if pd.notna(wind_speed):
                weather_data['wind_speed'] = float(wind_speed)
        
        # Rainfall detection
        if 'Rainfall' in weather_df.columns:
            # If rainfall column exists and has any True values
            if weather_df['Rainfall'].any():
                weather_data['rainfall'] = 1
        
    except Exception as e:
        # Return defaults on error
        pass
    
    return weather_data


if __name__ == "__main__":
    # Test config loading
    rules, tracks = load_configs()
    
    if rules and tracks:
        print(f"\n✅ Successfully loaded configs")
        print(f"   Rules years available: {[k for k in rules.keys() if k.isdigit()]}")
        print(f"   Tracks available: {len([k for k in tracks.keys() if k != '_metadata'])}")
        
        # Test year coefficients
        coeffs_2024 = get_year_coefficients(rules, 2024)
        coeffs_2026 = get_year_coefficients(rules, 2026)
        
        print(f"\n📊 2024 Fuel Penalty: {coeffs_2024['fuel_speed_penalty_coefficient']} kph/kg")
        print(f"   2026 Fuel Penalty: {coeffs_2026['fuel_speed_penalty_coefficient']} kph/kg")
        
        print(f"\n📊 2024 ERS Max: {coeffs_2024['ers_deployment_max_kj']} kJ")
        print(f"   2026 ERS Max: {coeffs_2026['ers_deployment_max_kj']} kJ")
        
        # Test track params
        abu_dhabi_2024 = get_track_params(tracks, 'Abu Dhabi Grand Prix', 2024)
        abu_dhabi_2026 = get_track_params(tracks, 'Abu Dhabi Grand Prix', 2026)
        
        print(f"\n🏁 Abu Dhabi 2024 Clipping: {abu_dhabi_2024['clipping_threshold']} kph")
        print(f"   Abu Dhabi 2026 Clipping: {abu_dhabi_2026['clipping_threshold']} kph")