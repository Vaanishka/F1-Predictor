"""
Extract Race Features - FIXED (Grid Position Bug Resolved)

CRITICAL FIX: Grid position now extracted correctly from Q session!
"""
import os
import sys

# Set UTF-8 encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Reconfigure stdout/stderr
if sys.platform == 'win32':
    import io
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import fastf1
import pandas as pd
import numpy as np
import sqlite3
import json
import gc
import sys
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    load_configs, get_year_coefficients, get_track_params,
    estimate_fuel_load, extract_session_weather
)


BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / 'database' / 'f1_predictions.db'
CACHE_DIR = BASE_DIR / 'fastf1_cache'

fastf1.Cache.enable_cache(str(CACHE_DIR))


def get_driver_grid_position(session, driver_num):
    """FIXED: Get grid position from Q or Race session"""
    try:
        # For Qualifying - use 'Position' column
        if 'Q' in str(session.name) or 'Qualifying' in str(session.name):
            if hasattr(session, 'results') and session.results is not None:
                if driver_num in session.results.index:
                    pos = session.results.loc[driver_num, 'Position']
                    if pd.notna(pos):
                        grid_pos = int(pos)
                        if 1 <= grid_pos <= 20:
                            return grid_pos
        
        # For Race - use 'GridPosition' column
        if hasattr(session, 'results') and session.results is not None:
            if driver_num in session.results.index:
                if 'GridPosition' in session.results.columns:
                    grid = session.results.loc[driver_num, 'GridPosition']
                    if pd.notna(grid):
                        grid_pos = int(grid)
                        if 1 <= grid_pos <= 20:
                            return grid_pos
        
        print(f"       Grid P20 (fallback) for driver {driver_num}")
        return 20
        
    except Exception as e:
        print(f"       Grid extraction error for {driver_num}: {type(e).__name__}")
        return 20


def extract_driver_features(session, driver, track_params, coeffs, sess_type, race_name, 
                            weather_data, session_stats):
    """Extract 25 features"""
    
    try:
        driver_laps = session.laps.pick_drivers(driver).pick_accurate()
        driver_laps = driver_laps[driver_laps['LapTime'].notna()].copy()
        
        if len(driver_laps) == 0:
            return None, None
        
        fastest_lap = driver_laps.pick_fastest()
        if fastest_lap is None or fastest_lap.empty:
            return None, None
        
        driver_num = fastest_lap['DriverNumber'] if 'DriverNumber' in fastest_lap.index else driver
        driver_abbr = fastest_lap['Driver'] if 'Driver' in fastest_lap.index else str(driver_num)
        
        # Telemetry
        has_telemetry = False
        tel_speed = []
        
        try:
            tel = fastest_lap.get_telemetry()
            if tel is not None and len(tel) > 0 and 'Speed' in tel.columns:
                speed_series = pd.to_numeric(tel['Speed'], errors='coerce').dropna()
                tel_speed = speed_series.tolist()
                if len(tel_speed) > 10:
                    has_telemetry = True
                del tel
        except:
            pass
        
        data_quality = "TELEMETRY_GOLD" if has_telemetry else "LAP_DATA_ONLY"
        
        features = {}
        
        # Grid - FIXED!
        grid_pos = get_driver_grid_position(session, driver_num)
        features['Grid_Position'] = grid_pos
        features['Grid_Position_Pct'] = (grid_pos / 20.0) * 100
        
        # Speed
        if has_telemetry and len(tel_speed) > 0:
            tel_speed_clean = [s for s in tel_speed if 50 < s < 400]
            if len(tel_speed_clean) > 0:
                avg_speed = float(np.mean(tel_speed_clean))
                features['Max_Speed'] = float(np.max(tel_speed_clean))
                features['P95_Speed'] = float(np.percentile(tel_speed_clean, 95))
            else:
                avg_speed = 200.0
                features['Max_Speed'] = 320.0
                features['P95_Speed'] = 310.0
        else:
            speeds = []
            for field in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
                try:
                    if field in fastest_lap.index and pd.notna(fastest_lap[field]):
                        val = float(fastest_lap[field])
                        if 50 < val < 400:
                            speeds.append(val)
                except:
                    pass
            
            if len(speeds) > 0:
                avg_speed = float(np.mean(speeds))
                features['Max_Speed'] = float(np.max(speeds))
                features['P95_Speed'] = float(np.max(speeds))
            else:
                lap_sec = fastest_lap['LapTime'].total_seconds()
                avg_speed = (5.0 / lap_sec) * 3600
                features['Max_Speed'] = avg_speed * 1.15
                features['P95_Speed'] = avg_speed * 1.10
        
        session_avg_speed = session_stats.get('avg_speed', 200.0)
        features['Speed_Delta_From_Avg_Pct'] = ((avg_speed - session_avg_speed) / session_avg_speed) * 100
        
        # Tire
        compound = fastest_lap['Compound']
        tire_age = fastest_lap['TyreLife'] if pd.notna(fastest_lap['TyreLife']) else 0
        tire_age = min(tire_age, 50)
        
        comp_map = {'SOFT': 3, 'MEDIUM': 2, 'HARD': 1, 'INTERMEDIATE': 0.5, 'WET': 0.2}
        features['Tyre_Grip'] = comp_map.get(str(compound), 2)
        features['Starting_Tyre_Age'] = int(tire_age)
        features['Avg_Tyre_Age'] = float(tire_age + 5)
        features['Total_Tire_Degradation'] = np.clip(0.05 * 1.0 * tire_age, 0, 10)
        
        # Fuel
        fuel_kg, fuel_conf = estimate_fuel_load(fastest_lap, sess_type, coeffs)
        fuel_kg = np.clip(fuel_kg, 0, 110)
        
        features['Starting_Fuel_kg'] = float(fuel_kg)
        features['Avg_Fuel_kg'] = float(fuel_kg)
        features['Fuel_Confidence'] = float(fuel_conf)
        
        # Pace
        fuel_excess = fuel_kg - 10.0
        features['Avg_Fuel_Adjusted_Pace'] = avg_speed + (fuel_excess * 0.30 * fuel_conf)
        features['Avg_True_Pace'] = features['Avg_Fuel_Adjusted_Pace']
        features['Max_True_Pace'] = features['Max_Speed'] + (fuel_excess * 0.30 * fuel_conf)
        
        chassis_score = track_params.get('chassis_score', 1.0)
        features['Avg_Pace_Correction'] = avg_speed / chassis_score
        
        # Driver Consistency
        all_lap_times = driver_laps['LapTime'].dt.total_seconds().dropna()
        if len(all_lap_times) > 3:
            cv = (all_lap_times.std() / all_lap_times.mean()) * 100
            features['Driver_Consistency'] = np.clip(100 - cv, 0, 100)
        else:
            features['Driver_Consistency'] = 85.0
        
        # Track Intelligence
        features['Overtake_Factor'] = track_params.get('overtake_factor', 0.5)
        features['Grid_Importance'] = track_params.get('overtake_factor', 0.5) * 10
        features['Projected_Stability'] = features['Grid_Importance'] / max(features['Grid_Position'], 1)
        
        # Weather
        features['Air_Temp_Celsius'] = weather_data.get('air_temp', 25.0)
        
        # Corrections
        features['Pace_Delta_From_Avg'] = 0.0
        
        # Risk
        features['Risk_Factor_Pct'] = 100 - features['Driver_Consistency']
        
        # Interactions
        features['Grid_X_Overtake'] = features['Grid_Position'] * features['Overtake_Factor']
        features['Tire_X_Fuel'] = features['Tyre_Grip'] * features['Avg_Fuel_kg']
        features['Speed_X_Grip'] = avg_speed * features['Tyre_Grip']
        
        metadata = {
            'driver': str(driver_abbr),
            'session_type': sess_type,
            'data_quality': data_quality
        }
        
        del driver_laps
        gc.collect()
        
        return features, metadata
        
    except Exception as e:
        gc.collect()
        return None, None


def process_session(session, race_name, sess_type, rules, tracks):
    """Process session"""
    print(f"\n   {sess_type} Session ({session.name})")
    
    year = session.date.year
    coeffs = get_year_coefficients(rules, year)
    track_params = get_track_params(tracks, race_name, year)
    
    if track_params is None:
        print("   No track DNA")
        return None
    
    weather_data = extract_session_weather(session)
    print(f"        {weather_data.get('air_temp', 25):.1f}°C")

    session_stats = {'avg_speed': 200.0}
    try:
        all_laps = session.laps.pick_accurate()
        if len(all_laps) > 0:
            lap_times = all_laps['LapTime'].dt.total_seconds().dropna()
            if len(lap_times) > 0:
                avg_lap_time = lap_times.median()
                session_stats['avg_speed'] = (5.0 / avg_lap_time) * 3600
    except:
        pass
    
    all_data = []
    grid_warnings = 0
    
    for driver in session.drivers:
        features, metadata = extract_driver_features(
            session, driver, track_params, coeffs, sess_type, race_name,
            weather_data, session_stats
        )
        
        if features is not None and metadata is not None:
            all_data.append({'features': features, 'metadata': metadata})
            quality_symbol = "G" if metadata['data_quality'] == "TELEMETRY_GOLD" else "B"
            
            if features['Grid_Position'] == 20:
                grid_warnings += 1
            
            print(f"      {quality_symbol} {metadata['driver']}: Grid P{features['Grid_Position']}")
        
        gc.collect()
    
    if len(all_data) == 0:
        return None
    
    # Warning if too many P20 defaults
    if grid_warnings > 15:
        print(f"\n        WARNING: {grid_warnings}/20 drivers have Grid P20 (default)")
        print(f"      This means grid extraction failed - check session type")
    
    all_paces = [d['features']['Avg_True_Pace'] for d in all_data]
    avg_pace = np.mean(all_paces)
    
    for data in all_data:
        data['features']['Pace_Delta_From_Avg'] = data['features']['Avg_True_Pace'] - avg_pace
    
    print(f"   {len(all_data)} drivers | 25 features")
    return all_data


def extract_race(race_name, year, sessions):
    """Main extraction"""
    
    print("\n" + "="*70)
    print("  EXTRACTING RACE (Grid Bug FIXED)")
    print("="*70)
    print(f"\n {race_name} {year}")
    
    rules, tracks = load_configs()
    if not rules or not tracks:
        print(" Config error!")
        return False
    
    race_map = {
        'Bahrain Grand Prix': 'Bahrain',
        'Monaco Grand Prix': 'Monaco',
        'Canadian Grand Prix': 'Canada',
        'Austrian Grand Prix': 'Austria',
        'Singapore Grand Prix': 'Singapore',
        'São Paulo Grand Prix': 'São Paulo',
        'Abu Dhabi Grand Prix': 'Abu Dhabi',
        'Miami Grand Prix': 'Miami',
        'Belgian Grand Prix': 'Belgium',
        'Hungarian Grand Prix': 'Hungary',
        'Saudi Arabian Grand Prix': 'Saudi Arabia',
        'Japanese Grand Prix': 'Japan',
        'Spanish Grand Prix': 'Spain'
    }
    
    fastf1_name = race_map.get(race_name, race_name)
    all_session_data = []
    
    for sess_type in sessions:
        try:
            print(f"\n Loading {sess_type}...")
            session = fastf1.get_session(year, fastf1_name, sess_type)
            session.load(laps=True, telemetry=True, weather=True, messages=False)
            
            session_data = process_session(session, race_name, sess_type, rules, tracks)
            if session_data:
                all_session_data.extend(session_data)
            
            del session
            gc.collect()
            
        except Exception as e:
            print(f"   {sess_type}: {type(e).__name__}")
    
    if not all_session_data:
        print("\n No data!")
        return False
    
    print(f"\n Storing...")
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO races (year, race_name, circuit, status)
            VALUES (?, ?, ?, ?)
        """, (year, race_name, race_name, 'features_ready'))
        
        cursor.execute("SELECT race_id FROM races WHERE year = ? AND race_name = ?", (year, race_name))
        race_id = cursor.fetchone()[0]
        
        cursor.execute("DELETE FROM race_features WHERE race_id = ?", (race_id,))
        
        inserted = 0
        for data in all_session_data:
            cursor.execute("""
                INSERT INTO race_features (
                    race_id, driver, session_type, grid_position, avg_speed,
                    data_quality, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id,
                data['metadata']['driver'],
                data['metadata']['session_type'],
                data['features']['Grid_Position'],
                data['features'].get('Avg_True_Pace', 200.0),
                data['metadata']['data_quality'],
                json.dumps(data['features'])
            ))
            inserted += 1
        
        cursor.execute("""
            UPDATE races 
            SET features_extracted_at = ?, status = 'features_ready'
            WHERE race_id = ?
        """, (datetime.now().isoformat(), race_id))
        
        conn.commit()
        
        print(f" Stored {inserted} records")
        print(f"\n Next: python predict_race.py --race \"{race_name}\" --year {year}")
        
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        return False
    
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--race', required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--sessions', default='FP2,Q')
    
    args = parser.parse_args()
    sessions = [s.strip() for s in args.sessions.split(',')]
    
    success = extract_race(args.race, args.year, sessions)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()