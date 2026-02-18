"""
Predict Race - Production (FIXED: Clear Rankings)

Shows RANK (1st, 2nd, 3rd) instead of rounded position integers.
Much clearer when predictions are close!

Usage:
    python predict_race.py --race "Belgian Grand Prix" --year 2024
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

import sqlite3
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
import argparse
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / 'database' / 'f1_predictions.db'
MODEL_DIR = BASE_DIR / 'models'


def load_model_artifacts():
    """Load model + scaler + metadata"""
    
    model_file = MODEL_DIR / 'xgboost_model.pkl'
    scaler_file = MODEL_DIR / 'scaler.pkl'
    metadata_file = MODEL_DIR / 'model_metadata.json'
    features_file = MODEL_DIR / 'feature_names.json'
    
    if not all([f.exists() for f in [model_file, scaler_file, metadata_file, features_file]]):
        print(" Model files not found!")
        print(f"   Expected in: {MODEL_DIR}")
        print("   Run: python train_model.py first!")
        return None, None, None, None
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    with open(features_file, 'r') as f:
        feature_names = json.load(f)
    
    return model, scaler, metadata, feature_names


def load_race_features(race_id, conn, feature_names):
    """Load ALL 25 features from JSON"""
    
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT driver, features_json
        FROM race_features
        WHERE race_id = ? AND session_type IN ('FP2', 'Q')
        ORDER BY driver, session_type
    """, (race_id,))
    
    rows = cursor.fetchall()
    
    if not rows:
        return None
    
    # Parse JSON
    driver_features = {}
    
    for driver, features_json in rows:
        features = json.loads(features_json)
        
        if driver not in driver_features:
            driver_features[driver] = []
        
        driver_features[driver].append(features)
    
    # Aggregate (average FP2 + Q)
    aggregated = []
    
    for driver, feature_list in driver_features.items():
        avg_features = {}
        
        for feat_name in feature_names:
            values = [f.get(feat_name, 0) for f in feature_list if feat_name in f]
            if values:
                # Grid: take last (from Q)
                if 'Grid' in feat_name:
                    avg_features[feat_name] = values[-1]
                else:
                    avg_features[feat_name] = np.mean(values)
            else:
                avg_features[feat_name] = 0.0
        
        aggregated.append({
            'driver': driver,
            **avg_features
        })
    
    return pd.DataFrame(aggregated)


def generate_predictions(model, scaler, features_df, feature_names, model_mae):
    """Generate predictions"""
    
    predictions = []
    
    for idx, row in features_df.iterrows():
        driver = row['driver']
        grid_pos = int(row['Grid_Position'])
        
        # Features in order
        feature_values = [row[fname] for fname in feature_names]
        feature_values = np.array(feature_values).reshape(1, -1)
        
        # Scale & predict
        feature_values_scaled = scaler.transform(feature_values)
        predicted_pos = model.predict(feature_values_scaled)[0]
        predicted_pos = np.clip(predicted_pos, 1, 20)
        
        # DON'T round yet - keep raw prediction for ranking
        
        expected_change = grid_pos - predicted_pos
        
        # Confidence intervals
        conf_68_lower = max(1, int(predicted_pos - model_mae))
        conf_68_upper = min(20, int(predicted_pos + model_mae))
        
        predictions.append({
            'driver': driver,
            'grid_position': grid_pos,
            'predicted_position_raw': predicted_pos,  # Raw (8.7, 8.9, etc.)
            'predicted_position_int': int(round(predicted_pos)),  # Rounded
            'expected_change': expected_change,
            'confidence_68_lower': conf_68_lower,
            'confidence_68_upper': conf_68_upper
        })
    
    df = pd.DataFrame(predictions)
    
    # Sort by RAW prediction (keeps order)
    df = df.sort_values('predicted_position_raw')
    
    # Add RANK (1st, 2nd, 3rd)
    df['predicted_rank'] = range(1, len(df) + 1)
    
    return df


def predict_race(race_name, year):
    """Main prediction"""
    
    print("\n" + "="*70)
    print(" GENERATING PREDICTIONS")
    print("="*70)
    print(f"\n {race_name} {year}")
    
    # Load model
    print("\n Loading model...")
    model, scaler, metadata, feature_names = load_model_artifacts()
    
    if model is None:
        return False
    
    model_version = metadata.get('model_type', 'v1.0_unbiased_25features')
    model_mae = metadata.get('test_mae', 2.34)
    
    print(f" Model: {model_version}")
    print(f"   MAE: {model_mae:.2f} | Features: {len(feature_names)}")
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT race_id, status FROM races WHERE year = ? AND race_name = ?", (year, race_name))
        result = cursor.fetchone()
        
        if result is None:
            print(f"\n Race not found!")
            print(f"   Run: python extract_race.py --race \"{race_name}\" --year {year}")
            return False
        
        race_id, status = result
        
        if status != 'features_ready':
            print(f"\n  Race status: {status}")
            if status == 'upcoming':
                print(f"   Run extract_race.py first")
                return False
        
        # Load features
        print("\n Loading features...")
        features_df = load_race_features(race_id, conn, feature_names)
        
        if features_df is None or len(features_df) == 0:
            print(" No features!")
            return False
        
        print(f" Loaded {len(features_df)} drivers")
        
        # Predict
        print("\n Predicting...")
        predictions_df = generate_predictions(model, scaler, features_df, feature_names, model_mae)
        
        # Display with RANKS (not repeated integers!)
        print("\n" + "="*70)
        print(" PREDICTED FINISH ORDER (by Model Ranking)")
        print("="*70)
        print(f"{'Rank':<6} {'Driver':<6} {'Grid':<6} {'Pred Pos':<10} {'Change':<8} {'68% CI':<12}")
        print("-" * 70)
        
        for idx, row in predictions_df.iterrows():
            rank = row['predicted_rank']
            driver = row['driver']
            grid = row['grid_position']
            pred_raw = row['predicted_position_raw']
            pred_int = row['predicted_position_int']
            
            change = row['expected_change']
            if change > 0:
                change_str = f"↑{abs(int(change))}"
            elif change < 0:
                change_str = f"↓{abs(int(change))}"
            else:
                change_str = "="
            
            ci_str = f"P{row['confidence_68_lower']}-{row['confidence_68_upper']}"
            
            # Show rank (clear!) and raw prediction (shows differences)
            print(f"{rank:<5} {driver:<6} P{grid:<4} P{pred_raw:<8.1f} {change_str:<8} {ci_str:<12}")
        
        # Store predictions
        print("\n Storing...")
        
        cursor.execute("DELETE FROM predictions WHERE race_id = ?", (race_id,))
        
        for idx, row in predictions_df.iterrows():
            cursor.execute("""
                INSERT INTO predictions (
                    race_id, driver, grid_position,
                    predicted_position, predicted_position_int, expected_change,
                    confidence_68_lower, confidence_68_upper,
                    confidence_95_lower, confidence_95_upper,
                    model_version, model_mae
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id, row['driver'], row['grid_position'],
                row['predicted_position_raw'], row['predicted_position_int'], row['expected_change'],
                row['confidence_68_lower'], row['confidence_68_upper'],
                row['confidence_68_lower'] - 2, row['confidence_68_upper'] + 2,  # 95% CI
                model_version, model_mae
            ))
        
        cursor.execute("""
            UPDATE races 
            SET predictions_generated_at = ?, status = 'predicted'
            WHERE race_id = ?
        """, (datetime.now().isoformat(), race_id))
        
        conn.commit()
        
        print(f" Stored {len(predictions_df)} predictions")
        
        top3 = predictions_df.head(3)
        print(f"\n Predicted Podium:")
        for idx, row in top3.iterrows():
            rank = row['predicted_rank']
            print(f"   #{rank}: {row['driver']} (Grid P{row['grid_position']} → Pred P{row['predicted_position_raw']:.1f})")
        
        print(f"\n After race: python persist_results.py --race \"{race_name}\" --year {year}")
        
        return True
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--race', required=True)
    parser.add_argument('--year', type=int, required=True)
    
    args = parser.parse_args()
    
    success = predict_race(args.race, args.year)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()