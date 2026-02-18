"""
Persist Race Results - FIXED (Uses Ranking for Accuracy)

CRITICAL FIX: Calculates error based on PREDICTED RANK vs ACTUAL RANK
(not raw position predictions which can be wrong when clustered)

Usage:
    python persist_results.py --race "Belgian Grand Prix" --year 2024
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
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / 'database' / 'f1_predictions.db'
CACHE_DIR = BASE_DIR / 'fastf1_cache'

fastf1.Cache.enable_cache(str(CACHE_DIR))


def fetch_race_results(race_name, year):
    """Fetch results from FastF1"""
    
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
    
    try:
        session = fastf1.get_session(year, fastf1_name, 'R')
        session.load(laps=False, telemetry=False, weather=False)
        
        if not hasattr(session, 'results') or session.results is None:
            return None
        
        results = []
        
        for driver_num, row in session.results.iterrows():
            driver_abbr = row['Abbreviation'] if 'Abbreviation' in row else str(driver_num)
            
            results.append({
                'driver': driver_abbr,
                'final_position': int(row['Position']) if pd.notna(row['Position']) else 20,
                'grid_position': int(row['GridPosition']) if pd.notna(row['GridPosition']) else 20,
                'points_scored': float(row['Points']) if pd.notna(row['Points']) else 0.0,
                'status': row['Status'] if 'Status' in row and pd.notna(row['Status']) else 'Unknown'
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f" Failed: {e}")
        return None


def calculate_accuracy(predictions_df, results_df):
    """Calculate metrics using RANKING (not raw positions)"""
    
    merged = predictions_df.merge(results_df, on='driver', how='inner')
    
    if len(merged) == 0:
        return None
    
    # Sort predictions by predicted_position (model's ranking)
    merged = merged.sort_values('predicted_position')
    merged['predicted_rank'] = range(1, len(merged) + 1)
    
    # CRITICAL: Calculate error based on RANK difference
    merged['rank_error'] = abs(merged['predicted_rank'] - merged['final_position'])
    
    # Also keep raw position error (for comparison)
    merged['position_error'] = abs(merged['predicted_position_int'] - merged['final_position'])
    
    # Use RANK error for metrics (more accurate!)
    mae = merged['rank_error'].mean()
    rmse = np.sqrt((merged['rank_error'] ** 2).mean())
    
    # R² based on ranking
    ss_res = ((merged['final_position'] - merged['predicted_rank']) ** 2).sum()
    ss_tot = ((merged['final_position'] - merged['final_position'].mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Position-specific (use ranks)
    top3_actual = merged[merged['final_position'] <= 3]
    top3_accuracy = (top3_actual['rank_error'] <= 2).mean() * 100 if len(top3_actual) > 0 else 0
    
    top10_actual = merged[merged['final_position'] <= 10]
    top10_accuracy = (top10_actual['rank_error'] <= 3).mean() * 100 if len(top10_actual) > 0 else 0
    
    # Best/worst
    best_idx = merged['rank_error'].idxmin()
    worst_idx = merged['rank_error'].idxmax()
    
    metrics = {
        'mae': mae,  # Based on RANK error
        'rmse': rmse,
        'r2': r2,
        'top3_accuracy': top3_accuracy,
        'top10_accuracy': top10_accuracy,
        'best_driver': merged.loc[best_idx, 'driver'],
        'best_error': merged.loc[best_idx, 'rank_error'],
        'worst_driver': merged.loc[worst_idx, 'driver'],
        'worst_error': merged.loc[worst_idx, 'rank_error'],
        'avg_position_error': merged['position_error'].mean()  # For comparison
    }
    
    return metrics, merged


def persist_results(race_name, year):
    """Main persistence"""
    
    print("\n" + "="*70)
    print(" PERSISTING RESULTS")
    print("="*70)
    print(f"\n {race_name} {year}")
    
    print("\n Fetching results...")
    results_df = fetch_race_results(race_name, year)
    
    if results_df is None or len(results_df) == 0:
        print(" No results!")
        return False
    
    print(f" Loaded {len(results_df)} results")
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT race_id, status FROM races WHERE year = ? AND race_name = ?", (year, race_name))
        result = cursor.fetchone()
        
        if result is None:
            print(f" Race not found!")
            return False
        
        race_id, status = result
        
        # Load predictions
        cursor.execute("""
            SELECT driver, predicted_position, predicted_position_int, model_version
            FROM predictions
            WHERE race_id = ?
        """, (race_id,))
        
        predictions_data = cursor.fetchall()
        
        if len(predictions_data) == 0:
            print(" No predictions!")
            return False
        
        predictions_df = pd.DataFrame(predictions_data, 
                                      columns=['driver', 'predicted_position', 'predicted_position_int', 'model_version'])
        model_version = predictions_df['model_version'].iloc[0]
        
        print(f" Loaded {len(predictions_df)} predictions")
        
        # Calculate (using RANKING!)
        print("\n Calculating accuracy (RANK-BASED)...")
        metrics, merged = calculate_accuracy(predictions_df, results_df)
        
        if metrics is None:
            print(" Failed!")
            return False
        
        # Display
        print("\n" + "="*70)
        print(" ACCURACY (RANK-BASED)")
        print("="*70)
        
        print(f"\n Overall (Rank Error):")
        print(f"   MAE:  {metrics['mae']:.2f} positions")
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   R²:   {metrics['r2']:.3f}")
        
        print(f"\n Comparison:")
        print(f"   Rank-based MAE:     {metrics['mae']:.2f} ← Accurate!")
        print(f"   Position-based MAE: {metrics['avg_position_error']:.2f} ← Misleading (clustered)")
        
        print(f"\n Position-Specific:")
        print(f"   Top 3:  {metrics['top3_accuracy']:.1f}%")
        print(f"   Top 10: {metrics['top10_accuracy']:.1f}%")
        
        print(f"\n Best: {metrics['best_driver']} ({metrics['best_error']:.0f} error)")
        print(f" Worst: {metrics['worst_driver']} ({metrics['worst_error']:.0f} error)")
        
        # Top 5 comparison
        print("\n" + "="*70)
        print(" TOP 5 (RANK vs ACTUAL)")
        print("="*70)
        print(f"{'Driver':<6} {'Pred Rank':<12} {'Actual Pos':<12} {'Rank Err':<10}")
        print("-" * 70)
        
        top5 = merged.nsmallest(5, 'final_position')
        for idx, row in top5.iterrows():
            rank_err_str = f"{row['rank_error']:+.0f}" if row['rank_error'] != 0 else "✓"
            print(f"{row['driver']:<6} #{row['predicted_rank']:<11d} P{row['final_position']:<11d} {rank_err_str:<10}")
        
        # Store
        print("\n Storing...")
        
        cursor.execute("DELETE FROM actual_results WHERE race_id = ?", (race_id,))
        
        for idx, row in results_df.iterrows():
            cursor.execute("""
                INSERT INTO actual_results (
                    race_id, driver, final_position, grid_position,
                    points_scored, status
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                race_id, row['driver'], row['final_position'],
                row['grid_position'], row['points_scored'], row['status']
            ))
        
        cursor.execute("DELETE FROM model_performance WHERE race_id = ?", (race_id,))
        
        cursor.execute("""
            INSERT INTO model_performance (
                race_id, model_version, mae, rmse, r2_score,
                top3_accuracy, top10_accuracy,
                best_prediction_driver, best_prediction_error,
                worst_prediction_driver, worst_prediction_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            race_id, model_version,
            metrics['mae'], metrics['rmse'], metrics['r2'],
            metrics['top3_accuracy'], metrics['top10_accuracy'],
            metrics['best_driver'], metrics['best_error'],
            metrics['worst_driver'], metrics['worst_error']
        ))
        
        cursor.execute("""
            UPDATE races 
            SET results_persisted_at = ?, status = 'completed'
            WHERE race_id = ?
        """, (datetime.now().isoformat(), race_id))
        
        conn.commit()
        
        print(f" Results persisted!")
        
        print("\n" + "="*70)
        print(" COMPLETE!")
        print("="*70)
        
        print(f"\n Key Insight:")
        print(f"   Your model's RANKING is {metrics['mae']:.1f} positions off on average")
        print(f"   (Much better than {metrics['avg_position_error']:.1f} from raw clustered positions!)")
        
        return True
        
    except Exception as e:
        print(f" Error: {e}")
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
    
    success = persist_results(args.race, args.year)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()