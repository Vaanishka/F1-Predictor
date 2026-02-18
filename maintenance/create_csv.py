"""
Retrain Model from Database

Uses completed races in database instead of re-extracting from FastF1.
Much faster: <1 minute vs 10+ minutes!

"""

import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / 'database' / 'f1_predictions.db'
OUTPUT_DIR = BASE_DIR / 'data' / 'processed'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_completed_races_from_db():
    """Load all completed races with features + actual results"""
    
    conn = sqlite3.connect(DB_FILE)
    
    # Get completed races
    query = """
        SELECT 
            r.race_id,
            r.race_name,
            r.year,
            rf.driver,
            rf.session_type,
            rf.features_json,
            ar.final_position
        FROM races r
        JOIN race_features rf ON r.race_id = rf.race_id
        JOIN actual_results ar ON r.race_id = ar.race_id AND rf.driver = ar.driver
        WHERE r.status = 'completed'
        AND rf.session_type IN ('FP2', 'Q')
        ORDER BY r.year, r.race_name, rf.driver, rf.session_type
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def aggregate_features(df):
    """Aggregate FP2 + Q features per driver per race"""
    
    all_race_data = []
    
    # Group by race and driver
    for (race_id, race_name, year, driver), group in df.groupby(['race_id', 'race_name', 'year', 'driver']):
        
        # Parse JSON features
        features_list = []
        for _, row in group.iterrows():
            features = json.loads(row['features_json'])
            features_list.append(features)
        
        if len(features_list) == 0:
            continue
        
        # Aggregate features (average FP2 + Q)
        agg_features = {}
        
        # Get all feature names from first record
        feature_names = list(features_list[0].keys())
        
        for feat_name in feature_names:
            values = [f.get(feat_name, 0) for f in features_list]
            
            # Grid features: take last (from Q)
            if 'Grid' in feat_name:
                agg_features[feat_name] = values[-1]
            else:
                agg_features[feat_name] = np.mean(values)
        
        # Add metadata
        agg_features['Driver'] = driver
        agg_features['Race'] = race_name
        agg_features['Session_Type'] = 'R'
        agg_features['Target_Final_Position'] = group['final_position'].iloc[0]
        
        # Session flags
        agg_features['Is_Qualy'] = 0
        agg_features['Is_Practice'] = 0
        agg_features['Is_Race'] = 1
        
        all_race_data.append(agg_features)
    
    return pd.DataFrame(all_race_data)


def retrain_from_db(min_races=6):
    """Main retraining pipeline"""
    
    print("\n" + "="*70)
    print("🔄 RETRAINING MODEL FROM DATABASE")
    print("="*70)
    
    # Load data
    print("\n📥 Loading completed races from database...")
    df_raw = load_completed_races_from_db()
    
    if len(df_raw) == 0:
        print("❌ No completed races in database!")
        print("   Run persist_results.py on some races first")
        return False
    
    unique_races = df_raw.groupby(['race_name', 'year']).size()
    print(f"✅ Found {len(unique_races)} completed races:")
    for (race, year), count in unique_races.items():
        print(f"   {year} {race}: {count} driver records")
    
    if len(unique_races) < min_races:
        print(f"\n⚠️  Only {len(unique_races)} races (need {min_races} minimum)")
        print("   Add more races or use --min-races to lower threshold")
        return False
    
    # Aggregate features
    print(f"\n🔧 Aggregating features...")
    training_df = aggregate_features(df_raw)
    
    print(f"✅ Created training dataset:")
    print(f"   Total records: {len(training_df)}")
    print(f"   Races: {training_df['Race'].nunique()}")
    print(f"   Drivers: {training_df['Driver'].nunique()}")
    
    # Save
    output_file = OUTPUT_DIR / 'training_data_from_db.csv'
    training_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved: {output_file}")
    
    print("\n" + "="*70)
    print("✅ READY FOR TRAINING!")
    print("="*70)
    print(f"\n🎯 Next step:")
    print(f"   python train_model_from_db.py")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Retrain from database')
    parser.add_argument('--min-races', type=int, default=6, 
                       help='Minimum races required (default: 6)')
    
    args = parser.parse_args()
    
    success = retrain_from_db(args.min_races)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()