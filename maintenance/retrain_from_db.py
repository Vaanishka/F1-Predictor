"""
Train Model from Database Data

Reads training_data_from_db.csv (created by create_csv.py)
Trains new model, updates database with new version.

Usage:
    python train_model_from_db.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import sqlite3
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'
DB_FILE = BASE_DIR / 'database' / 'f1_predictions.db'
VIZ_DIR = BASE_DIR / 'visualizations'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load training data from database export"""
    
    data_file = DATA_DIR / 'training_data_from_db.csv'
    
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        print("   Run: python create_csv.py first!")
        return None
    
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} records from {df['Race'].nunique()} races")
    
    return df


def prepare_features(df):
    """Prepare features and split data"""
    
    race_df = df[df['Is_Race'] == 1].copy()
    race_df = race_df[race_df['Target_Final_Position'].notna()]
    race_df = race_df[
        (race_df['Target_Final_Position'] >= 1) &
        (race_df['Target_Final_Position'] <= 20)
    ].copy()
    
    print(f"✅ Race data: {len(race_df)} records from {race_df['Race'].nunique()} races")
    
    # 25 features
    feature_cols = [
        'Grid_Position', 'Grid_Position_Pct',
        'Speed_Delta_From_Avg_Pct', 'P95_Speed', 'Max_Speed',
        'Avg_True_Pace', 'Max_True_Pace', 'Avg_Fuel_Adjusted_Pace', 'Avg_Pace_Correction',
        'Starting_Fuel_kg', 'Avg_Fuel_kg', 'Fuel_Confidence',
        'Tyre_Grip', 'Starting_Tyre_Age', 'Avg_Tyre_Age', 'Total_Tire_Degradation',
        'Driver_Consistency',
        'Overtake_Factor', 'Grid_Importance', 'Projected_Stability',
        'Air_Temp_Celsius',
        'Pace_Delta_From_Avg',
        'Risk_Factor_Pct',
        'Grid_X_Overtake', 'Tire_X_Fuel', 'Speed_X_Grip'
    ]
    
    # Stratified split
    unique_races = race_df['Race'].unique()
    
    if len(unique_races) < 3:
        print(f"❌ Need at least 3 races (have {len(unique_races)})")
        return None, None, None, None, None
    
    # Simple split: last 20% for test
    n_test = max(1, len(unique_races) // 5)
    test_races = unique_races[-n_test:]
    train_races = unique_races[:-n_test]
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Training: {len(train_races)} races")
    for r in train_races:
        print(f"      - {r}")
    print(f"   Test: {len(test_races)} races")
    for r in test_races:
        print(f"      - {r}")
    
    train_df = race_df[race_df['Race'].isin(train_races)].copy()
    test_df = race_df[race_df['Race'].isin(test_races)].copy()
    
    print(f"\n   Training: {len(train_df)} records")
    print(f"   Test: {len(test_df)} records")
    
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['Target_Final_Position'].values
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['Target_Final_Position'].values
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train, feature_names):
    """Train XGBoost model"""
    
    print("\n🔧 Training model...")
    
    model = XGBRegressor(
        n_estimators=80,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=2.0,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("✅ Model trained")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate performance"""
    
    print("\n📊 Evaluation:")
    
    y_train_pred = np.clip(model.predict(X_train), 1, 20)
    y_test_pred = np.clip(model.predict(X_test), 1, 20)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n   Train MAE: {train_mae:.2f}")
    print(f"   Test MAE:  {test_mae:.2f}")
    print(f"   Test R²:   {test_r2:.3f}")
    
    gap = test_mae - train_mae
    if gap < 0.8:
        print(f"   ✅ Excellent generalization (gap: {gap:.2f})")
    elif gap < 1.5:
        print(f"   ✅ Good generalization (gap: {gap:.2f})")
    else:
        print(f"   ⚠️  Some overfitting (gap: {gap:.2f})")
    
    return {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'test_r2': test_r2
    }


def save_model(model, scaler, feature_names, metrics, race_count):
    """Save model and update database"""
    
    # Save files
    joblib.dump(model, MODEL_DIR / 'xgboost_model.pkl')
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
    
    with open(MODEL_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    model_version = f"v1.1_{race_count}races_db"
    
    metadata = {
        'model_type': f'XGBoost Regressor ({race_count} races from database)',
        'n_features': len(feature_names),
        'train_mae': float(metrics['train_mae']),
        'test_mae': float(metrics['test_mae']),
        'test_r2': float(metrics['test_r2']),
        'trained_from': 'database',
        'trained_at': datetime.now().isoformat()
    }
    
    with open(MODEL_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Deactivate old models
    cursor.execute("UPDATE model_metadata SET is_active = 0")
    
    # Insert new model
    cursor.execute("""
        INSERT OR REPLACE INTO model_metadata (
            model_version, training_mae, test_mae, test_r2,
            n_features, model_file_path, scaler_file_path,
            is_active, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_version,
        metrics['train_mae'],
        metrics['test_mae'],
        metrics['test_r2'],
        len(feature_names),
        'data/models/xgboost_model.pkl',
        'data/models/scaler.pkl',
        1,
        f'Retrained from database with {race_count} races. MAE: {metrics["test_mae"]:.2f}'
    ))
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Model saved:")
    print(f"   Version: {model_version}")
    print(f"   Files: {MODEL_DIR}")
    print(f"   Database: Updated")


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("🔄 TRAINING MODEL FROM DATABASE")
    print("="*70)
    
    # Load
    df = load_data()
    if df is None:
        return
    
    race_count = df['Race'].nunique()
    
    # Prepare
    result = prepare_features(df)
    if result[0] is None:
        return
    
    X_train, X_test, y_train, y_test, feature_names = result
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = train_model(X_train_scaled, y_train, feature_names)
    
    # Evaluate
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save
    save_model(model, scaler, feature_names, metrics, race_count)
    
    print("\n" + "="*70)
    print("✅ RETRAINING COMPLETE!")
    print("="*70)
    print(f"\n🎯 New model active:")
    print(f"   MAE: {metrics['test_mae']:.2f}")
    print(f"   R²:  {metrics['test_r2']:.3f}")
    print(f"\n🎯 Next predictions will use this model automatically!")


if __name__ == "__main__":
    main()