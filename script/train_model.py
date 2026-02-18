"""
F1 Model Training - FINAL (25 Features - 100% Data-Driven)

NO HARDCODED BIAS:
- No constructor standings
- No driver ratings
- All features from actual race data

Expected: MAE 1.8-2.1, R² 0.76-0.83
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR  / 'models'
VIZ_DIR = BASE_DIR / 'visualizations'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load training data"""
    data_file = DATA_DIR / 'training_data_2024_final.csv'
    
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        print("   Run: python scripts/load_data.py first!")
        return None
    
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} records")
    
    return df


def prepare_features(df):
    """Prepare 25 features with stratified split"""
    
    # Filter to race data
    race_df = df[df['Is_Race'] == 1].copy()
    race_df = race_df[race_df['Target_Final_Position'].notna()]
    
    print(f"✅ Race data: {len(race_df)} records from {race_df['Race'].nunique()} races")
    
    # Show distribution
    print(f"\n📊 Race distribution:")
    for race, count in race_df['Race'].value_counts().items():
        print(f"   {race}: {count} drivers")
    
    # Remove target outliers
    race_df = race_df[
        (race_df['Target_Final_Position'] >= 1) &
        (race_df['Target_Final_Position'] <= 20)
    ].copy()
    
    # STRATIFIED SPLIT by track type
    unique_races = race_df['Race'].unique()
    
    if len(unique_races) < 3:
        print(f"\n❌ ERROR: Need at least 3 races")
        return None, None, None, None, None
    
    # Categorize by track type
    street_circuits = ['Monaco Grand Prix', 'Singapore Grand Prix']
    permanent = ['Bahrain Grand Prix', 'Canadian Grand Prix', 'Austrian Grand Prix', 'São Paulo Grand Prix']
    
    # Stratified test set (1 street + 1 permanent)
    test_races = []
    train_races = []
    
    street_in_data = [r for r in street_circuits if r in unique_races]
    if street_in_data:
        test_races.append(street_in_data[0])
    
    permanent_in_data = [r for r in permanent if r in unique_races and r not in test_races]
    if permanent_in_data:
        test_races.append(permanent_in_data[0])
    
    train_races = [r for r in unique_races if r not in test_races]
    
    print(f"\n📊 Stratified Train/Test Split:")
    print(f"   Training races ({len(train_races)}):")
    for r in train_races:
        print(f"      - {r}")
    print(f"   Test races ({len(test_races)}):")
    for r in test_races:
        print(f"      - {r}")
    
    # Split data
    train_df = race_df[race_df['Race'].isin(train_races)].copy()
    test_df = race_df[race_df['Race'].isin(test_races)].copy()
    
    print(f"\n   Training: {len(train_df)} records")
    print(f"   Test: {len(test_df)} records")
    print(f"   Split ratio: {len(test_df)/(len(train_df)+len(test_df))*100:.1f}% test")
    
    # Define 25 feature columns (IN ORDER!)
    feature_cols = [
        # Grid (2)
        'Grid_Position', 'Grid_Position_Pct',
        # Speed (3)
        'Speed_Delta_From_Avg_Pct', 'P95_Speed', 'Max_Speed',
        # Pace (7)
        'Avg_True_Pace', 'Max_True_Pace', 'Avg_Fuel_Adjusted_Pace', 'Avg_Pace_Correction',
        # Fuel (3)
        'Starting_Fuel_kg', 'Avg_Fuel_kg', 'Fuel_Confidence',
        # Tire (4)
        'Tyre_Grip', 'Starting_Tyre_Age', 'Avg_Tyre_Age', 'Total_Tire_Degradation',
        # Driver Consistency (1) - CALCULATED FROM LAPS
        'Driver_Consistency',
        # Track Intelligence (3)
        'Overtake_Factor', 'Grid_Importance', 'Projected_Stability',
        # Weather (1)
        'Air_Temp_Celsius',
        # Corrections (1)
        'Pace_Delta_From_Avg',
        # Risk (1)
        'Risk_Factor_Pct',
        # Interactions (3)
        'Grid_X_Overtake', 'Tire_X_Fuel', 'Speed_X_Grip'
    ]
    
    # Prepare X, y
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['Target_Final_Position'].values
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['Target_Final_Position'].values
    
    print(f"\n✅ Features: {len(feature_cols)} (100% data-driven)")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train, feature_names):
    """Train with strong regularization"""
    
    print("\n" + "="*70)
    print("🔧 TRAINING MODEL (25 UNBIASED FEATURES)")
    print("="*70)
    
    # Optimized hyperparameters for 25 features
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
    
    # Fit
    model.fit(X_train, y_train)
    
    # Feature importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Top 15 Feature Importance:")
    for idx, row in feature_importance_df.head(15).iterrows():
        print(f"   {row['Feature']:<30s} {row['Importance']:.4f}")
    
    # Check track intelligence
    track_features = ['Grid_X_Overtake', 'Overtake_Factor', 'Grid_Importance', 'Projected_Stability']
    track_importance = feature_importance_df[feature_importance_df['Feature'].isin(track_features)]
    track_total = track_importance['Importance'].sum()
    
    print(f"\n📊 Track Intelligence Total: {track_total:.4f} ({track_total*100:.1f}%)")
    
    grid_importance = feature_importance_df[feature_importance_df['Feature'] == 'Grid_Position']['Importance'].values[0]
    print(f"   Grid_Position alone: {grid_importance:.4f} ({grid_importance*100:.1f}%)")
    
    ratio = track_total / grid_importance
    if ratio > 0.6:
        print(f"   ✅ EXCELLENT: Track intelligence is {ratio:.1f}x grid importance")
    elif ratio > 0.4:
        print(f"   ✅ GOOD: Track intelligence is {ratio:.1f}x grid importance")
    else:
        print(f"   ⚠️  Track intelligence could be stronger ({ratio:.1f}x)")
    
    print("\n✅ Model trained")
    
    return model, feature_importance_df


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model"""
    
    print("\n" + "="*70)
    print("📊 MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    y_train_pred = np.clip(model.predict(X_train), 1, 20)
    y_test_pred = np.clip(model.predict(X_test), 1, 20)
    
    # Training metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n📊 TRAINING SET:")
    print(f"   MAE:  {train_mae:.2f} positions")
    print(f"   RMSE: {train_rmse:.2f}")
    print(f"   R²:   {train_r2:.3f}")
    
    print("\n📊 TEST SET:")
    print(f"   MAE:  {test_mae:.2f} positions")
    print(f"   RMSE: {test_rmse:.2f}")
    print(f"   R²:   {test_r2:.3f}")
    
    # Overfitting check
    print("\n🔍 GENERALIZATION CHECK:")
    mae_gap = test_mae - train_mae
    r2_gap = train_r2 - test_r2
    
    if mae_gap < 0.5 and r2_gap < 0.15:
        print(f"   ✅ EXCELLENT: Minimal overfitting")
        print(f"      MAE gap: {mae_gap:.2f}, R² gap: {r2_gap:.3f}")
    elif mae_gap < 1.0 and r2_gap < 0.25:
        print(f"   ✅ GOOD: Acceptable generalization")
        print(f"      MAE gap: {mae_gap:.2f}, R² gap: {r2_gap:.3f}")
    elif test_mae < train_mae:
        print(f"   🎉 GREAT: Test better than train!")
    else:
        print(f"   ⚠️  CAUTION: Some overfitting")
        print(f"      MAE gap: {mae_gap:.2f}, R² gap: {r2_gap:.3f}")
    
    # Prediction distribution
    print("\n📊 Prediction Distribution:")
    print(f"   Test predictions - Min: {y_test_pred.min():.1f}, Max: {y_test_pred.max():.1f}, Mean: {y_test_pred.mean():.1f}")
    print(f"   Test actual      - Min: {y_test.min():.0f}, Max: {y_test.max():.0f}, Mean: {y_test.mean():.1f}")
    
    return {
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }


def save_model(model, scaler, feature_names, metrics, feature_importance_df):
    """Save model artifacts"""
    
    joblib.dump(model, MODEL_DIR / 'xgboost_model.pkl')
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
    
    with open(MODEL_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    feature_importance_df.to_csv(MODEL_DIR / 'feature_importance.csv', index=False)
    
    metadata = {
        'model_type': 'XGBoost Regressor (25 Features - 100% Data-Driven)',
        'n_features': len(feature_names),
        'train_mae': float(metrics['train_mae']),
        'test_mae': float(metrics['test_mae']),
        'test_rmse': float(metrics['test_rmse']),
        'test_r2': float(metrics['test_r2']),
        'hyperparameters': {
            'n_estimators': 80,
            'learning_rate': 0.04,
            'max_depth': 3,
            'min_child_weight': 5,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'gamma': 0.1
        },
        'bias_removed': 'Constructor standings (3), Driver ratings (3)',
        'features_kept': '25 data-driven features only',
        'training_races': 'Bahrain, Monaco, Canada, Austria, Singapore, Brazil (stratified)',
        'data_sources': {
            'Grid': 'Qualifying results',
            'Speed': 'Telemetry/lap data',
            'Tire': 'Session data',
            'Fuel': 'Calculated from session type',
            'Driver_Consistency': 'Calculated from lap times',
            'Track': 'track_dna.json',
            'Weather': 'Session weather data'
        },
        'improvements': [
            'ZERO hardcoded bias',
            'All features from actual data',
            'Works for any driver (including rookies)',
            'Enhanced track intelligence',
            'Stronger regularization',
            'Stratified train/test split'
        ]
    }
    
    with open(MODEL_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Models saved to {MODEL_DIR}")


def create_visualizations(model, X_train, y_train, X_test, y_test):
    """Create evaluation plots"""
    
    y_train_pred = np.clip(model.predict(X_train), 1, 20)
    y_test_pred = np.clip(model.predict(X_test), 1, 20)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Train scatter
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, color='blue', s=50)
    axes[0, 0].plot([1, 20], [1, 20], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Position', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Position', fontsize=11)
    axes[0, 0].set_title(f'Training Set (MAE: {mean_absolute_error(y_train, y_train_pred):.2f})', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 21])
    axes[0, 0].set_ylim([0, 21])
    
    # Test scatter
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.7, color='green', s=60)
    axes[0, 1].plot([1, 20], [1, 20], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Position', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Position', fontsize=11)
    axes[0, 1].set_title(f'Test Set (MAE: {mean_absolute_error(y_test, y_test_pred):.2f})', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 21])
    axes[0, 1].set_ylim([0, 21])
    
    # Train errors
    train_errors = y_train - y_train_pred
    axes[1, 0].hist(train_errors, bins=15, color='blue', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error (positions)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Training Error Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Test errors
    test_errors = y_test - y_test_pred
    axes[1, 1].hist(test_errors, bins=15, color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Prediction Error (positions)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Test Error Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {VIZ_DIR}")


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("🏎️  F1 MODEL TRAINING - 25 FEATURES (100% DATA-DRIVEN)")
    print("="*70)
    print("✨ ZERO hardcoded bias - portfolio ready!")
    
    # Load
    df = load_data()
    if df is None:
        return
    
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
    model, feature_importance_df = train_model(X_train_scaled, y_train, feature_names)
    
    # Evaluate
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save
    save_model(model, scaler, feature_names, metrics, feature_importance_df)
    
    # Visualize
    create_visualizations(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    # Success criteria
    if metrics['test_mae'] < 2.5 and abs(metrics['train_mae'] - metrics['test_mae']) < 0.8:
        print("\n🎉 SUCCESS: Unbiased model generalizes well!")
        print(f"   Test MAE: {metrics['test_mae']:.2f} (target: < 2.5)")
        print(f"   Train-Test gap: {abs(metrics['train_mae'] - metrics['test_mae']):.2f} (target: < 0.8)")
        print(f"\n✨ Model Features:")
        print(f"   • ZERO hardcoded bias")
        print(f"   • Works for any driver (rookies included)")
        print(f"   • Track-intelligent (Monaco ≠ Bahrain)")
        print(f"\n🎯 Next Steps:")
        print(f"   1. cd database && python init_db.py")
        print(f"   2. cd scripts && python extract_race.py --race 'Bahrain Grand Prix' --year 2024")
    else:
        print("\n⚠️  Model trained, check metrics above")
        print(f"   Test MAE: {metrics['test_mae']:.2f}")


if __name__ == "__main__":
    main()