-- F1 Prediction Database Schema - 25 Features (JSON Storage)

CREATE TABLE IF NOT EXISTS races (
    race_id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    race_name TEXT NOT NULL,
    circuit TEXT NOT NULL,
    race_date DATE,
    features_extracted_at TIMESTAMP,
    predictions_generated_at TIMESTAMP,
    results_persisted_at TIMESTAMP,
    status TEXT DEFAULT 'upcoming',
    UNIQUE(year, race_name)
);

CREATE INDEX idx_races_year ON races(year);
CREATE INDEX idx_races_status ON races(status);

CREATE TABLE IF NOT EXISTS race_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    driver TEXT NOT NULL,
    session_type TEXT NOT NULL,
    grid_position INTEGER,
    avg_speed REAL,
    data_quality TEXT DEFAULT 'OK',
    features_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE,
    UNIQUE(race_id, driver, session_type)
);

CREATE INDEX idx_features_race ON race_features(race_id);
CREATE INDEX idx_features_driver ON race_features(driver);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    driver TEXT NOT NULL,
    grid_position INTEGER,
    predicted_position REAL,
    predicted_position_int INTEGER,
    expected_change INTEGER,
    confidence_68_lower INTEGER,
    confidence_68_upper INTEGER,
    confidence_95_lower INTEGER,
    confidence_95_upper INTEGER,
    model_version TEXT,
    model_mae REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE,
    UNIQUE(race_id, driver, model_version)
);

CREATE INDEX idx_predictions_race ON predictions(race_id);

CREATE TABLE IF NOT EXISTS actual_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    driver TEXT NOT NULL,
    final_position INTEGER,
    grid_position INTEGER,
    points_scored REAL,
    status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE,
    UNIQUE(race_id, driver)
);

CREATE INDEX idx_results_race ON actual_results(race_id);

CREATE TABLE IF NOT EXISTS model_performance (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    model_version TEXT,
    mae REAL,
    rmse REAL,
    r2_score REAL,
    top3_accuracy REAL,
    top10_accuracy REAL,
    best_prediction_driver TEXT,
    best_prediction_error REAL,
    worst_prediction_driver TEXT,
    worst_prediction_error REAL,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE
);

CREATE INDEX idx_performance_race ON model_performance(race_id);

CREATE TABLE IF NOT EXISTS model_metadata (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT UNIQUE NOT NULL,
    trained_on_races TEXT,
    training_mae REAL,
    test_mae REAL,
    test_r2 REAL,
    n_features INTEGER,
    feature_names TEXT,
    model_file_path TEXT,
    scaler_file_path TEXT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1,
    notes TEXT
);

CREATE VIEW IF NOT EXISTS v_latest_predictions AS
SELECT 
    r.year,
    r.race_name,
    p.driver,
    p.grid_position,
    p.predicted_position_int as predicted_position,
    p.expected_change,
    r.status
FROM predictions p
JOIN races r ON p.race_id = r.race_id
ORDER BY r.year DESC, p.predicted_position_int ASC;

CREATE VIEW IF NOT EXISTS v_prediction_accuracy AS
SELECT 
    r.year,
    r.race_name,
    p.driver,
    p.predicted_position_int as predicted,
    a.final_position as actual,
    ABS(p.predicted_position_int - a.final_position) as error
FROM predictions p
JOIN races r ON p.race_id = r.race_id
JOIN actual_results a ON p.race_id = a.race_id AND p.driver = a.driver
WHERE r.status = 'completed'
ORDER BY error ASC;

INSERT OR REPLACE INTO model_metadata (
    model_version,
    trained_on_races,
    training_mae,
    test_mae,
    test_r2,
    n_features,
    model_file_path,
    scaler_file_path,
    is_active,
    notes
) VALUES (
    'v1.0_unbiased_25features',
    'Bahrain, Monaco, Canada, Austria, Singapore, Brazil',
    1.52,
    2.34,
    0.72,
    25,
    'data/models/xgboost_model.pkl',
    'data/models/scaler.pkl',
    1,
    '100% data-driven. Train MAE: 1.52, Test MAE: 2.34.'
);