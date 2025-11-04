-- Database schema for learning system
-- This tracks all predictions and outcomes

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    stock TEXT NOT NULL,
    prediction TEXT NOT NULL,  -- BUY/HOLD/SELL
    confidence FLOAT,
    target_date DATE,
    entry_price FLOAT,
    predicted_price FLOAT,
    llm_model TEXT,
    reasoning TEXT,
    rsi FLOAT,
    macd FLOAT,
    volume_ratio FLOAT,
    patterns TEXT  -- Comma-separated patterns
);

CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    check_date DATE DEFAULT CURRENT_DATE,
    actual_price FLOAT,
    actual_move_pct FLOAT,
    success BOOLEAN,
    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
);

CREATE TABLE IF NOT EXISTS accuracy_tracking (
    stock TEXT,
    llm_model TEXT,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy_pct FLOAT DEFAULT 0,
    last_updated DATE DEFAULT CURRENT_DATE,
    PRIMARY KEY(stock, llm_model)
);
