"""
Prediction Tracker Module
Logs all predictions with full context for learning
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

class PredictionTracker:
    def __init__(self, db_path: str = "learning.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure all required tables exist with proper schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enhanced predictions table with more context
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    stock TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence FLOAT,
                    target_date DATE,
                    entry_price FLOAT,
                    predicted_price FLOAT,
                    llm_model TEXT,
                    reasoning TEXT,
                    rsi FLOAT,
                    macd FLOAT,
                    volume_ratio FLOAT,
                    patterns TEXT,
                    market_regime TEXT,
                    macro_score FLOAT,
                    sector TEXT,
                    news_sentiment FLOAT
                )
            """)
            
            # Enhanced outcomes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    check_date DATE DEFAULT CURRENT_DATE,
                    actual_price FLOAT,
                    actual_move_pct FLOAT,
                    success BOOLEAN,
                    failure_reason TEXT,
                    market_condition TEXT,
                    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
                )
            """)
            
            # Pattern performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_performance (
                    pattern_name TEXT PRIMARY KEY,
                    success_count INTEGER DEFAULT 0,
                    fail_count INTEGER DEFAULT 0,
                    success_rate FLOAT DEFAULT 0,
                    avg_return FLOAT DEFAULT 0,
                    last_updated DATE DEFAULT CURRENT_DATE,
                    market_regime TEXT
                )
            """)
            
            # LLM performance by context
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_context_performance (
                    llm_model TEXT,
                    context_type TEXT,
                    context_value TEXT,
                    success_rate FLOAT,
                    sample_size INTEGER,
                    PRIMARY KEY(llm_model, context_type, context_value)
                )
            """)
            
            conn.commit()
    
    def log_prediction(self, 
                      stock: str,
                      prediction: str,
                      confidence: float,
                      price: float,
                      llm_model: str,
                      reasoning: str,
                      indicators: Dict,
                      patterns: List[str],
                      market_context: Dict) -> int:
        """
        Log a prediction with full context
        Returns prediction_id for future reference
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate target date (1 day ahead for your requirement)
                target_date = (datetime.now() + timedelta(days=1)).date()
                
                # Calculate predicted price based on prediction type
                predicted_price = self._calculate_predicted_price(
                    prediction, price, confidence
                )
                
                cursor.execute("""
                    INSERT INTO predictions (
                        stock, prediction, confidence, target_date,
                        entry_price, predicted_price, llm_model, reasoning,
                        rsi, macd, volume_ratio, patterns,
                        market_regime, macro_score, sector, news_sentiment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stock, 
                    prediction, 
                    confidence,
                    target_date,
                    price,
                    predicted_price,
                    llm_model,
                    reasoning,
                    indicators.get('rsi'),
                    indicators.get('macd'),
                    indicators.get('volume_ratio'),
                    json.dumps(patterns),
                    market_context.get('regime', 'unknown'),
                    market_context.get('macro_score', 0),
                    market_context.get('sector', ''),
                    market_context.get('news_sentiment', 0)
                ))
                
                prediction_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Logged prediction #{prediction_id}: {stock} {prediction} @ {price}")
                return prediction_id
                
        except Exception as e:
            self.logger.error(f"Failed to log prediction: {e}")
            return -1
    
    def _calculate_predicted_price(self, prediction: str, current_price: float, confidence: float) -> float:
        """Calculate expected price based on prediction type"""
        # Your 2% threshold for success
        if prediction == "BUY":
            return current_price * 1.02  # Expecting 2% gain
        elif prediction == "SELL":
            return current_price * 0.98  # Expecting 2% drop
        else:  # HOLD
            return current_price  # Expecting no significant change
    
    def get_pending_checks(self) -> List[Dict]:
        """Get predictions that need outcome checking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get predictions that haven't been checked yet
            cursor.execute("""
                SELECT p.id, p.stock, p.prediction, p.entry_price, 
                       p.predicted_price, p.llm_model, p.target_date,
                       p.confidence, p.patterns, p.reasoning
                FROM predictions p
                LEFT JOIN outcomes o ON p.id = o.prediction_id
                WHERE o.id IS NULL 
                AND date(p.target_date) <= date('now')
                ORDER BY p.timestamp DESC
            """)
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_historical_accuracy(self, llm_model: str = None, stock: str = None, 
                               days_back: int = 30) -> Dict:
        """Get historical accuracy metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    p.llm_model,
                    COUNT(*) as total,
                    SUM(CASE WHEN o.success = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(o.actual_move_pct) as avg_move,
                    AVG(p.confidence) as avg_confidence
                FROM predictions p
                JOIN outcomes o ON p.id = o.prediction_id
                WHERE date(p.timestamp) >= date('now', '-{} days')
            """.format(days_back)
            
            params = []
            if llm_model:
                query += " AND p.llm_model = ?"
                params.append(llm_model)
            if stock:
                query += " AND p.stock = ?"
                params.append(stock)
            
            query += " GROUP BY p.llm_model"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            accuracy_data = {}
            for row in results:
                model = row[0]
                accuracy_data[model] = {
                    'total_predictions': row[1],
                    'correct_predictions': row[2],
                    'accuracy': (row[2] / row[1] * 100) if row[1] > 0 else 0,
                    'avg_move': row[3],
                    'avg_confidence': row[4]
                }
            
            return accuracy_data
    
    def get_pattern_performance(self) -> Dict:
        """Get performance metrics for each pattern"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pattern_name, success_rate, success_count + fail_count as total,
                       avg_return, market_regime
                FROM pattern_performance
                WHERE success_count + fail_count > 5
                ORDER BY success_rate DESC
            """)
            
            patterns = {}
            for row in cursor.fetchall():
                patterns[row[0]] = {
                    'success_rate': row[1],
                    'sample_size': row[2],
                    'avg_return': row[3],
                    'best_regime': row[4]
                }
            
            return patterns
