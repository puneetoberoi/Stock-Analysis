"""
Outcome Checker - Verifies if predictions were correct
Pure data-driven, no hardcoded rules
"""

import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
import logging

class OutcomeChecker:
    def __init__(self, db_path='src/modules/data/learning.db'):
        self.db_path = db_path
        
        # Success thresholds (only hardcoded constraint)
        self.thresholds = {
            'BUY': 1.5,      # Must go up >1.5%
            'SELL': -1.5,    # Must go down >1.5%
            'HOLD': (-1.0, 1.0)  # Within -1% to +1%
        }
    
    def check_yesterdays_predictions(self):
        """Check all predictions from yesterday"""
        logging.info("🔍 Checking yesterday's predictions...")
        
        # Get predictions from yesterday that haven't been checked
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT p.id, p.stock, p.prediction, p.confidence, p.entry_price, p.timestamp
            FROM predictions p
            LEFT JOIN outcomes o ON p.id = o.prediction_id
            WHERE DATE(p.timestamp) = ? AND o.id IS NULL
        """, (yesterday,))
        
        unchecked = cursor.fetchall()
        conn.close()
        
        logging.info(f"📊 Found {len(unchecked)} predictions to check from {yesterday}")
        
        results = []
        for pred_id, stock, action, confidence, entry_price, timestamp in unchecked:
            result = self._check_single_prediction(
                pred_id, stock, action, confidence, entry_price, timestamp
            )
            if result:
                results.append(result)
        
        logging.info(f"✅ Checked {len(results)} predictions")
        return results
    
    def _check_single_prediction(self, pred_id, stock, action, confidence, entry_price, timestamp):
        """Check a single prediction outcome"""
        try:
            # Get current price
            ticker = yf.Ticker(stock)
            current_data = ticker.history(period='1d')
            
            if current_data.empty:
                logging.warning(f"⚠️ No price data for {stock}")
                return None
            
            current_price = current_data['Close'].iloc[-1]
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Determine if prediction was correct (pure logic, no bias)
            was_correct = self._evaluate_prediction(action, price_change_pct)
            
            # Store outcome
            self._store_outcome(pred_id, current_price, price_change_pct, was_correct)
            
            result = {
                'stock': stock,
                'action': action,
                'entry_price': entry_price,
                'current_price': current_price,
                'change_pct': price_change_pct,
                'was_correct': was_correct,
                'confidence': confidence
            }
            
            status = "✅ RIGHT" if was_correct else "❌ WRONG"
            logging.info(f"{status}: {stock} {action} @${entry_price:.2f} → ${current_price:.2f} ({price_change_pct:+.1f}%)")
            
            return result
            
        except Exception as e:
            logging.error(f"Error checking {stock}: {e}")
            return None
    
    def _evaluate_prediction(self, action, price_change_pct):
        """Evaluate if prediction was correct (pure math, no interpretation)"""
        if action == 'BUY':
            return price_change_pct >= self.thresholds['BUY']
        elif action == 'SELL':
            return price_change_pct <= self.thresholds['SELL']
        elif action == 'HOLD':
            return self.thresholds['HOLD'][0] <= price_change_pct <= self.thresholds['HOLD'][1]
        return False
    
    def _store_outcome(self, pred_id, current_price, price_change_pct, was_correct):
        """Store outcome in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO outcomes (prediction_id, actual_price, actual_move_pct, success)
            VALUES (?, ?, ?, ?)
        """, (pred_id, current_price, price_change_pct, was_correct))
        
        conn.commit()
        conn.close()
    
    def get_performance_summary(self, days=7):
        """Get performance summary for learning context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Overall performance
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN o.success = 1 THEN 1 ELSE 0 END) as correct,
                AVG(o.actual_move_pct) as avg_move
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            WHERE DATE(p.timestamp) >= ?
        """, (cutoff_date,))
        
        total, correct, avg_move = cursor.fetchone()
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Get recent failures for learning
        cursor.execute("""
            SELECT p.stock, p.prediction, p.entry_price, o.actual_price, 
                   o.actual_move_pct, p.reasoning, p.rsi, p.volume_ratio
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            WHERE o.success = 0 AND DATE(p.timestamp) >= ?
            ORDER BY p.timestamp DESC
            LIMIT 5
        """, (cutoff_date,))
        
        failures = cursor.fetchall()
        
        # Get successful patterns for learning
        cursor.execute("""
            SELECT p.stock, p.prediction, o.actual_move_pct, p.rsi, p.volume_ratio
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            WHERE o.success = 1 AND DATE(p.timestamp) >= ?
            ORDER BY p.timestamp DESC
            LIMIT 5
        """, (cutoff_date,))
        
        successes = cursor.fetchall()
        
        conn.close()
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'avg_move': avg_move,
            'failures': failures,
            'successes': successes
        }
