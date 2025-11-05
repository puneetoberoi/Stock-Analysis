"""
Learning Brain Module - Complete version with outcome checking
"""

import sqlite3
import json
from datetime import datetime, timedelta
import os
import yfinance as yf

class LearningBrain:
    def __init__(self, db_path=None):
        """Initialize the learning brain with database"""
        if db_path is None:
            # Create path relative to THIS FILE's location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, 'data')
            db_path = os.path.join(data_dir, 'learning.db')
        
        self.db_path = db_path
        print(f"🔍 Database will be created at: {self.db_path}")
        self.init_database()
        
    def init_database(self):
        """Create database and tables if they don't exist"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect and create tables
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables directly (no external schema file)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                stock TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence FLOAT,
                target_date DATE,
                entry_price FLOAT,
                llm_model TEXT,
                reasoning TEXT,
                rsi FLOAT,
                macd FLOAT,
                volume_ratio FLOAT,
                patterns TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                check_date DATE DEFAULT CURRENT_DATE,
                actual_price FLOAT,
                actual_move_pct FLOAT,
                success BOOLEAN,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accuracy_tracking (
                stock TEXT,
                llm_model TEXT,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy_pct FLOAT DEFAULT 0,
                last_updated DATE DEFAULT CURRENT_DATE,
                PRIMARY KEY(stock, llm_model)
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized at {self.db_path}")
        
    def record_prediction(self, stock, prediction, confidence, price, 
                         llm_model, reasoning, indicators=None):
        """Record a new prediction"""
        if indicators is None:
            indicators = {}
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate target date (1 week from now)
        target_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        cursor.execute("""
            INSERT INTO predictions 
            (stock, prediction, confidence, target_date, entry_price, 
             llm_model, reasoning, rsi, macd, volume_ratio, patterns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stock, 
            prediction, 
            confidence,
            target_date,
            price,
            llm_model,
            reasoning,
            indicators.get('rsi', 0),
            indicators.get('macd', 0),
            indicators.get('volume_ratio', 0),
            indicators.get('patterns', '')
        ))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        print(f"📝 Recorded prediction #{prediction_id}: {stock} -> {prediction} (confidence: {confidence:.1f}%)")
        return prediction_id
    
    def check_outcomes(self, days_back=7):
        """Check predictions from past and record outcomes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find predictions that are due for checking (target_date has passed)
        check_date = datetime.now().date()
        
        cursor.execute("""
            SELECT id, stock, prediction, entry_price, target_date
            FROM predictions
            WHERE DATE(target_date) <= ?
            AND id NOT IN (SELECT prediction_id FROM outcomes)
            ORDER BY target_date DESC
            LIMIT 50
        """, (check_date,))
        
        due_predictions = cursor.fetchall()
        
        if not due_predictions:
            print("✅ No predictions due for checking")
            conn.close()
            return
            
        print(f"\n📊 Checking {len(due_predictions)} predictions...")
        
        checked_count = 0
        for pred_id, stock, prediction, entry_price, target_date in due_predictions:
            try:
                # Get current price
                ticker = yf.Ticker(stock)
                hist = ticker.history(period='5d')
                
                if hist.empty:
                    print(f"⚠️ No price data for {stock}, skipping")
                    continue
                
                current_price = hist['Close'].iloc[-1]
                
                # Calculate actual move
                actual_move_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Determine success based on prediction
                success = False
                if prediction == 'BUY' and actual_move_pct > 2:  # Gained more than 2%
                    success = True
                elif prediction == 'SELL' and actual_move_pct < -2:  # Dropped more than 2%
                    success = True
                elif prediction == 'HOLD' and abs(actual_move_pct) < 3:  # Stayed relatively flat
                    success = True
                
                # Record outcome
                cursor.execute("""
                    INSERT INTO outcomes 
                    (prediction_id, actual_price, actual_move_pct, success)
                    VALUES (?, ?, ?, ?)
                """, (pred_id, current_price, actual_move_pct, success))
                
                # Update accuracy tracking
                cursor.execute("SELECT llm_model FROM predictions WHERE id = ?", (pred_id,))
                llm_model = cursor.fetchone()[0]
                
                self._update_accuracy(cursor, stock, llm_model, success)
                
                result_emoji = "✅" if success else "❌"
                print(f"{result_emoji} {stock}: {prediction} -> {actual_move_pct:+.1f}% move")
                checked_count += 1
                
            except Exception as e:
                print(f"⚠️ Error checking {stock}: {e}")
                
        conn.commit()
        conn.close()
        print(f"✅ Checked {checked_count} predictions\n")
    
    def _update_accuracy(self, cursor, stock, llm_model, success):
        """Update accuracy statistics"""
        cursor.execute("""
            INSERT INTO accuracy_tracking (stock, llm_model, total_predictions, correct_predictions, accuracy_pct)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(stock, llm_model) DO UPDATE SET
                total_predictions = total_predictions + 1,
                correct_predictions = correct_predictions + ?,
                accuracy_pct = (CAST(correct_predictions AS FLOAT) + ?) * 100.0 / (total_predictions + 1),
                last_updated = CURRENT_DATE
        """, (
            stock, 
            llm_model, 
            1 if success else 0, 
            100.0 if success else 0,
            1 if success else 0,
            1 if success else 0
        ))
        
    def get_accuracy_report(self):
        """Generate accuracy report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get overall stats
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM outcomes")
        total_outcomes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM outcomes WHERE success = 1")
        successful_outcomes = cursor.fetchone()[0]
        
        # Get per-stock accuracy
        cursor.execute("""
            SELECT stock, llm_model, total_predictions, 
                   correct_predictions, accuracy_pct
            FROM accuracy_tracking
            WHERE total_predictions > 0
            ORDER BY accuracy_pct DESC
            LIMIT 10
        """)
        
        top_performers = cursor.fetchall()
        conn.close()
        
        if total_predictions == 0:
            return "No prediction history yet"
        
        # Build report
        overall_accuracy = (successful_outcomes / total_outcomes * 100) if total_outcomes > 0 else 0
        
        report = "\n📊 **LEARNING SYSTEM ACCURACY REPORT**\n"
        report += "=" * 50 + "\n"
        report += f"Total Predictions Made: {total_predictions}\n"
        report += f"Predictions Checked: {total_outcomes}\n"
        report += f"Successful Predictions: {successful_outcomes}\n"
        report += f"Overall Accuracy: {overall_accuracy:.1f}%\n"
        
        if top_performers:
            report += "\n🏆 Top Performing Stock/LLM Combinations:\n"
            report += "-" * 50 + "\n"
            for stock, model, total, correct, accuracy in top_performers:
                report += f"  {stock} ({model}): {accuracy:.1f}% ({correct}/{total})\n"
        
        return report


# Test function
if __name__ == "__main__":
    print("Starting Learning Brain test...")
    
    # Create instance
    brain = LearningBrain()
    
    # Test recording
    brain.record_prediction(
        stock='AAPL',
        prediction='BUY',
        confidence=75.5,
        price=175.50,
        llm_model='test',
        reasoning='Test prediction'
    )
    
    # Test checking outcomes
    brain.check_outcomes()
    
    # Show report
    print(brain.get_accuracy_report())
    print("✅ Test completed!")
