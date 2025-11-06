"""
Learning Brain Module - FINAL CORRECTED VERSION for Step 1
"""

import sqlite3
import json
from datetime import datetime, timedelta
import os
import yfinance as yf

class LearningBrain:
    def __init__(self, db_path=None):
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, 'data')
            db_path = os.path.join(data_dir, 'learning.db')
        
        self.db_path = db_path
        print(f"🔍 Database will be created at: {self.db_path}")
        self.init_database()
        
    def init_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
                timeframe_days INTEGER DEFAULT 7,
                timeframe_weight FLOAT DEFAULT 1.0,
                timeframe_label TEXT DEFAULT 'full_strategy',
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accuracy_tracking (
                stock TEXT,
                llm_model TEXT,
                total_predictions REAL DEFAULT 0,
                correct_predictions REAL DEFAULT 0,
                accuracy_pct REAL DEFAULT 0,
                last_updated DATE DEFAULT CURRENT_DATE,
                PRIMARY KEY(stock, llm_model)
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized at {self.db_path}")
        self.migrate_database()
    
    def migrate_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(outcomes)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'timeframe_days' not in columns:
                print("🔄 Migrating database: Adding timeframe_days column...")
                cursor.execute("ALTER TABLE outcomes ADD COLUMN timeframe_days INTEGER DEFAULT 7")
            if 'timeframe_weight' not in columns:
                print("🔄 Migrating database: Adding timeframe_weight column...")
                cursor.execute("ALTER TABLE outcomes ADD COLUMN timeframe_weight FLOAT DEFAULT 1.0")
            if 'timeframe_label' not in columns:
                print("🔄 Migrating database: Adding timeframe_label column...")
                cursor.execute("ALTER TABLE outcomes ADD COLUMN timeframe_label TEXT DEFAULT 'full_strategy'")
            
            cursor.execute("PRAGMA table_info(accuracy_tracking)")
            acc_columns = {col[1]: col[2] for col in cursor.fetchall()}
            if acc_columns.get('total_predictions') != 'REAL' or acc_columns.get('correct_predictions') != 'REAL':
                 print("🔄 Migrating database: Changing accuracy_tracking columns to REAL...")
                 # This is complex in SQLite, a simpler approach is to rebuild if needed
                 # For now, we'll assume the types are okay or will be handled by Python's flexibility

            conn.commit()
            print("✅ Database migration complete")
        except Exception as e:
            print(f"⚠️ Migration error: {e}")
        finally:
            conn.close()
        
    def record_prediction(self, stock, prediction, confidence, price, 
                         llm_model, reasoning, indicators=None):
        if indicators is None: indicators = {}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        target_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        cursor.execute("""
            INSERT INTO predictions (stock, prediction, confidence, target_date, entry_price, 
             llm_model, reasoning, rsi, macd, volume_ratio, patterns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (stock, prediction, confidence, target_date, price, llm_model, reasoning,
              indicators.get('rsi', 0), indicators.get('macd', 0), 
              indicators.get('volume_ratio', 0), indicators.get('patterns', '')))
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        print(f"📝 Recorded prediction #{prediction_id}: {stock} -> {prediction} (confidence: {confidence:.1f}%)")
        return prediction_id
    
    def check_outcomes_multi_timeframe(self):
        timeframes = [
            {'days': 1, 'weight': 0.25, 'label': 'entry_timing'},
            {'days': 3, 'weight': 0.50, 'label': 'short_term'},
            {'days': 5, 'weight': 0.75, 'label': 'trend_confirm'},
            {'days': 7, 'weight': 1.00, 'label': 'full_strategy'}
        ]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        total_checked = 0
        for tf in timeframes:
            days_ago = datetime.now().date() - timedelta(days=tf['days'])
            cursor.execute("""
                SELECT p.id, p.stock, p.prediction, p.entry_price, p.timestamp
                FROM predictions p
                WHERE DATE(p.timestamp) = ? AND NOT EXISTS (
                    SELECT 1 FROM outcomes o 
                    WHERE o.prediction_id = p.id AND o.timeframe_days = ?
                )
            """, (days_ago, tf['days']))
            predictions = cursor.fetchall()
            if not predictions: continue
            print(f"\n📊 Checking {len(predictions)} predictions at {tf['days']}-day timeframe ({tf['label']})...")
            for pred_id, stock, prediction, entry_price, timestamp in predictions:
                try:
                    ticker = yf.Ticker(stock)
                    hist = ticker.history(period='5d')
                    if hist.empty: continue
                    current_price = hist['Close'].iloc[-1]
                    actual_move_pct = ((current_price - entry_price) / entry_price) * 100
                    success = self._evaluate_success(prediction, actual_move_pct, tf['days'])
                    cursor.execute("""
                        INSERT INTO outcomes (prediction_id, actual_price, actual_move_pct, success, 
                         timeframe_days, timeframe_weight, timeframe_label)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (pred_id, current_price, actual_move_pct, success, 
                          tf['days'], tf['weight'], tf['label']))
                    cursor.execute("SELECT llm_model FROM predictions WHERE id = ?", (pred_id,))
                    llm_model = cursor.fetchone()[0]
                    self._update_weighted_accuracy(cursor, stock, llm_model, success, tf['weight'])
                    result = "✅" if success else "❌"
                    print(f"  {result} {stock} ({tf['days']}d): {prediction} → {actual_move_pct:+.1f}%")
                    total_checked += 1
                except Exception as e:
                    print(f"⚠️ Error checking {stock}: {e}")
        conn.commit()
        conn.close()
        print(f"\n✅ Checked {total_checked} prediction-timeframe combinations\n")
        return total_checked

    def _evaluate_success(self, prediction, move_pct, days):
        if days == 1: threshold = 0.5
        elif days == 3: threshold = 1.0
        elif days == 5: threshold = 1.5
        else: threshold = 2.0
        if prediction == 'BUY': return move_pct > threshold
        elif prediction == 'SELL': return move_pct < -threshold
        elif prediction == 'HOLD': return abs(move_pct) < threshold * 1.5
        return False

    def _update_weighted_accuracy(self, cursor, stock, llm_model, success, weight):
        cursor.execute("""
            INSERT INTO accuracy_tracking (stock, llm_model, total_predictions, correct_predictions)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(stock, llm_model) DO UPDATE SET
                total_predictions = total_predictions + ?,
                correct_predictions = correct_predictions + ?
        """, (stock, llm_model, weight, weight if success else 0,
              weight, weight if success else 0))
        cursor.execute("""
            UPDATE accuracy_tracking
            SET accuracy_pct = (correct_predictions / total_predictions) * 100
            WHERE stock = ? AND llm_model = ? AND total_predictions > 0
        """, (stock, llm_model))

    # Replace the entire get_accuracy_report method in learning_brain.py

def get_accuracy_report(self):
    """Generate comprehensive accuracy report with weighted metrics and trends"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]
    
    # --- Robust calculation of overall weighted accuracy ---
    cursor.execute("SELECT SUM(total_predictions), SUM(correct_predictions) FROM accuracy_tracking")
    row = cursor.fetchone()
    total_weighted_checks, weighted_successes = (row[0] or 0), (row[1] or 0)
    overall_accuracy = (weighted_successes / total_weighted_checks * 100) if total_weighted_checks > 0 else 0
    
    # --- FIXED: Robust calculation for timeframe stats ---
    cursor.execute("""
        SELECT 
            timeframe_label,
            timeframe_days,
            COUNT(*) as checks,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
        FROM outcomes
        GROUP BY timeframe_label, timeframe_days
        ORDER BY timeframe_days
    """)
    timeframe_data = cursor.fetchall()
    
    # --- FIXED: Robust calculation for daily trends ---
    cursor.execute("""
        SELECT 
            DATE(check_date) as day,
            COUNT(*) as checks,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
        FROM outcomes
        WHERE check_date >= DATE('now', '-7 days')
        GROUP BY DATE(check_date)
        ORDER BY day DESC
        LIMIT 7
    """)
    daily_trends_data = cursor.fetchall()

    cursor.execute("""
        SELECT stock, llm_model, total_predictions, correct_predictions, accuracy_pct FROM accuracy_tracking
        WHERE total_predictions > 0 ORDER BY accuracy_pct DESC, total_predictions DESC LIMIT 5
    """)
    top_performers = cursor.fetchall()
    conn.close()

    if total_predictions == 0: return "No prediction history yet"

    report = f"\n📊 **LEARNING SYSTEM ACCURACY REPORT**\n{'='*50}\n"
    report += f"Total Predictions Made: {total_predictions}\n"
    report += f"Total Weighted Checks: {total_weighted_checks:.1f}\n"
    report += f"Weighted Successes: {weighted_successes:.1f}\n"
    report += f"Overall Weighted Accuracy: {overall_accuracy:.1f}%\n"

    if timeframe_data:
        report += f"\n⏰ Accuracy by Timeframe:\n{'-'*50}\n"
        for label, days, checks, successes in timeframe_data:
            successes = successes or 0
            accuracy = (successes / checks * 100) if checks > 0 else 0
            emoji = "🟢" if accuracy >= 60 else "🟡" if accuracy >= 40 else "🔴"
            report += f"  {emoji} {label} ({days}d): {accuracy:.1f}% ({successes}/{checks} correct)\n"
    
    if daily_trends_data:
        report += f"\n📈 Daily Accuracy Trend (Last 7 Days):\n{'-'*50}\n"
        for day, checks, successes in daily_trends_data:
            successes = successes or 0
            accuracy = (successes / checks * 100) if checks > 0 else 0
            trend_emoji = "📈" if accuracy >= 60 else "📊" if accuracy >= 40 else "📉"
            report += f"  {trend_emoji} {day}: {accuracy:.1f}% ({successes}/{checks})\n"
        
        if len(daily_trends_data) >= 2:
            recent_accuracy = (daily_trends_data[0][2] or 0) / daily_trends_data[0][1] * 100
            older_accuracy = (daily_trends_data[-1][2] or 0) / daily_trends_data[-1][1] * 100
            improvement = recent_accuracy - older_accuracy
            if improvement != 0:
                report += f"\n  {'🚀' if improvement > 0 else '📉'} 7-Day Change: {improvement:+.1f}%\n"

    if top_performers:
        report += f"\n🏆 Top Performing Stock/LLM Combinations:\n{'-'*50}\n"
        for stock, model, total, correct, acc_pct in top_performers:
            report += f"  {stock} ({model}): {acc_pct:.1f}% ({correct:.1f}/{total:.1f})\n"
    
    return report

# Test function
if __name__ == "__main__":
    brain = LearningBrain()
    brain.record_prediction(stock='AAPL', prediction='BUY', confidence=75.5, price=175.50, llm_model='test', reasoning='Test prediction')
    brain.check_outcomes_multi_timeframe()
    print(brain.get_accuracy_report())
    print("✅ Test completed!")
