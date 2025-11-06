"""
Learning Brain Module - Complete version with multi-timeframe outcome checking
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
        
        # Migrate existing database
        self.migrate_database()
    
    def migrate_database(self):
        """Migrate existing database to add new columns if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if new columns exist in outcomes table
            cursor.execute("PRAGMA table_info(outcomes)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Add timeframe_days if it doesn't exist
            if 'timeframe_days' not in columns:
                print("🔄 Migrating database: Adding timeframe_days column...")
                cursor.execute("""
                    ALTER TABLE outcomes 
                    ADD COLUMN timeframe_days INTEGER DEFAULT 7
                """)
            
            # Add timeframe_weight if it doesn't exist
            if 'timeframe_weight' not in columns:
                print("🔄 Migrating database: Adding timeframe_weight column...")
                cursor.execute("""
                    ALTER TABLE outcomes 
                    ADD COLUMN timeframe_weight FLOAT DEFAULT 1.0
                """)
            
            # Add timeframe_label if it doesn't exist
            if 'timeframe_label' not in columns:
                print("🔄 Migrating database: Adding timeframe_label column...")
                cursor.execute("""
                    ALTER TABLE outcomes 
                    ADD COLUMN timeframe_label TEXT DEFAULT 'full_strategy'
                """)
            
            conn.commit()
            print("✅ Database migration complete")
            
        except Exception as e:
            print(f"⚠️ Migration error: {e}")
        finally:
            conn.close()
        
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
    
    def check_outcomes_multi_timeframe(self):
        """
        Check predictions at multiple timeframes with weighted learning
        1-day = 25% weight (entry timing)
        3-day = 50% weight (short-term direction)  
        5-day = 75% weight (trend confirmation)
        7-day = 100% weight (full strategy)
        """
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
            
            # Find predictions from exactly N days ago that haven't been checked at this timeframe
            cursor.execute("""
                SELECT p.id, p.stock, p.prediction, p.entry_price, p.timestamp
                FROM predictions p
                WHERE DATE(p.timestamp) = ?
                AND NOT EXISTS (
                    SELECT 1 FROM outcomes o 
                    WHERE o.prediction_id = p.id 
                    AND o.timeframe_days = ?
                )
            """, (days_ago, tf['days']))
            
            predictions = cursor.fetchall()
            
            if not predictions:
                continue
                
            print(f"\n📊 Checking {len(predictions)} predictions at {tf['days']}-day timeframe ({tf['label']})...")
            
            for pred_id, stock, prediction, entry_price, timestamp in predictions:
                try:
                    # Get current price
                    ticker = yf.Ticker(stock)
                    hist = ticker.history(period='5d')
                    
                    if hist.empty:
                        continue
                    
                    current_price = hist['Close'].iloc[-1]
                    actual_move_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Determine success based on timeframe
                    success = self._evaluate_success(
                        prediction, 
                        actual_move_pct, 
                        tf['days'],
                        tf['weight']
                    )
                    
                    # Record outcome with timeframe info
                    cursor.execute("""
                        INSERT INTO outcomes 
                        (prediction_id, actual_price, actual_move_pct, success, 
                         timeframe_days, timeframe_weight, timeframe_label)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (pred_id, current_price, actual_move_pct, success, 
                          tf['days'], tf['weight'], tf['label']))
                    
                    # Update weighted accuracy
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

    def _evaluate_success(self, prediction, move_pct, days, weight):
        """
        Evaluate success based on prediction type and timeframe
        Shorter timeframes have looser thresholds
        """
        # Adjust thresholds based on timeframe
        if days == 1:
            threshold = 0.5  # 0.5% move in 1 day
        elif days == 3:
            threshold = 1.0  # 1% move in 3 days
        elif days == 5:
            threshold = 1.5  # 1.5% move in 5 days
        else:  # 7 days
            threshold = 2.0  # 2% move in 7 days
        
        if prediction == 'BUY':
            return move_pct > threshold
        elif prediction == 'SELL':
            return move_pct < -threshold
        elif prediction == 'HOLD':
            return abs(move_pct) < threshold * 1.5
        
        return False

    def _update_weighted_accuracy(self, cursor, stock, llm_model, success, weight):
        """Update accuracy with weighted contributions"""
        cursor.execute("""
            INSERT INTO accuracy_tracking 
            (stock, llm_model, total_predictions, correct_predictions, accuracy_pct)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(stock, llm_model) DO UPDATE SET
                total_predictions = total_predictions + ?,
                correct_predictions = correct_predictions + ?,
                accuracy_pct = (CAST(correct_predictions AS FLOAT) + ?) * 100.0 / 
                              (total_predictions + ?),
                last_updated = CURRENT_DATE
        """, (
            stock, llm_model, 
            weight, weight if success else 0, (weight if success else 0) * 100,
            weight, weight if success else 0, weight if success else 0, weight
        ))
        
    def get_accuracy_report(self):
        """Generate comprehensive accuracy report with weighted metrics and trends"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get overall stats with weighted calculations
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        # Calculate weighted accuracy properly
        cursor.execute("""
            SELECT 
                SUM(timeframe_weight) as total_weighted,
                SUM(CASE WHEN success = 1 THEN timeframe_weight ELSE 0 END) as weighted_successes
            FROM outcomes
        """)
        row = cursor.fetchone()
        total_weighted_checks = row[0] if row[0] else 0
        weighted_successes = row[1] if row[1] else 0
        
        overall_accuracy = (weighted_successes / total_weighted_checks * 100) if total_weighted_checks > 0 else 0
        
        # Get accuracy by timeframe
        cursor.execute("""
            SELECT 
                timeframe_label,
                timeframe_days,
                COUNT(*) as checks,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                ROUND(AVG(CASE WHEN success = 1 THEN 100.0 ELSE 0.0 END), 1) as accuracy
            FROM outcomes
            GROUP BY timeframe_label, timeframe_days
            ORDER BY timeframe_days
        """)
        timeframe_stats = cursor.fetchall()
        
        # Get daily accuracy trend (last 7 days)
        cursor.execute("""
            SELECT 
                DATE(check_date) as day,
                COUNT(*) as checks,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                ROUND(AVG(CASE WHEN success = 1 THEN 100.0 ELSE 0.0 END), 1) as daily_accuracy
            FROM outcomes
            WHERE check_date >= DATE('now', '-7 days')
            GROUP BY DATE(check_date)
            ORDER BY day DESC
            LIMIT 7
        """)
        daily_trends = cursor.fetchall()
        
        # Get per-stock accuracy (top performers)
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
        
        # Build enhanced report
        report = "\n📊 **LEARNING SYSTEM ACCURACY REPORT**\n"
        report += "=" * 50 + "\n"
        report += f"Total Predictions Made: {total_predictions}\n"
        report += f"Total Weighted Checks: {total_weighted_checks:.1f}\n"
        report += f"Weighted Successes: {weighted_successes:.1f}\n"
        report += f"Overall Weighted Accuracy: {overall_accuracy:.1f}%\n"
        
        # Timeframe breakdown
        if timeframe_stats:
            report += "\n⏰ Accuracy by Timeframe:\n"
            report += "-" * 50 + "\n"
            for label, days, checks, successes, accuracy in timeframe_stats:
                emoji = "🟢" if accuracy >= 60 else "🟡" if accuracy >= 40 else "🔴"
                report += f"  {emoji} {label} ({days}d): {accuracy:.1f}% ({successes}/{checks} correct)\n"
        
        # Daily trend
        if daily_trends:
            report += "\n📈 Daily Accuracy Trend (Last 7 Days):\n"
            report += "-" * 50 + "\n"
            for day, checks, successes, accuracy in daily_trends:
                trend_emoji = "📈" if accuracy >= 60 else "📊" if accuracy >= 40 else "📉"
                report += f"  {trend_emoji} {day}: {accuracy:.1f}% ({successes}/{checks})\n"
            
            # Calculate improvement
            if len(daily_trends) >= 2:
                recent_accuracy = daily_trends[0][3]  # Most recent
                older_accuracy = daily_trends[-1][3]  # 7 days ago
                improvement = recent_accuracy - older_accuracy
                if improvement > 0:
                    report += f"\n  🚀 7-Day Improvement: +{improvement:.1f}%\n"
                elif improvement < 0:
                    report += f"\n  📉 7-Day Change: {improvement:.1f}%\n"
        
        # Top performers
        if top_performers:
            report += "\n🏆 Top Performing Stock/LLM Combinations:\n"
            report += "-" * 50 + "\n"
            for stock, model, total, correct, accuracy in top_performers[:5]:  # Top 5 only
                report += f"  {stock} ({model}): {accuracy:.1f}% ({correct:.1f}/{total:.1f})\n"
        
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
    
    # Test multi-timeframe checking
    brain.check_outcomes_multi_timeframe()
    
    # Show report
    print(brain.get_accuracy_report())
    print("✅ Test completed!")
