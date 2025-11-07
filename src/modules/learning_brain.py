"""
Learning Brain Module - Tracks predictions and learns from outcomes
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf

class LearningBrain:
    def __init__(self, db_path='data/learning.db'):
        """Initialize the learning brain with database"""
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Create database and tables if they don't exist"""
        # Create data directory if it doesn't exist
        Path(self.db_path).parent.mkdir(exist_ok=True)
        
        # Connect and create tables
        conn = sqlite3.connect(self.db_path)
        
        # Read and execute schema
        schema_path = Path(__file__).parent / 'schemas.sql'
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                conn.executescript(f.read())
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized at {self.db_path}")
        
    def record_prediction(self, stock, prediction, confidence, price, 
                         llm_model, reasoning, indicators):
        """Record a new prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate target date (1 week from now)
        target_date = (datetime.now() + timedelta(days=7)).date()
        
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
            indicators.get('rsi'),
            indicators.get('macd'),
            indicators.get('volume_ratio'),
            indicators.get('patterns', '')
        ))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        print(f"📝 Recorded prediction #{prediction_id}: {stock} -> {prediction} (confidence: {confidence:.1f}%)")
        return prediction_id
        
    def check_outcomes(self, days_back=7):
        """Check predictions from a week ago and record outcomes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find predictions that are due for checking
        check_date = datetime.now().date()
        past_date = (datetime.now() - timedelta(days=days_back)).date()
        
        cursor.execute("""
            SELECT id, stock, prediction, entry_price, predicted_price
            FROM predictions
            WHERE DATE(target_date) <= ?
            AND id NOT IN (SELECT prediction_id FROM outcomes)
        """, (check_date,))
        
        due_predictions = cursor.fetchall()
        
        if not due_predictions:
            print("✅ No predictions due for checking")
            conn.close()
            return
            
        print(f"\n📊 Checking {len(due_predictions)} predictions...")
        
        for pred_id, stock, prediction, entry_price, predicted_price in due_predictions:
            try:
                # Get current price
                ticker = yf.Ticker(stock)
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                
                # Calculate actual move
                actual_move_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Determine success
                success = False
                if prediction == 'BUY' and actual_move_pct > 1:
                    success = True
                elif prediction == 'SELL' and actual_move_pct < -1:
                    success = True
                elif prediction == 'HOLD' and abs(actual_move_pct) < 2:
                    success = True
                
                # Record outcome
                cursor.execute("""
                    INSERT INTO outcomes 
                    (prediction_id, actual_price, actual_move_pct, success)
                    VALUES (?, ?, ?, ?)
                """, (pred_id, current_price, actual_move_pct, success))
                
                # Update accuracy tracking
                cursor.execute("""
                    SELECT llm_model FROM predictions WHERE id = ?
                """, (pred_id,))
                llm_model = cursor.fetchone()[0]
                
                self._update_accuracy(cursor, stock, llm_model, success)
                
                result_emoji = "✅" if success else "❌"
                print(f"{result_emoji} {stock}: {prediction} -> {actual_move_pct:+.1f}% move")
                
            except Exception as e:
                print(f"⚠️ Error checking {stock}: {e}")
                
        conn.commit()
        conn.close()
        
    def _update_accuracy(self, cursor, stock, llm_model, success):
        """Update accuracy statistics"""
        cursor.execute("""
            INSERT INTO accuracy_tracking (stock, llm_model, total_predictions, correct_predictions, accuracy_pct)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(stock, llm_model) DO UPDATE SET
                total_predictions = total_predictions + 1,
                correct_predictions = correct_predictions + ?,
                accuracy_pct = (correct_predictions + ?) * 100.0 / (total_predictions + 1),
                last_updated = CURRENT_DATE
        """, (stock, llm_model, 1 if success else 0, 100.0 if success else 0, 
              1 if success else 0, 1 if success else 0))
        
    def get_accuracy_report(self):
        """Generate accuracy report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT stock, llm_model, total_predictions, 
                   correct_predictions, accuracy_pct
            FROM accuracy_tracking
            WHERE total_predictions > 0
            ORDER BY accuracy_pct DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "No prediction history yet"
            
        report = "\n📊 **LEARNING SYSTEM ACCURACY REPORT**\n"
        report += "-" * 50 + "\n"
        
        for stock, model, total, correct, accuracy in results:
            report += f"{stock} ({model}): {accuracy:.1f}% ({correct}/{total})\n"
            
        return report


# Test function
def test_learning_brain():
    """Test the learning brain functionality"""
    brain = LearningBrain()
    
    # Test recording a prediction
    test_indicators = {
        'rsi': 65.5,
        'macd': 0.25,
        'volume_ratio': 1.2,
        'patterns': 'bullish_engulfing,hammer'
    }
    
    pred_id = brain.record_prediction(
        stock='AAPL',
        prediction='BUY',
        confidence=75.5,
        price=175.50,
        llm_model='groq-llama',
        reasoning='Strong bullish patterns with good volume',
        indicators=test_indicators
    )
    
    print(f"\n✅ Test passed! Created prediction #{pred_id}")
    
    # Test checking outcomes
    brain.check_outcomes()
    
    # Get accuracy report
    report = brain.get_accuracy_report()
    print(report)


if __name__ == "__main__":
    test_learning_brain()
