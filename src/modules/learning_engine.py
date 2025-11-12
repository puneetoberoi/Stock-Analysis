import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
import json

class LearningEngine:
    def __init__(self):
        self.db_path = 'src/modules/data/learning.db'
        
    def check_past_predictions(self):
        """Check predictions from 7 days ago and mark if correct"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unchecked predictions from 7+ days ago
        week_ago = datetime.now() - timedelta(days=7)
        
        cursor.execute("""
            SELECT id, stock, prediction, entry_price, target_date
            FROM predictions
            WHERE DATE(timestamp) <= DATE(?)
            AND id NOT IN (SELECT prediction_id FROM outcomes)
        """, (week_ago,))
        
        unchecked = cursor.fetchall()
        print(f"🔍 Checking {len(unchecked)} predictions...")
        
        for pred_id, stock, action, entry_price, target_date in unchecked:
            try:
                # Get current price
                ticker = yf.Ticker(stock)
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                
                # Calculate performance
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Determine if correct (2% threshold)
                if action == 'BUY':
                    success = price_change_pct > 2
                elif action == 'SELL':
                    success = price_change_pct < -2
                else:  # HOLD
                    success = abs(price_change_pct) <= 2
                
                # Store outcome
                cursor.execute("""
                    INSERT INTO outcomes (prediction_id, actual_price, actual_move_pct, success)
                    VALUES (?, ?, ?, ?)
                """, (pred_id, current_price, price_change_pct, success))
                
                print(f"  {stock}: {action} was {'✅ CORRECT' if success else '❌ WRONG'} (moved {price_change_pct:.1f}%)")
                
            except Exception as e:
                print(f"  Error checking {stock}: {e}")
        
        conn.commit()
        conn.close()
        
    def generate_learning_insights(self):
        """Ask LLMs to reflect on their mistakes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent wrong predictions
        cursor.execute("""
            SELECT p.stock, p.prediction, p.reasoning, o.actual_move_pct
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            WHERE o.success = 0
            ORDER BY p.timestamp DESC
            LIMIT 10
        """)
        
        mistakes = cursor.fetchall()
        
        if mistakes:
            print("\n🧠 Learning from mistakes:")
            for stock, pred, reasoning, actual_move in mistakes:
                print(f"\n{stock}: Predicted {pred}, but moved {actual_move:.1f}%")
                print(f"Original reasoning: {reasoning[:100]}...")
                
                # Here you would ask LLMs: "Why was this wrong? What should I look for next time?"
                # This is where the real learning happens!
        
        conn.close()

if __name__ == "__main__":
    engine = LearningEngine()
    engine.check_past_predictions()
    engine.generate_learning_insights()
