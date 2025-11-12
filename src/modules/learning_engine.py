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
            SELECT id, stock, prediction, entry_price, target_date, llm_model
            FROM predictions
            WHERE DATE(timestamp) <= DATE(?)
            AND id NOT IN (SELECT prediction_id FROM outcomes)
        """, (week_ago,))
        
        unchecked = cursor.fetchall()
        print(f"🔍 Checking {len(unchecked)} predictions...")
        
        for pred_id, stock, action, entry_price, target_date, llm_model in unchecked:
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
                """, (pred_id, current_price, price_change_pct, 1 if success else 0))
                
                # Update accuracy tracking
                cursor.execute("""
                    INSERT OR REPLACE INTO accuracy_tracking (stock, llm_model, total_predictions, correct_predictions, accuracy_pct)
                    VALUES (
                        ?, ?, 
                        COALESCE((SELECT total_predictions FROM accuracy_tracking WHERE stock=? AND llm_model=?), 0) + 1,
                        COALESCE((SELECT correct_predictions FROM accuracy_tracking WHERE stock=? AND llm_model=?), 0) + ?,
                        0
                    )
                """, (stock, llm_model, stock, llm_model, stock, llm_model, 1 if success else 0))
                
                print(f"  {stock}: {action} was {'✅ CORRECT' if success else '❌ WRONG'} (moved {price_change_pct:.1f}%)")
                
            except Exception as e:
                print(f"  Error checking {stock}: {e}")
        
        # Update accuracy percentages
        cursor.execute("""
            UPDATE accuracy_tracking 
            SET accuracy_pct = ROUND(100.0 * correct_predictions / total_predictions, 1)
            WHERE total_predictions > 0
        """)
        
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

def show_learning_summary(self):
    """Show what we've learned so far"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    print("\n📊 LEARNING SUMMARY")
    print("=" * 50)
    
    # Overall accuracy by action type
    cursor.execute("""
        SELECT 
            p.prediction as action,
            COUNT(*) as total,
            SUM(CASE WHEN o.success = 1 THEN 1 ELSE 0 END) as correct,
            ROUND(100.0 * SUM(CASE WHEN o.success = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy
        FROM predictions p
        JOIN outcomes o ON p.id = o.prediction_id
        GROUP BY p.prediction
    """)
    
    results = cursor.fetchall()
    print("\n📈 Accuracy by Action Type:")
    for action, total, correct, accuracy in results:
        print(f"  {action}: {accuracy}% ({correct}/{total})")
    
    # Best performing stocks
    cursor.execute("""
        SELECT 
            p.stock,
            COUNT(*) as predictions,
            ROUND(AVG(o.actual_move_pct), 1) as avg_move,
            SUM(CASE WHEN o.success = 1 THEN 1 ELSE 0 END) as correct
        FROM predictions p
        JOIN outcomes o ON p.id = o.prediction_id
        GROUP BY p.stock
        HAVING predictions >= 5
        ORDER BY avg_move DESC
        LIMIT 5
    """)
    
    results = cursor.fetchall()
    print("\n🏆 Top Performing Stocks:")
    for stock, preds, avg_move, correct in results:
        accuracy = (correct/preds*100) if preds else 0
        print(f"  {stock}: {avg_move}% avg move, {accuracy:.0f}% accuracy")
    
    conn.close()

# Add to main execution
if __name__ == "__main__":
    engine = LearningEngine()
    engine.check_past_predictions()
    engine.generate_learning_insights()
    engine.show_learning_summary()  # ← ADD THIS
