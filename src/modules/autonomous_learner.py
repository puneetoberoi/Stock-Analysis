import sqlite3
import json
from datetime import datetime, timedelta
import os
import yfinance as yf
import logging

# --- New: Absolute Path Setup ---
# This makes the script work from any directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'data', 'learning.db')
LEARNING_FILE_PATH = os.path.join(SCRIPT_DIR, 'learning_insights.json')
# ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutonomousLearner:
    def __init__(self):
        # Use the absolute paths
        self.db_path = DB_PATH
        self.learning_file = LEARNING_FILE_PATH
        logging.info(f"Database path: {self.db_path}")
        logging.info(f"Learning file path: {self.learning_file}")

    def check_yesterdays_predictions(self):
        """Check all predictions from at least 1 day ago."""
        if not os.path.exists(self.db_path):
            logging.error(f"Database not found at {self.db_path}. Cannot check predictions.")
            return 0
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check predictions from yesterday or older
        one_day_ago = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT id, stock, prediction, entry_price, llm_model
            FROM predictions
            WHERE date(timestamp) <= ?
            AND id NOT IN (SELECT prediction_id FROM outcomes WHERE prediction_id IS NOT NULL)
            AND entry_price > 0
            LIMIT 100
        """, (one_day_ago,))
        
        unchecked = cursor.fetchall()
        logging.info(f"\n🔍 Found {len(unchecked)} unchecked predictions to verify...")
        
        checked_count = 0
        for pred_id, stock, action, entry_price, llm_model in unchecked:
            try:
                ticker_data = yf.Ticker(stock)
                hist = ticker_data.history(start=datetime.now() - timedelta(days=5), end=datetime.now())
                
                if hist.empty:
                    logging.warning(f"  - {stock}: No recent price data found to check outcome.")
                    continue
                    
                current_price = hist['Close'].iloc[-1]
                price_change_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price != 0 else 0
                
                # Aggressive 1% threshold
                if action == 'BUY': success = price_change_pct > 1
                elif action == 'SELL': success = price_change_pct < -1
                else: success = abs(price_change_pct) <= 1
                
                cursor.execute("""
                    INSERT INTO outcomes (prediction_id, actual_price, actual_move_pct, success)
                    VALUES (?, ?, ?, ?)
                """, (pred_id, current_price, price_change_pct, 1 if success else 0))
                
                checked_count += 1
                status = '✅ CORRECT' if success else '❌ WRONG'
                logging.info(f"  -> Graded {stock}: Predicted {action}, outcome was {status} (Move: {price_change_pct:+.1f}%)")
                
            except Exception as e:
                logging.warning(f"  - Error checking {stock} (ID: {pred_id}): {e}")
                continue
        
        conn.commit()
        conn.close()
        logging.info(f"✅ Graded {checked_count} predictions.")
        return checked_count
    
    # ... (The rest of the AutonomousLearner class remains the same) ...
    # The analyze_mistakes, save_learnings, and get_learning_prompt functions are good.

if __name__ == "__main__":
    learner = AutonomousLearner()
    checked_count = learner.check_yesterdays_predictions()
    
    if checked_count > 0:
        learnings = learner.analyze_mistakes()
        if learnings:
            total_insights = sum(len(v) for v in learnings.values())
            print(f"\n✅ Identified {total_insights} new learning patterns!")
            print("📚 These insights will be used in the next analysis.")
        else:
            print("\n✅ All checked predictions were correct! No new mistakes to learn from.")
    else:
        print("\n⏳ No new predictions were old enough to be checked today.")
