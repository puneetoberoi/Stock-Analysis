# src/modules/autonomous_learner.py
import sqlite3
import json
from datetime import datetime, timedelta
import os
import logging

try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance not found. Please run 'pip install yfinance'")
    yf = None

# --- ABSOLUTE PATH LOGIC (The Fix) ---
# This figures out the root directory of your project
# __file__ is the current file: /.../Stock-Analysis/src/modules/autonomous_learner.py
# os.path.dirname(__file__) is its directory: /.../Stock-Analysis/src/modules
# os.path.dirname(os.path.dirname(...)) is the parent: /.../Stock-Analysis/src
# os.path.dirname(os.path.dirname(os.path.dirname(...))) is the project root: /.../Stock-Analysis/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, 'src', 'modules', 'data', 'learning.db')
LEARNING_FILE_PATH = os.path.join(PROJECT_ROOT, 'src', 'modules', 'learning_insights.json')
# --- END OF FIX ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutonomousLearner:
    def __init__(self):
        self.db_path = DB_PATH
        self.learning_file = LEARNING_FILE_PATH
        logging.info(f"DB Path: {self.db_path}")
        logging.info(f"Learnings Path: {self.learning_file}")

    def check_past_predictions(self):
        """Checks all verifiable predictions made at least one day ago."""
        if not yf:
            logging.error("yfinance is not installed. Skipping prediction checks.")
            return 0
        if not os.path.exists(self.db_path):
            logging.error(f"Database not found at {self.db_path}. Cannot check predictions.")
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        one_day_ago = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # Grade today's predictions
        
        cursor.execute("""
            SELECT id, stock, prediction, entry_price, llm_model
            FROM predictions
            WHERE date(timestamp) <= ?
            AND id NOT IN (SELECT prediction_id FROM outcomes WHERE prediction_id IS NOT NULL)
            AND entry_price > 0
            LIMIT 100
        """, (one_day_ago,))
        
        unchecked = cursor.fetchall()
        logging.info(f"\n🔍 Found {len(unchecked)} unchecked predictions to grade...")
        
        checked_count = 0
        for pred_id, stock, action, entry_price, llm_model in unchecked:
            try:
                ticker_data = yf.Ticker(stock)
                # FIXED: Get the most recent trading day's data instead of future data
                hist = ticker_data.history(period='5d')  # Get last 5 days to ensure we have data
                
                if hist.empty:
                    logging.warning(f"  - {stock}: No recent price data found.")
                    continue
                
                # Use the most recent available closing price
                current_price = hist['Close'].iloc[-1]
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Aggressive 1% threshold for success
                if action == 'BUY': success = price_change_pct > 1.0
                elif action == 'SELL': success = price_change_pct < -1.0
                else: success = abs(price_change_pct) <= 1.0
                
                cursor.execute("""
                    INSERT INTO outcomes (prediction_id, check_date, actual_price, actual_move_pct, success)
                    VALUES (?, ?, ?, ?, ?)
                """, (pred_id, datetime.now().date(), current_price, price_change_pct, 1 if success else 0))
                
                checked_count += 1
                status = '✅ CORRECT' if success else '❌ WRONG'
                logging.info(f"  -> Graded {stock}: Predicted {action}, outcome was {status} (Move: {price_change_pct:+.1f}%)")
                
            except Exception as e:
                logging.error(f"  - Error checking {stock} (ID: {pred_id}): {e}", exc_info=False)
                continue
        
        conn.commit()
        conn.close()
        logging.info(f"✅ Graded {checked_count} predictions.")
        return checked_count

    def analyze_mistakes(self):
        """Finds patterns in mistakes and generates learning insights."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                p.stock, p.prediction, p.reasoning, p.rsi, p.patterns, o.actual_move_pct
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            WHERE o.success = 0 AND p.reasoning NOT LIKE 'Test prediction%'
            ORDER BY o.check_date DESC, ABS(o.actual_move_pct) DESC
            LIMIT 30
        """)
        
        mistakes = cursor.fetchall()
        learnings = {}
        logging.info(f"\n🔍 Analyzing {len(mistakes)} mistakes to find patterns...")

        for stock, pred, reasoning, rsi, patterns, actual_move in mistakes:
            rsi_val = rsi if rsi else 50
            if pred == "SELL" and actual_move > 1:
                mistake_type = "sold_too_early"
                learning = f"On {stock}, a SELL call was wrong (stock went up {actual_move:.1f}%). Re-evaluate bearish signals like {patterns} when RSI is around {rsi_val:.0f}."
            elif pred == "HOLD" and abs(actual_move) > 3:
                mistake_type = "missed_big_move"
                learning = f"On {stock}, a HOLD call missed a {actual_move:.1f}% move. Be more decisive with patterns like {patterns}, even with neutral RSI ({rsi_val:.0f})."
            elif pred == "BUY" and actual_move < -1:
                mistake_type = "bought_at_peak"
                learning = f"On {stock}, a BUY call was wrong (stock went down {actual_move:.1f}%). {patterns} might be a false signal when RSI is {rsi_val:.0f}."
            else:
                continue
            
            if mistake_type not in learnings: learnings[mistake_type] = []
            learnings[mistake_type].append(learning)
        
        conn.close()
        
        if learnings: self.save_learnings(learnings)
        return learnings

    def save_learnings(self, learnings):
        """Saves learning insights to a JSON file."""
        # Use a simpler, non-nested structure
        flat_learnings = []
        for category, items in learnings.items():
            flat_learnings.extend(items)

        # Keep only the most recent 10 unique learnings
        unique_learnings = list(dict.fromkeys(flat_learnings))[:10]

        with open(self.learning_file, 'w') as f:
            json.dump({'insights': unique_learnings}, f, indent=2)
        
        logging.info("\n🧠 AUTONOMOUS LEARNING COMPLETE")
        logging.info("=" * 50)
        for insight in unique_learnings:
            logging.info(f"  📝 {insight}")

    def get_learning_prompt(self):
        """Gets learning context for LLM prompts."""
        if not os.path.exists(self.learning_file):
            return ""
        
        try:
            with open(self.learning_file, 'r') as f:
                data = json.load(f)
            
            insights = data.get('insights', [])
            if not insights: return ""
            
            prompt = "\n🚨 CRITICAL LEARNINGS FROM PAST MISTAKES - ADJUST YOUR STRATEGY:\n"
            for insight in insights:
                prompt += f"- {insight}\n"
            prompt += "Apply these learnings to the new data. Be more decisive and avoid repeating these errors.\n"
            return prompt
        except (json.JSONDecodeError, FileNotFoundError):
            return ""

# Main execution block
if __name__ == "__main__":
    learner = AutonomousLearner()
    checked_count = learner.check_past_predictions()
    
    if checked_count > 0:
        learnings = learner.analyze_mistakes()
        if learnings:
            total_insights = sum(len(v) for v in learnings.values())
            logging.info(f"✅ Generated {total_insights} new learning insights.")
        else:
            logging.info("✅ All checked predictions were correct! No new mistakes found.")
    else:
        logging.info("⏳ No new predictions were old enough to be graded today.")
