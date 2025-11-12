import sqlite3
import json
from datetime import datetime, timedelta
import os

try:
    import yfinance as yf
except ImportError:
    print("⚠️ yfinance not installed. Install with: pip install yfinance")
    yf = None

class AutonomousLearner:
    def __init__(self):
        self.db_path = 'src/modules/data/learning.db'
        self.learning_file = 'src/modules/learning_insights.json'
    
    def check_yesterdays_predictions(self):
        """Check predictions from yesterday - aggressive daily learning"""
        if not yf:
            print("❌ Cannot check predictions without yfinance")
            return 0
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        yesterday = datetime.now() - timedelta(days=1)
        
        cursor.execute("""
            SELECT id, stock, prediction, entry_price, llm_model
            FROM predictions
            WHERE DATE(timestamp) <= DATE(?)
            AND id NOT IN (SELECT prediction_id FROM outcomes)
            AND entry_price > 0
            LIMIT 100
        """, (yesterday,))
        
        unchecked = cursor.fetchall()
        print(f"\n🔍 Checking {len(unchecked)} predictions...")
        
        checked_count = 0
        for pred_id, stock, action, entry_price, llm_model in unchecked:
            try:
                ticker = yf.Ticker(stock)
                hist = ticker.history(period='1d')
                if hist.empty:
                    continue
                    
                current_price = hist['Close'].iloc[-1]
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Aggressive 1% threshold
                if action == 'BUY':
                    success = price_change_pct > 1
                elif action == 'SELL':
                    success = price_change_pct < -1
                else:
                    success = abs(price_change_pct) <= 1
                
                cursor.execute("""
                    INSERT INTO outcomes (prediction_id, actual_price, actual_move_pct, success)
                    VALUES (?, ?, ?, ?)
                """, (pred_id, current_price, price_change_pct, 1 if success else 0))
                
                checked_count += 1
                status = '✅' if success else '❌'
                print(f"  {stock}: {action} {status} ({price_change_pct:+.1f}%)")
                
            except Exception as e:
                print(f"  ⚠️ {stock}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        return checked_count
    
    def analyze_mistakes(self):
        """Find patterns in mistakes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                p.stock, p.prediction, p.reasoning, p.rsi, p.patterns, o.actual_move_pct
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            WHERE o.success = 0
            ORDER BY ABS(o.actual_move_pct) DESC
            LIMIT 30
        """)
        
        mistakes = cursor.fetchall()
        learnings = {}
        
        print("\n🔍 ANALYZING MISTAKES")
        print("=" * 50)
        
        for stock, pred, reasoning, rsi, patterns, actual_move in mistakes:
            if pred == "SELL" and actual_move > 1:
                mistake_type = "sold_before_rally"
                learning = f"RSI {rsi:.0f}: rallied {actual_move:.1f}% - should BUY not SELL"
            elif pred == "HOLD" and abs(actual_move) > 3:
                mistake_type = "missed_big_move"
                learning = f"Moved {actual_move:.1f}% - be decisive!"
            elif pred == "BUY" and actual_move < -1:
                mistake_type = "bought_before_drop"
                learning = f"Dropped {actual_move:.1f}% - bearish signal missed"
            else:
                continue
                
            if mistake_type not in learnings:
                learnings[mistake_type] = []
            learnings[mistake_type].append({
                'stock': stock,
                'learning': learning,
                'move': actual_move
            })
        
        conn.close()
        
        if learnings:
            self.save_learnings(learnings)
            
        return learnings
    
    def save_learnings(self, learnings):
        """Save for tomorrow's predictions"""
        if os.path.exists(self.learning_file):
            with open(self.learning_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {}
        
        existing[datetime.now().isoformat()] = learnings
        
        # Keep last 3 days
        dates = sorted(existing.keys())
        if len(dates) > 3:
            for old_date in dates[:-3]:
                del existing[old_date]
        
        with open(self.learning_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        print("\n🧠 LEARNING COMPLETE")
        print("=" * 50)
        for mistake_type, lessons in learnings.items():
            print(f"\n❌ {mistake_type.upper()}: {len(lessons)} cases")
            for lesson in lessons[:2]:
                print(f"  📝 {lesson['learning']}")
    
    def get_learning_prompt(self):
        """Get learnings for LLM prompts"""
        if not os.path.exists(self.learning_file):
            return ""
        
        with open(self.learning_file, 'r') as f:
            learnings = json.load(f)
        
        if not learnings:
            return ""
        
        prompt = "\n🚨 LEARN FROM PAST MISTAKES:\n"
        for date in sorted(learnings.keys())[-2:]:
            for mistake_type, lessons in learnings[date].items():
                if lessons:
                    prompt += f"- {lessons[0]['learning']}\n"
        
        return prompt

if __name__ == "__main__":
    learner = AutonomousLearner()
    
    checked = learner.check_yesterdays_predictions()
    
    if checked > 0:
        learnings = learner.analyze_mistakes()
        
        if learnings:
            total = sum(len(v) for v in learnings.values())
            print(f"\n✅ Found {total} learning patterns!")
        else:
            print("\n✅ All predictions were correct!")
    else:
        print("\n⏳ No predictions ready to check yet")
