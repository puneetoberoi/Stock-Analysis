import sqlite3
import json
from datetime import datetime, timedelta
import os
import yfinance as yf

class AutonomousLearner:
    def __init__(self):
        self.db_path = 'src/modules/data/learning.db'
        self.learning_file = 'src/modules/learning_insights.json'
    
    def check_yesterdays_predictions(self):
        """Check ALL predictions from yesterday - aggressive learning!"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get ALL predictions at least 1 day old
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
        print(f"\n🔍 Checking {len(unchecked)} predictions from yesterday...")
        
        checked_count = 0
        for pred_id, stock, action, entry_price, llm_model in unchecked:
            try:
                ticker = yf.Ticker(stock)
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Quick aggressive thresholds (1% instead of 2%)
                if action == 'BUY':
                    success = price_change_pct > 1  # ← More aggressive
                elif action == 'SELL':
                    success = price_change_pct < -1  # ← More aggressive  
                else:  # HOLD
                    success = abs(price_change_pct) <= 1
                
                # Store outcome
                cursor.execute("""
                    INSERT INTO outcomes (prediction_id, actual_price, actual_move_pct, success)
                    VALUES (?, ?, ?, ?)
                """, (pred_id, current_price, price_change_pct, 1 if success else 0))
                
                checked_count += 1
                status = '✅' if success else '❌'
                print(f"  {stock}: {action} was {status} (moved {price_change_pct:.1f}%)")
                
            except Exception as e:
                continue  # Skip errors silently
        
        conn.commit()
        conn.close()
        
        return checked_count
    
    def analyze_mistakes(self):
        """Analyze what went wrong and create learning rules"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get ALL wrong predictions (not just recent)
        cursor.execute("""
            SELECT 
                p.stock, 
                p.prediction, 
                p.reasoning,
                p.rsi,
                p.patterns,
                o.actual_move_pct
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            WHERE o.success = 0
            ORDER BY o.check_date DESC
            LIMIT 50
        """)
        
        mistakes = cursor.fetchall()
        learnings = {}
        
        print("\n🔍 ANALYZING MISTAKES TO LEARN")
        print("=" * 50)
        
        for stock, pred, reasoning, rsi, patterns, actual_move in mistakes:
            # More aggressive learning patterns
            if pred == "SELL" and actual_move > 1:
                mistake_type = "sold_before_rally"
                learning = f"When RSI={rsi:.0f}, stock rallied {actual_move:.1f}% - should BUY not SELL"
            elif pred == "HOLD" and abs(actual_move) > 3:
                mistake_type = "missed_big_move"
                learning = f"Stock moved {actual_move:.1f}% - be decisive, not HOLD"
            elif pred == "BUY" and actual_move < -1:
                mistake_type = "bought_before_drop"
                learning = f"Stock dropped {actual_move:.1f}% - was bearish signal"
            else:
                continue
                
            if mistake_type not in learnings:
                learnings[mistake_type] = []
            learnings[mistake_type].append({
                'stock': stock,
                'learning': learning,
                'actual_move': actual_move
            })
        
        conn.close()
        
        # Save learnings
        if learnings:
            self.save_learnings(learnings)
        return learnings
    
    def save_learnings(self, learnings):
        """Save learning insights for tomorrow's predictions"""
        if os.path.exists(self.learning_file):
            with open(self.learning_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {}
        
        existing[datetime.now().isoformat()] = learnings
        
        # Keep only last 3 days (more aggressive rotation)
        dates = sorted(existing.keys())
        if len(dates) > 3:
            for old_date in dates[:-3]:
                del existing[old_date]
        
        with open(self.learning_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        print("\n🧠 AUTONOMOUS LEARNING COMPLETE")
        print("=" * 50)
        for mistake_type, lessons in learnings.items():
            print(f"\n❌ Pattern: {mistake_type.replace('_', ' ').upper()}")
            for lesson in lessons[:3]:
                print(f"  📝 {lesson['learning']}")
    
    def get_learning_prompt(self):
        """Get aggressive learning prompts for LLMs"""
        if not os.path.exists(self.learning_file):
            return ""
        
        with open(self.learning_file, 'r') as f:
            learnings = json.load(f)
        
        if not learnings:
            return ""
            
        # Get ALL recent learnings (not just latest)
        prompt = "\n🚨 CRITICAL LEARNINGS - ADJUST YOUR STRATEGY:\n"
        for date in sorted(learnings.keys())[-2:]:  # Last 2 days
            for mistake_type, lessons in learnings[date].items():
                if lessons:
                    prompt += f"- {lessons[0]['learning']}\n"
        
        prompt += "BE MORE AGGRESSIVE - ACT ON THESE LEARNINGS!\n"
        return prompt

if __name__ == "__main__":
    learner = AutonomousLearner()
    
    # First check yesterday's predictions
    checked = learner.check_yesterdays_predictions()
    
    if checked > 0:
        # Now analyze mistakes
        learnings = learner.analyze_mistakes()
        
        if learnings:
            print(f"\n✅ Identified {sum(len(v) for v in learnings.values())} learning opportunities")
            print("📚 These insights will improve tomorrow's predictions!")
        else:
            print("\n✅ No mistakes found - system is learning!")
    else:
        print("\n⏳ No predictions old enough to check yet (need 24 hours)")
