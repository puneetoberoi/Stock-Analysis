import sqlite3
import json
from datetime import datetime, timedelta
import os

class AutonomousLearner:
    def __init__(self):
        self.db_path = 'src/modules/data/learning.db'
        self.learning_file = 'src/modules/learning_insights.json'
        
    def analyze_mistakes(self):
        """Analyze what went wrong and create learning rules"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all wrong predictions with details
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
            AND p.reasoning NOT LIKE '%Test prediction%'
            ORDER BY ABS(o.actual_move_pct) DESC
            LIMIT 20
        """)
        
        mistakes = cursor.fetchall()
        learnings = {}
        
        print("\n🔍 ANALYZING MISTAKES TO LEARN")
        print("=" * 50)
        
        for stock, pred, reasoning, rsi, patterns, actual_move in mistakes:
            # Identify the mistake pattern
            if pred == "SELL" and actual_move > 10:
                mistake_type = "sold_before_rally"
                learning = f"When RSI={rsi:.0f} and pattern={patterns}, stock rallied {actual_move:.1f}% - should have been BUY not SELL"
            elif pred == "HOLD" and abs(actual_move) > 15:
                mistake_type = "missed_big_move"
                learning = f"Stock moved {actual_move:.1f}% - pattern {patterns} needs decisive action (BUY/SELL) not HOLD"
            elif pred == "BUY" and actual_move < -10:
                mistake_type = "bought_before_drop"
                learning = f"Stock dropped {actual_move:.1f}% - pattern {patterns} with RSI={rsi:.0f} was bearish not bullish"
            else:
                continue
                
            if mistake_type not in learnings:
                learnings[mistake_type] = []
            learnings[mistake_type].append({
                'stock': stock,
                'learning': learning,
                'actual_move': actual_move,
                'original_reasoning': reasoning[:100]
            })
        
        conn.close()
        
        # Save learnings
        self.save_learnings(learnings)
        return learnings
    
    def save_learnings(self, learnings):
        """Save learning insights for LLMs to use tomorrow"""
        if os.path.exists(self.learning_file):
            with open(self.learning_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {}
        
        existing[datetime.now().isoformat()] = learnings
        
        # Keep only last 7 days of learnings
        dates = sorted(existing.keys())
        if len(dates) > 7:
            for old_date in dates[:-7]:
                del existing[old_date]
        
        with open(self.learning_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        print("\n🧠 AUTONOMOUS LEARNING COMPLETE")
        print("=" * 50)
        for mistake_type, lessons in learnings.items():
            print(f"\n❌ Pattern: {mistake_type.replace('_', ' ').upper()}")
            for lesson in lessons[:3]:  # Top 3 lessons
                print(f"  📝 {lesson['learning']}")
    
    def get_learning_prompt(self):
        """Get learning context for LLMs to improve"""
        if not os.path.exists(self.learning_file):
            return ""
        
        with open(self.learning_file, 'r') as f:
            learnings = json.load(f)
        
        if not learnings:
            return ""
            
        # Get most recent learnings
        latest_date = sorted(learnings.keys())[-1]
        latest = learnings[latest_date]
        
        prompt = "\nIMPORTANT LEARNINGS FROM PAST MISTAKES:\n"
        for mistake_type, lessons in latest.items():
            if lessons:
                prompt += f"- {mistake_type.replace('_', ' ').upper()}: "
                prompt += lessons[0]['learning']
                prompt += "\n"
        
        prompt += "\nAdjust your analysis based on these learnings!\n"
        return prompt

def check_recent_predictions(self):
    """Check predictions from 3+ days ago (not just 7)"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    from datetime import datetime, timedelta
    import yfinance as yf
    
    # Check predictions from 3+ days ago
    days_ago = datetime.now() - timedelta(days=3)
    
    cursor.execute("""
        SELECT id, stock, prediction, entry_price
        FROM predictions
        WHERE DATE(timestamp) <= DATE(?)
        AND id NOT IN (SELECT prediction_id FROM outcomes)
        AND reasoning NOT LIKE '%Test%'
        LIMIT 20
    """, (days_ago,))
    
    unchecked = cursor.fetchall()
    print(f"\n🔍 Found {len(unchecked)} unchecked predictions to verify...")
    
    for pred_id, stock, action, entry_price in unchecked:
        try:
            ticker = yf.Ticker(stock)
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Determine success
            if action == 'BUY':
                success = price_change_pct > 2
            elif action == 'SELL':
                success = price_change_pct < -2
            else:
                success = abs(price_change_pct) <= 2
            
            # Store outcome
            cursor.execute("""
                INSERT INTO outcomes (prediction_id, actual_price, actual_move_pct, success)
                VALUES (?, ?, ?, ?)
            """, (pred_id, current_price, price_change_pct, 1 if success else 0))
            
            print(f"  {stock}: {action} - {'✅ CORRECT' if success else '❌ WRONG'} ({price_change_pct:.1f}%)")
        except Exception as e:
            print(f"  Error: {e}")
    
    conn.commit()
    conn.close()


if __name__ == "__main__":
    learner = AutonomousLearner()
    learner.check_recent_predictions()
    learnings = learner.analyze_mistakes()
    
    if learnings:
        print(f"\n✅ Identified {sum(len(v) for v in learnings.values())} learning opportunities")
        print("📚 These insights will improve tomorrow's predictions!")
    else:
        print("\n📊 No significant mistakes to learn from yet.")
