import sqlite3
from datetime import datetime, timedelta

def check_prediction_accuracy():
    """Check how our predictions performed"""
    # Use correct database path
    conn = sqlite3.connect('src/modules/data/learning.db')
    cursor = conn.cursor()
    
    # Get predictions from 7 days ago
    week_ago = datetime.now() - timedelta(days=7)
    
    cursor.execute("""
        SELECT stock, prediction, confidence, entry_price, reasoning
        FROM predictions
        WHERE DATE(timestamp) = DATE(?)
    """, (week_ago,))
    
    predictions = cursor.fetchall()
    
    print(f"📊 Checking {len(predictions)} predictions from {week_ago.date()}")
    
    for pred in predictions:
        stock, action, confidence, entry_price, reasoning = pred
        print(f"\n{stock}: {action} (Confidence: {confidence}%)")
        print(f"  Entry: ${entry_price:.2f}")
        print(f"  Reasoning: {reasoning[:100]}...")
        
        # Here you would check current price and calculate if prediction was correct
        # This is where the learning happens!
    
    conn.close()

if __name__ == "__main__":
    check_prediction_accuracy()
