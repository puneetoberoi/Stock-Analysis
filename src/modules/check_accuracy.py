import sqlite3
import os
from datetime import datetime, timedelta

def check_accuracy():
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'learning.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get accuracy for last 7, 30, and all-time
    periods = [
        ("Last 7 days", 7),
        ("Last 30 days", 30),
        ("All-time", 9999)
    ]
    
    print("\n" + "="*50)
    print("📊 AI PREDICTION ACCURACY REPORT")
    print("="*50)
    
    for period_name, days in periods:
        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(success) as correct,
                ROUND(CAST(SUM(success) AS FLOAT) / COUNT(*) * 100, 1) as accuracy
            FROM outcomes o
            JOIN predictions p ON o.prediction_id = p.id
            WHERE o.check_date >= ?
        """, (since_date,))
        
        result = cursor.fetchone()
        if result and result[0] > 0:
            total, correct, accuracy = result
            print(f"\n{period_name}:")
            print(f"  Predictions: {total}")
            print(f"  Correct: {correct}")
            print(f"  Accuracy: {accuracy}% {'📈' if accuracy > 33 else '📊'}")
            
            # Show improvement
            if accuracy > 50:
                print(f"  Status: 🎯 EXCEEDING EXPECTATIONS!")
            elif accuracy > 33:
                print(f"  Status: ✅ IMPROVING FROM BASELINE")
    
    # Show most improved stocks
    cursor.execute("""
        SELECT stock, 
               COUNT(*) as predictions,
               ROUND(CAST(SUM(success) AS FLOAT) / COUNT(*) * 100, 1) as accuracy
        FROM outcomes o
        JOIN predictions p ON o.prediction_id = p.id
        GROUP BY stock
        HAVING COUNT(*) >= 5
        ORDER BY accuracy DESC
        LIMIT 3
    """)
    
    print("\n🏆 TOP PERFORMING STOCKS:")
    for stock, count, acc in cursor.fetchall():
        print(f"  {stock}: {acc}% accuracy ({count} predictions)")
    
    conn.close()
    print("\n" + "="*50)

if __name__ == "__main__":
    check_accuracy()
