import sqlite3
import os
from datetime import datetime, timedelta

def check_accuracy():
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'learning.db')
    
    # Check if database exists
    if not os.path.exists(db_path):
        print("❌ Database not found. Run autonomous_learner.py first.")
        return
    
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
    
    has_data = False  # Track if we have any data to show
    
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
        
        # THIS IS THE KEY FIX - Better handling of empty/null results
        if result and result[0] and result[0] > 0:
            has_data = True
            total = result[0]
            correct = result[1] if result[1] else 0
            accuracy = result[2] if result[2] else 0.0
            
            print(f"\n{period_name}:")
            print(f"  Predictions: {total}")
            print(f"  Correct: {correct}")
            print(f"  Wrong: {total - correct}")
            print(f"  Accuracy: {accuracy}% {'📈' if accuracy > 33 else '📊'}")
            
            # Show improvement with better thresholds
            if accuracy >= 70:
                print(f"  Status: 🏆 EXCELLENT! Approaching target!")
            elif accuracy >= 50:
                print(f"  Status: 🎯 GREAT PROGRESS!")
            elif accuracy > 35:
                print(f"  Status: ✅ IMPROVING FROM BASELINE")
            else:
                print(f"  Status: 📊 LEARNING IN PROGRESS")
        else:
            print(f"\n{period_name}:")
            print(f"  No graded predictions yet for this period")
    
    # Show most improved stocks - with better error handling
    cursor.execute("""
        SELECT stock, 
               COUNT(*) as predictions,
               SUM(success) as correct,
               ROUND(CAST(SUM(success) AS FLOAT) / COUNT(*) * 100, 1) as accuracy
        FROM outcomes o
        JOIN predictions p ON o.prediction_id = p.id
        GROUP BY stock
        HAVING COUNT(*) >= 5
        ORDER BY accuracy DESC
        LIMIT 5
    """)
    
    top_stocks = cursor.fetchall()
    
    if top_stocks:
        print("\n🏆 TOP PERFORMING STOCKS:")
        for stock, count, correct, acc in top_stocks:
            print(f"  {stock}: {acc}% accuracy ({correct}/{count} correct)")
        
        # Also show worst performers so you know what to improve
        cursor.execute("""
            SELECT stock, 
                   COUNT(*) as predictions,
                   SUM(success) as correct,
                   ROUND(CAST(SUM(success) AS FLOAT) / COUNT(*) * 100, 1) as accuracy
            FROM outcomes o
            JOIN predictions p ON o.prediction_id = p.id
            GROUP BY stock
            HAVING COUNT(*) >= 5
            ORDER BY accuracy ASC
            LIMIT 3
        """)
        
        worst_stocks = cursor.fetchall()
        if worst_stocks:
            print("\n🎯 NEEDS IMPROVEMENT:")
            for stock, count, correct, acc in worst_stocks:
                print(f"  {stock}: {acc}% accuracy ({correct}/{count} correct)")
    
    if not has_data:
        print("\n⏳ No graded predictions found yet.")
        print("   The learning system will start grading predictions tomorrow.")
        
    # Show total statistics
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT stock) as stocks,
            COUNT(*) as total_predictions,
            COUNT(DISTINCT date(check_date)) as days_tracked
        FROM outcomes o
        JOIN predictions p ON o.prediction_id = p.id
    """)
    
    stats = cursor.fetchone()
    if stats and stats[1] > 0:
        print("\n📈 OVERALL STATISTICS:")
        print(f"  Tracking: {stats[0]} stocks")
        print(f"  Total Predictions Graded: {stats[1]}")
        print(f"  Days of Learning: {stats[2]}")
    
    conn.close()
    print("\n" + "="*50)

if __name__ == "__main__":
    check_accuracy()
