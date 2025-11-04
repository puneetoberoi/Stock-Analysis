"""
Manual test script for Learning Brain - designed for GitHub Actions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from learning_brain import LearningBrain
from datetime import datetime, timedelta
import random

def comprehensive_test():
    """Run comprehensive tests on Learning Brain"""
    
    print("="*60)
    print("🧪 LEARNING BRAIN COMPREHENSIVE TEST")
    print("="*60)
    
    # Initialize
    brain = LearningBrain()
    print("✅ Step 1: Brain initialized")
    
    # Test data
    test_stocks = [
        ('AAPL', 'BUY', 85.5, 175.50, 'groq-llama'),
        ('MSFT', 'HOLD', 65.0, 415.25, 'cohere'),
        ('GOOGL', 'SELL', 72.3, 155.80, 'gemini'),
        ('TSLA', 'BUY', 91.2, 245.60, 'groq-mixtral'),
        ('NVDA', 'HOLD', 55.5, 885.50, 'cohere')
    ]
    
    # Record multiple predictions
    print("\n📝 Step 2: Recording test predictions...")
    prediction_ids = []
    
    for stock, pred, conf, price, model in test_stocks:
        indicators = {
            'rsi': random.uniform(30, 70),
            'macd': random.uniform(-1, 1),
            'volume_ratio': random.uniform(0.5, 2.0),
            'patterns': 'test_pattern'
        }
        
        pred_id = brain.record_prediction(
            stock=stock,
            prediction=pred,
            confidence=conf,
            price=price,
            llm_model=model,
            reasoning=f"Test prediction for {stock}",
            indicators=indicators
        )
        prediction_ids.append(pred_id)
        print(f"   ✓ {stock}: {pred} @ ${price:.2f} (ID: {pred_id})")
    
    # Check current database state
    print("\n📊 Step 3: Database Statistics")
    import sqlite3
    conn = sqlite3.connect('data/learning.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    pred_count = cursor.fetchone()[0]
    print(f"   Total predictions: {pred_count}")
    
    cursor.execute("SELECT COUNT(*) FROM outcomes")
    outcome_count = cursor.fetchone()[0]
    print(f"   Total outcomes tracked: {outcome_count}")
    
    # Show sample predictions
    print("\n📋 Step 4: Sample Predictions in Database")
    cursor.execute("""
        SELECT stock, prediction, confidence, llm_model, 
               DATE(timestamp) as date
        FROM predictions 
        ORDER BY id DESC 
        LIMIT 5
    """)
    
    for row in cursor.fetchall():
        stock, pred, conf, model, date = row
        print(f"   {stock}: {pred} ({conf:.1f}% confidence) by {model} on {date}")
    
    conn.close()
    
    # Test outcome checking (simulate)
    print("\n🔍 Step 5: Testing Outcome Checker")
    brain.check_outcomes(days_back=0)  # Check today's predictions as test
    
    # Generate accuracy report
    print("\n📈 Step 6: Accuracy Report")
    report = brain.get_accuracy_report()
    print(report)
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Create summary for GitHub Actions
    print("\n## GitHub Actions Summary")
    print(f"- Predictions recorded: {len(prediction_ids)}")
    print(f"- Database size: {pred_count} predictions")
    print(f"- Test status: PASSED ✅")
    
    return True

if __name__ == "__main__":
    try:
        success = comprehensive_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
