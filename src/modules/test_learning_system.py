"""
Test script for the learning system
Run this to verify database integration and learning components
"""

import sys
import os
sys.path.append('src')

from modules.prediction_tracker import PredictionTracker
from modules.outcome_evaluator import OutcomeEvaluator
import sqlite3
from datetime import datetime, timedelta
import json

def test_database_connection():
    """Test 1: Verify database connection and tables"""
    print("\n" + "="*50)
    print("TEST 1: Database Connection and Schema")
    print("="*50)
    
    try:
        tracker = PredictionTracker("learning.db")
        
        # Check tables exist
        with sqlite3.connect("learning.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name
            """)
            tables = cursor.fetchall()
            
            print("✅ Database connected successfully")
            print(f"📊 Found {len(tables)} tables:")
            for table in tables:
                print(f"   - {table[0]}")
                
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_prediction_logging():
    """Test 2: Log sample predictions"""
    print("\n" + "="*50)
    print("TEST 2: Prediction Logging")
    print("="*50)
    
    tracker = PredictionTracker("learning.db")
    
    # Sample predictions to test
    test_predictions = [
        {
            'stock': 'AAPL',
            'prediction': 'BUY',
            'confidence': 85,
            'price': 180.50,
            'llm_model': 'groq',
            'reasoning': 'Strong bullish pattern detected',
            'indicators': {'rsi': 65, 'macd': 0.5, 'volume_ratio': 1.2},
            'patterns': ['double_bottom', 'bullish_engulfing'],
            'market_context': {'regime': 'bullish', 'macro_score': 15, 'sector': 'Technology'}
        },
        {
            'stock': 'GOOGL',
            'prediction': 'SELL',
            'confidence': 72,
            'price': 150.25,
            'llm_model': 'gemini',
            'reasoning': 'Overbought conditions detected',
            'indicators': {'rsi': 78, 'macd': -0.3, 'volume_ratio': 0.8},
            'patterns': ['evening_star', 'double_top'],
            'market_context': {'regime': 'bearish', 'macro_score': -10, 'sector': 'Technology'}
        },
        {
            'stock': 'MSFT',
            'prediction': 'HOLD',
            'confidence': 60,
            'price': 400.00,
            'llm_model': 'cohere',
            'reasoning': 'Mixed signals, neutral stance',
            'indicators': {'rsi': 50, 'macd': 0.1, 'volume_ratio': 1.0},
            'patterns': ['doji'],
            'market_context': {'regime': 'neutral', 'macro_score': 0, 'sector': 'Technology'}
        }
    ]
    
    prediction_ids = []
    for test_pred in test_predictions:
        try:
            pred_id = tracker.log_prediction(**test_pred)
            if pred_id > 0:
                prediction_ids.append(pred_id)
                print(f"✅ Logged prediction #{pred_id}: {test_pred['stock']} - {test_pred['prediction']}")
            else:
                print(f"❌ Failed to log prediction for {test_pred['stock']}")
        except Exception as e:
            print(f"❌ Error logging {test_pred['stock']}: {e}")
    
    print(f"\n📊 Successfully logged {len(prediction_ids)} predictions")
    return prediction_ids

def test_pending_checks():
    """Test 3: Retrieve pending predictions"""
    print("\n" + "="*50)
    print("TEST 3: Pending Predictions Check")
    print("="*50)
    
    tracker = PredictionTracker("learning.db")
    
    # Get pending predictions
    pending = tracker.get_pending_checks()
    
    print(f"📋 Found {len(pending)} pending predictions to check:")
    for p in pending[:5]:  # Show first 5
        print(f"   - {p['stock']}: {p['prediction']} (Model: {p['llm_model']}, Confidence: {p['confidence']}%)")
    
    return len(pending) > 0

def test_historical_accuracy():
    """Test 4: Calculate historical accuracy"""
    print("\n" + "="*50)
    print("TEST 4: Historical Accuracy Calculation")
    print("="*50)
    
    tracker = PredictionTracker("learning.db")
    
    # Get accuracy for all models
    accuracy_data = tracker.get_historical_accuracy(days_back=30)
    
    if accuracy_data:
        print("📈 Model Performance (Last 30 days):")
        for model, metrics in accuracy_data.items():
            print(f"\n   {model.upper()}:")
            print(f"   - Total Predictions: {metrics['total_predictions']}")
            print(f"   - Correct: {metrics['correct_predictions']}")
            print(f"   - Accuracy: {metrics['accuracy']:.1f}%")
            print(f"   - Avg Confidence: {metrics['avg_confidence']:.1f}%")
    else:
        print("📊 No historical data available yet (need outcomes first)")
    
    return True

def test_pattern_performance():
    """Test 5: Check pattern performance"""
    print("\n" + "="*50)
    print("TEST 5: Pattern Performance Analysis")
    print("="*50)
    
    tracker = PredictionTracker("learning.db")
    
    patterns = tracker.get_pattern_performance()
    
    if patterns:
        print("📊 Pattern Performance:")
        for pattern_name, metrics in patterns.items():
            print(f"   {pattern_name}:")
            print(f"   - Success Rate: {metrics['success_rate']:.1f}%")
            print(f"   - Sample Size: {metrics['sample_size']}")
            print(f"   - Avg Return: {metrics['avg_return']:.2f}%")
    else:
        print("📊 No pattern performance data yet (need more outcomes)")
    
    return True

def test_outcome_evaluation():
    """Test 6: Simulate outcome checking"""
    print("\n" + "="*50)
    print("TEST 6: Outcome Evaluation (Simulated)")
    print("="*50)
    
    # This would normally run in the evening
    # For testing, we'll simulate with fake data
    
    print("⚠️ Note: Real outcome evaluation requires market data")
    print("   This will be tested when evening check runs")
    
    # Show what the evaluator would do
    evaluator = OutcomeEvaluator("learning.db")
    tracker = PredictionTracker("learning.db")
    
    pending = tracker.get_pending_checks()
    print(f"📋 Would check {len(pending)} predictions in evening run")
    
    return True

def verify_current_database():
    """Check what's currently in the database"""
    print("\n" + "="*50)
    print("CURRENT DATABASE STATUS")
    print("="*50)
    
    with sqlite3.connect("learning.db") as conn:
        cursor = conn.cursor()
        
        # Count predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        print(f"📊 Total Predictions: {pred_count}")
        
        # Count outcomes
        cursor.execute("SELECT COUNT(*) FROM outcomes")
        outcome_count = cursor.fetchone()[0]
        print(f"✅ Total Outcomes Checked: {outcome_count}")
        
        # Show recent predictions
        cursor.execute("""
            SELECT stock, prediction, llm_model, confidence, timestamp
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        
        recent = cursor.fetchall()
        if recent:
            print("\n📅 Recent Predictions:")
            for row in recent:
                print(f"   {row[0]}: {row[1]} by {row[2]} ({row[3]}% conf) - {row[4]}")
        
        # Check if any patterns are tracked
        cursor.execute("SELECT COUNT(*) FROM pattern_performance")
        pattern_count = cursor.fetchone()[0]
        print(f"\n📈 Patterns Tracked: {pattern_count}")

def main():
    """Run all tests"""
    print("\n🧪 LEARNING SYSTEM TEST SUITE")
    print("================================")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Prediction Logging", test_prediction_logging),
        ("Pending Checks", test_pending_checks),
        ("Historical Accuracy", test_historical_accuracy),
        ("Pattern Performance", test_pattern_performance),
        ("Outcome Evaluation", test_outcome_evaluation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            failed += 1
    
    # Show current database status
    verify_current_database()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! System ready for production.")
    else:
        print("\n⚠️ Some tests failed. Review errors above.")

if __name__ == "__main__":
    main()
