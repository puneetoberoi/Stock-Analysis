"""
Manual test script for learning system
Can be run locally or in GitHub Actions for testing
"""

import sys
import os
import sqlite3
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_exists():
    """Test if learning database exists and has correct schema"""
    logger.info("🔍 Testing database existence...")
    
    db_path = "learning.db"
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found at {db_path}")
        return False
    
    logger.info(f"✅ Database found at {db_path}")
    
    # Check tables
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['predictions', 'outcomes', 'accuracy_tracking']
        missing_tables = [t for t in required_tables if t not in tables]
        
        if missing_tables:
            logger.error(f"❌ Missing tables: {missing_tables}")
            return False
        
        logger.info(f"✅ All required tables exist: {required_tables}")
        
        # Check prediction count
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        logger.info(f"📊 Total predictions in database: {count}")
        
        # Check today's predictions
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE date(timestamp) = date('now')")
        today_count = cursor.fetchone()[0]
        logger.info(f"📅 Predictions from today: {today_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return False

def test_prediction_tracker():
    """Test if PredictionTracker can be imported and initialized"""
    logger.info("\n🔍 Testing PredictionTracker...")
    
    try:
        from modules.prediction_tracker import PredictionTracker
        tracker = PredictionTracker("learning.db")
        logger.info("✅ PredictionTracker initialized successfully")
        
        # Test getting pending checks
        pending = tracker.get_pending_checks()
        logger.info(f"📋 Pending predictions to check: {len(pending)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PredictionTracker error: {e}")
        return False

def test_outcome_evaluator():
    """Test if OutcomeEvaluator can be imported"""
    logger.info("\n🔍 Testing OutcomeEvaluator...")
    
    try:
        from modules.outcome_evaluator import OutcomeEvaluator
        evaluator = OutcomeEvaluator("learning.db")
        logger.info("✅ OutcomeEvaluator initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ OutcomeEvaluator error: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("🧪 LEARNING SYSTEM MANUAL TEST")
    logger.info("="*60)
    logger.info(f"📅 Test run at: {datetime.now()}")
    logger.info("="*60 + "\n")
    
    tests = [
        ("Database Exists", test_database_exists),
        ("PredictionTracker", test_prediction_tracker),
        ("OutcomeEvaluator", test_outcome_evaluator),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info("="*60)
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"⚠️ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
