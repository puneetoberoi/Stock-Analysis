"""
Evening Outcome Checker
Runs at 6pm to evaluate today's predictions
"""

import sys
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.prediction_tracker import PredictionTracker
    from modules.outcome_evaluator import OutcomeEvaluator
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

def main():
    """Check outcomes of pending predictions"""
    logger.info("="*60)
    logger.info("🌅 EVENING OUTCOME CHECK - START")
    logger.info("="*60)
    
    try:
        # Initialize
        tracker = PredictionTracker("learning.db")
        evaluator = OutcomeEvaluator("learning.db")
        
        # Get predictions that need checking
        pending = tracker.get_pending_checks()
        logger.info(f"📋 Found {len(pending)} predictions to check")
        
        if not pending:
            logger.info("✅ No pending predictions to check today")
            return 0
        
        # Evaluate outcomes
        logger.info("🔍 Evaluating prediction outcomes...")
        results = evaluator.check_outcomes(tracker)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 OUTCOME SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Checked: {results['checked']}")
        logger.info(f"🎯 Successful: {results['successful']}")
        logger.info(f"❌ Failed: {results['failed']}")
        
        if results['checked'] > 0:
            accuracy = (results['successful'] / results['checked']) * 100
            logger.info(f"📈 Accuracy: {accuracy:.1f}%")
        
        # Show lessons learned
        if results.get('lessons'):
            logger.info("\n💡 TOP LESSONS LEARNED:")
            for i, lesson in enumerate(results['lessons'][:5], 1):
                logger.info(f"\n{i}. {lesson.get('stock', 'Unknown')}: {lesson.get('lesson', 'No lesson')}")
                if lesson.get('adjustment'):
                    logger.info(f"   → Adjustment: {lesson['adjustment']}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Evening check complete!")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evening check failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
