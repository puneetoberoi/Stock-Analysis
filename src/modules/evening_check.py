"""
Evening Outcome Checker
Runs at 6pm to evaluate today's predictions
"""

import sys
import logging
from modules.prediction_tracker import PredictionTracker
from modules.outcome_evaluator import OutcomeEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Check outcomes of pending predictions"""
    logger.info("🌅 EVENING OUTCOME CHECK - START")
    logger.info("="*60)
    
    tracker = PredictionTracker("learning.db")
    evaluator = OutcomeEvaluator("learning.db")
    
    # Get predictions that need checking
    pending = tracker.get_pending_checks()
    logger.info(f"📋 Found {len(pending)} predictions to check")
    
    if not pending:
        logger.info("✅ No pending predictions to check")
        return
    
    # Evaluate outcomes
    results = evaluator.check_outcomes(tracker)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 OUTCOME SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Checked: {results['checked']}")
    logger.info(f"🎯 Successful: {results['successful']}")
    logger.info(f"❌ Failed: {results['failed']}")
    
    if results['successful'] > 0:
        accuracy = (results['successful'] / results['checked']) * 100
        logger.info(f"📈 Accuracy: {accuracy:.1f}%")
    
    # Show lessons learned
    if results['lessons']:
        logger.info("\n💡 LESSONS LEARNED:")
        for lesson in results['lessons'][:5]:  # Top 5
            logger.info(f"   • {lesson['lesson']}")
            logger.info(f"     Adjustment: {lesson['adjustment']}")
    
    logger.info("\n✅ Evening check complete!")

if __name__ == "__main__":
    main()
