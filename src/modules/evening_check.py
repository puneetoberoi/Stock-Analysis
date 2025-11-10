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

# Fix Python path - add src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) if 'modules' in current_dir else current_dir
project_root = os.path.dirname(src_dir) if os.path.basename(src_dir) == 'src' else os.path.dirname(current_dir)

# Add both src and project root to path
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

logger.info(f"Python path: {sys.path[:3]}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Script location: {current_dir}")

try:
    # Try multiple import paths
    try:
        from modules.prediction_tracker import PredictionTracker
        from modules.outcome_evaluator import OutcomeEvaluator
        logger.info("✅ Imported from modules.*")
    except ImportError:
        from src.modules.prediction_tracker import PredictionTracker
        from src.modules.outcome_evaluator import OutcomeEvaluator
        logger.info("✅ Imported from src.modules.*")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.error(f"Tried importing from: {sys.path}")
    sys.exit(1)

def main():
    """Check outcomes of pending predictions"""
    logger.info("="*60)
    logger.info("🌅 EVENING OUTCOME CHECK - START")
    logger.info("="*60)
    
    try:
        # Find database file
        db_paths_to_try = [
            "learning.db",
            os.path.join(project_root, "learning.db"),
            os.path.join(os.getcwd(), "learning.db")
        ]
        
        db_path = None
        for path in db_paths_to_try:
            if os.path.exists(path):
                db_path = path
                logger.info(f"✅ Found database at: {db_path}")
                break
        
        if not db_path:
            logger.error(f"❌ Database not found. Tried: {db_paths_to_try}")
            return 1
        
        # Initialize
        tracker = PredictionTracker(db_path)
        evaluator = OutcomeEvaluator(db_path)
        
        # Get predictions that need checking
        pending = tracker.get_pending_checks()
        logger.info(f"📋 Found {len(pending)} predictions to check")
        
        if not pending:
            logger.info("✅ No pending predictions to check today")
            logger.info("💡 Predictions are checked 1 day after they're made")
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
