"""Manual test for learning system"""

import sys
import os
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🧪 Testing Learning System...")
    
    # Check database
    if not os.path.exists("learning.db"):
        logger.error("❌ learning.db not found!")
        return 1
    
    logger.info("✅ learning.db exists")
    
    # Check tables
    try:
        conn = sqlite3.connect("learning.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        logger.info(f"✅ Predictions in DB: {count}")
        
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE date(timestamp) = date('now')")
        today = cursor.fetchone()[0]
        logger.info(f"✅ Today's predictions: {today}")
        
        conn.close()
        
        if count == 0:
            logger.warning("⚠️ No predictions in database yet")
        
        logger.info("✅ All tests passed!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
