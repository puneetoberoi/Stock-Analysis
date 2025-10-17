"""
Standalone Email Bot Monitor
Runs continuously to check for email questions
"""

import asyncio
import logging
from main import MarketIntelligenceDB, EmailConversationBot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_bot():
    """Monitor emails and respond to questions"""
    db = MarketIntelligenceDB()
    bot = EmailConversationBot(db)
    
    logging.info("🤖 Email bot started - monitoring inbox...")
    
    while True:
        try:
            bot.check_for_questions()
            await asyncio.sleep(300)  # Check every 5 minutes
        except Exception as e:
            logging.error(f"Bot error: {e}")
            await asyncio.sleep(60)  # Wait 1 min on error

if __name__ == "__main__":
    asyncio.run(run_bot())
