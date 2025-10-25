# /src/check_outcomes.py
#
# This is the "teacher" script. It runs in the evening to grade the morning's predictions.

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, date
from pathlib import Path

# --- Setup Paths and Logging ---
# This allows the script to find your main.py and its components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import The Learning Components from main.py ---
# We are re-using the same instances you defined in main.py
try:
    from main import prediction_tracker, candle_analyzer, learning_memory, yf
except ImportError as e:
    logging.critical(f"Failed to import components from main.py: {e}. Make sure they are defined at the global scope.")
    sys.exit(1)


async def check_single_prediction_outcome(pred_id, pred):
    """Checks the outcome of a single past prediction."""
    try:
        # Check if already processed or too recent
        if pred.get('outcome') is not None:
            return None # Skip
            
        pred_date = datetime.fromisoformat(pred['timestamp'])
        # Give it at least one full trading day to mature
        if datetime.now() < pred_date + timedelta(days=1):
            return None # Skip, too recent

        ticker = pred['ticker']
        stock = yf.Ticker(ticker)
        
        # Get price data from the prediction date until now
        hist = await asyncio.to_thread(stock.history, start=pred_date.date(), end=(pred_date.date() + timedelta(days=5)))
        
        if len(hist) < 2:
            logging.warning(f"Not enough history to check outcome for {ticker} on {pred_date.date()}")
            return None

        # Price at prediction time and price now (or next day's close)
        pred_price = pred.get('price_at_prediction')
        if not pred_price:
            logging.warning(f"Missing price_at_prediction for {ticker} on {pred_date.date()}")
            return None
            
        actual_price_after_1d = hist['Close'].iloc[1] # Next day's close
        
        price_change_pct = ((actual_price_after_1d - pred_price) / pred_price) * 100
        
        # Determine if the prediction was correct based on a simple threshold
        was_correct = False
        action = pred['action']
        
        if action == 'BUY' and price_change_pct > 0.5:
            was_correct = True
        elif action == 'SELL' and price_change_pct < -0.5:
            was_correct = True
        elif action == 'HOLD' and abs(price_change_pct) <= 1.0: # Wider threshold for HOLD
            was_correct = True
        
        # Prepare the outcome data
        outcome_data = {
            'checked_date': datetime.now().isoformat(),
            'price_after_1d': float(actual_price_after_1d),
            'price_change_pct_1d': price_change_pct,
            'was_correct': was_correct
        }
        
        logging.info(f"✅ Outcome for {ticker} ({pred_date.strftime('%Y-%m-%d')}): Predicted {action}, Actual Change: {price_change_pct:+.2f}%. Prediction was {'CORRECT' if was_correct else 'WRONG'}.")
        
        return pred_id, outcome_data

    except Exception as e:
        logging.error(f"Error checking outcome for prediction {pred_id} ({pred.get('ticker')}): {e}")
        return None


async def run_learning_process():
    """The main process to check outcomes and update learning memory."""
    logging.info("🧠 Starting evening learning process...")
    
    # Load all predictions that haven't been checked yet
    unchecked_predictions = {pid: p for pid, p in prediction_tracker.predictions.items() if p.get('outcome') is None}
    
    if not unchecked_predictions:
        logging.info("No new predictions to check. Learning complete for today.")
        return

    logging.info(f"Found {len(unchecked_predictions)} unchecked predictions to process.")

    # Check outcomes concurrently
    tasks = [check_single_prediction_outcome(pid, p) for pid, p in unchecked_predictions.items()]
    results = await asyncio.gather(*tasks)

    # --- This is the "Learning" Part ---
    correct_count = 0
    total_checked = 0
    
    for result in results:
        if result is None:
            continue
            
        pred_id, outcome_data = result
        total_checked += 1
        
        # Update the prediction in our tracker
        prediction_tracker.predictions[pred_id]['outcome'] = outcome_data
        prediction_tracker.predictions[pred_id]['was_correct'] = outcome_data['was_correct']
        
        # Get the original prediction context
        original_pred = prediction_tracker.predictions[pred_id]
        was_correct = outcome_data['was_correct']
        
        if was_correct:
            correct_count += 1
            
        # Learn from candlestick patterns
        if original_pred.get('candle_pattern'):
            candle_analyzer.update_pattern_outcome(
                original_pred['candle_pattern'],
                original_pred['ticker'],
                was_correct
            )
            
        # Learn about LLM accuracy (if LLM data was stored)
        # We need to enhance the prediction storage to include which LLMs were used
        # For now, this is a placeholder for future enhancement
        # Example: learning_memory.update_llm_accuracy('groq', was_correct)
        
    # Save all the updated predictions and pattern history to their JSON files
    prediction_tracker._save_predictions()
    candle_analyzer._save_patterns()
    
    # Add a summary insight to the learning memory
    if total_checked > 0:
        accuracy = (correct_count / total_checked) * 100
        insight = f"Graded {total_checked} predictions with an accuracy of {accuracy:.1f}%."
        learning_memory.add_insight(insight)
        logging.info(f"📈 Learning Summary: {insight}")
    
    logging.info("✅ Evening learning process complete.")


async def main():
    """Entry point for the script."""
    await run_learning_process()

if __name__ == "__main__":
    # This allows the script to be run directly from the command line
    asyncio.run(main())
