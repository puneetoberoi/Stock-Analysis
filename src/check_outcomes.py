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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import The Learning Components from main.py ---
try:
    from main import prediction_tracker, candle_analyzer, learning_memory, yf
except ImportError as e:
    logging.critical(f"Failed to import components from main.py: {e}")
    sys.exit(1)


def fix_date_if_needed(date_str):
    """Fix the year if it's showing 2025 instead of 2024"""
    if '2025' in date_str:
        return date_str.replace('2025', '2024')
    return date_str


def is_market_day(check_date):
    """Check if a date is a trading day (Mon-Fri, not a holiday)"""
    return check_date.weekday() < 5  # Monday = 0, Friday = 4


def get_trading_days_between(start_date, end_date):
    """Count trading days between two dates"""
    trading_days = 0
    current = start_date
    while current <= end_date:
        if is_market_day(current):
            trading_days += 1
        current += timedelta(days=1)
    return trading_days


async def check_single_prediction_outcome(pred_id, pred):
    """Checks the outcome of a single past prediction with smarter date checking."""
    try:
        # Skip if already checked
        if pred.get('outcome') is not None:
            logging.debug(f"Prediction {pred_id} already has outcome, skipping")
            return None

        # Fix the date issue
        pred_timestamp_str = fix_date_if_needed(pred['timestamp'])
        pred_date = datetime.fromisoformat(pred_timestamp_str)
        current_date = datetime.now()
        
        # Calculate how many trading days have passed
        trading_days_passed = get_trading_days_between(pred_date.date(), current_date.date())
        
        # Need at least 2 trading days to check outcome
        MIN_TRADING_DAYS = 2
        if trading_days_passed < MIN_TRADING_DAYS:
            logging.info(f"⏳ {pred['ticker']}: Only {trading_days_passed} trading days since prediction. Need {MIN_TRADING_DAYS} days to check outcome.")
            return None

        ticker = pred['ticker']
        stock = yf.Ticker(ticker)
        
        # Get historical data from prediction date to now
        start_date = pred_date.date()
        end_date = min(pred_date.date() + timedelta(days=7), current_date.date())
        
        logging.debug(f"Fetching {ticker} data from {start_date} to {end_date}")
        hist = await asyncio.to_thread(
            stock.history, 
            start=start_date, 
            end=end_date
        )
        
        if len(hist) < 2:
            logging.warning(f"Not enough history for {ticker}. Got {len(hist)} days of data.")
            return None

        # Get prices
        pred_price = pred.get('price_at_prediction')
        if not pred_price:
            # Try to get it from the first day's data
            pred_price = hist['Close'].iloc[0]
            logging.info(f"Using close price from {start_date} as prediction price: ${pred_price:.2f}")
            
        # Use the most recent close price (or price from 2+ days later)
        if len(hist) >= 3:
            actual_price_after = hist['Close'].iloc[2]  # Price after 2 trading days
            days_held = 2
        else:
            actual_price_after = hist['Close'].iloc[-1]  # Most recent available
            days_held = len(hist) - 1
        
        price_change_pct = ((actual_price_after - pred_price) / pred_price) * 100
        
        # Determine if the prediction was correct
        was_correct = False
        action = pred['action']
        
        # Success thresholds
        BULLISH_SUCCESS_THRESHOLD = 1.0  # Need >1% gain for BUY to be correct
        BEARISH_SUCCESS_THRESHOLD = -1.0  # Need >1% loss for SELL to be correct
        HOLD_THRESHOLD = 2.0  # Need <2% movement for HOLD to be correct
        
        if action == 'BUY' and price_change_pct > BULLISH_SUCCESS_THRESHOLD:
            was_correct = True
        elif action == 'SELL' and price_change_pct < BEARISH_SUCCESS_THRESHOLD:
            was_correct = True
        elif action == 'HOLD' and abs(price_change_pct) <= HOLD_THRESHOLD:
            was_correct = True
        
        # Prepare the outcome data
        outcome_data = {
            'checked_date': datetime.now().isoformat(),
            'price_at_prediction': float(pred_price),
            'price_after': float(actual_price_after),
            'days_held': days_held,
            'price_change_pct': round(price_change_pct, 2),
            'was_correct': was_correct,
            'action_taken': action
        }
        
        # Log result with emoji
        emoji = "✅" if was_correct else "❌"
        logging.info(
            f"{emoji} {ticker} ({pred_date.strftime('%Y-%m-%d')}): "
            f"Predicted {action}, Price moved {price_change_pct:+.2f}% in {days_held} days. "
            f"Prediction was {'CORRECT' if was_correct else 'WRONG'}."
        )
        
        return pred_id, outcome_data

    except Exception as e:
        logging.error(f"Error checking outcome for {pred.get('ticker')}: {e}")
        return None


async def run_learning_process():
    """The main process to check outcomes and update learning memory."""
    logging.info("🧠 Starting evening learning process...")
    logging.info(f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Fix dates in predictions if needed
    for pred_id, pred in prediction_tracker.predictions.items():
        if '2025' in pred['timestamp']:
            pred['timestamp'] = pred['timestamp'].replace('2025', '2024')
    
    # Load predictions that haven't been checked yet
    unchecked_predictions = {
        pid: p for pid, p in prediction_tracker.predictions.items() 
        if p.get('outcome') is None
    }
    
    if not unchecked_predictions:
        logging.info("No unchecked predictions found.")
        return

    logging.info(f"Found {len(unchecked_predictions)} unchecked predictions.")
    
    # Group by age
    current_date = datetime.now()
    ready_to_check = []
    too_recent = []
    
    for pid, pred in unchecked_predictions.items():
        pred_date = datetime.fromisoformat(fix_date_if_needed(pred['timestamp']))
        days_old = get_trading_days_between(pred_date.date(), current_date.date())
        
        if days_old >= 2:
            ready_to_check.append((pid, pred))
        else:
            too_recent.append((pid, pred, days_old))
    
    logging.info(f"📊 Ready to check: {len(ready_to_check)} predictions")
    logging.info(f"⏳ Too recent: {len(too_recent)} predictions (need more time)")
    
    if not ready_to_check:
        logging.info("No predictions old enough to check. Check back tomorrow!")
        return

    # Check outcomes concurrently
    tasks = [check_single_prediction_outcome(pid, p) for pid, p in ready_to_check]
    results = await asyncio.gather(*tasks)

    # Process results and update learning
    correct_count = 0
    total_checked = 0
    pattern_performance = {}
    
    for result in results:
        if result is None:
            continue
            
        pred_id, outcome_data = result
        total_checked += 1
        
        # Update the prediction with outcome
        prediction_tracker.predictions[pred_id]['outcome'] = outcome_data
        prediction_tracker.predictions[pred_id]['was_correct'] = outcome_data['was_correct']
        
        original_pred = prediction_tracker.predictions[pred_id]
        was_correct = outcome_data['was_correct']
        
        if was_correct:
            correct_count += 1
            
        # Track pattern performance
        if original_pred.get('candle_pattern'):
            pattern = original_pred['candle_pattern']
            if pattern not in pattern_performance:
                pattern_performance[pattern] = {'correct': 0, 'total': 0}
            pattern_performance[pattern]['total'] += 1
            if was_correct:
                pattern_performance[pattern]['correct'] += 1
            
            # Update pattern analyzer
            candle_analyzer.update_pattern_outcome(
                pattern,
                original_pred['ticker'],
                was_correct
            )
    
    # Save updated predictions
    prediction_tracker._save_predictions()
    candle_analyzer._save_patterns()
    
    # Generate learning insights
    if total_checked > 0:
        accuracy = (correct_count / total_checked) * 100
        
        # Main insight
        insight = f"Checked {total_checked} predictions: {accuracy:.1f}% accuracy ({correct_count}/{total_checked} correct)"
        learning_memory.add_insight(insight)
        logging.info(f"📈 {insight}")
        
        # Pattern insights
        for pattern, stats in pattern_performance.items():
            pattern_acc = (stats['correct'] / stats['total']) * 100
            pattern_insight = f"Pattern '{pattern}': {pattern_acc:.0f}% accuracy ({stats['correct']}/{stats['total']})"
            learning_memory.add_insight(pattern_insight)
            logging.info(f"   🕯️ {pattern_insight}")
        
        # Save learning memory
        learning_memory._save_memory()
    
    # Summary
    logging.info("✅ Evening learning process complete!")
    logging.info(f"   - Checked: {total_checked} predictions")
    logging.info(f"   - Pending: {len(too_recent)} predictions (will check later)")
    logging.info(f"   - Accuracy: {accuracy:.1f}%" if total_checked > 0 else "   - No predictions checked")


async def main():
    """Entry point for the script."""
    await run_learning_process()

if __name__ == "__main__":
    asyncio.run(main())
