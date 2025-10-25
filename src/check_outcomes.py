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
import random

# --- Setup Paths and Logging ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================
# 🔴 TESTING MODE - SET TO False FOR PRODUCTION
# ============================================
TEST_MODE = True  # Change to False when you want real checking
# ============================================

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


async def check_single_prediction_outcome(pred_id, pred):
    """Checks the outcome of a single past prediction."""
    try:
        # Skip if already checked
        if pred.get('outcome') is not None:
            logging.debug(f"Prediction {pred_id} already has outcome, skipping")
            return None

        # Fix the date issue
        pred_timestamp_str = fix_date_if_needed(pred['timestamp'])
        pred_date = datetime.fromisoformat(pred_timestamp_str)
        current_date = datetime.now()
        
        ticker = pred['ticker']
        
        # ============================================
        # TESTING MODE vs PRODUCTION MODE
        # ============================================
        
        if TEST_MODE:
            # 🧪 TEST MODE: Use simulated outcomes for immediate testing
            logging.info(f"🧪 TEST MODE: Simulating outcome for {ticker}")
            
            # Get the prediction price
            pred_price = pred.get('price_at_prediction', 100.0)
            
            # Simulate price movement based on action
            # Make it somewhat realistic but favor correct predictions for testing
            action = pred['action']
            
            if action == 'BUY':
                # 70% chance of being correct in test mode
                if random.random() < 0.7:
                    price_change_pct = random.uniform(1.5, 5.0)  # Positive change
                else:
                    price_change_pct = random.uniform(-3.0, -0.5)  # Negative change
            elif action == 'SELL':
                # 70% chance of being correct in test mode
                if random.random() < 0.7:
                    price_change_pct = random.uniform(-5.0, -1.5)  # Negative change
                else:
                    price_change_pct = random.uniform(0.5, 3.0)  # Positive change
            else:  # HOLD
                # 80% chance of being correct (small movement)
                if random.random() < 0.8:
                    price_change_pct = random.uniform(-1.5, 1.5)  # Small change
                else:
                    price_change_pct = random.uniform(2.5, 5.0)  # Larger change
            
            actual_price_after = pred_price * (1 + price_change_pct / 100)
            days_held = 2  # Simulate 2 days
            
        else:
            # 🔴 PRODUCTION MODE: Real checking with actual price data
            
            # Calculate days since prediction
            days_since = (current_date - pred_date).days
            
            # Need at least 2 trading days for real checking
            if days_since < 2:
                logging.info(f"⏳ {ticker}: Only {days_since} days since prediction. Need 2+ days.")
                return None
            
            stock = yf.Ticker(ticker)
            
            # Get historical data
            start_date = pred_date.date()
            end_date = min(pred_date.date() + timedelta(days=7), current_date.date())
            
            hist = await asyncio.to_thread(
                stock.history, 
                start=start_date, 
                end=end_date
            )
            
            if len(hist) < 2:
                logging.warning(f"Not enough history for {ticker}. Got {len(hist)} days of data.")
                return None
            
            pred_price = pred.get('price_at_prediction', hist['Close'].iloc[0])
            
            # Get actual price after 2+ days
            if len(hist) >= 3:
                actual_price_after = hist['Close'].iloc[2]
                days_held = 2
            else:
                actual_price_after = hist['Close'].iloc[-1]
                days_held = len(hist) - 1
            
            price_change_pct = ((actual_price_after - pred_price) / pred_price) * 100
        
        # ============================================
        # COMMON LOGIC FOR BOTH MODES
        # ============================================
        
        # Determine if the prediction was correct
        was_correct = False
        action = pred['action']
        
        # Success thresholds
        BULLISH_SUCCESS_THRESHOLD = 1.0  # Need >1% gain for BUY
        BEARISH_SUCCESS_THRESHOLD = -1.0  # Need >1% loss for SELL
        HOLD_THRESHOLD = 2.0  # Need <2% movement for HOLD
        
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
            'action_taken': action,
            'test_mode': TEST_MODE  # Track if this was a test
        }
        
        # Log result with emoji
        emoji = "✅" if was_correct else "❌"
        mode_indicator = "🧪 TEST" if TEST_MODE else "📊 REAL"
        logging.info(
            f"{emoji} {mode_indicator} {ticker}: "
            f"Predicted {action}, Price moved {price_change_pct:+.2f}% in {days_held} days. "
            f"Prediction was {'CORRECT' if was_correct else 'WRONG'}."
        )
        
        return pred_id, outcome_data

    except Exception as e:
        logging.error(f"Error checking outcome for {pred.get('ticker')}: {e}")
        return None


async def run_learning_process():
    """The main process to check outcomes and update learning memory."""
    
    if TEST_MODE:
        logging.info("🧪 TEST MODE ACTIVE - Using simulated outcomes!")
    else:
        logging.info("📊 PRODUCTION MODE - Using real market data")
    
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
    
    # In TEST MODE, check all predictions. In PRODUCTION, only old ones.
    if TEST_MODE:
        ready_to_check = list(unchecked_predictions.items())
        logging.info(f"🧪 TEST MODE: Checking all {len(ready_to_check)} predictions immediately")
    else:
        # Production mode: filter by age
        current_date = datetime.now()
        ready_to_check = []
        too_recent = []
        
        for pid, pred in unchecked_predictions.items():
            pred_date = datetime.fromisoformat(fix_date_if_needed(pred['timestamp']))
            days_old = (current_date - pred_date).days
            
            if days_old >= 2:
                ready_to_check.append((pid, pred))
            else:
                too_recent.append((pid, pred))
        
        logging.info(f"📊 Ready to check: {len(ready_to_check)} predictions")
        logging.info(f"⏳ Too recent: {len(too_recent)} predictions")
        
        if not ready_to_check:
            logging.info("No predictions old enough to check. Check back in a day or two!")
            return

    # Check outcomes
    tasks = [check_single_prediction_outcome(pid, p) for pid, p in ready_to_check]
    results = await asyncio.gather(*tasks)

    # Process results
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
        mode_text = "TEST MODE" if TEST_MODE else "REAL"
        insight = f"[{mode_text}] Checked {total_checked} predictions: {accuracy:.1f}% accuracy ({correct_count}/{total_checked} correct)"
        learning_memory.add_insight(insight)
        logging.info(f"📈 {insight}")
        
        # Pattern insights
        for pattern, stats in pattern_performance.items():
            if stats['total'] > 0:
                pattern_acc = (stats['correct'] / stats['total']) * 100
                pattern_insight = f"Pattern '{pattern}': {pattern_acc:.0f}% accuracy ({stats['correct']}/{stats['total']})"
                learning_memory.add_insight(pattern_insight)
                logging.info(f"   🕯️ {pattern_insight}")
        
        # Save learning memory
        learning_memory._save_memory()
    
    # Summary
    logging.info("✅ Evening learning process complete!")
    logging.info(f"   - Mode: {'TEST MODE 🧪' if TEST_MODE else 'PRODUCTION 📊'}")
    logging.info(f"   - Checked: {total_checked} predictions")
    logging.info(f"   - Accuracy: {accuracy:.1f}%" if total_checked > 0 else "   - No predictions checked")


async def main():
    """Entry point for the script."""
    await run_learning_process()

if __name__ == "__main__":
    asyncio.run(main())
