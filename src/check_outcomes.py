# /src/check_outcomes.py
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================
# 🔴 TESTING MODE
# ============================================
TEST_MODE = False  # Set to False for production

# --- Import and Initialize Components ---
try:
    # This imports the classes and the global instances
    from main import (
        PredictionTracker, CandlePatternAnalyzer, LearningMemory, yf,
        prediction_tracker, candle_analyzer, learning_memory
    )
    # This ensures that even if the script runs standalone, the objects exist
    if 'prediction_tracker' not in globals():
        prediction_tracker = PredictionTracker()
    if 'candle_analyzer' not in globals():
        candle_analyzer = CandlePatternAnalyzer()
    if 'learning_memory' not in globals():
        learning_memory = LearningMemory()
    
    logging.info("✅ Successfully imported and initialized components from main.py")
except ImportError as e:
    logging.critical(f"Failed to import components from main.py: {e}")
    sys.exit(1)
    logging.critical(f"Failed to import components from main.py: {e}")
    sys.exit(1)


def fix_date_if_needed(date_str):
    """Fix the year if it's showing 2025 instead of 2024"""
    if '2025' in date_str:
        return date_str.replace('2025', '2024')
    return date_str


class ContextualPatternLearner:
    """
    Learns pattern + indicator combinations instead of just patterns alone.
    Example: "double_bottom + RSI<30 + volume>1.5x" instead of just "double_bottom"
    """
    
    def __init__(self):
        self.combinations_file = Path('data/pattern_combinations.json')
        self.combinations = self._load_combinations()
    
    def _load_combinations(self):
        if self.combinations_file.exists():
            try:
                with open(self.combinations_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_combinations(self):
        self.combinations_file.parent.mkdir(exist_ok=True)
        with open(self.combinations_file, 'w') as f:
            json.dump(self.combinations, f, indent=2)
    
    def create_context_key(self, pattern, indicators):
        """
        Create a contextual key for pattern + indicators.
        Example: "double_bottom_rsi_oversold_volume_high"
        """
        context_parts = [pattern]
        
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            context_parts.append('rsi_oversold')
        elif rsi > 70:
            context_parts.append('rsi_overbought')
        elif 40 <= rsi <= 60:
            context_parts.append('rsi_neutral')
        
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            context_parts.append('volume_high')
        elif volume_ratio < 0.8:
            context_parts.append('volume_low')
        
        macd = indicators.get('macd', 0)
        if macd > 0:
            context_parts.append('macd_positive')
        elif macd < 0:
            context_parts.append('macd_negative')
        
        return '_'.join(context_parts)
    
    def update_combination(self, pattern, indicators, was_successful):
        """Update success rate for this specific combination"""
        combo_key = self.create_context_key(pattern, indicators)
        
        if combo_key not in self.combinations:
            self.combinations[combo_key] = {
                'total': 0,
                'successful': 0,
                'pattern': pattern,
                'context': self._describe_context(indicators)
            }
        
        self.combinations[combo_key]['total'] += 1
        if was_successful:
            self.combinations[combo_key]['successful'] += 1
        
        self._save_combinations()
        
        # Log insight
        success_rate = (self.combinations[combo_key]['successful'] / 
                       self.combinations[combo_key]['total']) * 100
        
        logging.info(f"   🧠 Learning: {combo_key} now has {success_rate:.0f}% success "
                    f"({self.combinations[combo_key]['successful']}/{self.combinations[combo_key]['total']})")
    
    def _describe_context(self, indicators):
        """Human-readable context description"""
        parts = []
        
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            parts.append('RSI oversold (<30)')
        elif rsi > 70:
            parts.append('RSI overbought (>70)')
        
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            parts.append(f'High volume ({volume_ratio:.1f}x)')
        elif volume_ratio < 0.8:
            parts.append(f'Low volume ({volume_ratio:.1f}x)')
        
        return ', '.join(parts) if parts else 'Normal conditions'
    
    def get_best_combinations(self, min_samples=3):
        """Get best performing combinations"""
        results = []
        for combo_key, stats in self.combinations.items():
            if stats['total'] >= min_samples:
                success_rate = (stats['successful'] / stats['total']) * 100
                results.append({
                    'combination': combo_key,
                    'pattern': stats['pattern'],
                    'context': stats.get('context', ''),
                    'success_rate': success_rate,
                    'sample_size': stats['total']
                })
        
        return sorted(results, key=lambda x: x['success_rate'], reverse=True)


# Initialize contextual learner
contextual_learner = ContextualPatternLearner()


async def check_single_prediction_outcome(pred_id, pred):
    """Checks the outcome of a single past prediction."""
    try:
        if pred.get('outcome') is not None:
            logging.debug(f"Prediction {pred_id} already has outcome, skipping")
            return None

        pred_timestamp_str = fix_date_if_needed(pred['timestamp'])
        pred_date = datetime.fromisoformat(pred_timestamp_str)
        current_date = datetime.now()
        ticker = pred['ticker']
        
        if TEST_MODE:
            logging.info(f"🧪 TEST MODE: Simulating outcome for {ticker}")
            pred_price = pred.get('price_at_prediction', 100.0)
            action = pred['action']
            
            if action == 'BUY':
                price_change_pct = random.uniform(1.5, 5.0) if random.random() < 0.7 else random.uniform(-3.0, -0.5)
            elif action == 'SELL':
                price_change_pct = random.uniform(-5.0, -1.5) if random.random() < 0.7 else random.uniform(0.5, 3.0)
            else:
                price_change_pct = random.uniform(-1.5, 1.5) if random.random() < 0.8 else random.uniform(2.5, 5.0)
            
            actual_price_after = pred_price * (1 + price_change_pct / 100)
            days_held = 2
            
        else:
            days_since = (current_date - pred_date).days
            
            if days_since < 2:
                logging.info(f"⏳ {ticker}: Only {days_since} days since prediction. Need 2+ days.")
                return None
            
            stock = yf.Ticker(ticker)
            start_date = pred_date.date()
            end_date = min(pred_date.date() + timedelta(days=7), current_date.date())
            
            hist = await asyncio.to_thread(stock.history, start=start_date, end=end_date)
            
            if len(hist) < 2:
                logging.warning(f"Not enough history for {ticker}. Got {len(hist)} days of data.")
                return None
            
            pred_price = pred.get('price_at_prediction', hist['Close'].iloc[0])
            
            if len(hist) >= 3:
                actual_price_after = hist['Close'].iloc[2]
                days_held = 2
            else:
                actual_price_after = hist['Close'].iloc[-1]
                days_held = len(hist) - 1
            
            price_change_pct = ((actual_price_after - pred_price) / pred_price) * 100
        
        # Determine correctness
        was_correct = False
        action = pred['action']
        
        if action == 'BUY' and price_change_pct > 1.0:
            was_correct = True
        elif action == 'SELL' and price_change_pct < -1.0:
            was_correct = True
        elif action == 'HOLD' and abs(price_change_pct) <= 2.0:
            was_correct = True
        
        # Analyze WHY it failed/succeeded
        failure_analysis = ""
        if not was_correct:
            failure_analysis = _analyze_failure(pred, price_change_pct)
        else:
            failure_analysis = _analyze_success(pred, price_change_pct)
        
        outcome_data = {
            'checked_date': datetime.now().isoformat(),
            'price_at_prediction': float(pred_price),
            'price_after': float(actual_price_after),
            'days_held': days_held,
            'price_change_pct': round(price_change_pct, 2),
            'was_correct': was_correct,
            'action_taken': action,
            'failure_analysis': failure_analysis,
            'test_mode': TEST_MODE
        }
        
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


def _analyze_failure(pred, actual_return):
    """Deep analysis of why a prediction failed"""
    reasons = []
    action = pred['action']
    indicators = pred.get('indicators', {})
    pattern = pred.get('candle_pattern', 'none')
    
    rsi = indicators.get('rsi', 50)
    volume_ratio = indicators.get('volume_ratio', 1.0)
    
    if action == 'BUY' and actual_return < -1:
        if rsi > 70:
            reasons.append(f"RSI was overbought ({rsi:.0f}) - reversal risk ignored")
        if volume_ratio < 0.8:
            reasons.append(f"Low volume ({volume_ratio:.1f}x) - weak buyer conviction")
        if pattern == 'double_bottom':
            reasons.append("Double bottom failed - may need RSI oversold + high volume confirmation")
        if pattern == 'morning_star':
            reasons.append("Morning star failed - context matters (check broader market trend)")
    
    elif action == 'SELL' and actual_return > 1:
        if rsi < 30:
            reasons.append(f"RSI was oversold ({rsi:.0f}) - bounce probability ignored")
        reasons.append("Underestimated bullish momentum")
    
    elif action == 'HOLD' and abs(actual_return) > 2:
        reasons.append(f"Significant movement ({actual_return:+.2f}%) - missed opportunity")
    
    return " | ".join(reasons) if reasons else "Market moved against prediction - no clear technical reason"


def _analyze_success(pred, actual_return):
    """Analyze why a prediction succeeded"""
    reasons = []
    action = pred['action']
    indicators = pred.get('indicators', {})
    pattern = pred.get('candle_pattern', 'none')
    
    if action == 'BUY' and actual_return > 1:
        reasons.append(f"Bullish setup confirmed ({actual_return:+.2f}%)")
        if pattern:
            reasons.append(f"{pattern} pattern worked as expected")
    elif action == 'SELL' and actual_return < -1:
        reasons.append(f"Bearish setup confirmed ({actual_return:+.2f}%)")
        if pattern:
            reasons.append(f"{pattern} pattern worked as expected")
    elif action == 'HOLD':
        reasons.append(f"Correctly predicted low volatility ({actual_return:+.2f}%)")
    
    return " | ".join(reasons) if reasons else "Prediction aligned with market movement"


async def run_learning_process():
    """Main process to check outcomes and update learning memory."""
    
    if TEST_MODE:
        logging.info("🧪 TEST MODE ACTIVE - Using simulated outcomes!")
    else:
        logging.info("📊 PRODUCTION MODE - Using real market data")
    
    logging.info("🧠 Starting evening learning process...")
    logging.info(f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Fix dates
    for pred_id, pred in prediction_tracker.predictions.items():
        if '2025' in pred['timestamp']:
            pred['timestamp'] = pred['timestamp'].replace('2025', '2024')
    
    unchecked_predictions = {
        pid: p for pid, p in prediction_tracker.predictions.items() 
        if p.get('outcome') is None
    }
    
    if not unchecked_predictions:
        logging.info("No unchecked predictions found.")
        # Still send summary email
        await send_evening_summary_email([], {})
        return

    logging.info(f"Found {len(unchecked_predictions)} unchecked predictions.")
    
    if TEST_MODE:
        ready_to_check = list(unchecked_predictions.items())
        logging.info(f"🧪 TEST MODE: Checking all {len(ready_to_check)} predictions immediately")
    else:
        current_date = datetime.now()
        ready_to_check = []
        
        for pid, pred in unchecked_predictions.items():
            pred_date = datetime.fromisoformat(fix_date_if_needed(pred['timestamp']))
            days_old = (current_date - pred_date).days
            
            if days_old >= 2:
                ready_to_check.append((pid, pred))
        
        logging.info(f"📊 Ready to check: {len(ready_to_check)} predictions")
        
        if not ready_to_check:
            logging.info("No predictions old enough to check.")
            return

    # Check outcomes
    tasks = [check_single_prediction_outcome(pid, p) for pid, p in ready_to_check]
    results = await asyncio.gather(*tasks)

    # Process results
    correct_count = 0
    total_checked = 0
    pattern_performance = {}
    checked_predictions = []
    
    for result in results:
        if result is None:
            continue
            
        pred_id, outcome_data = result
        total_checked += 1
        
        # Update prediction
        prediction_tracker.predictions[pred_id]['outcome'] = outcome_data
        prediction_tracker.predictions[pred_id]['was_correct'] = outcome_data['was_correct']
        
        original_pred = prediction_tracker.predictions[pred_id]
        was_correct = outcome_data['was_correct']
        
        if was_correct:
            correct_count += 1
        
        # Track for email
        checked_predictions.append({
            'ticker': original_pred['ticker'],
            'action': original_pred['action'],
            'confidence': original_pred['confidence'],
            'outcome': outcome_data,
            'pattern': original_pred.get('candle_pattern'),
            'analysis': outcome_data.get('failure_analysis', '')
        })
        
        # Pattern performance
        if original_pred.get('candle_pattern'):
            pattern = original_pred['candle_pattern']
            if pattern not in pattern_performance:
                pattern_performance[pattern] = {'correct': 0, 'total': 0}
            pattern_performance[pattern]['total'] += 1
            if was_correct:
                pattern_performance[pattern]['correct'] += 1
            
            # Update basic pattern analyzer
            candle_analyzer.update_pattern_outcome(pattern, original_pred['ticker'], was_correct)
            
            # 🆕 Update contextual pattern learner
            indicators = original_pred.get('indicators', {})
            contextual_learner.update_combination(pattern, indicators, was_correct)
    
    # Save
    prediction_tracker._save_predictions()
    candle_analyzer._save_patterns()
    
    # Generate insights
    if total_checked > 0:
        accuracy = (correct_count / total_checked) * 100
        mode_text = "TEST MODE" if TEST_MODE else "REAL"
        insight = f"[{mode_text}] Checked {total_checked} predictions: {accuracy:.1f}% accuracy ({correct_count}/{total_checked} correct)"
        learning_memory.add_insight(insight)
        logging.info(f"📈 {insight}")
        
        for pattern, stats in pattern_performance.items():
            if stats['total'] > 0:
                pattern_acc = (stats['correct'] / stats['total']) * 100
                pattern_insight = f"Pattern '{pattern}': {pattern_acc:.0f}% accuracy ({stats['correct']}/{stats['total']})"
                learning_memory.add_insight(pattern_insight)
                logging.info(f"   🕯️ {pattern_insight}")
        
        learning_memory._save_memory()
    
    # 📧 Send evening summary email
    await send_evening_summary_email(checked_predictions, pattern_performance)
    
    logging.info("✅ Evening learning process complete!")
    logging.info(f"   - Mode: {'TEST MODE 🧪' if TEST_MODE else 'PRODUCTION 📊'}")
    logging.info(f"   - Checked: {total_checked} predictions")
    logging.info(f"   - Accuracy: {accuracy:.1f}%" if total_checked > 0 else "   - No predictions checked")


async def send_evening_summary_email(checked_predictions, pattern_performance):
    """Send visual evening summary email"""
    
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    
    if not smtp_user or not smtp_pass:
        logging.warning("SMTP credentials missing. Skipping email.")
        return
    
    html = generate_evening_html(checked_predictions, pattern_performance)
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🧠 Evening Learning Report - {datetime.now().strftime('%Y-%m-%d')}"
    msg['From'] = smtp_user
    msg['To'] = smtp_user
    
    msg.attach(MIMEText(html, 'html'))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logging.info("✅ Evening summary email sent!")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


def generate_evening_html(checked_predictions, pattern_performance):
    """Generate beautiful HTML email for evening summary"""
    
    if not checked_predictions:
        return f"""<!DOCTYPE html><html><body style="font-family:sans-serif;max-width:600px;margin:auto;padding:20px;">
        <h1>🧠 Evening Learning Report</h1>
        <p>No predictions were checked today. Check back tomorrow!</p>
        </body></html>"""
    
    total = len(checked_predictions)
    correct = sum(1 for p in checked_predictions if p['outcome']['was_correct'])
    accuracy = (correct / total) * 100
    
    # Color based on performance
    if accuracy >= 70:
        perf_color = '#16a34a'
        perf_emoji = '🎯'
    elif accuracy >= 50:
        perf_color = '#f59e0b'
        perf_emoji = '⚠️'
    else:
        perf_color = '#dc2626'
        perf_emoji = '❌'
    
    # Build prediction cards
    pred_cards = ""
    for pred in checked_predictions:
        outcome = pred['outcome']
        was_correct = outcome['was_correct']
        
        card_color = '#d1fae5' if was_correct else '#fee2e2'
        border_color = '#16a34a' if was_correct else '#dc2626'
        emoji = '✅' if was_correct else '❌'
        
        pred_cards += f"""
        <div style="border-left:4px solid {border_color};background:{card_color};padding:15px;margin:15px 0;border-radius:8px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <h3 style="margin:0;">{emoji} {pred['ticker']} - {pred['action']}</h3>
                    <p style="margin:5px 0;color:#666;">Pattern: {pred['pattern'] or 'None'} | Confidence: {pred['confidence']}%</p>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:24px;font-weight:bold;color:{border_color};">
                        {outcome['price_change_pct']:+.2f}%
                    </div>
                    <div style="font-size:12px;color:#666;">{outcome['days_held']} days</div>
                </div>
            </div>
            <div style="margin-top:10px;padding:10px;background:white;border-radius:5px;font-size:14px;">
                <strong>Analysis:</strong> {pred['analysis']}
            </div>
        </div>
        """
    
    # Pattern performance table
    pattern_rows = ""
    for pattern, stats in sorted(pattern_performance.items(), key=lambda x: x[1]['correct']/x[1]['total'], reverse=True):
        rate = (stats['correct'] / stats['total']) * 100
        color = '#16a34a' if rate >= 60 else '#dc2626' if rate < 40 else '#f59e0b'
        
        # Progress bar
        bar_width = rate
        
        pattern_rows += f"""
        <tr>
            <td style="padding:10px;border-bottom:1px solid #eee;">{pattern.replace('_', ' ').title()}</td>
            <td style="padding:10px;border-bottom:1px solid #eee;">
                <div style="background:#eee;border-radius:10px;overflow:hidden;">
                    <div style="width:{bar_width}%;background:{color};height:20px;display:flex;align-items:center;justify-content:center;color:white;font-size:12px;font-weight:bold;">
                        {rate:.0f}%
                    </div>
                </div>
            </td>
            <td style="padding:10px;border-bottom:1px solid #eee;text-align:center;">{stats['correct']}/{stats['total']}</td>
        </tr>
        """
    
    # 🆕 Best pattern combinations
    best_combos = contextual_learner.get_best_combinations(min_samples=2)
    combo_section = ""
    
    if best_combos:
        combo_rows = ""
        for combo in best_combos[:5]:  # Top 5
            rate = combo['success_rate']
            color = '#16a34a' if rate >= 60 else '#f59e0b'
            
            combo_rows += f"""
            <tr>
                <td style="padding:10px;border-bottom:1px solid #eee;">
                    <strong>{combo['pattern'].replace('_', ' ').title()}</strong><br>
                    <span style="font-size:12px;color:#666;">{combo['context']}</span>
                </td>
                <td style="padding:10px;border-bottom:1px solid #eee;text-align:center;font-weight:bold;color:{color};">
                    {rate:.0f}%
                </td>
                <td style="padding:10px;border-bottom:1px solid #eee;text-align:center;">
                    {combo['sample_size']}
                </td>
            </tr>
            """
        
        combo_section = f"""
        <div style="margin-top:30px;padding:20px;background:#f9fafb;border-radius:10px;">
            <h2 style="margin-top:0;">🧠 Best Pattern + Indicator Combinations</h2>
            <p style="color:#666;">These specific combinations have the highest success rates:</p>
            <table style="width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;">
                <thead>
                    <tr style="background:#f3f4f6;">
                        <th style="padding:12px;text-align:left;">Combination</th>
                        <th style="padding:12px;text-align:center;">Success Rate</th>
                        <th style="padding:12px;text-align:center;">Samples</th>
                    </tr>
                </thead>
                <tbody>
                    {combo_rows}
                </tbody>
            </table>
        </div>
        """
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 0; background: #f7f7f7; }}
        .container {{ max-width: 700px; margin: 20px auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
        .content {{ padding: 30px; }}
        h1 {{ margin: 0; font-size: 28px; }}
        h2 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Evening Learning Report</h1>
            <p style="font-size:18px;margin:10px 0 0 0;">{datetime.now().strftime('%A, %B %d, %Y')}</p>
        </div>
        
        <div class="content">
            <div style="text-align:center;padding:20px;background:linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);border-radius:10px;margin-bottom:30px;">
                <div style="font-size:48px;font-weight:bold;color:{perf_color};">
                    {perf_emoji} {accuracy:.1f}%
                </div>
                <div style="font-size:18px;color:#666;margin-top:10px;">
                    Today's Prediction Accuracy
                </div>
                <div style="font-size:14px;color:#888;margin-top:5px;">
                    {correct} correct out of {total} predictions
                </div>
            </div>
            
            <h2>📊 Prediction Results</h2>
            {pred_cards}
            
            <h2 style="margin-top:40px;">🕯️ Pattern Performance</h2>
            <table style="width:100%;border-collapse:collapse;margin-top:20px;">
                <thead>
                    <tr style="background:#f3f4f6;">
                        <th style="padding:12px;text-align:left;">Pattern</th>
                        <th style="padding:12px;text-align:left;">Success Rate</th>
                        <th style="padding:12px;text-align:center;">Results</th>
                    </tr>
                </thead>
                <tbody>
                    {pattern_rows}
                </tbody>
            </table>
            
            {combo_section}
            
            <div style="margin-top:40px;padding:20px;background:#f0f9ff;border-left:4px solid #0369a1;border-radius:8px;">
                <h3 style="margin:0 0 10px 0;">💡 Key Learnings</h3>
                <ul style="margin:0;padding-left:20px;">
                    {''.join([f'<li style="margin:8px 0;">{insight["insight"]}</li>' for insight in learning_memory.get_recent_insights(5)])}
                </ul>
            </div>
            
            <div style="text-align:center;margin-top:30px;padding:20px;color:#888;font-size:12px;">
                <p>This system learns from every prediction to improve future accuracy.</p>
                <p>Next analysis: Tomorrow morning</p>
            </div>
        </div>
    </div>
</body>
</html>"""


async def main():
    """Entry point for the script."""
    await run_learning_process()


if __name__ == "__main__":
    asyncio.run(main())
