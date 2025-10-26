# /src/reports/weekly_dashboard.py
# Generates comprehensive weekly learning reports

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WeeklyDashboard:
    def __init__(self):
        self.predictions_file = 'data/predictions.json'
        self.patterns_file = 'data/patterns.json'
        self.learning_file = 'data/learning_memory.json'
        self.reports_dir = 'data/reports'
        
        # Create reports directory
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
        
      def generate_weekly_report(self, days_back=30):  # Changed from 7 to 30 for testing
        """Generate comprehensive weekly learning report"""
        logging.info("📊 Generating Weekly Learning Dashboard...")
        
        # Load all data
        predictions = self._load_json(self.predictions_file)
        patterns_data = self._load_json(self.patterns_file)
        learning = self._load_json(self.learning_file)
        
        if not predictions:
            logging.error("❌ No predictions found in predictions.json")
            return None, None
        
        # Get predictions from specified days back
        cutoff_date = datetime.now() - timedelta(days=days_back)
        weekly_predictions = self._filter_by_date(predictions, cutoff_date)
        
        if not weekly_predictions:
            logging.warning(f"⚠️ No predictions found in the last {days_back} days")
            logging.info(f"📊 Total predictions in file: {len(predictions)}")
            
            # Show sample dates
            if predictions:
                sample = list(predictions.values())[0]
                logging.info(f"Sample prediction date: {sample.get('timestamp', 'N/A')}")
            
            # For testing: Use ALL predictions if none in time window
            logging.info("📊 Using ALL predictions for report (testing mode)")
            weekly_predictions = predictions
        
        # Calculate all metrics
        overall_stats = self._calculate_overall_stats(weekly_predictions)
        
        # Check if we have any checked predictions
        if overall_stats['checked_predictions'] == 0:
            logging.warning("⚠️ No predictions have been checked yet (need to wait for outcomes)")
            logging.info("📊 Run the evening learner to check prediction outcomes")
            return None, None
        
        daily_breakdown = self._calculate_daily_breakdown(weekly_predictions)
        pattern_performance = self._calculate_pattern_performance(weekly_predictions)
        llm_performance = self._calculate_llm_performance(weekly_predictions)
        top_predictions = self._get_top_predictions(weekly_predictions)
        missed_calls = self._get_missed_calls(weekly_predictions)
        insights = self._generate_insights(overall_stats, pattern_performance, llm_performance)
        
        # Build report
        report = {
            'week_ending': datetime.now().strftime('%Y-%m-%d'),
            'days_analyzed': days_back,
            'overall_stats': overall_stats,
            'daily_breakdown': daily_breakdown,
            'pattern_performance': pattern_performance,
            'llm_performance': llm_performance,
            'top_predictions': top_predictions,
            'missed_calls': missed_calls,
            'insights': insights
        }
        
        # Save report
        self._save_report(report)
        
        # Generate email HTML
        email_html = self._generate_email_html(report)
        
        # Save email HTML
        email_file = f"{self.reports_dir}/weekly_email_{datetime.now().strftime('%Y%m%d')}.html"
        with open(email_file, 'w') as f:
            f.write(email_html)
        
        logging.info(f"✅ Weekly report generated: {email_file}")
        
        return report, email_html

    
    def _load_json(self, filepath):
        """Load JSON file"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading {filepath}: {e}")
                return {}
        return {}
    
    def _filter_by_date(self, predictions, start_date):
        """Get predictions from the last week"""
        weekly = {}
        for pred_id, pred in predictions.items():
            if 'timestamp' in pred:
                # Fix date issue (2025 -> 2024)
                timestamp_str = pred['timestamp'].replace('2025', '2024')
                try:
                    pred_date = datetime.fromisoformat(timestamp_str.split('T')[0] if 'T' in timestamp_str else timestamp_str)
                    if pred_date >= start_date.replace(hour=0, minute=0, second=0):
                        weekly[pred_id] = pred
                except Exception as e:
                    logging.debug(f"Error parsing date for {pred_id}: {e}")
        return weekly
    
    def _calculate_overall_stats(self, predictions):
        """Calculate overall performance stats"""
        total = len(predictions)
        checked = sum(1 for p in predictions.values() if p.get('outcome'))
        correct = sum(1 for p in predictions.values() if p.get('was_correct'))
        
        accuracy = (correct / checked * 100) if checked > 0 else 0
        
        # Calculate by action type
        buy_correct = sum(1 for p in predictions.values() 
                         if p.get('action') == 'BUY' and p.get('was_correct'))
        buy_total = sum(1 for p in predictions.values() 
                       if p.get('action') == 'BUY' and p.get('outcome'))
        
        sell_correct = sum(1 for p in predictions.values() 
                          if p.get('action') == 'SELL' and p.get('was_correct'))
        sell_total = sum(1 for p in predictions.values() 
                        if p.get('action') == 'SELL' and p.get('outcome'))
        
        hold_correct = sum(1 for p in predictions.values() 
                          if p.get('action') == 'HOLD' and p.get('was_correct'))
        hold_total = sum(1 for p in predictions.values() 
                        if p.get('action') == 'HOLD' and p.get('outcome'))
        
        return {
            'total_predictions': total,
            'checked_predictions': checked,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'pending_predictions': total - checked,
            'buy_accuracy': (buy_correct / buy_total * 100) if buy_total > 0 else 0,
            'sell_accuracy': (sell_correct / sell_total * 100) if sell_total > 0 else 0,
            'hold_accuracy': (hold_correct / hold_total * 100) if hold_total > 0 else 0,
            'buy_count': buy_total,
            'sell_count': sell_total,
            'hold_count': hold_total
        }
    
    def _calculate_daily_breakdown(self, predictions):
        """Calculate accuracy by day of week"""
        daily_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred in predictions.values():
            if pred.get('outcome'):
                timestamp_str = pred['timestamp'].replace('2025', '2024')
                try:
                    pred_date = datetime.fromisoformat(timestamp_str.split('T')[0] if 'T' in timestamp_str else timestamp_str)
                    day_name = pred_date.strftime('%A')
                    
                    daily_stats[day_name]['total'] += 1
                    if pred.get('was_correct'):
                        daily_stats[day_name]['correct'] += 1
                except:
                    pass
        
        # Calculate accuracy for each day
        results = {}
        for day, stats in daily_stats.items():
            if stats['total'] > 0:
                results[day] = {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': (stats['correct'] / stats['total']) * 100
                }
        
        # Sort by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sorted_results = {day: results[day] for day in day_order if day in results}
        
        return sorted_results
    
    def _calculate_pattern_performance(self, predictions):
        """Calculate performance by pattern"""
        pattern_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred in predictions.values():
            if pred.get('outcome') and pred.get('candle_pattern'):
                pattern = pred['candle_pattern']
                pattern_stats[pattern]['total'] += 1
                if pred.get('was_correct'):
                    pattern_stats[pattern]['correct'] += 1
        
        # Calculate accuracy for each pattern
        results = {}
        for pattern, stats in pattern_stats.items():
            if stats['total'] > 0:
                results[pattern] = {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': (stats['correct'] / stats['total']) * 100
                }
        
        # Sort by accuracy
        sorted_results = dict(sorted(results.items(), 
                                    key=lambda x: x[1]['accuracy'], 
                                    reverse=True))
        
        return sorted_results
    
    def _calculate_llm_performance(self, predictions):
        """Calculate performance by LLM (if tracked)"""
        # This is a placeholder - you'll need to track which LLM made which prediction
        # For now, return structure for future implementation
        return {
            'groq': {'total': 0, 'correct': 0, 'accuracy': 0},
            'gemini': {'total': 0, 'correct': 0, 'accuracy': 0},
            'cohere': {'total': 0, 'correct': 0, 'accuracy': 0}
        }
    
    def _get_top_predictions(self, predictions):
        """Get best performing predictions"""
        successful = []
        
        for pred_id, pred in predictions.items():
            if pred.get('outcome') and pred.get('was_correct'):
                outcome = pred['outcome']
                price_change = outcome.get('price_change_pct', 0)
                
                successful.append({
                    'ticker': pred['ticker'],
                    'action': pred['action'],
                    'date': pred['timestamp'].split('T')[0],
                    'price_change': price_change,
                    'confidence': pred.get('confidence', 0)
                })
        
        # Sort by absolute price change
        successful.sort(key=lambda x: abs(x['price_change']), reverse=True)
        
        return successful[:5]  # Top 5
    
    def _get_missed_calls(self, predictions):
        """Get predictions that were wrong"""
        missed = []
        
        for pred_id, pred in predictions.items():
            if pred.get('outcome') and not pred.get('was_correct'):
                outcome = pred['outcome']
                price_change = outcome.get('price_change_pct', 0)
                
                missed.append({
                    'ticker': pred['ticker'],
                    'action': pred['action'],
                    'date': pred['timestamp'].split('T')[0],
                    'price_change': price_change,
                    'confidence': pred.get('confidence', 0)
                })
        
        # Sort by how wrong (biggest miss)
        missed.sort(key=lambda x: abs(x['price_change']), reverse=True)
        
        return missed[:5]  # Top 5 misses
    
    def _generate_insights(self, overall_stats, pattern_performance, llm_performance):
        """Generate actionable insights"""
        insights = []
        
        # Overall performance insight
        accuracy = overall_stats['accuracy']
        if accuracy >= 70:
            insights.append(f"🎯 Excellent week! {accuracy:.1f}% accuracy exceeds our 65% target")
        elif accuracy >= 60:
            insights.append(f"✅ Good performance at {accuracy:.1f}% accuracy")
        elif accuracy >= 50:
            insights.append(f"⚠️ Below target at {accuracy:.1f}% - review strategy")
        else:
            insights.append(f"❌ Poor week at {accuracy:.1f}% - significant adjustments needed")
        
        # Best pattern insight
        if pattern_performance:
            best_pattern = list(pattern_performance.items())[0]
            if best_pattern[1]['total'] >= 3 and best_pattern[1]['accuracy'] >= 80:
                insights.append(f"🌟 {best_pattern[0].replace('_', ' ').title()} is highly reliable at {best_pattern[1]['accuracy']:.0f}% ({best_pattern[1]['correct']}/{best_pattern[1]['total']})")
        
        # Worst pattern warning
        if pattern_performance:
            worst_pattern = list(pattern_performance.items())[-1]
            if worst_pattern[1]['total'] >= 3 and worst_pattern[1]['accuracy'] < 50:
                insights.append(f"⚠️ {worst_pattern[0].replace('_', ' ').title()} underperforming at {worst_pattern[1]['accuracy']:.0f}% - consider reducing weight")
        
        # Action type insights
        if overall_stats['buy_count'] >= 3:
            insights.append(f"📈 BUY signals: {overall_stats['buy_accuracy']:.0f}% accurate ({overall_stats['buy_count']} calls)")
        if overall_stats['sell_count'] >= 3:
            insights.append(f"📉 SELL signals: {overall_stats['sell_accuracy']:.0f}% accurate ({overall_stats['sell_count']} calls)")
        if overall_stats['hold_count'] >= 3:
            insights.append(f"⏸️ HOLD signals: {overall_stats['hold_accuracy']:.0f}% accurate ({overall_stats['hold_count']} calls)")
        
        # Recommendations
        if overall_stats['accuracy'] < 65:
            insights.append("💡 Recommendation: Focus on high-confidence signals (>75%) only")
        
        return insights
    
    def _generate_email_html(self, report):
        """Generate HTML email for weekly report"""
        
        overall = report['overall_stats']
        patterns = report['pattern_performance']
        daily = report['daily_breakdown']
        top_preds = report['top_predictions']
        missed = report['missed_calls']
        insights = report['insights']
        
        # Determine emoji for overall performance
        if overall['accuracy'] >= 70:
            perf_emoji = "🎉"
            perf_color = "#10b981"
        elif overall['accuracy'] >= 60:
            perf_emoji = "✅"
            perf_color = "#3b82f6"
        else:
            perf_emoji = "⚠️"
            perf_color = "#f59e0b"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 3px solid {perf_color}; padding-bottom: 20px; margin-bottom: 30px; }}
        .header h1 {{ margin: 0; font-size: 28px; color: #1f2937; }}
        .header p {{ margin: 5px 0; color: #6b7280; }}
        .section {{ margin: 30px 0; }}
        .section-title {{ font-size: 20px; font-weight: bold; color: #1f2937; border-left: 4px solid {perf_color}; padding-left: 12px; margin-bottom: 15px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f9fafb; padding: 15px; border-radius: 8px; border-left: 4px solid {perf_color}; }}
        .stat-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #1f2937; margin: 5px 0; }}
        .pattern-row {{ padding: 10px; border-bottom: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; }}
        .pattern-row:last-child {{ border-bottom: none; }}
        .pattern-name {{ font-weight: 500; }}
        .accuracy-badge {{ padding: 4px 12px; border-radius: 12px; font-size: 14px; font-weight: bold; }}
        .accuracy-high {{ background: #d1fae5; color: #065f46; }}
        .accuracy-medium {{ background: #dbeafe; color: #1e40af; }}
        .accuracy-low {{ background: #fee2e2; color: #991b1b; }}
        .insight-box {{ background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        .prediction-row {{ padding: 12px; background: #f9fafb; margin: 8px 0; border-radius: 6px; display: flex; justify-content: space-between; }}
        .ticker {{ font-weight: bold; font-size: 16px; }}
        .change-positive {{ color: #10b981; font-weight: bold; }}
        .change-negative {{ color: #ef4444; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 2px solid #e5e7eb; color: #6b7280; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{perf_emoji} Weekly Learning Report</h1>
            <p>Week Ending: {report['week_ending']}</p>
        </div>

        <!-- OVERALL PERFORMANCE -->
        <div class="section">
            <div class="section-title">🎯 Overall Performance</div>
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-label">Accuracy</div>
                    <div class="stat-value" style="color: {perf_color};">{overall['accuracy']:.1f}%</div>
                    <div class="stat-label">{overall['correct_predictions']}/{overall['checked_predictions']} Correct</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Predictions</div>
                    <div class="stat-value">{overall['total_predictions']}</div>
                    <div class="stat-label">{overall['pending_predictions']} Pending Review</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">BUY Accuracy</div>
                    <div class="stat-value">{overall['buy_accuracy']:.0f}%</div>
                    <div class="stat-label">{overall['buy_count']} Calls</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">SELL Accuracy</div>
                    <div class="stat-value">{overall['sell_accuracy']:.0f}%</div>
                    <div class="stat-label">{overall['sell_count']} Calls</div>
                </div>
            </div>
        </div>

        <!-- DAILY BREAKDOWN -->
        <div class="section">
            <div class="section-title">📅 Daily Performance</div>
"""
        
        for day, stats in daily.items():
            acc_class = 'accuracy-high' if stats['accuracy'] >= 70 else 'accuracy-medium' if stats['accuracy'] >= 55 else 'accuracy-low'
            html += f"""
            <div class="pattern-row">
                <span class="pattern-name">{day}</span>
                <span class="accuracy-badge {acc_class}">{stats['accuracy']:.0f}% ({stats['correct']}/{stats['total']})</span>
            </div>
"""
        
        # PATTERN PERFORMANCE
        html += """
        </div>

        <!-- PATTERN PERFORMANCE -->
        <div class="section">
            <div class="section-title">🕯️ Pattern Performance</div>
"""
        
        for pattern, stats in list(patterns.items())[:10]:  # Top 10
            acc_class = 'accuracy-high' if stats['accuracy'] >= 70 else 'accuracy-medium' if stats['accuracy'] >= 55 else 'accuracy-low'
            pattern_name = pattern.replace('_', ' ').title()
            html += f"""
            <div class="pattern-row">
                <span class="pattern-name">{pattern_name}</span>
                <span class="accuracy-badge {acc_class}">{stats['accuracy']:.0f}% ({stats['correct']}/{stats['total']})</span>
            </div>
"""
        
        # TOP PREDICTIONS
        html += """
        </div>

        <!-- TOP PREDICTIONS -->
        <div class="section">
            <div class="section-title">🏆 Top Predictions This Week</div>
"""
        
        for pred in top_preds:
            change_class = 'change-positive' if pred['price_change'] > 0 else 'change-negative'
            html += f"""
            <div class="prediction-row">
                <div>
                    <span class="ticker">{pred['ticker']}</span> - {pred['action']}
                    <br><small style="color: #6b7280;">{pred['date']}</small>
                </div>
                <div style="text-align: right;">
                    <span class="{change_class}">{pred['price_change']:+.1f}%</span>
                    <br><small style="color: #6b7280;">{pred['confidence']}% confidence</small>
                </div>
            </div>
"""
        
        # MISSED CALLS
        html += """
        </div>

        <!-- MISSED CALLS -->
        <div class="section">
            <div class="section-title">❌ Missed Calls</div>
"""
        
        for pred in missed:
            html += f"""
            <div class="prediction-row">
                <div>
                    <span class="ticker">{pred['ticker']}</span> - {pred['action']} (Wrong)
                    <br><small style="color: #6b7280;">{pred['date']}</small>
                </div>
                <div style="text-align: right;">
                    <span>Actually: {pred['price_change']:+.1f}%</span>
                    <br><small style="color: #6b7280;">{pred['confidence']}% confidence</small>
                </div>
            </div>
"""
        
        # KEY INSIGHTS
        html += """
        </div>

        <!-- KEY INSIGHTS -->
        <div class="section">
            <div class="section-title">💡 Key Insights & Learnings</div>
"""
        
        for insight in insights:
            html += f"""
            <div class="insight-box">
                {insight}
            </div>
"""
        
        html += """
        </div>

        <div class="footer">
            <p>🤖 Generated by AI Stock Analysis System</p>
            <p>Keep learning, keep improving! 🚀</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _save_report(self, report):
        """Save report to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{self.reports_dir}/weekly_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logging.info(f"✅ Report saved to {filename}")


def main():
    """Main entry point"""
    dashboard = WeeklyDashboard()
    
    # Generate report (looking at last 30 days for testing)
    result = dashboard.generate_weekly_report(days_back=30)
    
    # Handle None return
    if result is None or result == (None, None):
        print("\n" + "="*80)
        print("⚠️ CANNOT GENERATE REPORT")
        print("="*80)
        print("Reasons:")
        print("1. No predictions found in data/predictions.json, OR")
        print("2. No predictions have outcomes yet (need to run evening learner)")
        print("\n💡 Solutions:")
        print("1. Run morning analysis to create predictions")
        print("2. Wait 2+ days, then run evening learner to check outcomes")
        print("3. Set TEST_MODE=True in check_outcomes.py to simulate outcomes")
        print("="*80 + "\n")
        return
    
    report, email_html = result
    
    if report:
        print("\n" + "="*80)
        print("📊 WEEKLY REPORT GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"Period: Last {report.get('days_analyzed', 7)} days")
        print(f"Accuracy: {report['overall_stats']['accuracy']:.1f}%")
        print(f"Total Predictions: {report['overall_stats']['total_predictions']}")
        print(f"Checked: {report['overall_stats']['checked_predictions']}")
        print(f"Correct: {report['overall_stats']['correct_predictions']}")
        print("="*80)
        
        # Show top insights
        if report['insights']:
            print("\n💡 KEY INSIGHTS:")
            for insight in report['insights'][:3]:
                print(f"  • {insight}")
        
        print("\n" + "="*80)
        print(f"📧 Email HTML: data/reports/weekly_email_{datetime.now().strftime('%Y%m%d')}.html")
        print("="*80 + "\n")
    else:
        print("❌ Report generation failed")
