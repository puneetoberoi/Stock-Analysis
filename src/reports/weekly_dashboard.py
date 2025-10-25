# /src/reports/weekly_dashboard.py
# Generates a weekly learning report showing system performance and insights

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeeklyDashboard:
    def __init__(self):
        self.predictions_file = 'data/predictions.json'
        self.patterns_file = 'data/patterns.json'
        self.learning_file = 'data/learning_memory.json'
        self.reports_dir = 'data/reports'
        
        # Create reports directory if needed
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
        
    def generate_weekly_report(self):
        """Generate comprehensive weekly learning report"""
        logging.info("📊 Generating Weekly Learning Dashboard...")
        
        # Load all data
        predictions = self._load_json(self.predictions_file)
        patterns = self._load_json(self.patterns_file)
        learning = self._load_json(self.learning_file)
        
        # Get predictions from last 7 days
        week_ago = datetime.now() - timedelta(days=7)
        weekly_predictions = self._filter_by_date(predictions, week_ago)
        
        # Calculate metrics
        overall_stats = self._calculate_overall_stats(weekly_predictions)
        pattern_stats = self._calculate_pattern_stats(weekly_predictions)
        ticker_stats = self._calculate_ticker_stats(weekly_predictions)
        action_stats = self._calculate_action_stats(weekly_predictions)
        llm_stats = self._calculate_llm_stats(weekly_predictions)
        insights = self._generate_insights(overall_stats, pattern_stats, action_stats, llm_stats)
        
        # Build report
        report = self._build_report(
            overall_stats,
            pattern_stats,
            ticker_stats,
            action_stats,
            llm_stats,
            insights
        )
        
        # Save report
        self._save_report(report)
        
        # Print to console
        print(report)
        
        return report
    
    def _load_json(self, filepath):
        """Load JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def _filter_by_date(self, predictions, start_date):
        """Get predictions from the last week"""
        weekly = {}
        for pred_id, pred in predictions.items():
            if 'timestamp' in pred:
                # Fix date issue
                timestamp_str = pred['timestamp'].replace('2025', '2024')
                pred_date = datetime.fromisoformat(timestamp_str)
                if pred_date >= start_date:
                    weekly[pred_id] = pred
        return weekly
    
    def _calculate_overall_stats(self, predictions):
        """Calculate overall performance stats"""
        total = len(predictions)
        checked = sum(1 for p in predictions.values() if p.get('outcome'))
        correct = sum(1 for p in predictions.values() if p.get('was_correct'))
        
        accuracy = (correct / checked * 100) if checked > 0 else 0
        
        return {
            'total_predictions': total,
            'checked_predictions': checked,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'pending_predictions': total - checked
        }
    
    def _calculate_pattern_stats(self, predictions):
        """Calculate performance by candlestick pattern"""
        pattern_performance = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred in predictions.values():
            if pred.get('outcome') and pred.get('candle_pattern'):
                pattern = pred['candle_pattern']
                pattern_performance[pattern]['total'] += 1
                if pred.get('was_correct'):
                    pattern_performance[pattern]['correct'] += 1
        
        # Calculate accuracy for each pattern
        results = {}
        for pattern, stats in pattern_performance.items():
            if stats['total'] > 0:
                results[pattern] = {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': (stats['correct'] / stats['total']) * 100
                }
        
        # Sort by accuracy
        return dict(sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True))
    
    def _calculate_ticker_stats(self, predictions):
        """Calculate performance by ticker"""
        ticker_performance = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred in predictions.values():
            if pred.get('outcome'):
                ticker = pred['ticker']
                ticker_performance[ticker]['total'] += 1
                if pred.get('was_correct'):
                    ticker_performance[ticker]['correct'] += 1
        
        results = {}
        for ticker, stats in ticker_performance.items():
            if stats['total'] > 0:
                results[ticker] = {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': (stats['correct'] / stats['total']) * 100
                }
        
        return dict(sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True))
    
    def _calculate_action_stats(self, predictions):
        """Calculate performance by action (BUY/SELL/HOLD)"""
        action_performance = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred in predictions.values():
            if pred.get('outcome'):
                action = pred['action']
                action_performance[action]['total'] += 1
                if pred.get('was_correct'):
                    action_performance[action]['correct'] += 1
        
        results = {}
        for action, stats in action_performance.items():
            if stats['total'] > 0:
                results[action] = {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': (stats['correct'] / stats['total']) * 100
                }
        
        return results
    
    def _calculate_llm_stats(self, predictions):
        """Calculate performance by LLM (if tracked)"""
        # This would need LLM tracking in predictions
        # For now, return placeholder
        return {
            'groq': {'accuracy': 0, 'total': 0},
            'gemini': {'accuracy': 0, 'total': 0},
            'cohere': {'accuracy': 0, 'total': 0}
        }
    
    def _generate_insights(self, overall, patterns, actions, llms):
        """Generate actionable insights"""
        insights = []
        
        # Overall performance insight
        if overall['accuracy'] >= 70:
            insights.append(f"✅ Strong performance this week at {overall['accuracy']:.1f}% accuracy!")
        elif overall['accuracy'] >= 50:
            insights.append(f"⚠️ Moderate performance at {overall['accuracy']:.1f}% - room for improvement")
        else:
            insights.append(f"❌ Below target at {overall['accuracy']:.1f}% - review strategy")
        
        # Best pattern
        if patterns:
            best_pattern = list(patterns.items())[0]
            if best_pattern[1]['accuracy'] >= 80:
                insights.append(f"🌟 Best pattern: '{best_pattern[0]}' at {best_pattern[1]['accuracy']:.0f}% accuracy")
        
        # Worst pattern
        if patterns:
            worst_pattern = list(patterns.items())[-1]
            if worst_pattern[1]['accuracy'] < 50 and worst_pattern[1]['total'] >= 3:
                insights.append(f"⚠️ Struggling with '{worst_pattern[0]}' pattern - only {worst_pattern[1]['accuracy']:.0f}% accurate")
        
        # Best action type
        if actions:
            best_action = max(actions.items(), key=lambda x: x[1]['accuracy'])
            insights.append(f"📈 Best at {best_action[0]} signals ({best_action[1]['accuracy']:.0f}% accurate)")
        
        # Recommendations
        if overall['accuracy'] < 60:
            insights.append("💡 RECOMMENDATION: Focus on high-confidence signals only (>70%)")
        
        if patterns and len(patterns) > 5:
            weak_patterns = [p for p, s in patterns.items() if s['accuracy'] < 50 and s['total'] >= 3]
            if weak_patterns:
                insights.append(f"💡 RECOMMENDATION: Avoid these patterns: {', '.join(weak_patterns)}")
        
        return insights
    
    def _build_report(self, overall, patterns, tickers, actions, llms, insights):
        """Build formatted report"""
        report = []
        report.append("=" * 70)
        report.append("📊 WEEKLY LEARNING DASHBOARD")
        report.append(f"📅 Week Ending: {datetime.now().strftime('%Y-%m-%d')}")
        report.append("=" * 70)
        report.append("")
        
        # Overall Performance
        report.append("📈 OVERALL PERFORMANCE")
        report.append("-" * 70)
        report.append(f"Total Predictions: {overall['total_predictions']}")
        report.append(f"Checked (Graded): {overall['checked_predictions']}")
        report.append(f"Correct: {overall['correct_predictions']}")
        report.append(f"Accuracy: {overall['accuracy']:.1f}%")
        report.append(f"Pending Review: {overall['pending_predictions']}")
        report.append("")
        
        # Pattern Performance
        if patterns:
            report.append("🕯️ CANDLESTICK PATTERN PERFORMANCE")
            report.append("-" * 70)
            for pattern, stats in list(patterns.items())[:5]:  # Top 5
                emoji = "✅" if stats['accuracy'] >= 70 else "⚠️" if stats['accuracy'] >= 50 else "❌"
                report.append(f"{emoji} {pattern.ljust(20)} {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")
            report.append("")
        
        # Ticker Performance
        if tickers:
            report.append("📊 BEST PERFORMING STOCKS")
            report.append("-" * 70)
            for ticker, stats in list(tickers.items())[:5]:  # Top 5
                emoji = "✅" if stats['accuracy'] >= 70 else "⚠️"
                report.append(f"{emoji} {ticker.ljust(8)} {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")
            report.append("")
        
        # Action Performance
        if actions:
            report.append("🎯 PREDICTION TYPE PERFORMANCE")
            report.append("-" * 70)
            for action, stats in actions.items():
                emoji = "✅" if stats['accuracy'] >= 70 else "⚠️" if stats['accuracy'] >= 50 else "❌"
                report.append(f"{emoji} {action.ljust(8)} {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")
            report.append("")
        
        # Key Insights
        report.append("💡 KEY INSIGHTS & RECOMMENDATIONS")
        report.append("-" * 70)
        for insight in insights:
            report.append(f"  {insight}")
        report.append("")
        
        report.append("=" * 70)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def _save_report(self, report):
        """Save report to file"""
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{self.reports_dir}/weekly_report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(report)
        
        logging.info(f"✅ Report saved to {filename}")


def main():
    dashboard = WeeklyDashboard()
    dashboard.generate_weekly_report()

if __name__ == "__main__":
    main()
