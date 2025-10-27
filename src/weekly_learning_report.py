# src/weekly_learning_report.py
import json
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'data'


def generate_weekly_report():
    """Generate comprehensive weekly learning report"""
    
    # Load data
    predictions_file = Path('data/predictions.json')
    patterns_file = Path('data/patterns.json')
    combinations_file = Path('data/pattern_combinations.json')
    
    if not predictions_file.exists():
        logging.warning("No predictions data found")
        return
    
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    patterns = {}
    if patterns_file.exists():
        with open(patterns_file, 'r') as f:
            patterns = json.load(f)
    
    combinations = {}
    if combinations_file.exists():
        with open(combinations_file, 'r') as f:
            combinations = json.load(f)
    
    # Filter last 7 days
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    recent_preds = [
        p for p in predictions.values()
        if p.get('timestamp', '') >= week_ago and p.get('was_correct') is not None
    ]
    
    if not recent_preds:
        logging.info("No predictions from last 7 days")
        return
    
    # Calculate stats
    total = len(recent_preds)
    correct = sum(1 for p in recent_preds if p.get('was_correct'))
    accuracy = (correct / total) * 100
    
    # By action
    buy_preds = [p for p in recent_preds if p['action'] == 'BUY']
    sell_preds = [p for p in recent_preds if p['action'] == 'SELL']
    hold_preds = [p for p in recent_preds if p['action'] == 'HOLD']
    
    buy_accuracy = (sum(1 for p in buy_preds if p['was_correct']) / len(buy_preds) * 100) if buy_preds else 0
    sell_accuracy = (sum(1 for p in sell_preds if p['was_correct']) / len(sell_preds) * 100) if sell_preds else 0
    hold_accuracy = (sum(1 for p in hold_preds if p['was_correct']) / len(hold_preds) * 100) if hold_preds else 0
    
    # Average returns
    avg_return = sum(p['outcome']['price_change_pct'] for p in recent_preds) / total
    
    # Best and worst
    sorted_by_return = sorted(recent_preds, key=lambda x: x['outcome']['price_change_pct'], reverse=True)
    best = sorted_by_return[0]
    worst = sorted_by_return[-1]
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 0; background: #f7f7f7; }}
        .container {{ max-width: 800px; margin: 20px auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; padding: 40px; text-align: center; }}
        .content {{ padding: 40px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 30px 0; }}
        .stat-card {{ padding: 20px; background: #f9fafb; border-radius: 10px; text-align: center; }}
        .stat-value {{ font-size: 36px; font-weight: bold; margin: 10px 0; }}
        .chart-bar {{ background: #e5e7eb; border-radius: 8px; overflow: hidden; margin: 10px 0; }}
        .chart-fill {{ height: 30px; background: linear-gradient(90deg, #10b981 0%, #059669 100%); display: flex; align-items: center; padding: 0 15px; color: white; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Weekly Learning Report</h1>
            <p style="font-size:20px;margin:10px 0 0 0;">{(datetime.now() - timedelta(days=7)).strftime('%b %d')} - {datetime.now().strftime('%b %d, %Y')}</p>
        </div>
        
        <div class="content">
            <div class="stat-grid">
                <div class="stat-card">
                    <div style="color:#666;">Overall Accuracy</div>
                    <div class="stat-value" style="color:{'#16a34a' if accuracy >= 60 else '#dc2626'};">{accuracy:.1f}%</div>
                    <div style="color:#888;font-size:14px;">{correct}/{total} correct</div>
                </div>
                
                <div class="stat-card">
                    <div style="color:#666;">Avg Return</div>
                    <div class="stat-value" style="color:{'#16a34a' if avg_return > 0 else '#dc2626'};">{avg_return:+.2f}%</div>
                    <div style="color:#888;font-size:14px;">Per prediction</div>
                </div>
            </div>
            
            <h2 style="border-bottom:3px solid #3b82f6;padding-bottom:10px;">📊 Performance by Action</h2>
            
            <div class="chart-bar">
                <div class="chart-fill" style="width:{buy_accuracy}%;background:linear-gradient(90deg, #10b981 0%, #059669 100%);">
                    BUY: {buy_accuracy:.0f}% ({len(buy_preds)} predictions)
                </div>
            </div>
            
            <div class="chart-bar">
                <div class="chart-fill" style="width:{sell_accuracy}%;background:linear-gradient(90deg, #ef4444 0%, #dc2626 100%);">
                    SELL: {sell_accuracy:.0f}% ({len(sell_preds)} predictions)
                </div>
            </div>
            
            <div class="chart-bar">
                <div class="chart-fill" style="width:{hold_accuracy}%;background:linear-gradient(90deg, #f59e0b 0%, #d97706 100%);">
                    HOLD: {hold_accuracy:.0f}% ({len(hold_preds)} predictions)
                </div>
            </div>
            
            <h2 style="border-bottom:3px solid #3b82f6;padding-bottom:10px;margin-top:40px;">🏆 Best & Worst Predictions</h2>
            
            <div style="padding:20px;background:#d1fae5;border-left:4px solid #16a34a;border-radius:8px;margin:20px 0;">
                <div style="font-size:18px;font-weight:bold;color:#16a34a;">✅ Best: {best['ticker']} ({best['action']})</div>
                <div style="margin:10px 0;font-size:24px;font-weight:bold;">{best['outcome']['price_change_pct']:+.2f}%</div>
                <div style="color:#666;">Pattern: {best.get('candle_pattern', 'None')}</div>
            </div>
            
            <div style="padding:20px;background:#fee2e2;border-left:4px solid #dc2626;border-radius:8px;">
                <div style="font-size:18px;font-weight:bold;color:#dc2626;">❌ Worst: {worst['ticker']} ({worst['action']})</div>
                <div style="margin:10px 0;font-size:24px;font-weight:bold;">{worst['outcome']['price_change_pct']:+.2f}%</div>
                <div style="color:#666;">Pattern: {worst.get('candle_pattern', 'None')}</div>
            </div>
            
            <h2 style="border-bottom:3px solid #3b82f6;padding-bottom:10px;margin-top:40px;">🧠 Top Pattern Combinations</h2>
            <p style="color:#666;">Patterns that work best with specific market conditions:</p>
            
            {_generate_combo_table(combinations)}
            
            <div style="margin-top:40px;padding:30px;background:linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);border-radius:10px;text-align:center;">
                <h3 style="margin:0 0 15px 0;">💡 This Week's Key Learning</h3>
                <p style="font-size:16px;line-height:1.8;margin:0;">
                    {_generate_key_insight(recent_preds, combinations)}
                </p>
            </div>
            
            <div style="text-align:center;margin-top:30px;color:#888;font-size:14px;">
                <p>Keep learning, keep improving! 📈</p>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    send_email(html, f"📊 Weekly Learning Report - {datetime.now().strftime('%b %d, %Y')}")


def _generate_combo_table(combinations):
    """Generate HTML table for best combinations"""
    if not combinations:
        return "<p style='color:#888;'>Not enough data yet. Check back next week!</p>"
    
    # Get top performing combinations
    sorted_combos = sorted(
        [
            (k, v) for k, v in combinations.items()
            if v['total'] >= 3  # At least 3 samples
        ],
        key=lambda x: x[1]['successful'] / x[1]['total'],
        reverse=True
    )[:5]
    
    if not sorted_combos:
        return "<p style='color:#888;'>Not enough data yet. Check back next week!</p>"
    
    rows = ""
    for combo_key, stats in sorted_combos:
        success_rate = (stats['successful'] / stats['total']) * 100
        color = '#16a34a' if success_rate >= 60 else '#f59e0b'
        
        rows += f"""
        <tr>
            <td style="padding:15px;border-bottom:1px solid #eee;">
                <strong>{stats['pattern'].replace('_', ' ').title()}</strong><br>
                <span style="font-size:13px;color:#666;">{stats.get('context', '')}</span>
            </td>
            <td style="padding:15px;border-bottom:1px solid #eee;text-align:center;">
                <div style="font-size:24px;font-weight:bold;color:{color};">{success_rate:.0f}%</div>
            </td>
            <td style="padding:15px;border-bottom:1px solid #eee;text-align:center;">
                {stats['total']}
            </td>
        </tr>
        """
    
    return f"""
    <table style="width:100%;border-collapse:collapse;background:white;border-radius:10px;overflow:hidden;">
        <thead>
            <tr style="background:#f3f4f6;">
                <th style="padding:15px;text-align:left;">Combination</th>
                <th style="padding:15px;text-align:center;">Success Rate</th>
                <th style="padding:15px;text-align:center;">Samples</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


def _generate_key_insight(predictions, combinations):
    """Generate a key insight from the week's data"""
    
    # Find the most successful pattern combination
    if combinations:
        sorted_combos = sorted(
            [(k, v) for k, v in combinations.items() if v['total'] >= 3],
            key=lambda x: x[1]['successful'] / x[1]['total'],
            reverse=True
        )
        
        if sorted_combos:
            best = sorted_combos[0]
            success_rate = (best[1]['successful'] / best[1]['total']) * 100
            return f"The '{best[1]['pattern'].replace('_', ' ')}' pattern with {best[1].get('context', 'specific conditions')} has shown a {success_rate:.0f}% success rate. Consider this when making future predictions!"
    
    # Fallback insights
    buy_success = sum(1 for p in predictions if p['action'] == 'BUY' and p.get('was_correct'))
    sell_success = sum(1 for p in predictions if p['action'] == 'SELL' and p.get('was_correct'))
    
    if sell_success > buy_success:
        return "SELL signals are currently more reliable than BUY signals. The market may be in a corrective phase."
    else:
        return "BUY signals are performing well. Bullish setups are working in the current market environment."


def send_email(html, subject):
    """Send the weekly report email"""
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    
    if not smtp_user or not smtp_pass:
        logging.warning("SMTP credentials missing")
        return
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = smtp_user
    
    msg.attach(MIMEText(html, 'html'))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logging.info("✅ Weekly report email sent!")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


if __name__ == "__main__":
    generate_weekly_report()
