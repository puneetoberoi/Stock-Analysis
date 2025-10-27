import json
from datetime import datetime

def view_learning_stats():
    # Load predictions
    with open('data/predictions.json', 'r') as f:
        predictions = json.load(f)
    
    # Load patterns
    with open('data/patterns.json', 'r') as f:
        patterns = json.load(f)
    
    print("="*60)
    print("📊 LEARNING SYSTEM STATISTICS")
    print("="*60)
    
    # Overall stats
    total = len(predictions)
    checked = sum(1 for p in predictions.values() if p.get('was_correct') is not None)
    correct = sum(1 for p in predictions.values() if p.get('was_correct') == True)
    
    print(f"\n📈 Overall Performance:")
    print(f"   Total Predictions: {total}")
    print(f"   Checked: {checked}")
    print(f"   Correct: {correct}")
    if checked > 0:
        print(f"   Accuracy: {(correct/checked)*100:.1f}%")
    
    # By action
    print(f"\n🎯 By Action Type:")
    for action in ['BUY', 'SELL', 'HOLD']:
        action_preds = [p for p in predictions.values() if p.get('action') == action]
        action_checked = [p for p in action_preds if p.get('was_correct') is not None]
        action_correct = [p for p in action_checked if p.get('was_correct') == True]
        
        if action_checked:
            accuracy = (len(action_correct) / len(action_checked)) * 100
            print(f"   {action}: {accuracy:.1f}% ({len(action_correct)}/{len(action_checked)})")
    
    # Pattern performance
    print(f"\n🕯️ Pattern Performance:")
    # Get global patterns (not ticker-specific)
    global_patterns = {k: v for k, v in patterns.items() if '_' not in k}
    
    sorted_patterns = sorted(global_patterns.items(), 
                            key=lambda x: x[1]['successful']/x[1]['total'] if x[1]['total'] > 0 else 0, 
                            reverse=True)
    
    for pattern, stats in sorted_patterns[:10]:  # Top 10
        if stats['total'] >= 2:  # Only show patterns with 2+ samples
            success_rate = (stats['successful'] / stats['total']) * 100
            emoji = "🟢" if success_rate >= 60 else "🔴" if success_rate < 40 else "🟡"
            print(f"   {emoji} {pattern}: {success_rate:.0f}% ({stats['successful']}/{stats['total']})")
    
    # Recent predictions
    print(f"\n📅 Recent Predictions:")
    recent = sorted(predictions.values(), 
                   key=lambda x: x.get('timestamp', ''), 
                   reverse=True)[:5]
    
    for pred in recent:
        ticker = pred.get('ticker', '?')
        action = pred.get('action', '?')
        outcome = pred.get('outcome', {})
        
        if outcome:
            change = outcome.get('price_change_pct', 0)
            correct = pred.get('was_correct', False)
            emoji = "✅" if correct else "❌"
            print(f"   {emoji} {ticker} {action}: {change:+.2f}%")
        else:
            print(f"   ⏳ {ticker} {action}: Not checked yet")

if __name__ == "__main__":
    view_learning_stats()
