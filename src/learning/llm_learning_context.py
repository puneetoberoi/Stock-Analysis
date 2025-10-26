import json
import os
from datetime import datetime, timedelta

class LLMLearningContext:
    """Provides learning context to LLMs based on past performance"""
    
    def __init__(self):
        self.predictions_file = 'data/predictions.json'
        self.llm_accuracy_file = 'data/llm_accuracy.json'
        
    def get_learning_context_for_ticker(self, ticker, llm_name=None, days_back=30):
        """Get what the system has learned about this ticker"""
        
        # Load predictions
        with open(self.predictions_file, 'r') as f:
            all_preds = json.load(f)
        
        # Filter for this ticker with outcomes
        ticker_preds = [
            p for p in all_preds.values() 
            if p['ticker'] == ticker and p.get('outcome')
        ]
        
        if not ticker_preds:
            return "No prior predictions for this ticker."
        
        # Calculate recent performance
        recent_preds = ticker_preds[-10:]  # Last 10
        correct_count = sum(1 for p in recent_preds if p.get('was_correct'))
        accuracy = (correct_count / len(recent_preds)) * 100 if recent_preds else 0
        
        # Find patterns that worked/failed
        pattern_performance = {}
        for pred in ticker_preds:
            pattern = pred.get('candle_pattern')
            if pattern:
                if pattern not in pattern_performance:
                    pattern_performance[pattern] = {'correct': 0, 'total': 0}
                pattern_performance[pattern]['total'] += 1
                if pred.get('was_correct'):
                    pattern_performance[pattern]['correct'] += 1
        
        # Build learning context
        context = f"""
PAST PERFORMANCE ON {ticker}:
- Recent accuracy: {accuracy:.0f}% ({correct_count}/{len(recent_preds)} correct)
"""
        
        # Add last prediction
        if recent_preds:
            last = recent_preds[-1]
            outcome = last.get('outcome', {})
            result = "CORRECT ✅" if last.get('was_correct') else "WRONG ❌"
            
            context += f"""
- Last prediction: {last['action']} → {result}
  (Stock moved {outcome.get('price_change_pct', 0):+.1f}%)
"""
        
        # Add pattern insights
        if pattern_performance:
            context += "\nPATTERN PERFORMANCE:\n"
            for pattern, stats in pattern_performance.items():
                acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                context += f"- {pattern}: {acc:.0f}% accurate ({stats['correct']}/{stats['total']})\n"
        
        # Add recent mistakes to learn from
        recent_mistakes = [p for p in recent_preds[-5:] if not p.get('was_correct')]
        if recent_mistakes:
            context += "\nRECENT MISTAKES TO AVOID:\n"
            for mistake in recent_mistakes:
                outcome = mistake.get('outcome', {})
                context += f"- Predicted {mistake['action']} but stock moved {outcome.get('price_change_pct', 0):+.1f}%\n"
                context += f"  Reason given: {mistake.get('reasoning', 'N/A')[:100]}...\n"
        
        return context
    
    def get_llm_overall_accuracy(self, llm_name):
        """Get this LLM's overall accuracy"""
        if not os.path.exists(self.llm_accuracy_file):
            return None
        
        with open(self.llm_accuracy_file, 'r') as f:
            accuracy_data = json.load(f)
        
        return accuracy_data.get(llm_name, {}).get('accuracy', 0)
    
    def update_llm_accuracy(self, llm_name, ticker, was_correct):
        """Update LLM accuracy after outcome is known"""
        
        if os.path.exists(self.llm_accuracy_file):
            with open(self.llm_accuracy_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        
        # Initialize if needed
        if llm_name not in data:
            data[llm_name] = {
                'total_predictions': 0,
                'correct': 0,
                'accuracy': 0,
                'by_ticker': {}
            }
        
        # Update overall stats
        data[llm_name]['total_predictions'] += 1
        if was_correct:
            data[llm_name]['correct'] += 1
        data[llm_name]['accuracy'] = (
            data[llm_name]['correct'] / data[llm_name]['total_predictions'] * 100
        )
        
        # Update ticker-specific stats
        if ticker not in data[llm_name]['by_ticker']:
            data[llm_name]['by_ticker'][ticker] = {'total': 0, 'correct': 0}
        
        data[llm_name]['by_ticker'][ticker]['total'] += 1
        if was_correct:
            data[llm_name]['by_ticker'][ticker]['correct'] += 1
        data[llm_name]['by_ticker'][ticker]['accuracy'] = (
            data[llm_name]['by_ticker'][ticker]['correct'] / 
            data[llm_name]['by_ticker'][ticker]['total'] * 100
        )
        
        # Save
        with open(self.llm_accuracy_file, 'w') as f:
            json.dump(data, f, indent=2)
