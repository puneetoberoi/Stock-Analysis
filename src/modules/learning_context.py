"""
Learning Context Generator - Creates autonomous learning prompts
No hardcoded rules, just data presentation
"""

import sqlite3
from datetime import datetime, timedelta
import logging

class LearningContextGenerator:
    def __init__(self, db_path='src/modules/data/learning.db'):
        self.db_path = db_path
    
    def generate_autonomous_context(self, stock, current_data, performance_summary):
        """
        Generate autonomous learning context
        Pure data presentation - LLM discovers patterns on its own
        """
        
        # Get stock-specific performance
        stock_performance = self._get_stock_specific_performance(stock)
        
        # Get recent failures (raw data only)
        recent_failures = self._format_failures(performance_summary['failures'])
        
        # Get recent successes (raw data only)
        recent_successes = self._format_successes(performance_summary['successes'])
        
        # Build autonomous prompt (no suggestions, just data)
        context = f"""You are an autonomous learning AI with one goal: Achieve 90% prediction accuracy.

CURRENT ACCURACY: {performance_summary['accuracy']:.1f}% ({performance_summary['correct']}/{performance_summary['total']})
TARGET: 90%
GAP: {90 - performance_summary['accuracy']:.1f}% to go

YOUR RECENT PERFORMANCE:
{recent_failures}

YOUR RECENT SUCCESSES:
{recent_successes}

STOCK-SPECIFIC PERFORMANCE ({stock}):
{stock_performance}

TODAY'S ANALYSIS - {stock}:
Current Price: ${current_data.get('current_price', 0):.2f}
RSI: {current_data.get('rsi', 0):.1f}
MACD: {current_data.get('macd', 0):.2f}
Volume: {current_data.get('volume_ratio', 0):.1f}x average
Bollinger Band Position: {current_data.get('bb_position', 0):.0f}%
Patterns Detected: {', '.join(current_data.get('patterns', []))}
Macro Score: {current_data.get('macro_score', 0):.1f}
Gap Info: {current_data.get('gap_info', 'None')}

AUTONOMOUS LEARNING TASK:
1. Make your prediction: BUY, HOLD, or SELL
2. Set your confidence: 0-100%
3. Explain your reasoning
4. CRITICAL: What did you learn from your past failures?
5. What variables are you weighting most today?
6. What new pattern or rule are you testing?
7. Self-assessment: How confident are you this will be correct?

Remember: You have complete freedom to analyze this however you want.
No rules are imposed. Discover what works through experimentation.
"""
        
        return context
    
    def _get_stock_specific_performance(self, stock):
        """Get performance data for specific stock (raw data only)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN o.success = 1 THEN 1 ELSE 0 END) as correct
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            WHERE p.stock = ?
        """, (stock,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] > 0:
            total, correct = result
            accuracy = (correct / total * 100) if total > 0 else 0
            return f"Total predictions: {total} | Correct: {correct} | Accuracy: {accuracy:.1f}%"
        else:
            return "No previous predictions for this stock (First time!)"
    
    def _format_failures(self, failures):
        """Format failures as pure data (no interpretation)"""
        if not failures:
            return "No recent failures - Great job!"
        
        formatted = "FAILURES (Last 5):\n"
        for i, (stock, action, entry, actual, change_pct, reasoning, rsi, vol) in enumerate(failures, 1):
            formatted += f"\n{i}. {stock} {action} @${entry:.2f} → ${actual:.2f} ({change_pct:+.1f}%)\n"
            formatted += f"   Data at prediction: RSI={rsi:.1f}, Volume={vol:.1f}x\n"
            formatted += f"   Your reasoning: {reasoning[:100]}...\n"
        
        return formatted
    
    def _format_successes(self, successes):
        """Format successes as pure data (no interpretation)"""
        if not successes:
            return "No recent successes yet - Keep learning!"
        
        formatted = "SUCCESSES (Last 5):\n"
        for i, (stock, action, change_pct, rsi, vol) in enumerate(successes, 1):
            formatted += f"\n{i}. {stock} {action} → {change_pct:+.1f}% ✅\n"
            formatted += f"   Data at prediction: RSI={rsi:.1f}, Volume={vol:.1f}x\n"
        
        return formatted
    
    def extract_llm_learnings(self, llm_response):
        """
        Extract what the LLM learned from its response
        Store for future reference
        """
        learnings = {
            'variables_used': [],
            'new_rules_tested': [],
            'self_reflection': '',
            'confidence_reasoning': ''
        }
        
        # Simple extraction (LLM will structure its response)
        response_lower = llm_response.lower()
        
        # Extract mentioned variables
        variables = ['rsi', 'macd', 'volume', 'bollinger', 'macro', 'pattern']
        for var in variables:
            if var in response_lower:
                learnings['variables_used'].append(var)
        
        # Store the full response as learning
        learnings['self_reflection'] = llm_response
        
        return learnings
