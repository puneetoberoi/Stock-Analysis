"""
Context Generator - Builds learning context from prediction history
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class ContextGenerator:
    def __init__(self, learning_brain):
        """Initialize with learning brain instance"""
        self.learning_brain = learning_brain
        self.db_path = learning_brain.db_path
    
    def get_stock_history(self, stock: str, limit: int = 10) -> List[Dict]:
        """Get prediction history for a specific stock"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT p.id, p.timestamp, p.prediction, p.confidence,
                   p.entry_price, p.patterns, p.rsi, p.llm_model,
                   o.actual_price, o.actual_move_pct, o.success
            FROM predictions p
            LEFT JOIN outcomes o ON p.id = o.prediction_id
            WHERE p.stock = ?
            ORDER BY p.timestamp DESC
            LIMIT ?
        """, (stock, limit))
        
        columns = ['id', 'timestamp', 'prediction', 'confidence', 
                  'entry_price', 'patterns', 'rsi', 'llm_model',
                  'actual_price', 'actual_move_pct', 'success']
        
        history = []
        for row in cursor.fetchall():
            history.append(dict(zip(columns, row)))
        
        conn.close()
        return history
    
    def calculate_pattern_accuracy(self, stock: str, pattern: str) -> Dict:
        """Calculate success rate for a specific pattern on a stock"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN o.success = 1 THEN 1 ELSE 0 END) as successful
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            WHERE p.stock = ? AND p.patterns LIKE ?
        """, (stock, f'%{pattern}%'))
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0] > 0:
            total, successful = row
            return {
                'total': total,
                'successful': successful,
                'accuracy': (successful / total * 100) if total > 0 else 0
            }
        return {'total': 0, 'successful': 0, 'accuracy': 0}
    
    def build_learning_context(self, stock: str, current_indicators: Dict) -> str:
        """Build comprehensive learning context for LLM prompt"""
        
        # Get stock history
        history = self.get_stock_history(stock, limit=5)
        
        if not history:
            return ""  # No history yet
        
        # Build context string
        context = f"\n\n📚 YOUR PAST PERFORMANCE ON {stock}:\n"
        context += "=" * 50 + "\n"
        
        # Overall stats
        checked_predictions = [h for h in history if h['success'] is not None]
        if checked_predictions:
            total = len(checked_predictions)
            successful = sum(1 for h in checked_predictions if h['success'])
            accuracy = (successful / total * 100) if total > 0 else 0
            
            context += f"📊 Accuracy: {accuracy:.1f}% ({successful}/{total} correct)\n\n"
            
            # Recent predictions
            context += "Recent predictions:\n"
            for h in checked_predictions[:3]:
                result = "✅ CORRECT" if h['success'] else "❌ WRONG"
                move = h['actual_move_pct'] if h['actual_move_pct'] else 0
                context += f"  {result}: {h['prediction']} @ ${h['entry_price']:.2f} → {move:+.1f}% move\n"
                context += f"    Pattern: {h['patterns']}, RSI: {h['rsi']:.0f}\n"
            
            # Pattern-specific insights
            current_pattern = current_indicators.get('patterns', '')
            if current_pattern:
                pattern_stats = self.calculate_pattern_accuracy(stock, current_pattern)
                if pattern_stats['total'] > 0:
                    context += f"\n🎯 Pattern '{current_pattern}' on {stock}:\n"
                    context += f"   Historical success: {pattern_stats['accuracy']:.0f}% "
                    context += f"({pattern_stats['successful']}/{pattern_stats['total']} times)\n"
            
            # Specific lessons learned
            failures = [h for h in checked_predictions if not h['success']]
            if failures:
                context += f"\n⚠️ LEARN FROM MISTAKES:\n"
                for f in failures[:2]:
                    context += f"  • {f['prediction']} failed on {f['patterns']} "
                    context += f"(RSI: {f['rsi']:.0f}) → {f['actual_move_pct']:+.1f}%\n"
            
            context += "\n💡 Use this knowledge to make a better prediction!\n"
            context += "=" * 50 + "\n"
        
        return context
    
    def get_overall_accuracy(self) -> Dict:
        """Get overall system accuracy stats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) * 100 as accuracy
            FROM outcomes
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0] > 0:
            return {
                'total_checked': row[0],
                'successful': row[1],
                'accuracy': row[2]
            }
        return {'total_checked': 0, 'successful': 0, 'accuracy': 0}


# Test function
if __name__ == "__main__":
    from learning_brain import LearningBrain
    
    brain = LearningBrain()
    context_gen = ContextGenerator(brain)
    
    # Test context generation
    test_indicators = {
        'patterns': 'swing_highs_lows',
        'rsi': 65.5
    }
    
    context = context_gen.build_learning_context('AAPL', test_indicators)
    print("Generated Context:")
    print(context)
    
    # Test overall accuracy
    stats = context_gen.get_overall_accuracy()
    print(f"\nOverall Stats: {stats}")
