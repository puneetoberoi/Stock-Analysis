"""
Outcome Evaluator Module
Checks prediction outcomes and extracts learning insights
"""

import sqlite3
import json
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

class OutcomeEvaluator:
    def __init__(self, db_path: str = "learning.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
    def check_outcomes(self, tracker) -> Dict:
        """Check all pending predictions and evaluate their success"""
        pending = tracker.get_pending_checks()
        
        results = {
            'checked': 0,
            'successful': 0,
            'failed': 0,
            'details': [],
            'lessons': []
        }
        
        for prediction in pending:
            outcome = self._evaluate_single_prediction(prediction)
            if outcome:
                results['checked'] += 1
                if outcome['success']:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    # Extract lesson from failure
                    lesson = self._extract_lesson(prediction, outcome)
                    if lesson:
                        results['lessons'].append(lesson)
                
                results['details'].append(outcome)
                self._store_outcome(prediction['id'], outcome)
                self._update_pattern_performance(prediction, outcome)
                self._update_llm_context_performance(prediction, outcome)
        
        # Update accuracy tracking
        self._update_accuracy_metrics()
        
        return results
    
    def _evaluate_single_prediction(self, prediction: Dict) -> Optional[Dict]:
        """Evaluate a single prediction against actual market movement"""
        try:
            stock = prediction['stock']
            ticker = yf.Ticker(stock)
            
            # Get current price
            current_data = ticker.history(period="2d")
            if current_data.empty:
                return None
            
            current_price = current_data['Close'].iloc[-1]
            entry_price = prediction['entry_price']
            
            # Calculate actual movement
            actual_move_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Determine success based on prediction type and 2% threshold
            success = self._determine_success(
                prediction['prediction'],
                actual_move_pct
            )
            
            # Analyze why it succeeded or failed
            failure_reason = None if success else self._analyze_failure(
                prediction, actual_move_pct
            )
            
            return {
                'stock': stock,
                'prediction': prediction['prediction'],
                'entry_price': entry_price,
                'actual_price': current_price,
                'actual_move_pct': actual_move_pct,
                'expected_move': 2.0 if prediction['prediction'] != 'HOLD' else 0,
                'success': success,
                'failure_reason': failure_reason,
                'llm_model': prediction['llm_model'],
                'confidence': prediction['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate {prediction['stock']}: {e}")
            return None
    
    def _determine_success(self, prediction_type: str, actual_move_pct: float) -> bool:
        """Determine if prediction was successful based on 2% threshold"""
        if prediction_type == "BUY":
            return actual_move_pct >= 2.0  # Success if gained 2%+
        elif prediction_type == "SELL":
            return actual_move_pct <= -2.0  # Success if dropped 2%+
        else:  # HOLD
            return abs(actual_move_pct) < 2.0  # Success if stayed within ±2%
    
    def _analyze_failure(self, prediction: Dict, actual_move: float) -> str:
        """Analyze why a prediction failed"""
        reasons = []
        
        # Check if market moved opposite to prediction
        if prediction['prediction'] == "BUY" and actual_move < 0:
            reasons.append("market_reversal")
        elif prediction['prediction'] == "SELL" and actual_move > 0:
            reasons.append("unexpected_rally")
        
        # Check if movement was too small
        if abs(actual_move) < 1:
            reasons.append("insufficient_momentum")
        
        # Check pattern reliability
        if prediction.get('patterns'):
            patterns = json.loads(prediction['patterns']) if isinstance(prediction['patterns'], str) else prediction['patterns']
            if patterns:
                reasons.append(f"pattern_failed:{patterns[0]}")
        
        # Check if confidence was too high for failed prediction
        if prediction['confidence'] > 80:
            reasons.append("overconfidence")
        
        return "; ".join(reasons) if reasons else "unknown"
    
    def _extract_lesson(self, prediction: Dict, outcome: Dict) -> Dict:
        """Extract a learning lesson from a failed prediction"""
        return {
            'stock': prediction['stock'],
            'pattern': json.loads(prediction['patterns'])[0] if prediction.get('patterns') else None,
            'prediction': prediction['prediction'],
            'llm_model': prediction['llm_model'],
            'failure_type': outcome['failure_reason'],
            'actual_move': outcome['actual_move_pct'],
            'lesson': self._generate_lesson_text(prediction, outcome),
            'adjustment': self._suggest_adjustment(prediction, outcome)
        }
    
    def _generate_lesson_text(self, prediction: Dict, outcome: Dict) -> str:
        """Generate human-readable lesson from failure"""
        lessons = []
        
        if "market_reversal" in outcome['failure_reason']:
            lessons.append(f"Market reversed against {prediction['prediction']} signal")
        
        if "insufficient_momentum" in outcome['failure_reason']:
            lessons.append("Price movement was too weak - need stronger signals")
        
        if "overconfidence" in outcome['failure_reason']:
            lessons.append(f"High confidence ({prediction['confidence']}%) was misplaced")
        
        if "pattern_failed" in outcome['failure_reason']:
            pattern = outcome['failure_reason'].split(':')[1] if ':' in outcome['failure_reason'] else 'unknown'
            lessons.append(f"Pattern '{pattern}' gave false signal")
        
        return ". ".join(lessons)
    
    def _suggest_adjustment(self, prediction: Dict, outcome: Dict) -> str:
        """Suggest how to adjust future predictions"""
        adjustments = []
        
        if abs(outcome['actual_move_pct']) < 1:
            adjustments.append("Require stronger momentum indicators")
        
        if prediction['confidence'] > 80 and not outcome['success']:
            adjustments.append("Reduce confidence weighting for this pattern")
        
        if "market_reversal" in outcome['failure_reason']:
            adjustments.append("Check broader market sentiment before signaling")
        
        return "; ".join(adjustments)
    
    def _store_outcome(self, prediction_id: int, outcome: Dict):
        """Store outcome in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO outcomes (
                    prediction_id, actual_price, actual_move_pct,
                    success, failure_reason, market_condition
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                outcome['actual_price'],
                outcome['actual_move_pct'],
                outcome['success'],
                outcome.get('failure_reason', ''),
                'normal'  # This could be enhanced with actual market regime detection
            ))
            conn.commit()
    
    def _update_pattern_performance(self, prediction: Dict, outcome: Dict):
        """Update performance metrics for patterns"""
        if not prediction.get('patterns'):
            return
            
        patterns = json.loads(prediction['patterns']) if isinstance(prediction['patterns'], str) else prediction['patterns']
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for pattern in patterns:
                # Check if pattern exists
                cursor.execute("""
                    SELECT success_count, fail_count, avg_return 
                    FROM pattern_performance 
                    WHERE pattern_name = ?
                """, (pattern,))
                
                row = cursor.fetchone()
                
                if row:
                    success_count = row[0] + (1 if outcome['success'] else 0)
                    fail_count = row[1] + (0 if outcome['success'] else 1)
                    total = success_count + fail_count
                    
                    # Update running average return
                    old_avg = row[2] or 0
                    new_avg = ((old_avg * (total - 1)) + outcome['actual_move_pct']) / total
                    
                    cursor.execute("""
                        UPDATE pattern_performance
                        SET success_count = ?, fail_count = ?, 
                            success_rate = ?, avg_return = ?,
                            last_updated = date('now')
                        WHERE pattern_name = ?
                    """, (
                        success_count, fail_count,
                        (success_count / total * 100),
                        new_avg, pattern
                    ))
                else:
                    # Insert new pattern
                    cursor.execute("""
                        INSERT INTO pattern_performance (
                            pattern_name, success_count, fail_count,
                            success_rate, avg_return
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        pattern,
                        1 if outcome['success'] else 0,
                        0 if outcome['success'] else 1,
                        100.0 if outcome['success'] else 0.0,
                        outcome['actual_move_pct']
                    ))
            
            conn.commit()
    
    def _update_llm_context_performance(self, prediction: Dict, outcome: Dict):
        """Track LLM performance in different contexts"""
        contexts = [
            ('rsi_range', self._categorize_rsi(prediction.get('rsi', 50))),
            ('volume', 'high' if prediction.get('volume_ratio', 1) > 1.5 else 'normal'),
            ('confidence', 'high' if prediction['confidence'] > 80 else 'normal')
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for context_type, context_value in contexts:
                # Update or insert context performance
                cursor.execute("""
                    INSERT INTO llm_context_performance (
                        llm_model, context_type, context_value,
                        success_rate, sample_size
                    ) VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(llm_model, context_type, context_value) DO UPDATE SET
                        success_rate = (success_rate * sample_size + ?) / (sample_size + 1),
                        sample_size = sample_size + 1
                """, (
                    prediction['llm_model'], context_type, context_value,
                    100.0 if outcome['success'] else 0.0,
                    100.0 if outcome['success'] else 0.0
                ))
            
            conn.commit()
    
    def _categorize_rsi(self, rsi: float) -> str:
        """Categorize RSI into ranges"""
        if rsi < 30:
            return 'oversold'
        elif rsi > 70:
            return 'overbought'
        else:
            return 'neutral'
    
    def _update_accuracy_metrics(self):
        """Update overall accuracy metrics table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Update accuracy_tracking table
            cursor.execute("""
                INSERT OR REPLACE INTO accuracy_tracking (stock, llm_model, total_predictions, correct_predictions, accuracy_pct, last_updated)
                SELECT 
                    p.stock,
                    p.llm_model,
                    COUNT(*) as total,
                    SUM(CASE WHEN o.success = 1 THEN 1 ELSE 0 END) as correct,
                    (SUM(CASE WHEN o.success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as accuracy,
                    date('now')
                FROM predictions p
                JOIN outcomes o ON p.id = o.prediction_id
                GROUP BY p.stock, p.llm_model
            """)
            
            conn.commit()
