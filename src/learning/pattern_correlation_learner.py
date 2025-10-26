# src/learning/pattern_correlation_learner.py
"""
Advanced learning system that analyzes WHY patterns fail/succeed
by correlating with RSI, MACD, volume, news, etc.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
import logging

class PatternCorrelationLearner:
    """Learns which combinations of indicators make patterns succeed/fail"""
    
    def __init__(self):
        self.correlations_file = 'data/pattern_correlations.json'
        self.correlations = self._load_correlations()
        
    def _load_correlations(self):
        """Load learned pattern correlations"""
        if os.path.exists(self.correlations_file):
            with open(self.correlations_file, 'r') as f:
                return json.load(f)
        return {
            'pattern_rules': {},  # Learned rules
            'success_conditions': {},  # What makes patterns work
            'failure_conditions': {}   # What makes patterns fail
        }
    
    def analyze_prediction_outcome(self, prediction, outcome):
        """
        Analyze WHY a prediction succeeded or failed
        
        Args:
            prediction: The original prediction with all context
            outcome: The actual outcome with price change
        """
        pattern = prediction.get('candle_pattern')
        if not pattern:
            return
        
        was_correct = prediction.get('was_correct', False)
        
        # Extract all the conditions present when prediction was made
        conditions = self._extract_conditions(prediction)
        
        # Learn from this outcome
        if was_correct:
            self._learn_success_conditions(pattern, conditions, outcome)
        else:
            self._learn_failure_conditions(pattern, conditions, outcome)
        
        # Update pattern rules
        self._update_pattern_rules(pattern)
        
        # Save learnings
        self._save_correlations()
    
    def _extract_conditions(self, prediction):
        """Extract all conditions that were present"""
        indicators = prediction.get('indicators', {})
        
        conditions = {
            'rsi': indicators.get('rsi', 50),
            'rsi_zone': self._get_rsi_zone(indicators.get('rsi', 50)),
            'volume_ratio': prediction.get('volume_ratio', 1.0),
            'volume_level': self._get_volume_level(prediction.get('volume_ratio', 1.0)),
            'confidence': prediction.get('confidence', 50),
            'action': prediction.get('action'),
            'market_context': prediction.get('market_context', 0),
            'market_sentiment': self._get_market_sentiment(prediction.get('market_context', 0)),
            'price_at_prediction': prediction.get('price_at_prediction', 0)
        }
        
        return conditions
    
    def _get_rsi_zone(self, rsi):
        """Categorize RSI into zones"""
        if rsi < 30:
            return 'oversold'
        elif rsi < 45:
            return 'low'
        elif rsi < 55:
            return 'neutral'
        elif rsi < 70:
            return 'high'
        else:
            return 'overbought'
    
    def _get_volume_level(self, volume_ratio):
        """Categorize volume"""
        if volume_ratio < 0.5:
            return 'very_low'
        elif volume_ratio < 0.8:
            return 'low'
        elif volume_ratio < 1.2:
            return 'normal'
        elif volume_ratio < 2.0:
            return 'high'
        else:
            return 'very_high'
    
    def _get_market_sentiment(self, macro_score):
        """Categorize market context"""
        if macro_score > 10:
            return 'very_positive'
        elif macro_score > 0:
            return 'positive'
        elif macro_score > -10:
            return 'neutral'
        elif macro_score > -20:
            return 'negative'
        else:
            return 'very_negative'
    
    def _learn_success_conditions(self, pattern, conditions, outcome):
        """Record conditions when pattern succeeded"""
        if pattern not in self.correlations['success_conditions']:
            self.correlations['success_conditions'][pattern] = []
        
        self.correlations['success_conditions'][pattern].append({
            'conditions': conditions,
            'outcome': {
                'price_change': outcome.get('price_change_pct', 0),
                'days_held': outcome.get('days_held', 0)
            },
            'timestamp': datetime.now().isoformat()
        })
        
        logging.info(f"✅ Learned success: {pattern} with RSI={conditions['rsi']:.0f}, Volume={conditions['volume_level']}")
    
    def _learn_failure_conditions(self, pattern, conditions, outcome):
        """Record conditions when pattern failed"""
        if pattern not in self.correlations['failure_conditions']:
            self.correlations['failure_conditions'][pattern] = []
        
        self.correlations['failure_conditions'][pattern].append({
            'conditions': conditions,
            'outcome': {
                'price_change': outcome.get('price_change_pct', 0),
                'days_held': outcome.get('days_held', 0)
            },
            'timestamp': datetime.now().isoformat()
        })
        
        logging.info(f"❌ Learned failure: {pattern} with RSI={conditions['rsi']:.0f}, Volume={conditions['volume_level']}")
    
    def _update_pattern_rules(self, pattern):
        """Generate rules based on success/failure patterns"""
        successes = self.correlations['success_conditions'].get(pattern, [])
        failures = self.correlations['failure_conditions'].get(pattern, [])
        
        if len(successes) < 3 and len(failures) < 3:
            return  # Not enough data yet
        
        # Analyze common conditions in failures
        failure_rules = []
        
        if failures:
            # Check RSI correlation
            high_rsi_failures = [f for f in failures if f['conditions']['rsi'] > 70]
            if len(high_rsi_failures) / len(failures) > 0.6:
                failure_rules.append({
                    'rule': f"{pattern} fails when RSI > 70",
                    'confidence': len(high_rsi_failures) / len(failures) * 100,
                    'sample_size': len(failures)
                })
            
            # Check volume correlation
            low_vol_failures = [f for f in failures if f['conditions']['volume_ratio'] < 0.8]
            if len(low_vol_failures) / len(failures) > 0.6:
                failure_rules.append({
                    'rule': f"{pattern} fails with low volume",
                    'confidence': len(low_vol_failures) / len(failures) * 100,
                    'sample_size': len(failures)
                })
            
            # Check market sentiment
            negative_market_failures = [f for f in failures if f['conditions']['market_context'] < -10]
            if len(negative_market_failures) / len(failures) > 0.6:
                failure_rules.append({
                    'rule': f"{pattern} fails in negative market context",
                    'confidence': len(negative_market_failures) / len(failures) * 100,
                    'sample_size': len(failures)
                })
        
        # Analyze common conditions in successes
        success_rules = []
        
        if successes:
            # Check what makes it work
            good_rsi_successes = [s for s in successes if 30 < s['conditions']['rsi'] < 70]
            if len(good_rsi_successes) / len(successes) > 0.7:
                success_rules.append({
                    'rule': f"{pattern} works best with balanced RSI (30-70)",
                    'confidence': len(good_rsi_successes) / len(successes) * 100,
                    'sample_size': len(successes)
                })
            
            high_vol_successes = [s for s in successes if s['conditions']['volume_ratio'] > 1.2]
            if len(high_vol_successes) / len(successes) > 0.6:
                success_rules.append({
                    'rule': f"{pattern} works best with high volume",
                    'confidence': len(high_vol_successes) / len(successes) * 100,
                    'sample_size': len(successes)
                })
        
        # Store rules
        self.correlations['pattern_rules'][pattern] = {
            'failure_rules': failure_rules,
            'success_rules': success_rules,
            'total_observations': len(successes) + len(failures),
            'success_rate': len(successes) / (len(successes) + len(failures)) * 100 if (successes or failures) else 0,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_pattern_warnings(self, pattern, current_conditions):
        """
        Check if current conditions match known failure patterns
        
        Returns warnings if pattern likely to fail
        """
        if pattern not in self.correlations['pattern_rules']:
            return []
        
        rules = self.correlations['pattern_rules'][pattern]
        warnings = []
        
        # Check failure rules
        for rule in rules.get('failure_rules', []):
            rule_text = rule['rule'].lower()
            
            if 'rsi > 70' in rule_text and current_conditions.get('rsi', 50) > 70:
                warnings.append({
                    'warning': f"⚠️ {pattern} has {rule['confidence']:.0f}% failure rate when RSI > 70",
                    'severity': 'high',
                    'recommendation': 'Consider HOLD instead'
                })
            
            if 'low volume' in rule_text and current_conditions.get('volume_ratio', 1.0) < 0.8:
                warnings.append({
                    'warning': f"⚠️ {pattern} has {rule['confidence']:.0f}% failure rate with low volume",
                    'severity': 'medium',
                    'recommendation': 'Wait for volume confirmation'
                })
            
            if 'negative market' in rule_text and current_conditions.get('market_context', 0) < -10:
                warnings.append({
                    'warning': f"⚠️ {pattern} has {rule['confidence']:.0f}% failure rate in negative markets",
                    'severity': 'medium',
                    'recommendation': 'Be cautious in current market environment'
                })
        
        return warnings
    
    def get_learning_context(self, pattern, current_conditions):
        """
        Get learning context to feed to LLMs
        
        This is what we'll add to the LLM prompt
        """
        if pattern not in self.correlations['pattern_rules']:
            return f"No historical data for {pattern} pattern yet."
        
        rules = self.correlations['pattern_rules'][pattern]
        success_rate = rules.get('success_rate', 0)
        observations = rules.get('total_observations', 0)
        
        context = f"""
HISTORICAL PERFORMANCE OF {pattern.upper()} PATTERN:
- Overall success rate: {success_rate:.0f}% (based on {observations} observations)
"""
        
        # Add failure warnings if applicable
        warnings = self.get_pattern_warnings(pattern, current_conditions)
        if warnings:
            context += "\n⚠️ WARNING SIGNALS:\n"
            for warning in warnings:
                context += f"- {warning['warning']}\n"
                context += f"  Recommendation: {warning['recommendation']}\n"
        
        # Add success conditions
        success_rules = rules.get('success_rules', [])
        if success_rules:
            context += "\n✅ THIS PATTERN WORKS BEST WHEN:\n"
            for rule in success_rules[:3]:  # Top 3
                context += f"- {rule['rule']} ({rule['confidence']:.0f}% confidence)\n"
        
        return context
    
    def _save_correlations(self):
        """Save learned correlations"""
        with open(self.correlations_file, 'w') as f:
            json.dump(self.correlations, f, indent=2, default=str)
