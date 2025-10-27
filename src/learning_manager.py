
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class PredictionOutcome:
    ticker: str
    date: str
    prediction: str  # BUY/SELL/HOLD
    confidence: float
    actual_return: float
    success: bool
    technical_context: Dict  # RSI, MACD, volume etc at prediction time
    failure_analysis: Optional[str] = None
    
@dataclass
class LearningPattern:
    pattern_name: str
    conditions: Dict
    success_rate: float
    sample_size: int
    last_updated: str

class LearningManager:
    def __init__(self, github_token: str, repo_name: str):
        self.github_token = github_token
        self.repo_name = repo_name
        self.learning_file = "data/learning_history.json"
        self.pattern_file = "data/learned_patterns.json"
        
    def load_learning_history(self) -> Dict:
        """Load historical predictions and their outcomes"""
        try:
            # Load from GitHub
            from github import Github
            g = Github(self.github_token)
            repo = g.get_repo(self.repo_name)
            
            try:
                content = repo.get_contents(self.learning_file)
                return json.loads(content.decoded_content)
            except:
                return {"predictions": [], "patterns": {}, "ticker_accuracy": {}}
        except Exception as e:
            print(f"Error loading learning history: {e}")
            return {"predictions": [], "patterns": {}, "ticker_accuracy": {}}
    
    def analyze_failure_patterns(self, ticker: str, prediction: str, 
                               technical_data: Dict, actual_return: float) -> str:
        """Analyze why a prediction failed"""
        analysis = []
        
        # Check for false signals
        if prediction == "BUY" and actual_return < -2:
            if technical_data.get('rsi', 0) > 70:
                analysis.append("Failed BUY despite overbought RSI (>70) - possible reversal ignored")
            if technical_data.get('macd_histogram', 0) < 0:
                analysis.append("Failed BUY with negative MACD histogram - momentum wasn't confirmed")
            if technical_data.get('volume_ratio', 1) < 0.8:
                analysis.append("Failed BUY with low volume - lack of conviction in move")
                
        elif prediction == "SELL" and actual_return > 2:
            if technical_data.get('rsi', 0) < 30:
                analysis.append("Failed SELL despite oversold RSI (<30) - possible bounce ignored")
            if technical_data.get('near_support', False):
                analysis.append("Failed SELL near support level - support held stronger than expected")
                
        # Check for pattern conflicts
        if technical_data.get('double_bottom', False) and prediction == "SELL":
            analysis.append("Missed double bottom pattern - bullish reversal pattern ignored")
            
        return " | ".join(analysis) if analysis else "No clear pattern identified"
    
    def get_ticker_learning_context(self, ticker: str, current_technicals: Dict) -> str:
        """Generate learning context for a specific ticker"""
        history = self.load_learning_history()
        
        # Filter predictions for this ticker
        ticker_predictions = [p for p in history.get('predictions', []) 
                            if p.get('ticker') == ticker]
        
        if not ticker_predictions:
            return "No historical data for learning."
        
        # Calculate accuracy metrics
        recent_predictions = ticker_predictions[-10:]  # Last 10 predictions
        accuracy = sum(1 for p in recent_predictions if p.get('success', False)) / len(recent_predictions)
        
        # Find similar technical setups
        similar_setups = self._find_similar_setups(current_technicals, ticker_predictions)
        
        # Build context
        context_parts = [
            f"Historical accuracy on {ticker}: {accuracy:.1%} (last {len(recent_predictions)} trades)",
        ]
        
        # Add pattern-specific insights
        if similar_setups:
            success_rate = sum(1 for s in similar_setups if s.get('success', False)) / len(similar_setups)
            context_parts.append(
                f"Similar technical setups occurred {len(similar_setups)} times with {success_rate:.1%} success rate"
            )
            
            # Add specific failure lessons
            failures = [s for s in similar_setups if not s.get('success', False)]
            if failures:
                common_failure = failures[0].get('failure_analysis', '')
                if common_failure:
                    context_parts.append(f"Common failure pattern: {common_failure}")
        
        # Add recent performance trend
        if len(recent_predictions) >= 3:
            recent_returns = [p.get('actual_return', 0) for p in recent_predictions[-3:]]
            avg_return = np.mean(recent_returns)
            context_parts.append(f"Recent average return: {avg_return:.2f}%")
        
        return " | ".join(context_parts)
    
    def _find_similar_setups(self, current: Dict, historical: List[Dict]) -> List[Dict]:
        """Find historically similar technical setups"""
        similar = []
        
        for hist in historical:
            if not hist.get('technical_context'):
                continue
                
            hist_tech = hist['technical_context']
            
            # Define similarity criteria
            rsi_similar = abs(current.get('rsi', 50) - hist_tech.get('rsi', 50)) < 10
            macd_similar_sign = (current.get('macd_histogram', 0) * hist_tech.get('macd_histogram', 0)) > 0
            volume_similar = abs(current.get('volume_ratio', 1) - hist_tech.get('volume_ratio', 1)) < 0.3
            
            if rsi_similar and macd_similar_sign and volume_similar:
                similar.append(hist)
                
        return similar[-5:]  # Return last 5 similar setups
    
    def update_learning_history(self, outcome: PredictionOutcome):
        """Update learning history with new outcome"""
        history = self.load_learning_history()
        
        # Add new prediction outcome
        history['predictions'].append(asdict(outcome))
        
        # Update ticker accuracy
        ticker = outcome.ticker
        if ticker not in history['ticker_accuracy']:
            history['ticker_accuracy'][ticker] = {'correct': 0, 'total': 0}
        
        history['ticker_accuracy'][ticker]['total'] += 1
        if outcome.success:
            history['ticker_accuracy'][ticker]['correct'] += 1
        
        # Save back to GitHub
        self._save_to_github(history)
        
    def _save_to_github(self, data: Dict):
        """Save learning history to GitHub"""
        from github import Github
        g = Github(self.github_token)
        repo = g.get_repo(self.repo_name)
        
        content = json.dumps(data, indent=2)
        
        try:
            # Try to update existing file
            file = repo.get_contents(self.learning_file)
            repo.update_file(
                self.learning_file,
                f"Update learning history - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                content,
                file.sha
            )
        except:
            # Create new file
            repo.create_file(
                self.learning_file,
                f"Initialize learning history - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                content
            )
    
    def get_pattern_recommendations(self, ticker: str, current_technicals: Dict) -> Dict:
        """Get recommendations based on learned patterns"""
        history = self.load_learning_history()
        patterns = history.get('patterns', {})
        
        recommendations = {
            'confidence_adjustment': 1.0,
            'warning_flags': [],
            'positive_signals': [],
            'suggested_action': None
        }
        
        # Check for known failure patterns
        for pattern_name, pattern_data in patterns.items():
            if self._matches_pattern(current_technicals, pattern_data['conditions']):
                if pattern_data['success_rate'] < 0.3:
                    recommendations['warning_flags'].append(
                        f"{pattern_name}: {pattern_data['success_rate']:.1%} success rate"
                    )
                    recommendations['confidence_adjustment'] *= 0.7
                elif pattern_data['success_rate'] > 0.7:
                    recommendations['positive_signals'].append(
                        f"{pattern_name}: {pattern_data['success_rate']:.1%} success rate"
                    )
                    recommendations['confidence_adjustment'] *= 1.3
        
        return recommendations
    
    def _matches_pattern(self, current: Dict, conditions: Dict) -> bool:
        """Check if current technicals match a pattern"""
        for key, value in conditions.items():
            if key not in current:
                return False
            
            if isinstance(value, dict):
                # Range check
                if 'min' in value and current[key] < value['min']:
                    return False
                if 'max' in value and current[key] > value['max']:
                    return False
            else:
                # Direct comparison
                if current[key] != value:
                    return False
        
        return True
