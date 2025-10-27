# src/learning_manager.py
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, asdict
import numpy as np
from github import Github

@dataclass
class PredictionOutcome:
    ticker: str
    date: str
    prediction: str  # BUY/SELL/HOLD
    confidence: float
    actual_return: float
    success: bool
    technical_context: Dict
    failure_analysis: Optional[str] = None
    llm_model: str = "gpt-4"  # Track which LLM made the prediction

class LearningManager:
    def __init__(self, github_token: str, repo_name: str):
        self.github_token = github_token
        self.repo_name = repo_name
        self.learning_file = "data/learning_history.json"
        self.pattern_file = "data/learned_patterns.json"
        
    def load_learning_history(self) -> Dict:
        """Load historical predictions and their outcomes"""
        try:
            g = Github(self.github_token)
            repo = g.get_repo(self.repo_name)
            
            try:
                content = repo.get_contents(self.learning_file)
                return json.loads(content.decoded_content)
            except:
                return {
                    "predictions": [], 
                    "patterns": {}, 
                    "ticker_accuracy": {},
                    "llm_performance": {}
                }
        except Exception as e:
            print(f"Error loading learning history: {e}")
            return {
                "predictions": [], 
                "patterns": {}, 
                "ticker_accuracy": {},
                "llm_performance": {}
            }
    
    def analyze_failure_patterns(self, ticker: str, prediction: str, 
                               technical_data: Dict, actual_return: float) -> str:
        """Analyze why a prediction failed"""
        analysis = []
        
        # Check for false signals based on RSI
        rsi = technical_data.get('rsi', 50)
        macd = technical_data.get('macd', 0)
        volume_ratio = technical_data.get('volume_ratio', 1)
        
        if prediction == "BUY" and actual_return < -2:
            if rsi > 70:
                analysis.append("Failed BUY with RSI>70 (overbought) - reversal pattern ignored")
            if rsi > 60 and macd < 0:
                analysis.append("Failed BUY with negative MACD divergence - momentum wasn't confirmed")
            if volume_ratio < 0.8:
                analysis.append("Failed BUY with low volume - lack of buyer conviction")
                
        elif prediction == "SELL" and actual_return > 2:
            if rsi < 30:
                analysis.append("Failed SELL with RSI<30 (oversold) - bounce probability ignored")
            if rsi < 40 and macd > 0:
                analysis.append("Failed SELL with positive MACD - upward momentum building")
                
        elif prediction == "HOLD" and abs(actual_return) > 5:
            analysis.append(f"Failed HOLD - significant move of {actual_return:.2f}% missed")
            
        return " | ".join(analysis) if analysis else "No clear failure pattern identified"
    
    def get_ticker_learning_context(self, ticker: str, current_technicals: Dict) -> str:
        """Generate learning context for a specific ticker"""
        history = self.load_learning_history()
        
        # Filter predictions for this ticker
        ticker_predictions = [p for p in history.get('predictions', []) 
                            if p.get('ticker') == ticker]
        
        if not ticker_predictions:
            return "No historical data for learning."
        
        # Get last 10 predictions for this ticker
        recent_predictions = ticker_predictions[-10:]
        correct = sum(1 for p in recent_predictions if p.get('success', False))
        accuracy = (correct / len(recent_predictions)) * 100
        
        # Find similar technical setups
        similar_setups = self._find_similar_setups(current_technicals, ticker_predictions)
        
        context_parts = [
            f"Historical accuracy on {ticker}: {accuracy:.1f}% ({correct}/{len(recent_predictions)} correct)"
        ]
        
        # Add pattern-specific insights
        if similar_setups:
            similar_success = sum(1 for s in similar_setups if s.get('success', False))
            similar_accuracy = (similar_success / len(similar_setups)) * 100
            context_parts.append(
                f"Similar RSI/MACD setups: {len(similar_setups)} times, {similar_accuracy:.1f}% success"
            )
            
            # Get the most recent failure
            recent_failures = [s for s in similar_setups if not s.get('success', False)]
            if recent_failures:
                last_failure = recent_failures[-1]
                if last_failure.get('failure_analysis'):
                    context_parts.append(f"Last similar setup failed: {last_failure['failure_analysis']}")
        
        # Add recent trend
        if len(recent_predictions) >= 3:
            last_3_success = sum(1 for p in recent_predictions[-3:] if p.get('success', False))
            if last_3_success == 0:
                context_parts.append("WARNING: Last 3 predictions all failed")
            elif last_3_success == 3:
                context_parts.append("POSITIVE: Last 3 predictions all successful")
        
        return " | ".join(context_parts)
    
    def _find_similar_setups(self, current: Dict, historical: List[Dict]) -> List[Dict]:
        """Find historically similar technical setups"""
        similar = []
        
        current_rsi = current.get('rsi', 50)
        current_macd = current.get('macd', 0)
        
        for hist in historical:
            if not hist.get('technical_context'):
                continue
                
            hist_tech = hist['technical_context']
            hist_rsi = hist_tech.get('rsi', 50)
            hist_macd = hist_tech.get('macd', 0)
            
            # Check if RSI is in similar range (±10)
            rsi_similar = abs(current_rsi - hist_rsi) < 10
            
            # Check if MACD has same sign (both positive or both negative)
            macd_similar = (current_macd * hist_macd) > 0
            
            if rsi_similar and macd_similar:
                similar.append(hist)
        
        return similar[-5:] if similar else []  # Return up to 5 most recent similar setups
    
    def save_prediction_with_context(self, ticker: str, prediction: str, 
                                    confidence: float, technical_data: Dict, 
                                    current_price: float, llm_model: str = "gpt-4"):
        """Save a new prediction with technical context"""
        history = self.load_learning_history()
        
        new_prediction = {
            'ticker': ticker,
            'date': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'price_at_prediction': current_price,
            'technical_context': technical_data,
            'llm_model': llm_model,
            'success': None,  # Will be updated by evening learner
            'actual_return': None,  # Will be updated by evening learner
            'failure_analysis': None  # Will be updated if prediction fails
        }
        
        history['predictions'].append(new_prediction)
        self._save_to_github(history)
        
        return new_prediction
    
    def _save_to_github(self, data: Dict):
        """Save learning history to GitHub"""
        g = Github(self.github_token)
        repo = g.get_repo(self.repo_name)
        
        content = json.dumps(data, indent=2)
        
        try:
            file = repo.get_contents(self.learning_file)
            repo.update_file(
                self.learning_file,
                f"Update learning history - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                content,
                file.sha
            )
        except:
            repo.create_file(
                self.learning_file,
                f"Initialize learning history - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                content
            )
