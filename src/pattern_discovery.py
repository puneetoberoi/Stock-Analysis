import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple

class PatternDiscovery:
    def __init__(self, learning_manager):
        self.learning_manager = learning_manager
        
    def discover_patterns(self, min_samples: int = 10) -> List[Dict]:
        """Discover patterns that lead to successful predictions"""
        
        history = self.learning_manager.load_learning_history()
        predictions = history.get('predictions', [])
        
        if len(predictions) < min_samples:
            return []
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(predictions)
        
        # Extract technical features
        features = []
        targets = []
        
        for _, row in df.iterrows():
            if 'technical_context' in row and row['technical_context']:
                tech = row['technical_context']
                features.append([
                    tech.get('rsi', 50),
                    tech.get('macd_histogram', 0),
                    tech.get('volume_ratio', 1),
                    1 if tech.get('near_support', False) else 0,
                    1 if tech.get('near_resistance', False) else 0,
                ])
                targets.append(1 if row['success'] else 0)
        
        if len(features) < min_samples:
            return []
        
        # Train pattern recognition model
        X = np.array(features)
        y = np.array(targets)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Extract important patterns
        feature_importance = model.feature_importances_
        feature_names = ['RSI', 'MACD_Hist', 'Volume_Ratio', 'Near_Support', 'Near_Resistance']
        
        patterns = []
        
        # Find successful pattern combinations
        for i in range(len(X)):
            if y[i] == 1:  # Successful prediction
                pattern = {
                    'conditions': {
                        name: {'min': X[i][j] - 0.1, 'max': X[i][j] + 0.1}
                        for j, name in enumerate(feature_names)
                        if feature_importance[j] > 0.15  # Only important features
                    },
                    'success_rate': self._calculate_pattern_success(X, y, X[i], feature_importance),
                    'sample_size': 1
                }
                patterns.append(pattern)
        
        # Cluster similar patterns
        return self._cluster_patterns(patterns)
    
    def _calculate_pattern_success(self, X, y, pattern, importance):
        """Calculate success rate of a specific pattern"""
        matches = []
        for i in range(len(X)):
            distance = np.sum(importance * np.abs(X[i] - pattern))
            if distance < 0.5:  # Similar pattern
                matches.append(y[i])
        
        if matches:
            return np.mean(matches)
        return 0.5
    
    def _cluster_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Cluster similar patterns together"""
        # Simplified clustering - in production, use proper clustering algorithm
        clustered = {}
        
        for pattern in patterns:
            key = str(sorted(pattern['conditions'].keys()))
            if key not in clustered:
                clustered[key] = {
                    'patterns': [],
                    'total_success': 0,
                    'count': 0
                }
            clustered[key]['patterns'].append(pattern)
            clustered[key]['total_success'] += pattern['success_rate']
            clustered[key]['count'] += 1
        
        result = []
        for key, cluster in clustered.items():
            if cluster['count'] >= 3:  # Minimum 3 similar patterns
                avg_success = cluster['total_success'] / cluster['count']
                result.append({
                    'pattern_name': f"Pattern_{key[:20]}",
                    'conditions': cluster['patterns'][0]['conditions'],
                    'success_rate': avg_success,
                    'sample_size': cluster['count']
                })
        
        return result
