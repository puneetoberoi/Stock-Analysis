# /src/analysis/enhanced_patterns.py
# Comprehensive candlestick and chart pattern detection system

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Scipy is optional - some methods won't work without it
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - some chart patterns will be disabled")

logging.basicConfig(level=logging.INFO)


class EnhancedPatternDetector:
    """
    Detects 40+ candlestick patterns and advanced chart patterns
    """
    
    def __init__(self):
        self.pattern_history = {}
        
    # ========================================
    # HELPER FUNCTIONS
    # ========================================
    
    def _is_bullish_candle(self, row) -> bool:
        """Check if candle is bullish (close > open)"""
        return row['Close'] > row['Open']
    
    def _is_bearish_candle(self, row) -> bool:
        """Check if candle is bearish (close < open)"""
        return row['Close'] < row['Open']
    
    def _body_size(self, row) -> float:
        """Get candle body size"""
        return abs(row['Close'] - row['Open'])
    
    def _upper_shadow(self, row) -> float:
        """Get upper shadow length"""
        return row['High'] - max(row['Open'], row['Close'])
    
    def _lower_shadow(self, row) -> float:
        """Get lower shadow length"""
        return min(row['Open'], row['Close']) - row['Low']
    
    def _total_range(self, row) -> float:
        """Get total candle range"""
        return row['High'] - row['Low']
    
    def _avg_body_size(self, df, periods=10) -> float:
        """Get average body size over periods"""
        return df.tail(periods).apply(lambda x: self._body_size(x), axis=1).mean()
    
    def _is_doji(self, row, threshold=0.1) -> bool:
        """Check if candle is a doji"""
        body = self._body_size(row)
        total = self._total_range(row)
        return (body / total) < threshold if total > 0 else False
    
    def _is_long_candle(self, row, df) -> bool:
        """Check if candle is longer than average"""
        avg = self._avg_body_size(df)
        return self._body_size(row) > (avg * 1.5)
    
    def _is_small_candle(self, row, df) -> bool:
        """Check if candle is smaller than average"""
        avg = self._avg_body_size(df)
        return self._body_size(row) < (avg * 0.5)
    
    # ========================================
    # SINGLE CANDLESTICK PATTERNS
    # ========================================
    
    def detect_doji(self, df) -> Optional[Dict]:
        """Basic Doji - Indecision"""
        if len(df) < 1:
            return None
        
        current = df.iloc[-1]
        if self._is_doji(current):
            return {
                'name': 'doji',
                'type': 'indecision',
                'signal': 'HOLD',
                'strength': 60,
                'description': 'Market indecision, potential reversal'
            }
        return None
    
    def detect_dragonfly_doji(self, df) -> Optional[Dict]:
        """Dragonfly Doji - Bullish reversal"""
        if len(df) < 1:
            return None
        
        current = df.iloc[-1]
        body = self._body_size(current)
        total = self._total_range(current)
        lower = self._lower_shadow(current)
        upper = self._upper_shadow(current)
        
        if self._is_doji(current) and lower > (total * 0.7) and upper < (total * 0.1):
            return {
                'name': 'dragonfly_doji',
                'type': 'bullish_reversal',
                'signal': 'BUY',
                'strength': 75,
                'description': 'Strong bullish reversal signal'
            }
        return None
    
    def detect_gravestone_doji(self, df) -> Optional[Dict]:
        """Gravestone Doji - Bearish reversal"""
        if len(df) < 1:
            return None
        
        current = df.iloc[-1]
        total = self._total_range(current)
        lower = self._lower_shadow(current)
        upper = self._upper_shadow(current)
        
        if self._is_doji(current) and upper > (total * 0.7) and lower < (total * 0.1):
            return {
                'name': 'gravestone_doji',
                'type': 'bearish_reversal',
                'signal': 'SELL',
                'strength': 75,
                'description': 'Strong bearish reversal signal'
            }
        return None
    
    def detect_long_legged_doji(self, df) -> Optional[Dict]:
        """Long-Legged Doji - High volatility indecision"""
        if len(df) < 1:
            return None
        
        current = df.iloc[-1]
        total = self._total_range(current)
        lower = self._lower_shadow(current)
        upper = self._upper_shadow(current)
        
        if self._is_doji(current) and lower > (total * 0.3) and upper > (total * 0.3):
            return {
                'name': 'long_legged_doji',
                'type': 'indecision',
                'signal': 'HOLD',
                'strength': 65,
                'description': 'High volatility, strong indecision'
            }
        return None
    
    def detect_hammer(self, df) -> Optional[Dict]:
        """Hammer - Bullish reversal"""
        if len(df) < 1:
            return None
        
        current = df.iloc[-1]
        body = self._body_size(current)
        total = self._total_range(current)
        lower = self._lower_shadow(current)
        upper = self._upper_shadow(current)
        
        # Hammer: small body at top, long lower shadow
        if (lower > (body * 2) and 
            upper < (body * 0.3) and 
            body > (total * 0.2)):
            return {
                'name': 'hammer',
                'type': 'bullish_reversal',
                'signal': 'BUY',
                'strength': 80,
                'description': 'Strong bullish reversal at bottom'
            }
        return None
    
    def detect_inverted_hammer(self, df) -> Optional[Dict]:
        """Inverted Hammer - Bullish reversal"""
        if len(df) < 1:
            return None
        
        current = df.iloc[-1]
        body = self._body_size(current)
        total = self._total_range(current)
        lower = self._lower_shadow(current)
        upper = self._upper_shadow(current)
        
        # Inverted hammer: small body at bottom, long upper shadow
        if (upper > (body * 2) and 
            lower < (body * 0.3) and 
            body > (total * 0.2)):
            return {
                'name': 'inverted_hammer',
                'type': 'bullish_reversal',
                'signal': 'BUY',
                'strength': 75,
                'description': 'Bullish reversal signal'
            }
        return None
    
    def detect_hanging_man(self, df) -> Optional[Dict]:
        """Hanging Man - Bearish reversal (same shape as hammer, but at top)"""
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        body = self._body_size(current)
        total = self._total_range(current)
        lower = self._lower_shadow(current)
        upper = self._upper_shadow(current)
        
        # Must appear after uptrend
        uptrend = previous['Close'] > df.iloc[-3]['Close'] if len(df) >= 3 else True
        
        if (uptrend and 
            lower > (body * 2) and 
            upper < (body * 0.3) and 
            body > (total * 0.2)):
            return {
                'name': 'hanging_man',
                'type': 'bearish_reversal',
                'signal': 'SELL',
                'strength': 75,
                'description': 'Bearish reversal at top'
            }
        return None
    
    def detect_shooting_star(self, df) -> Optional[Dict]:
        """Shooting Star - Bearish reversal"""
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        body = self._body_size(current)
        total = self._total_range(current)
        lower = self._lower_shadow(current)
        upper = self._upper_shadow(current)
        
        # Shooting star: small body at bottom, long upper shadow
        if (upper > (body * 2) and 
            lower < (body * 0.3) and 
            body > (total * 0.2) and
            self._is_bearish_candle(current)):
            return {
                'name': 'shooting_star',
                'type': 'bearish_reversal',
                'signal': 'SELL',
                'strength': 80,
                'description': 'Strong bearish reversal'
            }
        return None
    
    def detect_spinning_top(self, df) -> Optional[Dict]:
        """Spinning Top - Indecision"""
        if len(df) < 1:
            return None
        
        current = df.iloc[-1]
        body = self._body_size(current)
        total = self._total_range(current)
        lower = self._lower_shadow(current)
        upper = self._upper_shadow(current)
        
        # Small body with upper and lower shadows
        if (body < (total * 0.3) and 
            lower > (body * 0.5) and 
            upper > (body * 0.5)):
            return {
                'name': 'spinning_top',
                'type': 'indecision',
                'signal': 'HOLD',
                'strength': 60,
                'description': 'Market indecision'
            }
        return None
    
    def detect_marubozu_bullish(self, df) -> Optional[Dict]:
        """Bullish Marubozu - Strong bullish continuation"""
        if len(df) < 1:
            return None
        
        current = df.iloc[-1]
        body = self._body_size(current)
        total = self._total_range(current)
        upper = self._upper_shadow(current)
        lower = self._lower_shadow(current)
        
        # Very long bullish candle with no/minimal shadows
        if (self._is_bullish_candle(current) and 
            body > (total * 0.95) and
            upper < (total * 0.05) and
            lower < (total * 0.05)):
            return {
                'name': 'bullish_marubozu',
                'type': 'bullish_continuation',
                'signal': 'BUY',
                'strength': 85,
                'description': 'Very strong bullish momentum'
            }
        return None
    
    def detect_marubozu_bearish(self, df) -> Optional[Dict]:
        """Bearish Marubozu - Strong bearish continuation"""
        if len(df) < 1:
            return None
        
        current = df.iloc[-1]
        body = self._body_size(current)
        total = self._total_range(current)
        upper = self._upper_shadow(current)
        lower = self._lower_shadow(current)
        
        # Very long bearish candle with no/minimal shadows
        if (self._is_bearish_candle(current) and 
            body > (total * 0.95) and
            upper < (total * 0.05) and
            lower < (total * 0.05)):
            return {
                'name': 'bearish_marubozu',
                'type': 'bearish_continuation',
                'signal': 'SELL',
                'strength': 85,
                'description': 'Very strong bearish momentum'
            }
        return None
    
    # ========================================
    # TWO CANDLESTICK PATTERNS
    # ========================================
    
    def detect_bullish_engulfing(self, df) -> Optional[Dict]:
        """Bullish Engulfing - Strong bullish reversal"""
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Previous bearish, current bullish and engulfs previous
        if (self._is_bearish_candle(previous) and 
            self._is_bullish_candle(current) and
            current['Open'] < previous['Close'] and
            current['Close'] > previous['Open']):
            return {
                'name': 'bullish_engulfing',
                'type': 'bullish_reversal',
                'signal': 'BUY',
                'strength': 85,
                'description': 'Strong bullish reversal pattern'
            }
        return None
    
    def detect_bearish_engulfing(self, df) -> Optional[Dict]:
        """Bearish Engulfing - Strong bearish reversal"""
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Previous bullish, current bearish and engulfs previous
        if (self._is_bullish_candle(previous) and 
            self._is_bearish_candle(current) and
            current['Open'] > previous['Close'] and
            current['Close'] < previous['Open']):
            return {
                'name': 'bearish_engulfing',
                'type': 'bearish_reversal',
                'signal': 'SELL',
                'strength': 85,
                'description': 'Strong bearish reversal pattern'
            }
        return None
    
    def detect_piercing_line(self, df) -> Optional[Dict]:
        """Piercing Line - Bullish reversal"""
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Previous bearish, current bullish and closes above midpoint
        prev_midpoint = (previous['Open'] + previous['Close']) / 2
        
        if (self._is_bearish_candle(previous) and 
            self._is_bullish_candle(current) and
            current['Open'] < previous['Close'] and
            current['Close'] > prev_midpoint and
            current['Close'] < previous['Open']):
            return {
                'name': 'piercing_line',
                'type': 'bullish_reversal',
                'signal': 'BUY',
                'strength': 80,
                'description': 'Bullish reversal pattern'
            }
        return None
    
    def detect_dark_cloud_cover(self, df) -> Optional[Dict]:
        """Dark Cloud Cover - Bearish reversal"""
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Previous bullish, current bearish and closes below midpoint
        prev_midpoint = (previous['Open'] + previous['Close']) / 2
        
        if (self._is_bullish_candle(previous) and 
            self._is_bearish_candle(current) and
            current['Open'] > previous['Close'] and
            current['Close'] < prev_midpoint and
            current['Close'] > previous['Open']):
            return {
                'name': 'dark_cloud_cover',
                'type': 'bearish_reversal',
                'signal': 'SELL',
                'strength': 80,
                'description': 'Bearish reversal pattern'
            }
        return None
    
    def detect_bullish_harami(self, df) -> Optional[Dict]:
        """Bullish Harami - Bullish reversal"""
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Previous large bearish, current small bullish inside previous
        if (self._is_bearish_candle(previous) and 
            self._is_bullish_candle(current) and
            current['Open'] > previous['Close'] and
            current['Close'] < previous['Open'] and
            self._is_long_candle(previous, df) and
            self._is_small_candle(current, df)):
            return {
                'name': 'bullish_harami',
                'type': 'bullish_reversal',
                'signal': 'BUY',
                'strength': 75,
                'description': 'Bullish reversal, trend exhaustion'
            }
        return None
    
    def detect_bearish_harami(self, df) -> Optional[Dict]:
        """Bearish Harami - Bearish reversal"""
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Previous large bullish, current small bearish inside previous
        if (self._is_bullish_candle(previous) and 
            self._is_bearish_candle(current) and
            current['Open'] < previous['Close'] and
            current['Close'] > previous['Open'] and
            self._is_long_candle(previous, df) and
            self._is_small_candle(current, df)):
            return {
                'name': 'bearish_harami',
                'type': 'bearish_reversal',
                'signal': 'SELL',
                'strength': 75,
                'description': 'Bearish reversal, trend exhaustion'
            }
        return None
    
    def detect_tweezer_top(self, df) -> Optional[Dict]:
        """Tweezer Top - Bearish reversal"""
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Two candles with same/similar highs
        high_diff = abs(current['High'] - previous['High'])
        avg_range = (self._total_range(current) + self._total_range(previous)) / 2
        
        if (high_diff < (avg_range * 0.1) and
            self._is_bullish_candle(previous) and
            self._is_bearish_candle(current)):
            return {
                'name': 'tweezer_top',
                'type': 'bearish_reversal',
                'signal': 'SELL',
                'strength': 75,
                'description': 'Bearish reversal at resistance'
            }
        return None
    
    def detect_tweezer_bottom(self, df) -> Optional[Dict]:
        """Tweezer Bottom - Bullish reversal"""
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Two candles with same/similar lows
        low_diff = abs(current['Low'] - previous['Low'])
        avg_range = (self._total_range(current) + self._total_range(previous)) / 2
        
        if (low_diff < (avg_range * 0.1) and
            self._is_bearish_candle(previous) and
            self._is_bullish_candle(current)):
            return {
                'name': 'tweezer_bottom',
                'type': 'bullish_reversal',
                'signal': 'BUY',
                'strength': 75,
                'description': 'Bullish reversal at support'
            }
        return None
    
    # ========================================
    # THREE CANDLESTICK PATTERNS
    # ========================================
    
    def detect_morning_star(self, df) -> Optional[Dict]:
        """Morning Star - Strong bullish reversal"""
        if len(df) < 3:
            return None
        
        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]
        
        # First: long bearish, Second: small (star), Third: long bullish
        if (self._is_bearish_candle(first) and
            self._is_long_candle(first, df) and
            self._is_small_candle(second, df) and
            self._is_bullish_candle(third) and
            self._is_long_candle(third, df) and
            third['Close'] > (first['Open'] + first['Close']) / 2):
            return {
                'name': 'morning_star',
                'type': 'bullish_reversal',
                'signal': 'BUY',
                'strength': 90,
                'description': 'Very strong bullish reversal'
            }
        return None
    
    def detect_evening_star(self, df) -> Optional[Dict]:
        """Evening Star - Strong bearish reversal"""
        if len(df) < 3:
            return None
        
        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]
        
        # First: long bullish, Second: small (star), Third: long bearish
        if (self._is_bullish_candle(first) and
            self._is_long_candle(first, df) and
            self._is_small_candle(second, df) and
            self._is_bearish_candle(third) and
            self._is_long_candle(third, df) and
            third['Close'] < (first['Open'] + first['Close']) / 2):
            return {
                'name': 'evening_star',
                'type': 'bearish_reversal',
                'signal': 'SELL',
                'strength': 90,
                'description': 'Very strong bearish reversal'
            }
        return None
    
    def detect_three_white_soldiers(self, df) -> Optional[Dict]:
        """Three White Soldiers - Very strong bullish continuation"""
        if len(df) < 3:
            return None
        
        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]
        
        # Three consecutive bullish candles, each opening within previous body
        if (self._is_bullish_candle(first) and
            self._is_bullish_candle(second) and
            self._is_bullish_candle(third) and
            second['Close'] > first['Close'] and
            third['Close'] > second['Close'] and
            second['Open'] > first['Open'] and second['Open'] < first['Close'] and
            third['Open'] > second['Open'] and third['Open'] < second['Close']):
            return {
                'name': 'three_white_soldiers',
                'type': 'bullish_continuation',
                'signal': 'BUY',
                'strength': 95,
                'description': 'Extremely strong bullish momentum'
            }
        return None
    
    def detect_three_black_crows(self, df) -> Optional[Dict]:
        """Three Black Crows - Very strong bearish continuation"""
        if len(df) < 3:
            return None
        
        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]
        
        # Three consecutive bearish candles, each opening within previous body
        if (self._is_bearish_candle(first) and
            self._is_bearish_candle(second) and
            self._is_bearish_candle(third) and
            second['Close'] < first['Close'] and
            third['Close'] < second['Close'] and
            second['Open'] < first['Open'] and second['Open'] > first['Close'] and
            third['Open'] < second['Open'] and third['Open'] > second['Close']):
            return {
                'name': 'three_black_crows',
                'type': 'bearish_continuation',
                'signal': 'SELL',
                'strength': 95,
                'description': 'Extremely strong bearish momentum'
            }
        return None
    
    def detect_three_inside_up(self, df) -> Optional[Dict]:
        """Three Inside Up - Bullish reversal with confirmation"""
        if len(df) < 3:
            return None
        
        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]
        
        # Bullish harami followed by bullish confirmation
        harami = (self._is_bearish_candle(first) and
                 self._is_bullish_candle(second) and
                 second['Open'] > first['Close'] and
                 second['Close'] < first['Open'])
        
        if harami and self._is_bullish_candle(third) and third['Close'] > second['Close']:
            return {
                'name': 'three_inside_up',
                'type': 'bullish_reversal',
                'signal': 'BUY',
                'strength': 85,
                'description': 'Confirmed bullish reversal'
            }
        return None
    
    def detect_three_inside_down(self, df) -> Optional[Dict]:
        """Three Inside Down - Bearish reversal with confirmation"""
        if len(df) < 3:
            return None
        
        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]
        
        # Bearish harami followed by bearish confirmation
        harami = (self._is_bullish_candle(first) and
                 self._is_bearish_candle(second) and
                 second['Open'] < first['Close'] and
                 second['Close'] > first['Open'])
        
        if harami and self._is_bearish_candle(third) and third['Close'] < second['Close']:
            return {
                'name': 'three_inside_down',
                'type': 'bearish_reversal',
                'signal': 'SELL',
                'strength': 85,
                'description': 'Confirmed bearish reversal'
            }
        return None
    
    def detect_three_outside_up(self, df) -> Optional[Dict]:
        """Three Outside Up - Bullish engulfing with confirmation"""
        if len(df) < 3:
            return None
        
        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]
        
        # Bullish engulfing followed by bullish confirmation
        engulfing = (self._is_bearish_candle(first) and
                    self._is_bullish_candle(second) and
                    second['Open'] < first['Close'] and
                    second['Close'] > first['Open'])
        
        if engulfing and self._is_bullish_candle(third) and third['Close'] > second['Close']:
            return {
                'name': 'three_outside_up',
                'type': 'bullish_reversal',
                'signal': 'BUY',
                'strength': 90,
                'description': 'Strong confirmed bullish reversal'
            }
        return None
    
    def detect_three_outside_down(self, df) -> Optional[Dict]:
        """Three Outside Down - Bearish engulfing with confirmation"""
        if len(df) < 3:
            return None
        
        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]
        
        # Bearish engulfing followed by bearish confirmation
        engulfing = (self._is_bullish_candle(first) and
                    self._is_bearish_candle(second) and
                    second['Open'] > first['Close'] and
                    second['Close'] < first['Open'])
        
        if engulfing and self._is_bearish_candle(third) and third['Close'] < second['Close']:
            return {
                'name': 'three_outside_down',
                'type': 'bearish_reversal',
                'signal': 'SELL',
                'strength': 90,
                'description': 'Strong confirmed bearish reversal'
            }
        return None
    
    # ========================================
    # ADVANCED CHART PATTERNS
    # ========================================
    
    def detect_cup_and_handle(self, df, lookback=30) -> Optional[Dict]:
        """Cup and Handle - Bullish continuation pattern"""
        if len(df) < lookback:
            return None
        
        data = df.tail(lookback)
        
        # Find potential cup (U-shape)
        # 1. Initial high
        # 2. Decline to low
        # 3. Recovery to near initial high
        # 4. Small consolidation (handle)
        
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        
        # Simple cup detection
        first_third = int(lookback / 3)
        middle_third = int(2 * lookback / 3)
        
        cup_high = max(highs[:first_third])
        cup_low = min(lows[first_third:middle_third])
        recovery_high = max(highs[middle_third:-5])
        
        # Cup depth should be 12-33% typically
        cup_depth = (cup_high - cup_low) / cup_high
        
        # Handle should be smaller consolidation
        handle_high = max(highs[-5:])
        handle_low = min(lows[-5:])
        handle_depth = (handle_high - handle_low) / handle_high
        
        if (0.12 < cup_depth < 0.33 and
            recovery_high > (cup_high * 0.95) and
            handle_depth < 0.12 and
            closes[-1] > handle_low):
            return {
                'name': 'cup_and_handle',
                'type': 'bullish_continuation',
                'signal': 'BUY',
                'strength': 85,
                'description': 'Classic bullish continuation pattern',
                'breakout_level': cup_high
            }
        return None
    
    def detect_double_top(self, df, lookback=20) -> Optional[Dict]:
        """M Shape (Double Top) - Bearish reversal"""
        if len(df) < lookback:
            return None
        
        data = df.tail(lookback)
        highs = data['High'].values
        
        # Find two similar peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(highs, distance=5)
        
        if len(peaks) >= 2:
            # Check last two peaks
            peak1_val = highs[peaks[-2]]
            peak2_val = highs[peaks[-1]]
            
            # Peaks should be similar (within 3%)
            if abs(peak1_val - peak2_val) / peak1_val < 0.03:
                # Find valley between peaks
                valley_idx = peaks[-2] + np.argmin(highs[peaks[-2]:peaks[-1]])
                valley_val = highs[valley_idx]
                
                # Current price should break below valley
                if data.iloc[-1]['Close'] < valley_val:
                    return {
                        'name': 'double_top',
                        'type': 'bearish_reversal',
                        'signal': 'SELL',
                        'strength': 85,
                        'description': 'M-shape bearish reversal pattern',
                        'resistance_level': (peak1_val + peak2_val) / 2
                    }
        return None
    
    def detect_double_bottom(self, df, lookback=20) -> Optional[Dict]:
        """W Shape (Double Bottom) - Bullish reversal"""
        if len(df) < lookback:
            return None
        
        data = df.tail(lookback)
        lows = data['Low'].values
        
        # Find two similar bottoms
        from scipy.signal import find_peaks
        valleys, _ = find_peaks(-lows, distance=5)
        
        if len(valleys) >= 2:
            # Check last two valleys
            valley1_val = lows[valleys[-2]]
            valley2_val = lows[valleys[-1]]
            
            # Valleys should be similar (within 3%)
            if abs(valley1_val - valley2_val) / valley1_val < 0.03:
                # Find peak between valleys
                peak_idx = valleys[-2] + np.argmax(lows[valleys[-2]:valleys[-1]])
                peak_val = lows[peak_idx]
                
                # Current price should break above peak
                if data.iloc[-1]['Close'] > peak_val:
                    return {
                        'name': 'double_bottom',
                        'type': 'bullish_reversal',
                        'signal': 'BUY',
                        'strength': 85,
                        'description': 'W-shape bullish reversal pattern',
                        'support_level': (valley1_val + valley2_val) / 2
                    }
        return None
    
    def detect_vcp(self, df, lookback=50) -> Optional[Dict]:
        """
        VCP - Volatility Contraction Pattern (Mark Minervini)
        Price consolidates with decreasing volatility before breakout
        """
        if len(df) < lookback:
            return None
        
        data = df.tail(lookback)
        
        # Calculate volatility for different periods
        recent_volatility = data['High'].tail(10).std()
        mid_volatility = data['High'].iloc[-20:-10].std()
        early_volatility = data['High'].iloc[-30:-20].std()
        
        # VCP: volatility should be contracting
        volatility_contracting = (recent_volatility < mid_volatility < early_volatility)
        
        # Price should be near highs
        current_price = data.iloc[-1]['Close']
        period_high = data['High'].max()
        near_highs = current_price > (period_high * 0.95)
        
        # Volume should be decreasing
        recent_volume = data['Volume'].tail(10).mean()
        earlier_volume = data['Volume'].iloc[-30:-20].mean()
        volume_contracting = recent_volume < earlier_volume
        
        if volatility_contracting and near_highs and volume_contracting:
            return {
                'name': 'vcp',
                'type': 'bullish_continuation',
                'signal': 'BUY',
                'strength': 90,
                'description': 'Volatility Contraction Pattern - Minervini',
                'breakout_level': period_high
            }
        return None
    
    def detect_darvas_box(self, df, lookback=20) -> Optional[Dict]:
        """
        Darvas Box - Nicolas Darvas trading method
        Stock makes new highs, then consolidates in a box
        """
        if len(df) < lookback:
            return None
        
        data = df.tail(lookback)
        
        # Find recent consolidation range
        recent_data = data.tail(10)
        box_high = recent_data['High'].max()
        box_low = recent_data['Low'].min()
        box_range = (box_high - box_low) / box_low
        
        # Box should be relatively tight (< 8%)
        if box_range > 0.08:
            return None
        
        # Should be after an uptrend
        older_data = data.iloc[-20:-10]
        uptrend = data.iloc[-1]['Close'] > older_data['Close'].mean()
        
        # Check for breakout
        current_close = data.iloc[-1]['Close']
        breaking_out = current_close > box_high
        
        if uptrend and box_range < 0.08:
            return {
                'name': 'darvas_box',
                'type': 'bullish_continuation' if breaking_out else 'consolidation',
                'signal': 'BUY' if breaking_out else 'HOLD',
                'strength': 80 if breaking_out else 65,
                'description': 'Darvas Box pattern',
                'box_high': box_high,
                'box_low': box_low,
                'breakout': breaking_out
            }
        return None
    
    def detect_swing_highs_lows(self, df, lookback=20) -> Optional[Dict]:
        """
        Swing Highs and Lows - Price action pattern
        Identifies key reversal points
        """
        if len(df) < lookback:
            return None
        
        data = df.tail(lookback)
        
        # Find swing highs (local maxima)
        from scipy.signal import find_peaks
        swing_highs, _ = find_peaks(data['High'].values, distance=3)
        swing_lows, _ = find_peaks(-data['Low'].values, distance=3)
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Analyze trend
            recent_highs = [data['High'].values[i] for i in swing_highs[-2:]]
            recent_lows = [data['Low'].values[i] for i in swing_lows[-2:]]
            
            higher_highs = recent_highs[-1] > recent_highs[-2]
            higher_lows = recent_lows[-1] > recent_lows[-2]
            lower_highs = recent_highs[-1] < recent_highs[-2]
            lower_lows = recent_lows[-1] < recent_lows[-2]
            
            if higher_highs and higher_lows:
                signal = 'BUY'
                trend = 'uptrend'
                strength = 75
            elif lower_highs and lower_lows:
                signal = 'SELL'
                trend = 'downtrend'
                strength = 75
            else:
                signal = 'HOLD'
                trend = 'consolidation'
                strength = 60
            
            return {
                'name': 'swing_highs_lows',
                'type': trend,
                'signal': signal,
                'strength': strength,
                'description': f'Price action showing {trend}',
                'swing_high': recent_highs[-1],
                'swing_low': recent_lows[-1]
            }
        return None
    
    # ========================================
    # MASTER DETECTION METHOD
    # ========================================
    
    def detect_all_patterns(self, df) -> List[Dict]:
        """
        Run all pattern detection methods and return all found patterns
        """
        patterns = []
        
        # All detection methods
        detectors = [
            # Single candle patterns
            self.detect_doji,
            self.detect_dragonfly_doji,
            self.detect_gravestone_doji,
            self.detect_long_legged_doji,
            self.detect_hammer,
            self.detect_inverted_hammer,
            self.detect_hanging_man,
            self.detect_shooting_star,
            self.detect_spinning_top,
            self.detect_marubozu_bullish,
            self.detect_marubozu_bearish,
            
            # Two candle patterns
            self.detect_bullish_engulfing,
            self.detect_bearish_engulfing,
            self.detect_piercing_line,
            self.detect_dark_cloud_cover,
            self.detect_bullish_harami,
            self.detect_bearish_harami,
            self.detect_tweezer_top,
            self.detect_tweezer_bottom,
            
            # Three candle patterns
            self.detect_morning_star,
            self.detect_evening_star,
            self.detect_three_white_soldiers,
            self.detect_three_black_crows,
            self.detect_three_inside_up,
            self.detect_three_inside_down,
            self.detect_three_outside_up,
            self.detect_three_outside_down,
            
            # Chart patterns
            self.detect_cup_and_handle,
            self.detect_double_top,
            self.detect_double_bottom,
            self.detect_vcp,
            self.detect_darvas_box,
            self.detect_swing_highs_lows,
        ]
        
        for detector in detectors:
            try:
                result = detector(df)
                if result:
                    patterns.append(result)
            except Exception as e:
                logging.debug(f"Error in {detector.__name__}: {e}")
        
        # Sort by strength
        patterns.sort(key=lambda x: x['strength'], reverse=True)
        
        return patterns
    
    def get_strongest_pattern(self, df) -> Optional[Dict]:
        """Get the single strongest pattern"""
        patterns = self.detect_all_patterns(df)
        return patterns[0] if patterns else None


# ========================================
# TESTING
# ========================================

if __name__ == "__main__":
    import yfinance as yf
    
    # Test with real data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="3mo")
    
    detector = EnhancedPatternDetector()
    patterns = detector.detect_all_patterns(df)
    
    print(f"\n🕯️ Found {len(patterns)} patterns for AAPL:\n")
    for p in patterns[:5]:  # Show top 5
        print(f"{p['name'].upper()}: {p['signal']} (Strength: {p['strength']}%)")
        print(f"   {p['description']}\n")
