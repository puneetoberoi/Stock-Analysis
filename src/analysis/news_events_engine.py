# /src/analysis/news_events_engine.py
# Detects and scores market-moving events for stocks

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)


class NewsEventsEngine:
    """
    Detects and scores market-moving events:
    - Earnings reports (beat/miss)
    - Corporate announcements (partnerships, products)
    - Analyst ratings (upgrades/downgrades)
    - News sentiment
    """
    
    def __init__(self):
        # API Keys
        self.finnhub_key = os.getenv('FINNHUB_API_KEY', '')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', '')
        
        # Event scoring weights
        self.event_weights = {
            'earnings_beat_large': 20,      # Beat by >10%
            'earnings_beat_small': 10,      # Beat by 5-10%
            'earnings_miss_large': -20,     # Miss by >10%
            'earnings_miss_small': -10,     # Miss by 5-10%
            'analyst_upgrade': 10,
            'analyst_downgrade': -10,
            'major_partnership': 15,
            'product_launch': 10,
            'positive_news': 8,
            'negative_news': -8,
            'insider_buying': 5,
            'insider_selling': -5
        }
        
        # Cache for API calls (avoid hitting rate limits)
        self.cache = {}
        
    def analyze_events(self, ticker: str, days_back: int = 7) -> Dict:
        """
        Analyze all events for a ticker in the last N days
        
        Returns:
            {
                'total_score': int,
                'events': [list of events],
                'earnings': {dict of earnings data},
                'news_sentiment': float,
                'analyst_activity': [list of ratings]
            }
        """
        logging.info(f"📰 Analyzing events for {ticker} (last {days_back} days)...")
        
        events = []
        total_score = 0
        
        # 1. Check earnings (most important)
        earnings_data = self._check_earnings(ticker, days_back)
        if earnings_data:
            events.append(earnings_data)
            total_score += earnings_data.get('score', 0)
        
        # 2. Check analyst ratings
        analyst_data = self._check_analyst_ratings(ticker, days_back)
        if analyst_data:
            events.extend(analyst_data)
            total_score += sum(a.get('score', 0) for a in analyst_data)
        
        # 3. Check news sentiment
        news_data = self._check_news_sentiment(ticker, days_back)
        if news_data:
            events.extend(news_data['events'])
            total_score += news_data.get('total_score', 0)
        
        # 4. Check corporate announcements
        announcements = self._check_corporate_announcements(ticker, days_back)
        if announcements:
            events.extend(announcements)
            total_score += sum(a.get('score', 0) for a in announcements)
        
        result = {
            'ticker': ticker,
            'total_score': total_score,
            'events': events,
            'event_count': len(events),
            'has_major_events': total_score >= 15 or total_score <= -15,
            'sentiment': 'bullish' if total_score > 5 else 'bearish' if total_score < -5 else 'neutral'
        }
        
        logging.info(f"✅ {ticker}: Found {len(events)} events, Score: {total_score:+d}")
        
        return result
    
    def _check_earnings(self, ticker: str, days_back: int) -> Optional[Dict]:
        """Check for recent earnings reports"""
        try:
            # Use Finnhub for earnings calendar
            if not self.finnhub_key:
                return None
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f"https://finnhub.io/api/v1/calendar/earnings"
            params = {
                'token': self.finnhub_key,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'symbol': ticker
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            earnings_reports = data.get('earningsCalendar', [])
            
            if not earnings_reports:
                return None
            
            # Get most recent earnings
            latest_earnings = earnings_reports[0]
            
            eps_actual = latest_earnings.get('epsActual')
            eps_estimate = latest_earnings.get('epsEstimate')
            
            if eps_actual is None or eps_estimate is None or eps_estimate == 0:
                return None
            
            # Calculate surprise percentage
            surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
            
            # Determine score
            if surprise_pct > 10:
                score = self.event_weights['earnings_beat_large']
                description = f"Earnings beat by {surprise_pct:.1f}%"
                event_type = 'earnings_beat_large'
            elif surprise_pct > 0:
                score = self.event_weights['earnings_beat_small']
                description = f"Earnings beat by {surprise_pct:.1f}%"
                event_type = 'earnings_beat_small'
            elif surprise_pct < -10:
                score = self.event_weights['earnings_miss_large']
                description = f"Earnings missed by {abs(surprise_pct):.1f}%"
                event_type = 'earnings_miss_large'
            else:
                score = self.event_weights['earnings_miss_small']
                description = f"Earnings missed by {abs(surprise_pct):.1f}%"
                event_type = 'earnings_miss_small'
            
            return {
                'type': event_type,
                'description': description,
                'score': score,
                'date': latest_earnings.get('date'),
                'emoji': '🎯' if score > 0 else '❌',
                'details': {
                    'eps_actual': eps_actual,
                    'eps_estimate': eps_estimate,
                    'surprise_pct': surprise_pct
                }
            }
            
        except Exception as e:
            logging.debug(f"Error checking earnings for {ticker}: {e}")
            return None
    
    def _check_analyst_ratings(self, ticker: str, days_back: int) -> List[Dict]:
        """Check for analyst upgrades/downgrades"""
        try:
            if not self.finnhub_key:
                return []
            
            # Get recommendation trends from Finnhub
            url = f"https://finnhub.io/api/v1/stock/recommendation"
            params = {
                'token': self.finnhub_key,
                'symbol': ticker
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return []
            
            recommendations = response.json()
            
            if not recommendations:
                return []
            
            # Get recent recommendations (last month)
            recent = recommendations[0] if recommendations else None
            
            if not recent:
                return []
            
            events = []
            
            # Count upgrades vs downgrades
            strong_buy = recent.get('strongBuy', 0)
            buy = recent.get('buy', 0)
            hold = recent.get('hold', 0)
            sell = recent.get('sell', 0)
            strong_sell = recent.get('strongSell', 0)
            
            bullish_count = strong_buy + buy
            bearish_count = sell + strong_sell
            
            # If significant analyst activity
            if bullish_count > bearish_count + 2:
                events.append({
                    'type': 'analyst_upgrade',
                    'description': f"{bullish_count} analyst upgrades",
                    'score': self.event_weights['analyst_upgrade'],
                    'date': recent.get('period'),
                    'emoji': '📊',
                    'details': {
                        'strong_buy': strong_buy,
                        'buy': buy,
                        'hold': hold
                    }
                })
            elif bearish_count > bullish_count + 2:
                events.append({
                    'type': 'analyst_downgrade',
                    'description': f"{bearish_count} analyst downgrades",
                    'score': self.event_weights['analyst_downgrade'],
                    'date': recent.get('period'),
                    'emoji': '📉',
                    'details': {
                        'sell': sell,
                        'strong_sell': strong_sell
                    }
                })
            
            return events
            
        except Exception as e:
            logging.debug(f"Error checking analyst ratings for {ticker}: {e}")
            return []
    
    def _check_news_sentiment(self, ticker: str, days_back: int) -> Optional[Dict]:
        """Check news sentiment using Finnhub"""
        try:
            if not self.finnhub_key:
                return None
            
            # Get company news from Finnhub
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'token': self.finnhub_key,
                'symbol': ticker,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return None
            
            news_items = response.json()
            
            if not news_items:
                return None
            
            # Analyze sentiment of headlines
            total_sentiment = 0
            positive_count = 0
            negative_count = 0
            events = []
            
            for item in news_items[:10]:  # Analyze last 10 articles
                headline = item.get('headline', '').lower()
                
                # Simple keyword-based sentiment
                positive_words = ['beat', 'surge', 'rally', 'gain', 'upgrade', 'partnership', 
                                'launch', 'growth', 'profit', 'strong', 'positive', 'up']
                negative_words = ['miss', 'fall', 'drop', 'loss', 'downgrade', 'concern',
                                'weak', 'decline', 'negative', 'down', 'cut', 'warning']
                
                sentiment_score = 0
                for word in positive_words:
                    if word in headline:
                        sentiment_score += 1
                        positive_count += 1
                
                for word in negative_words:
                    if word in headline:
                        sentiment_score -= 1
                        negative_count += 1
                
                total_sentiment += sentiment_score
            
            # Calculate overall news score
            if total_sentiment > 3:
                score = self.event_weights['positive_news']
                description = f"Positive news sentiment ({positive_count} bullish articles)"
                event_type = 'positive_news'
                emoji = '📰'
            elif total_sentiment < -3:
                score = self.event_weights['negative_news']
                description = f"Negative news sentiment ({negative_count} bearish articles)"
                event_type = 'negative_news'
                emoji = '⚠️'
            else:
                return None  # Neutral news
            
            events.append({
                'type': event_type,
                'description': description,
                'score': score,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'emoji': emoji,
                'details': {
                    'articles_analyzed': len(news_items[:10]),
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'top_headline': news_items[0].get('headline') if news_items else None
                }
            })
            
            return {
                'events': events,
                'total_score': score,
                'sentiment': total_sentiment
            }
            
        except Exception as e:
            logging.debug(f"Error checking news sentiment for {ticker}: {e}")
            return None
    
    def _check_corporate_announcements(self, ticker: str, days_back: int) -> List[Dict]:
        """Check for major corporate announcements (partnerships, acquisitions, etc.)"""
        try:
            if not self.finnhub_key:
                return []
            
            # Get company news
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'token': self.finnhub_key,
                'symbol': ticker,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return []
            
            news_items = response.json()
            events = []
            
            # Look for specific announcement types
            for item in news_items[:20]:  # Check last 20 articles
                headline = item.get('headline', '').lower()
                summary = item.get('summary', '').lower()
                combined_text = f"{headline} {summary}"
                
                # Partnership detection
                if any(word in combined_text for word in ['partnership', 'partners with', 'collaboration', 'joint venture']):
                    # Check if it's with a major company
                    major_companies = ['google', 'amazon', 'microsoft', 'apple', 'meta', 'tesla', 
                                      'nvidia', 'openai', 'volvo', 'ford', 'gm']
                    
                    if any(company in combined_text for company in major_companies):
                        events.append({
                            'type': 'major_partnership',
                            'description': f"Major partnership announced",
                            'score': self.event_weights['major_partnership'],
                            'date': item.get('datetime', 0),
                            'emoji': '🤝',
                            'details': {
                                'headline': item.get('headline')
                            }
                        })
                        break  # Only count once
                
                # Product launch detection
                elif any(word in combined_text for word in ['launch', 'unveil', 'introduce', 'new product']):
                    events.append({
                        'type': 'product_launch',
                        'description': f"Product launch announced",
                        'score': self.event_weights['product_launch'],
                        'date': item.get('datetime', 0),
                        'emoji': '🚀',
                        'details': {
                            'headline': item.get('headline')
                        }
                    })
                    break  # Only count once
            
            return events
            
        except Exception as e:
            logging.debug(f"Error checking corporate announcements for {ticker}: {e}")
            return []
    
    def format_events_for_display(self, events_data: Dict) -> str:
        """Format events for display in emails/reports"""
        if not events_data or events_data.get('event_count', 0) == 0:
            return "No significant events detected"
        
        lines = []
        for event in events_data.get('events', []):
            emoji = event.get('emoji', '📌')
            desc = event.get('description', 'Event')
            score = event.get('score', 0)
            
            sign = '+' if score > 0 else ''
            lines.append(f"{emoji} {desc}: {sign}{score}")
        
        return '\n'.join(lines)


# Quick test function
def test_news_engine():
    """Test the news engine with a sample ticker"""
    engine = NewsEventsEngine()
    
    # Test with NVDA (usually has lots of news)
    result = engine.analyze_events('NVDA', days_back=7)
    
    print("\n" + "="*80)
    print(f"📰 NEWS/EVENTS ANALYSIS FOR {result['ticker']}")
    print("="*80)
    print(f"Total Events: {result['event_count']}")
    print(f"Total Score: {result['total_score']:+d}")
    print(f"Sentiment: {result['sentiment'].upper()}")
    print("\nEvents Detected:")
    print(engine.format_events_for_display(result))
    print("="*80 + "\n")


if __name__ == "__main__":
    test_news_engine()
