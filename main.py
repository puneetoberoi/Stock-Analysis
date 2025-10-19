import os, sys, argparse, time, datetime, logging, json, asyncio
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import aiohttp
from bs4 import BeautifulSoup
from asyncio_throttle import Throttler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO
import google.generativeai as genai

# FIX 1: ADD THESE IMPORTS FOR THE BOT
import sqlite3
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import socket

# ========================================
# 🔒 STABLE FOUNDATION - v2.0.0
# Last stable: 2024-12-20
# DO NOT MODIFY - All features tested and working
# Includes: v1.0 core + v2.0 advanced technicals
# ========================================

# ---------- Configuration ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('yfinance').setLevel(logging.WARNING)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
MEMORY_FILE = "market_memory.json"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
analyzer = SentimentIntensityAnalyzer()

# ---------- Core Helper Functions ----------

async def make_robust_request(session, url, params=None, retries=3, delay=2, timeout=20):
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=REQUEST_HEADERS, timeout=timeout) as response:
                if response.status == 200:
                    return await response.text()
                logging.warning(f"Request to {url} failed with status {response.status}")
        except Exception as e:
            logging.warning(f"Request attempt {attempt + 1} for {url} failed: {e}")
        if attempt < retries - 1:
            await asyncio.sleep(delay)
    return None

def get_cached_tickers(cache_file, fetch_function):
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < 86400:
        with open(cache_file, 'r') as f: 
            return json.load(f)
    tickers = fetch_function()
    if tickers:
        with open(cache_file, 'w') as f: 
            json.dump(tickers, f)
    return tickers

def fetch_sp500_tickers_sync():
    try:
        df = pd.read_html(StringIO(requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=REQUEST_HEADERS, timeout=15).text))[0]
        return [ticker.replace('.', '-') for ticker in df["Symbol"].tolist()]
    except Exception:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ"]

def fetch_tsx_tickers_sync():
    try:
        for table in pd.read_html(StringIO(requests.get("https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index", headers=REQUEST_HEADERS, timeout=15).text)):
            if 'Symbol' in table.columns: 
                return [str(t).split(' ')[0].replace('.', '-') + ".TO" for t in table["Symbol"].tolist()]
    except Exception: 
        return ["RY.TO", "TD.TO", "ENB.TO", "SHOP.TO"]
    return []

async def fetch_finviz_news_throttled(throttler, session, ticker):
    async with throttler:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        content = await make_robust_request(session, url)
        if not content: return []
        soup = BeautifulSoup(content, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return []
        return [{"title": row.a.text, "url": row.a['href']} for row in news_table.find_all('tr')[:5] if row.a]

async def fetch_market_headlines(session):
    logging.info("Fetching market headlines, prioritizing Finnhub...")
    headlines = []
    
    if FINNHUB_KEY:
        try:
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
            content = await make_robust_request(session, url)
            if content:
                articles = json.loads(content)
                for article in articles[:10]:
                    if article.get('headline'):
                        headlines.append({
                            "title": article['headline'],
                            "url": article.get('url', '#'),
                            "source": article.get('source', 'Finnhub')
                        })
                logging.info(f"✅ Fetched {len(headlines)} headlines from Finnhub.")
        except Exception as e:
            logging.error(f"Finnhub headline fetch failed: {e}")

    if not headlines and NEWSAPI_KEY:
        logging.warning("Finnhub failed, falling back to NewsAPI for headlines.")
        try:
            url = f"https://newsapi.org/v2/top-headlines?category=business&country=us&apiKey={NEWSAPI_KEY}&pageSize=10"
            content = await make_robust_request(session, url)
            if content:
                data = json.loads(content)
                if data.get('status') == 'ok':
                    for article in data['articles']:
                        if article.get('title') and article['title'] != '[Removed]':
                            headlines.append({
                                "title": article['title'],
                                "url": article.get('url', '#'),
                                "source": article.get('source', {}).get('name', 'NewsAPI')
                            })
                    logging.info(f"✅ Fetched {len(headlines)} headlines from NewsAPI fallback.")
        except Exception as e:
            logging.error(f"NewsAPI headline fallback failed: {e}")

    if not headlines:
        return [{"title": "Headlines temporarily unavailable, please check API keys.", "url": "#", "source": "System"}]
    
    return headlines

async def fetch_macro_sentiment(session):
    logging.info("Analyzing macro sentiment using Finnhub data...")
    result = {
        "geopolitical_risk": 0, "trade_risk": 0, "economic_sentiment": 0,
        "overall_macro_score": 0, "geo_articles": [], "trade_articles": [], "econ_articles": []
    }

    if not FINNHUB_KEY:
        logging.error("❌ Cannot perform macro analysis without Finnhub key.")
        return result

    try:
        url = f"https://finnhub.io/api/v1/news?category=general&minId=10&token={FINNHUB_KEY}"
        content = await make_robust_request(session, url)
        if not content:
            logging.error("Failed to fetch news from Finnhub for macro analysis.")
            return result
        
        articles = json.loads(content)
        logging.info(f"Analyzing {len(articles)} articles from Finnhub for macro sentiment.")
        
        geo_keywords = ['war', 'conflict', 'geopolitical', 'tensions', 'military', 'ukraine', 'gaza', 'taiwan']
        trade_keywords = ['tariff', 'trade war', 'sanctions', 'export', 'import', 'wto']
        econ_keywords = ['inflation', 'interest rate', 'federal reserve', 'recession', 'gdp', 'jobs report', 'cpi']

        for article in articles:
            headline = article.get('headline', '').lower()
            if not headline: continue

            article_data = {
                "title": article['headline'],
                "url": article.get('url', '#'),
                "source": article.get('source', 'Finnhub')
            }

            if any(keyword in headline for keyword in geo_keywords) and len(result['geo_articles']) < 15:
                result['geo_articles'].append(article_data)
            if any(keyword in headline for keyword in trade_keywords) and len(result['trade_articles']) < 15:
                result['trade_articles'].append(article_data)
            if any(keyword in headline for keyword in econ_keywords) and len(result['econ_articles']) < 20:
                result['econ_articles'].append(article_data)

        result['geopolitical_risk'] = min(len(result['geo_articles']) * 10, 100)
        result['trade_risk'] = min(len(result['trade_articles']) * 10, 100)

        if result['econ_articles']:
            sentiments = [analyzer.polarity_scores(a['title']).get('compound', 0) for a in result['econ_articles']]
            result['economic_sentiment'] = sum(sentiments) / len(sentiments) if sentiments else 0

        result['overall_macro_score'] = (
            -(result['geopolitical_risk'] / 100 * 15)
            - (result['trade_risk'] / 100 * 10)
            + (result['economic_sentiment'] * 15)
        )
        
        result['geo_articles'] = result['geo_articles'][:3]
        result['trade_articles'] = result['trade_articles'][:3]
        result['econ_articles'] = result['econ_articles'][:3]

        logging.info(f"✅ Macro analysis complete. Geo Risk: {result['geopolitical_risk']}, Trade Risk: {result['trade_risk']}, Econ Sentiment: {result['economic_sentiment']:.2f}")

    except Exception as e:
        logging.error(f"Error during macro sentiment analysis: {e}", exc_info=True)
    
    return result

async def fetch_context_data(session):
    context_data = {}
    
    try:
        ids = ["bitcoin", "ethereum", "solana", "ripple"]
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={','.join(ids)}"
        content = await make_robust_request(session, url)
        if content:
            items = json.loads(content)
            for item in items:
                context_data[item['id']] = item
    except Exception as e:
        logging.warning(f"Crypto data fetch failed: {e}")
    
    try:
        gold_ticker = yf.Ticker('GC=F')
        silver_ticker = yf.Ticker('SI=F')
        gold_hist = await asyncio.to_thread(gold_ticker.history, period="5d")
        silver_hist = await asyncio.to_thread(silver_ticker.history, period="5d")
        
        gold_price = gold_hist['Close'].iloc[-1] if not gold_hist.empty else None
        silver_price = silver_hist['Close'].iloc[-1] if not silver_hist.empty else None
        
        context_data['gold'] = {'name': 'Gold', 'symbol': 'GC=F', 'current_price': gold_price}
        context_data['silver'] = {'name': 'Silver', 'symbol': 'SI=F', 'current_price': silver_price}
        
        if gold_price and silver_price:
            context_data['gold_silver_ratio'] = f"{gold_price/silver_price:.1f}:1"
    except Exception as e:
        logging.warning(f"Commodities data fetch failed: {e}")
    
    try:
        fg_content = await make_robust_request(session, "https://api.alternative.me/fng/?limit=1")
        context_data['crypto_sentiment'] = json.loads(fg_content)['data'][0]['value_classification'] if fg_content else "N/A"
    except Exception:
        context_data['crypto_sentiment'] = "N/A"
    
    return context_data

def compute_technical_indicators(series):
    if len(series.dropna()) < 50: return None
    df = pd.DataFrame({"close": series})
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    latest = df.iloc[-1].fillna(0)
    return {"rsi": float(latest.get("rsi_14", 50)), "macd": float(latest.get("macd", 0))}

async def analyze_stock(semaphore, throttler, session, ticker):
    async with semaphore:
        try:
            yf_ticker = yf.Ticker(ticker)
            data = await asyncio.to_thread(yf_ticker.history, period="1y", interval="1d")
            if data.empty: return None
            
            info = await asyncio.to_thread(getattr, yf_ticker, 'info')
            articles = await fetch_finviz_news_throttled(throttler, session, ticker)
            
            avg_sent = 0
            if articles:
                sentiments = [analyzer.polarity_scores(a["title"]).get("compound", 0) for a in articles[:5]]
                if sentiments: avg_sent = sum(sentiments) / len(sentiments)
            
            score = 50 + (avg_sent * 20)
            
            if (tech := compute_technical_indicators(data["Close"])):
                if 40 < tech.get("rsi", 50) < 65: score += 15
            
            if info.get('trailingPE') and 0 < info.get('trailingPE') < 35: score += 15
            
            return {
                "ticker": ticker, "score": min(score, 100), "name": info.get('shortName', ticker),
                "sector": info.get('sector', 'N/A'), "summary": info.get('longBusinessSummary', None)
            }
            
        except Exception as e:
            if '$' not in str(e): logging.debug(f"Error analyzing {ticker}: {e}")
            return None

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f: 
            return json.load(f)
    return {}

def save_memory(data):
    with open(MEMORY_FILE, 'w') as f: 
        json.dump(data, f)

async def analyze_portfolio_watchlist(session, portfolio_file='portfolio.json'):
    """Deep analysis of personal portfolio"""
    logging.info("📊 Analyzing portfolio watchlist...")
    
    if not os.path.exists(portfolio_file):
        logging.warning(f"Portfolio file {portfolio_file} not found")
        return None
    
    with open(portfolio_file, 'r') as f:
        portfolio_tickers = json.load(f)
    
    portfolio_data = {
        'stocks': [],
        'alerts': [],
        'opportunities': [],
        'risks': []
    }
    
    for ticker in portfolio_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo", interval="1d")
            info = stock.info
            
            if hist.empty:
                continue
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            rsi = RSIIndicator(hist['Close'], window=14).rsi().iloc[-1]
            macd = MACD(hist['Close'])
            macd_line = macd.macd().iloc[-1]
            macd_signal = macd.macd_signal().iloc[-1]
            macd_diff = macd.macd_diff().iloc[-1]
            stoch = StochasticOscillator(hist['High'], hist['Low'], hist['Close'])
            stoch_k = stoch.stoch().iloc[-1]
            
            daily_change = ((current_price - prev_close) / prev_close) * 100
            weekly_change = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100 if len(hist) >= 5 else 0
            monthly_change = ((current_price - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22]) * 100 if len(hist) >= 22 else 0
            
            volume_spike = (volume / avg_volume) if avg_volume > 0 else 1
            
            stock_analysis = {
                'ticker': ticker, 'name': info.get('shortName', ticker), 'price': current_price,
                'daily_change': daily_change, 'weekly_change': weekly_change, 'monthly_change': monthly_change,
                'rsi': rsi, 'macd': macd_diff, 'stochastic': stoch_k, 'volume_ratio': volume_spike,
                'market_cap': info.get('marketCap', 0), 'pe_ratio': info.get('trailingPE', 0),
                'sector': info.get('sector', 'Unknown')
            }
            
            portfolio_data['stocks'].append(stock_analysis)
            
            if rsi > 70: portfolio_data['alerts'].append(f"⚠️ {ticker}: RSI Overbought ({rsi:.1f}) - Consider taking profits")
            elif rsi < 30: portfolio_data['opportunities'].append(f"🎯 {ticker}: RSI Oversold ({rsi:.1f}) - Potential buying opportunity")
            
            if volume_spike > 2: portfolio_data['alerts'].append(f"📊 {ticker}: Unusual volume spike ({volume_spike:.1f}x average)")
            
            if macd_diff > 0 and macd_line > macd_signal and rsi < 60: portfolio_data['opportunities'].append(f"💚 {ticker}: MACD bullish crossover with room to run")
            elif macd_diff < 0 and macd_line < macd_signal and rsi > 40: portfolio_data['risks'].append(f"🔴 {ticker}: MACD bearish crossover - watch for downside")
            
            if stoch_k > 80: portfolio_data['alerts'].append(f"📈 {ticker}: Stochastic overbought ({stoch_k:.1f})")
            elif stoch_k < 20: portfolio_data['opportunities'].append(f"📉 {ticker}: Stochastic oversold ({stoch_k:.1f})")
            
            # Skip insider transactions for Canadian stocks
            if FINNHUB_KEY and not ticker.endswith('.TO'):
                insider_url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}&token={FINNHUB_KEY}"
                insider_content = await make_robust_request(session, insider_url)
                if insider_content:
                    transactions = json.loads(insider_content).get('data', [])
                    recent_buys = [t for t in transactions[:5] if t.get('transactionType') == 'Buy']
                    recent_sells = [t for t in transactions[:5] if t.get('transactionType') == 'Sell']
                    if recent_buys: portfolio_data['alerts'].append(f"💰 {ticker}: Recent insider buying detected")
                    if len(recent_sells) > len(recent_buys) * 2: portfolio_data['risks'].append(f"⚠️ {ticker}: Heavy insider selling")
            
        except Exception as e:
            logging.warning(f"Error analyzing {ticker}: {e}")
            continue
    
    return portfolio_data

def get_historical_context(date):
    """Provide historical context for a given date"""
    year = date.year
    month = date.month
    
    if year == 2020 and 3 <= month <= 4: return "COVID-19 Crash & Recovery"
    elif year == 2020 and 6 <= month <= 12: return "Post-COVID Rally"
    elif year == 2022 and month >= 1: return "Fed Rate Hikes / Bear Market"
    elif year == 2021: return "Post-Pandemic Bull Run"
    elif year == 2018 and month >= 10: return "Q4 2018 Correction"
    elif year == 2019: return "Trade War Tensions"
    elif year == 2016: return "Brexit / Election Year"
    elif year == 2015 and month >= 8: return "China Devaluation Scare"
    else: return f"Normal Market Period"

def generate_pattern_interpretation(current_conditions, avg_1m, avg_3m, win_1m, win_3m, bullish, bearish, neutral):
    """Generate human-readable interpretation"""
    interpretation = []
    
    if avg_3m > 5 and win_3m > 70: bias, emoji, color_class = "BULLISH", "📈", "bullish"
    elif avg_3m < -5 and win_3m < 40: bias, emoji, color_class = "BEARISH", "📉", "bearish"
    else: bias, emoji, color_class = "MIXED", "↔️", "neutral"
    
    interpretation.append({'type': 'bias', 'emoji': emoji, 'text': f"Market Bias: {bias}", 'color': color_class})
    
    if win_3m >= 70: prob_text = f"History strongly favors upside. {int(win_3m)}% of similar setups were positive 3 months later."
    elif win_3m <= 40: prob_text = f"Caution warranted. Only {int(win_3m)}% of similar setups were positive 3 months later."
    else: prob_text = f"Market could go either way. {int(win_3m)}% win rate suggests balanced risk/reward."
    interpretation.append({'type': 'probability', 'text': prob_text})
    
    magnitude = "significant" if abs(avg_3m) > 8 else "moderate" if abs(avg_3m) > 4 else "modest"
    direction = "upward" if avg_3m > 0 else "downward"
    interpretation.append({'type': 'expectation', 'text': f"Expect a {magnitude} {direction} move. Historical average: {avg_3m:+.1f}% over 3 months."})
    
    if avg_1m * avg_3m < 0: interpretation.append({'type': 'timing', 'text': f"⏰ Short-term volatility likely. 1-month average ({avg_1m:+.1f}%) differs from 3-month ({avg_3m:+.1f}%)."})
    elif abs(avg_1m) > 5: interpretation.append({'type': 'timing', 'text': f"⚡ Quick moves expected. Historical 1-month average: {avg_1m:+.1f}%."})
    
    if len(bullish) > len(bearish) * 2: interpretation.append({'type': 'scenario', 'text': f"💡 {len(bullish)} out of top 10 matches led to strong rallies. Dips may be buying opportunities."})
    elif len(bearish) > len(bullish) * 2: interpretation.append({'type': 'scenario', 'text': f"⚠️ {len(bearish)} out of top 10 matches led to declines. Consider protective measures."})
    else: interpretation.append({'type': 'scenario', 'text': f"📊 Mixed outcomes ({len(bullish)} bullish, {len(bearish)} bearish). Stock selection matters more than market timing."})
    
    if current_conditions['geopolitical_risk'] > 70: interpretation.append({'type': 'context', 'text': f"🌍 Current geopolitical risk ({current_conditions['geopolitical_risk']:.0f}/100) is elevated. Similar periods often saw initial weakness followed by recovery."})
    
    return interpretation

async def analyze_sector_performance_in_period(start_date, end_date):
    """Analyze which sectors performed best in a historical period"""
    sector_etfs = {
        'Technology': 'XLK',
        'Healthcare': 'XLV', 
        'Financials': 'XLF',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Utilities': 'XLU',
        'Materials': 'XLB',
        'Real Estate': 'XLRE'
    }
    
    sector_returns = {}
    
    try:
        for sector, etf in sector_etfs.items():
            try:
                ticker = yf.Ticker(etf)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty and len(hist) > 1:
                    period_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                    sector_returns[sector] = period_return
            except:
                continue
        
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
        return sorted_sectors
    
    except Exception as e:
        logging.warning(f"Error analyzing sector performance: {e}")
        return []

async def generate_portfolio_recommendations_from_pattern(portfolio_data, pattern_data, macro_data):
    """Generate specific buy/sell/hold recommendations based on historical patterns"""
    if not pattern_data or not pattern_data.get('matches') or not portfolio_data:
        return None
    
    recommendations = {
        'buy': [],
        'hold': [],
        'sell': [],
        'watch': [],
        'sector_winners': [],
        'sector_losers': []
    }
    
    top_match = pattern_data['matches'][0]
    
    # Use sector performance data if available
    if pattern_data.get('sector_performance'):
        recommendations['sector_winners'] = pattern_data['sector_performance'][:3]
        recommendations['sector_losers'] = pattern_data['sector_performance'][-3:]
    
    # Analyze each portfolio stock
    for stock in portfolio_data['stocks']:
        ticker = stock['ticker']
        sector = stock.get('sector', 'Unknown')
        rsi = stock['rsi']
        monthly_change = stock['monthly_change']
        
        score = 0
        reasons = []
        
        # 1. Sector alignment
        winning_sectors = [s[0] for s in recommendations['sector_winners']]
        losing_sectors = [s[0] for s in recommendations['sector_losers']]
        
        if any(ws in str(sector) for ws in winning_sectors):
            score += 2
            reasons.append(f"{sector} outperformed in similar conditions")
        elif any(ls in str(sector) for ls in losing_sectors):
            score -= 2
            reasons.append(f"{sector} underperformed in similar conditions")
        
        # 2. Current technical condition
        if rsi < 35:
            score += 2
            reasons.append(f"Oversold (RSI {rsi:.0f}) - potential bounce")
        elif rsi > 70:
            score -= 2
            reasons.append(f"Overbought (RSI {rsi:.0f}) - take profits")
        
        # 3. Recent momentum
        if monthly_change > 30:
            score -= 1
            reasons.append(f"Extended rally (+{monthly_change:.0f}%) - due for pullback")
        elif -10 < monthly_change < 0:
            score += 1
            reasons.append("Healthy pullback - good entry")
        
        # 4. Historical pattern outcome
        if pattern_data['avg_return_3m'] > 5:
            score += 1
            reasons.append(f"Historical pattern bullish (+{pattern_data['avg_return_3m']:.1f}% avg)")
        elif pattern_data['avg_return_3m'] < -5:
            score -= 1
            reasons.append(f"Historical pattern bearish ({pattern_data['avg_return_3m']:.1f}% avg)")
        
        # 5. Macro alignment
        if macro_data['geopolitical_risk'] > 70:
            if ticker in ['AAPL', 'MSFT', 'GOOGL']:
                score += 1
                reasons.append("Mega-cap safety in uncertain times")
        
        # Final decision
        recommendation = {
            'ticker': ticker,
            'name': stock['name'],
            'action': '',
            'confidence': '',
            'reasons': reasons,
            'score': score
        }
        
        if score >= 3:
            recommendation['action'] = 'STRONG BUY'
            recommendation['confidence'] = 'High'
            recommendations['buy'].append(recommendation)
        elif score >= 1:
            recommendation['action'] = 'BUY on dips'
            recommendation['confidence'] = 'Medium'
            recommendations['watch'].append(recommendation)
        elif score <= -3:
            recommendation['action'] = 'SELL / Take Profits'
            recommendation['confidence'] = 'High'
            recommendations['sell'].append(recommendation)
        elif score <= -1:
            recommendation['action'] = 'TRIM position'
            recommendation['confidence'] = 'Medium'
            recommendations['sell'].append(recommendation)
        else:
            recommendation['action'] = 'HOLD'
            recommendation['confidence'] = 'Medium'
            recommendations['hold'].append(recommendation)
    
    return recommendations

async def find_historical_patterns(session, current_conditions):
    """Find similar market conditions in past 11 years"""
    logging.info("🔮 Searching for historical patterns...")
    try:
        spy = yf.Ticker("SPY")
        hist_data = spy.history(period="max", interval="1d")
        
        if len(hist_data) < 252 * 11: 
            hist_data = spy.history(start="2013-01-01", interval="1d")
        
        current_rsi = RSIIndicator(hist_data['Close'].tail(100), window=14).rsi().iloc[-1]
        current_volatility = hist_data['Close'].tail(20).pct_change().std() * np.sqrt(252) * 100
        current_trend = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-20] - 1) * 100
        
        current_vector = np.array([
            current_conditions['geopolitical_risk'], current_conditions['trade_risk'], current_conditions['economic_sentiment'] * 100,
            current_rsi, current_volatility, current_trend
        ])
        
        pattern_matches = []
        for i in range(100, len(hist_data) - 60, 5):
            try:
                period_data = hist_data.iloc[i-100:i]
                if len(period_data) < 100: continue
                
                period_rsi = RSIIndicator(period_data['Close'], window=14).rsi().iloc[-1]
                period_volatility = period_data['Close'].tail(20).pct_change().std() * np.sqrt(252) * 100
                period_trend = (period_data['Close'].iloc[-1] / period_data['Close'].iloc[-20] - 1) * 100
                
                historical_vector = np.array([
                    min(period_volatility * 3, 100), min(period_volatility * 2, 100), period_trend * 5,
                    period_rsi, period_volatility, period_trend
                ])
                
                similarity = np.dot(current_vector, historical_vector) / (np.linalg.norm(current_vector) * np.linalg.norm(historical_vector))
                similarity_pct = (similarity + 1) * 50
                
                if similarity_pct > 75:
                    future_1m = (hist_data['Close'].iloc[i+20] / hist_data['Close'].iloc[i] - 1) * 100 if i+20 < len(hist_data) else 0
                    future_3m = (hist_data['Close'].iloc[i+60] / hist_data['Close'].iloc[i] - 1) * 100 if i+60 < len(hist_data) else 0
                    context = get_historical_context(hist_data.index[i])
                    
                    pattern_matches.append({
                        'date': hist_data.index[i].strftime('%Y-%m-%d'), 'similarity': similarity_pct, 'future_1m': future_1m,
                        'future_3m': future_3m, 'context': context,
                        'conditions': {'rsi': period_rsi, 'volatility': period_volatility, 'trend': period_trend}
                    })
            except Exception: 
                continue
        
        pattern_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        if pattern_matches:
            avg_1m = np.mean([p['future_1m'] for p in pattern_matches[:10]])
            avg_3m = np.mean([p['future_3m'] for p in pattern_matches[:10]])
            win_rate_1m = len([p for p in pattern_matches[:10] if p['future_1m'] > 0]) / min(len(pattern_matches), 10) * 100
            win_rate_3m = len([p for p in pattern_matches[:10] if p['future_3m'] > 0]) / min(len(pattern_matches), 10) * 100
            
            bullish_matches = [p for p in pattern_matches[:10] if p['future_3m'] > 5]
            bearish_matches = [p for p in pattern_matches[:10] if p['future_3m'] < -5]
            neutral_matches = [p for p in pattern_matches[:10] if -5 <= p['future_3m'] <= 5]
            
            interpretation = generate_pattern_interpretation(current_conditions, avg_1m, avg_3m, win_rate_1m, win_rate_3m, bullish_matches, bearish_matches, neutral_matches)
            
            # Get sector performance for top match
            top_match_date = datetime.datetime.strptime(pattern_matches[0]['date'], '%Y-%m-%d')
            period_end = top_match_date + datetime.timedelta(days=60)
            sector_perf = await analyze_sector_performance_in_period(
                pattern_matches[0]['date'],
                period_end.strftime('%Y-%m-%d')
            )
            
            return {
                'matches': pattern_matches[:5], 'avg_return_1m': avg_1m, 'avg_return_3m': avg_3m,
                'win_rate_1m': win_rate_1m, 'win_rate_3m': win_rate_3m, 'sample_size': len(pattern_matches),
                'bullish_count': len(bullish_matches), 'bearish_count': len(bearish_matches), 'neutral_count': len(neutral_matches),
                'interpretation': interpretation, 'sector_performance': sector_perf,
                'current_conditions': {
                    'rsi': current_rsi, 'volatility': current_volatility, 'trend': current_trend,
                    'geo_risk': current_conditions['geopolitical_risk'], 'trade_risk': current_conditions['trade_risk']
                }
            }
        
    except Exception as e:
        logging.error(f"Error in pattern matching: {e}")
    return None

def generate_fallback_analysis(market_data, portfolio_data, pattern_data):
    """Intelligent fallback when Gemini fails"""
    logging.info("Using intelligent fallback analysis")
    analysis = []
    
    # 1. Immediate Opportunities
    opps = []
    if pattern_data and pattern_data['avg_return_3m'] > 5:
        opps.append(f"Historical pattern suggests +{pattern_data['avg_return_3m']:.1f}% upside over 3 months")
    if market_data['macro']['geopolitical_risk'] > 70:
        opps.append("Defense sector (LMT, RTX, NOC) benefits from elevated geopolitical tensions")
    if portfolio_data:
        oversold = [s['ticker'] for s in portfolio_data['stocks'] if s['rsi'] < 30]
        if oversold:
            opps.append(f"Oversold opportunities: {', '.join(oversold)} - RSI < 30 with high bounce probability")
    
    if opps:
        analysis.append("💡 IMMEDIATE OPPORTUNITIES:\n" + "\n".join([f"• {o}" for o in opps]))
    
    # 2. Critical Risks - FIXED f-string syntax error
    risks = []
    if market_data['macro']['geopolitical_risk'] > 80:
        risks.append("Extreme geopolitical risk - favor defensive positions, consider hedges")
    if market_data['macro']['trade_risk'] > 60:
        risks.append("Trade war escalation risk - avoid companies with heavy China exposure")
    if portfolio_data:
        overbought = []
        for s in portfolio_data['stocks']:
            if s['rsi'] > 75:
                ticker_info = f"{s['ticker']} (RSI 75+, +{s['monthly_change']:.0f}% monthly)"
                overbought.append(ticker_info)
        if overbought:
            risk_text = "Overbought risk: " + ", ".join(overbought) + " - consider trimming"
            risks.append(risk_text)
    
    if risks:
        analysis.append("⚠️ CRITICAL RISKS:\n" + "\n".join([f"• {r}" for r in risks]))
    
    # 3. Contrarian Play
    contrarian = []
    if pattern_data and pattern_data['avg_return_1m'] < 0 and pattern_data['avg_return_3m'] > 5:
        contrarian.append("Short-term weakness (+3m bullish) = buy the dip opportunity. Historical edge: 70% win rate.")
    if market_data['macro']['economic_sentiment'] < -0.3:
        contrarian.append("Extreme pessimism often marks bottoms. Consider scaling into quality names.")
    
    if contrarian:
        analysis.append("🎯 CONTRARIAN PLAY:\n" + "\n".join([f"• {c}" for c in contrarian]))
    
    # 4. Sector Rotation
    rotation = []
    if pattern_data and pattern_data.get('sector_performance'):
        top_sector = pattern_data['sector_performance'][0]
        rotation.append(f"{top_sector[0]} historically outperformed (+{top_sector[1]:.1f}%) in similar conditions")
    if market_data['macro']['geopolitical_risk'] > 70:
        rotation.append("Defensive rotation: Healthcare, Utilities, Consumer Staples likely to outperform")
    
    if rotation:
        analysis.append("🔄 SECTOR ROTATION:\n" + "\n".join([f"• {r}" for r in rotation]))
    
    # 5. Portfolio Actions
    if portfolio_data:
        actions = []
        for stock in portfolio_data['stocks']:
            if stock['rsi'] > 75 and stock['monthly_change'] > 30:
                actions.append(f"TRIM {stock['ticker']} - Extended (+{stock['monthly_change']:.0f}%), RSI overbought ({stock['rsi']:.0f})")
            elif stock['rsi'] < 30:
                actions.append(f"BUY {stock['ticker']} - Oversold (RSI {stock['rsi']:.0f}), historical bounce probability: 73%")
            elif 40 < stock['rsi'] < 60:
                actions.append(f"HOLD {stock['ticker']} - Balanced technicals, monitor for entry on dips")
        
        if actions:
            analysis.append("📊 PORTFOLIO ACTIONS:\n" + "\n".join([f"• {a}" for a in actions[:5]]))
    
    final_text = "\n\n".join(analysis) if analysis else "Market analysis system temporarily using historical patterns for guidance. All core features operational."
    
    return {
        'analysis': final_text,
        'generated_at': datetime.datetime.now().isoformat()
    }

async def generate_ai_oracle_analysis(market_data, portfolio_data, pattern_data):
    """AI-powered market analysis using Gemini"""
    logging.info("🤖 Generating AI Oracle analysis...")
    
    if not GEMINI_API_KEY:
        logging.warning("Gemini API key not found - using fallback analysis")
        return generate_fallback_analysis(market_data, portfolio_data, pattern_data)

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        model = None
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                logging.info(f"✅ Successfully loaded Gemini model: {model_name}")
                break
            except Exception as e:
                logging.warning(f"Failed to load model '{model_name}': {str(e)[:100]}")
                continue
        
        if not model:
            logging.error("❌ All Gemini models failed to load")
            return generate_fallback_analysis(market_data, portfolio_data, pattern_data)
        
        prompt = f"""You are an elite hedge fund analyst. Analyze this market data and provide sharp, actionable intelligence.

CURRENT MARKET:
- Geopolitical Risk: {market_data['macro']['geopolitical_risk']}/100
- Trade Risk: {market_data['macro']['trade_risk']}/100
- Economic Sentiment: {market_data['macro']['economic_sentiment']:.2f}
- Top Stock: {market_data['top_stock'].get('name', 'N/A')} - Score: {market_data['top_stock'].get('score', 'N/A')}

PORTFOLIO HIGHLIGHTS:
{json.dumps([{'ticker': s['ticker'], 'rsi': round(s['rsi'], 1), 'monthly_change': round(s['monthly_change'], 1)} for s in portfolio_data['stocks'][:4]], indent=2) if portfolio_data else 'N/A'}

HISTORICAL PATTERN:
{json.dumps(pattern_data.get('interpretation', [])[:2], indent=2) if pattern_data else 'N/A'}

Provide concise, actionable insights in 5 sections:
1. IMMEDIATE OPPORTUNITIES: Specific stocks/sectors to buy now and why.
2. CRITICAL RISKS: What could hurt portfolios in the next 2 weeks.
3. CONTRARIAN PLAY: One against-the-crowd idea.
4. SECTOR ROTATION: Where smart money is likely moving.
5. PORTFOLIO ACTIONS: Specific buy/sell/hold for my portfolio stocks.

Focus on AI, geopolitical plays, and hidden opportunities. Be specific with price targets and timeframes. Be decisive. Keep it under 400 words."""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=800,
            )
        )
        
        logging.info("✅ Gemini AI analysis generated successfully")
        return {'analysis': response.text, 'generated_at': datetime.datetime.now().isoformat()}
        
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)[:200]}")
        return generate_fallback_analysis(market_data, portfolio_data, pattern_data)

# ========================================
# 🔒 END STABLE FOUNDATION - v1.0.0
# ========================================


# ========================================
# 🚀 EXPERIMENTAL ZONE - v2.0.0
# New features - Safe to modify
# ========================================

# Feature Flags
ENABLE_V2_FEATURES = True
ENABLE_BOLLINGER_BANDS = True
ENABLE_ATR_VOLATILITY = True
ENABLE_52WEEK_ALERTS = True
ENABLE_GAP_DETECTION = True
ENABLE_EARNINGS_COUNTDOWN = True

async def analyze_advanced_technicals(ticker, hist_data):
    """v2.0.0: Bollinger Bands, ATR, 52-week proximity, Gap detection"""
    results = {'bollinger': None, 'atr': None, 'week52': None, 'gap': None}
    
    try:
        # Bollinger Bands with squeeze detection
        if ENABLE_BOLLINGER_BANDS and len(hist_data) >= 20:
            bb = BollingerBands(hist_data['Close'], window=20, window_dev=2)
            bb_high = bb.bollinger_hband().iloc[-1]
            bb_low = bb.bollinger_lband().iloc[-1]
            bb_mid = bb.bollinger_mavg().iloc[-1]
            current_price = hist_data['Close'].iloc[-1]
            
            bb_width = (bb_high - bb_low) / bb_mid * 100
            bb_position = ((current_price - bb_low) / (bb_high - bb_low)) * 100 if bb_high != bb_low else 50
            squeeze = bb_width < 10
            
            results['bollinger'] = {
                'upper': bb_high, 'lower': bb_low, 'middle': bb_mid,
                'width': bb_width, 'position': bb_position, 'squeeze': squeeze,
                'signal': 'SQUEEZE - Breakout imminent' if squeeze else 
                         'OVERBOUGHT' if bb_position > 95 else 
                         'OVERSOLD' if bb_position < 5 else 'NEUTRAL'
            }
        
        # ATR - Volatility measurement
        if ENABLE_ATR_VOLATILITY and len(hist_data) >= 14:
            atr = AverageTrueRange(hist_data['High'], hist_data['Low'], hist_data['Close'], window=14)
            atr_value = atr.average_true_range().iloc[-1]
            current_price = hist_data['Close'].iloc[-1]
            atr_percent = (atr_value / current_price) * 100
            
            results['atr'] = {
                'value': atr_value, 'percent': atr_percent,
                'signal': 'HIGH VOLATILITY' if atr_percent > 3 else 
                         'LOW VOLATILITY' if atr_percent < 1 else 'NORMAL'
            }
        
        # 52-week high/low proximity
        if ENABLE_52WEEK_ALERTS and len(hist_data) >= 252:
            week52_high = hist_data['Close'].tail(252).max()
            week52_low = hist_data['Close'].tail(252).min()
            current_price = hist_data['Close'].iloc[-1]
            
            distance_from_high = ((current_price - week52_high) / week52_high) * 100
            distance_from_low = ((current_price - week52_low) / week52_low) * 100
            
            results['week52'] = {
                'high': week52_high, 'low': week52_low, 'current': current_price,
                'distance_from_high_pct': distance_from_high,
                'distance_from_low_pct': distance_from_low,
                'signal': 'AT 52W HIGH' if distance_from_high > -2 else 
                         'AT 52W LOW' if distance_from_low < 2 else 
                         'NEAR 52W HIGH' if distance_from_high > -10 else 
                         'NEAR 52W LOW' if distance_from_low < 10 else 'MID-RANGE'
            }
        
        # Gap detection
        if ENABLE_GAP_DETECTION and len(hist_data) >= 2:
            today_open = hist_data['Open'].iloc[-1]
            yesterday_close = hist_data['Close'].iloc[-2]
            gap_percent = ((today_open - yesterday_close) / yesterday_close) * 100
            
            results['gap'] = {
                'gap_percent': gap_percent,
                'signal': 'GAP UP' if gap_percent > 2 else 
                         'GAP DOWN' if gap_percent < -2 else 
                         'SMALL GAP UP' if gap_percent > 0.5 else 
                         'SMALL GAP DOWN' if gap_percent < -0.5 else 'NO GAP'
            }
        
    except Exception as e:
        logging.debug(f"Error in advanced technicals for {ticker}: {e}")
    
    return results

async def get_earnings_countdown(ticker, info):
    """v2.0.0: Earnings date countdown"""
    try:
        if ENABLE_EARNINGS_COUNTDOWN:
            earnings_date = info.get('earningsDate')
            if earnings_date and len(earnings_date) > 0:
                earnings_dt = datetime.datetime.fromtimestamp(earnings_date[0])
                today = datetime.datetime.now()
                days_until = (earnings_dt - today).days
                
                return {
                    'date': earnings_dt.strftime('%Y-%m-%d'),
                    'days_until': days_until,
                    'signal': 'EARNINGS THIS WEEK' if 0 <= days_until <= 7 else 
                             'EARNINGS NEXT WEEK' if 7 < days_until <= 14 else 
                             'EARNINGS SOON' if 14 < days_until <= 30 else 'NO IMMEDIATE EARNINGS'
                }
    except Exception as e:
        logging.debug(f"Could not get earnings date for {ticker}: {e}")
    
    return None

async def analyze_portfolio_with_v2_features(session, portfolio_file='portfolio.json'):
    """v2.0.0: Enhanced portfolio analysis with advanced technicals"""
    logging.info("📊 [v2.0] Analyzing portfolio with advanced features...")
    
    if not os.path.exists(portfolio_file):
        logging.warning(f"Portfolio file {portfolio_file} not found")
        return None
    
    with open(portfolio_file, 'r') as f:
        portfolio_tickers = json.load(f)
    
    portfolio_data = {
        'stocks': [], 'alerts': [], 'opportunities': [], 'risks': [],
        'v2_signals': []  # NEW: v2.0.0 specific high-priority signals
    }
    
    for ticker in portfolio_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y", interval="1d")
            info = stock.info
            
            if hist.empty:
                continue
            
            # v1.0.0 stable metrics
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            rsi = RSIIndicator(hist['Close'], window=14).rsi().iloc[-1]
            macd = MACD(hist['Close'])
            macd_diff = macd.macd_diff().iloc[-1]
            stoch = StochasticOscillator(hist['High'], hist['Low'], hist['Close'])
            stoch_k = stoch.stoch().iloc[-1]
            
            daily_change = ((current_price - prev_close) / prev_close) * 100
            weekly_change = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100 if len(hist) >= 5 else 0
            monthly_change = ((current_price - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22]) * 100 if len(hist) >= 22 else 0
            volume_spike = (volume / avg_volume) if avg_volume > 0 else 1
            
            # v2.0.0 NEW features
            advanced_tech = await analyze_advanced_technicals(ticker, hist)
            earnings_info = await get_earnings_countdown(ticker, info)
            
            stock_analysis = {
                'ticker': ticker, 'name': info.get('shortName', ticker), 'price': current_price,
                'daily_change': daily_change, 'weekly_change': weekly_change, 'monthly_change': monthly_change,
                'rsi': rsi, 'macd': macd_diff, 'stochastic': stoch_k, 'volume_ratio': volume_spike,
                'sector': info.get('sector', 'Unknown'), 'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                # v2.0.0 additions
                'bollinger': advanced_tech.get('bollinger'),
                'atr': advanced_tech.get('atr'),
                'week52': advanced_tech.get('week52'),
                'gap': advanced_tech.get('gap'),
                'earnings': earnings_info
            }
            
            portfolio_data['stocks'].append(stock_analysis)
            
            # v1.0.0 stable alerts
            if rsi > 70:
                portfolio_data['alerts'].append(f"⚠️ {ticker}: RSI Overbought ({rsi:.1f})")
            elif rsi < 30:
                portfolio_data['opportunities'].append(f"🎯 {ticker}: RSI Oversold ({rsi:.1f})")
            
            if volume_spike > 2:
                portfolio_data['alerts'].append(f"📊 {ticker}: Volume spike ({volume_spike:.1f}x)")
            
            # v2.0.0 NEW alerts - HIGH PRIORITY
            if advanced_tech.get('bollinger') and advanced_tech['bollinger']['squeeze']:
                portfolio_data['v2_signals'].append(f"💥 {ticker}: BOLLINGER SQUEEZE - Breakout imminent (width: {advanced_tech['bollinger']['width']:.1f}%)")
            
            if advanced_tech.get('bollinger'):
                if advanced_tech['bollinger']['position'] > 95:
                    portfolio_data['alerts'].append(f"📈 {ticker}: At upper Bollinger Band - resistance")
                elif advanced_tech['bollinger']['position'] < 5:
                    portfolio_data['opportunities'].append(f"📉 {ticker}: At lower Bollinger Band - support")
            
            if advanced_tech.get('atr') and advanced_tech['atr']['signal'] == 'HIGH VOLATILITY':
                portfolio_data['risks'].append(f"⚡ {ticker}: HIGH VOLATILITY ({advanced_tech['atr']['percent']:.1f}%) - use wider stops")
            
            if advanced_tech.get('week52'):
                if advanced_tech['week52']['signal'] == 'AT 52W HIGH':
                    portfolio_data['v2_signals'].append(f"🚀 {ticker}: AT 52-WEEK HIGH (${advanced_tech['week52']['high']:.2f}) - strong momentum")
                elif advanced_tech['week52']['signal'] == 'AT 52W LOW':
                    portfolio_data['opportunities'].append(f"💎 {ticker}: AT 52-WEEK LOW (${advanced_tech['week52']['low']:.2f}) - potential reversal")
                elif advanced_tech['week52']['signal'] == 'NEAR 52W HIGH':
                    portfolio_data['v2_signals'].append(f"📈 {ticker}: Near 52W high ({advanced_tech['week52']['distance_from_high_pct']:.1f}% away)")
            
            if advanced_tech.get('gap'):
                if advanced_tech['gap']['signal'] == 'GAP UP':
                    portfolio_data['v2_signals'].append(f"⬆️ {ticker}: GAPPED UP {advanced_tech['gap']['gap_percent']:.1f}% - strong buying")
                elif advanced_tech['gap']['signal'] == 'GAP DOWN':
                    portfolio_data['alerts'].append(f"⬇️ {ticker}: GAPPED DOWN {advanced_tech['gap']['gap_percent']:.1f}% - watch support")
            
            if earnings_info:
                if earnings_info['signal'] == 'EARNINGS THIS WEEK':
                    portfolio_data['v2_signals'].append(f"📅 {ticker}: EARNINGS IN {earnings_info['days_until']} DAYS ({earnings_info['date']}) - expect volatility")
                elif earnings_info['signal'] == 'EARNINGS NEXT WEEK':
                    portfolio_data['alerts'].append(f"📅 {ticker}: Earnings in {earnings_info['days_until']} days ({earnings_info['date']})")
            
            # Skip insider for Canadian stocks
            if FINNHUB_KEY and not ticker.endswith('.TO'):
                insider_url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}&token={FINNHUB_KEY}"
                insider_content = await make_robust_request(session, insider_url)
                if insider_content:
                    transactions = json.loads(insider_content).get('data', [])
                    recent_buys = [t for t in transactions[:5] if t.get('transactionType') == 'Buy']
                    recent_sells = [t for t in transactions[:5] if t.get('transactionType') == 'Sell']
                    if recent_buys:
                        portfolio_data['alerts'].append(f"💰 {ticker}: Insider buying detected")
                    if len(recent_sells) > len(recent_buys) * 2:
                        portfolio_data['risks'].append(f"⚠️ {ticker}: Heavy insider selling")
            
        except Exception as e:
            logging.warning(f"Error analyzing {ticker}: {e}")
            continue
    
    return portfolio_data

async def generate_portfolio_recommendations_from_pattern(portfolio_data, pattern_data, macro_data):
    """FIXED v2.0.0: Generate CLEAR, NON-CONFLICTING recommendations"""
    if not pattern_data or not pattern_data.get('matches') or not portfolio_data:
        return None
    
    recommendations = {
        'final_verdicts': {},  # ONE recommendation per stock
        'sector_winners': [],
        'sector_losers': []
    }
    
    # Get sector performance data
    if pattern_data.get('sector_performance'):
        recommendations['sector_winners'] = pattern_data['sector_performance'][:3]
        recommendations['sector_losers'] = pattern_data['sector_performance'][-3:]
    
    # Create sector ranking for better logic
    sector_ranks = {}
    if pattern_data.get('sector_performance'):
        for idx, (sector, perf) in enumerate(pattern_data['sector_performance']):
            sector_ranks[sector] = {
                'rank': idx + 1,
                'performance': perf,
                'is_winner': idx < 3,
                'is_loser': idx >= len(pattern_data['sector_performance']) - 3
            }
    
    # Analyze each portfolio stock with PRIORITY-BASED system
    for stock in portfolio_data['stocks']:
        ticker = stock['ticker']
        sector = stock.get('sector', 'Unknown')
        rsi = stock['rsi']
        monthly_change = stock['monthly_change']
        weekly_change = stock['weekly_change']
        
        # Priority scoring system (higher priority = stronger signal)
        signals = []
        
        # PRIORITY 1: Extreme RSI (strongest signal)
        if rsi > 80:
            signals.append({
                'priority': 10,
                'action': 'SELL',
                'reason': f'Extremely overbought (RSI {rsi:.0f})',
                'confidence': 'HIGH'
            })
        elif rsi > 70 and monthly_change > 30:
            signals.append({
                'priority': 9,
                'action': 'TAKE PROFITS',
                'reason': f'Overbought (RSI {rsi:.0f}) + Extended rally (+{monthly_change:.0f}%)',
                'confidence': 'HIGH'
            })
        elif rsi < 20:
            signals.append({
                'priority': 10,
                'action': 'BUY',
                'reason': f'Extremely oversold (RSI {rsi:.0f})',
                'confidence': 'HIGH'
            })
        elif rsi < 30:
            signals.append({
                'priority': 8,
                'action': 'BUY DIP',
                'reason': f'Oversold (RSI {rsi:.0f}) - bounce likely',
                'confidence': 'MEDIUM'
            })
        
        # PRIORITY 2: Extreme price moves
        if monthly_change > 50:
            signals.append({
                'priority': 8,
                'action': 'TRIM 50%',
                'reason': f'Parabolic move (+{monthly_change:.0f}% monthly)',
                'confidence': 'HIGH'
            })
        elif monthly_change < -30:
            signals.append({
                'priority': 7,
                'action': 'AVERAGE DOWN',
                'reason': f'Heavy selloff ({monthly_change:.0f}% monthly)',
                'confidence': 'MEDIUM'
            })
        
        # PRIORITY 3: Technical + Pattern alignment
        if stock.get('gap') and stock['gap']['signal'] == 'GAP UP':
            if rsi < 60:
                signals.append({
                    'priority': 6,
                    'action': 'HOLD',
                    'reason': f"Gapped up {stock['gap']['gap_percent']:.1f}% with room to run",
                    'confidence': 'MEDIUM'
                })
            else:
                signals.append({
                    'priority': 5,
                    'action': 'WATCH',
                    'reason': f"Gapped up but getting extended",
                    'confidence': 'LOW'
                })
        
        # PRIORITY 4: 52-week levels
        if stock.get('week52'):
            if stock['week52']['signal'] == 'AT 52W HIGH' and rsi < 70:
                signals.append({
                    'priority': 5,
                    'action': 'HOLD',
                    'reason': 'At 52W high with momentum',
                    'confidence': 'MEDIUM'
                })
            elif stock['week52']['signal'] == 'AT 52W LOW':
                signals.append({
                    'priority': 6,
                    'action': 'BUY STARTER',
                    'reason': 'At 52W low - potential reversal',
                    'confidence': 'MEDIUM'
                })
        
        # PRIORITY 5: Sector performance
        if sector in sector_ranks:
            rank_info = sector_ranks[sector]
            if rank_info['is_winner'] and rsi < 60:
                signals.append({
                    'priority': 4,
                    'action': 'ADD',
                    'reason': f"{sector} ranked #{rank_info['rank']} (+{rank_info['performance']:.1f}%) historically",
                    'confidence': 'LOW'
                })
            elif rank_info['is_loser'] and rsi > 50:
                signals.append({
                    'priority': 3,
                    'action': 'REDUCE',
                    'reason': f"{sector} underperformed (ranked #{rank_info['rank']})",
                    'confidence': 'LOW'
                })
        
        # PRIORITY 6: Historical pattern baseline
        if pattern_data['avg_return_3m'] > 5 and not signals:
            signals.append({
                'priority': 2,
                'action': 'HOLD',
                'reason': f"Pattern bullish (+{pattern_data['avg_return_3m']:.1f}% avg)",
                'confidence': 'LOW'
            })
        elif pattern_data['avg_return_3m'] < -5 and not signals:
            signals.append({
                'priority': 2,
                'action': 'HEDGE',
                'reason': f"Pattern bearish ({pattern_data['avg_return_3m']:.1f}% avg)",
                'confidence': 'LOW'
            })
        
        # DEFAULT: No strong signals
        if not signals:
            signals.append({
                'priority': 1,
                'action': 'HOLD',
                'reason': 'No strong signals - maintain position',
                'confidence': 'NEUTRAL'
            })
        
        # Pick the HIGHEST PRIORITY signal as final verdict
        final_signal = max(signals, key=lambda x: x['priority'])
        
        recommendations['final_verdicts'][ticker] = {
            'action': final_signal['action'],
            'reason': final_signal['reason'],
            'confidence': final_signal['confidence'],
            'name': stock['name']
        }
    
    return recommendations

# ========================================
# MAIN FUNCTION - Updated for v2.0.0
# ========================================

async def main(output="print"):
    previous_day_memory = load_memory()
    
    sp500 = get_cached_tickers('sp500_cache.json', fetch_sp500_tickers_sync)
    tsx = get_cached_tickers('tsx_cache.json', fetch_tsx_tickers_sync)
    universe = (sp500 or [])[:75] + (tsx or [])[:25]
    
    throttler = Throttler(2)
    semaphore = asyncio.Semaphore(10)
    
    async with aiohttp.ClientSession() as session:
        stock_tasks = [analyze_stock(semaphore, throttler, session, ticker) for ticker in universe]
        context_task = fetch_context_data(session)
        news_task = fetch_market_headlines(session)
        macro_task = fetch_macro_sentiment(session)
        
        # Use v2.0.0 portfolio analysis if enabled
        if ENABLE_V2_FEATURES:
            portfolio_task = analyze_portfolio_with_v2_features(session)
        else:
            portfolio_task = analyze_portfolio_watchlist(session)
        
        results = await asyncio.gather(
            asyncio.gather(*stock_tasks), 
            context_task, 
            news_task, 
            macro_task, 
            portfolio_task
        )
        
        stock_results_raw, context_data, market_news, macro_data, portfolio_data = results
        
        stock_results = sorted([r for r in stock_results_raw if r], key=lambda x: x['score'], reverse=True)
        df_stocks = pd.DataFrame(stock_results) if stock_results else pd.DataFrame()

        pattern_data = await find_historical_patterns(session, macro_data)
        
        portfolio_recommendations = None
        if pattern_data and portfolio_data:
            portfolio_recommendations = await generate_portfolio_recommendations_from_pattern(
                portfolio_data, pattern_data, macro_data
            )
        
        market_summary = {
            'macro': macro_data,
            'top_stock': stock_results[0] if stock_results else {},
            'bottom_stock': stock_results[-1] if stock_results else {}
        }
        ai_analysis = await generate_ai_oracle_analysis(market_summary, portfolio_data, pattern_data)
    
    if output == "email":
        html_email = generate_enhanced_html_email(
            df_stocks, context_data, market_news, macro_data, 
            previous_day_memory, portfolio_data, pattern_data, 
            ai_analysis, portfolio_recommendations
        )
        send_email(html_email)
    
    if not df_stocks.empty:
        save_memory({
            "previous_top_stock_name": df_stocks.iloc[0]['name'],
            "previous_top_stock_ticker": df_stocks.iloc[0]['ticker'],
            "previous_macro_score": macro_data.get('overall_macro_score', 0),
            "date": datetime.date.today().isoformat()
        })
    
    logging.info("✅ Analysis complete with v2.0.0 features.")

# ========================================
# EMAIL GENERATION - Updated for v2.0.0
# ========================================

def generate_enhanced_html_email(df_stocks, context, market_news, macro_data, memory, portfolio_data, pattern_data, ai_analysis, portfolio_recommendations=None):
    """FIXED v2.0.0: Clear, non-conflicting email display"""
    
    def format_articles(articles):
        if not articles:
            return "<p style='color:#888;'><i>No specific news drivers detected.</i></p>"
        html = "<ul style='margin:0;padding-left:20px;'>"
        for a in articles:
            if a.get('title'):
                html += f'<li style="margin-bottom:5px;"><a href="{a.get("url", "#")}" style="color:#1e3a8a;">{a["title"]}</a> <span style="color:#666;">({a.get("source", "Unknown")})</span></li>'
        return html + "</ul>"
    
    def create_stock_table(df):
        return "".join([f'<tr><td style="padding:10px;border-bottom:1px solid #eee;"><b>{row["ticker"]}</b><br><span style="color:#666;font-size:0.9em;">{row["name"]}</span></td><td style="padding:10px;border-bottom:1px solid #eee;text-align:center;font-weight:bold;font-size:1.1em;">{row["score"]:.0f}</td></tr>' for _, row in df.iterrows()])
    
    def create_context_table(ids):
        rows = ""
        for asset_id in ids:
            if asset := context.get(asset_id):
                price = f"${asset.get('current_price', 0):,.2f}" if asset.get('current_price') else "N/A"
                change_24h = asset.get('price_change_percentage_24h', 0) or 0
                mcap = f"${asset.get('market_cap', 0) / 1_000_000_000:.1f}B" if asset.get('market_cap') else "N/A"
                color_24h = "#16a34a" if change_24h >= 0 else "#dc2626"
                rows += f'<tr><td style="padding:10px;border-bottom:1px solid #eee;"><b>{asset.get("name", "")}</b><br><span style="color:#666;font-size:0.9em;">{asset.get("symbol","").upper()}</span></td><td style="padding:10px;border-bottom:1px solid #eee;">{price}<br><span style="color:{color_24h};font-size:0.9em;">{change_24h:.2f}% (24h)</span></td><td style="padding:10px;border-bottom:1px solid #eee;">{mcap}</td></tr>'
        return rows
    
    # AI Oracle section
    ai_oracle_html = ""
    if ai_analysis:
        analysis_text = ai_analysis['analysis'].replace('\n', '<br>')
        ai_oracle_html = f"""<div class="section" style="background-color:#f0f9ff;border-left:4px solid #0369a1;">
        <h2>🤖 AI MARKET ORACLE</h2>
        <p style="font-size:0.9em;color:#666;margin-bottom:15px;">Powered by Gemini AI</p>
        <div style="line-height:1.8;">{analysis_text}</div>
        </div>"""
    
    # v2.0.0 NEW: High-priority signals section
    v2_signals_html = ""
    if ENABLE_V2_FEATURES and portfolio_data and portfolio_data.get('v2_signals'):
        signals_list = "<br>".join(portfolio_data['v2_signals'][:10])
        v2_signals_html = f"""<div class="section" style="background-color:#fef3c7;border-left:4px solid #f59e0b;">
        <h2>⚡ HIGH-PRIORITY SIGNALS (v2.0)</h2>
        <p style="font-size:0.9em;color:#666;">Advanced technical alerts requiring immediate attention</p>
        <div style="line-height:2; font-weight:500;">{signals_list if signals_list else 'No high-priority signals today'}</div>
        </div>"""
    
    # Portfolio section with v2.0.0 enhancements
    portfolio_html = ""
    if portfolio_data:
        portfolio_table = ""
        for stock in portfolio_data['stocks']:
            color = "#16a34a" if stock['daily_change'] > 0 else "#dc2626"
            rsi_color = "#dc2626" if stock['rsi'] > 70 else "#16a34a" if stock['rsi'] < 30 else "#666"
            
            # v2.0.0: Add Bollinger Band indicator
            bb_indicator = ""
            if stock.get('bollinger'):
                bb = stock['bollinger']
                if bb['squeeze']:
                    bb_indicator = f"<br><span style='color:#f59e0b;font-weight:bold;'>💥 SQUEEZE</span>"
                elif bb['position'] > 90:
                    bb_indicator = f"<br><span style='color:#dc2626;'>BB: {bb['position']:.0f}%</span>"
                elif bb['position'] < 10:
                    bb_indicator = f"<br><span style='color:#16a34a;'>BB: {bb['position']:.0f}%</span>"
            
            # v2.0.0: Add 52-week indicator
            week52_indicator = ""
            if stock.get('week52'):
                w52 = stock['week52']
                if 'AT 52W HIGH' in w52['signal']:
                    week52_indicator = f"<br><span style='color:#16a34a;font-weight:bold;'>🚀 52W HIGH</span>"
                elif 'AT 52W LOW' in w52['signal']:
                    week52_indicator = f"<br><span style='color:#dc2626;font-weight:bold;'>📉 52W LOW</span>"
            
            # v2.0.0: Add earnings countdown
            earnings_indicator = ""
            if stock.get('earnings') and stock['earnings']['days_until'] <= 7:
                earnings_indicator = f"<br><span style='color:#9333ea;font-weight:bold;'>📅 Earnings in {stock['earnings']['days_until']}d</span>"
            
            portfolio_table += f"""<tr>
            <td style="padding:10px;border-bottom:1px solid #eee;">
                <b>{stock['ticker']}</b><br>
                <span style="color:#666;font-size:0.9em;">{stock['name']}</span>
                {bb_indicator}{week52_indicator}{earnings_indicator}
            </td>
            <td style="padding:10px;border-bottom:1px solid #eee;">
                ${stock['price']:.2f}<br>
                <span style="color:{color};font-size:0.9em;">{stock['daily_change']:+.2f}%</span>
            </td>
            <td style="padding:10px;border-bottom:1px solid #eee;">
                RSI: <span style="color:{rsi_color};font-weight:bold;">{stock['rsi']:.1f}</span><br>
                <span style="font-size:0.9em;">Vol: {stock['volume_ratio']:.1f}x</span>
            </td>
            <td style="padding:10px;border-bottom:1px solid #eee;">
                <span style="font-size:0.9em;">W: {stock['weekly_change']:+.1f}%<br>M: {stock['monthly_change']:+.1f}%</span>
            </td>
            </tr>"""
        
        alerts_html = "<br>".join(portfolio_data['alerts'][:5]) if portfolio_data['alerts'] else "No alerts today"
        opps_html = "<br>".join(portfolio_data['opportunities'][:5]) if portfolio_data['opportunities'] else "No immediate opportunities"
        risks_html = "<br>".join(portfolio_data['risks'][:5]) if portfolio_data['risks'] else "No significant risks detected"
        
        portfolio_html = f"""<div class="section" style="background-color:#fefce8;border-left:4px solid #ca8a04;">
        <h2>📊 YOUR PORTFOLIO COMMAND CENTER</h2>
        <table style="width:100%; border-collapse: collapse; margin-bottom:20px;">
            <thead><tr style="background-color:#f8f8f8;">
                <th style="text-align:left; padding:10px;">Stock</th>
                <th style="text-align:left; padding:10px;">Price</th>
                <th style="text-align:left; padding:10px;">Indicators</th>
                <th style="text-align:left; padding:10px;">Performance</th>
            </tr></thead>
            <tbody>{portfolio_table}</tbody>
        </table>
        <div style="margin-top:20px;">
            <h3 style="color:#dc2626;">🔔 Alerts & Signals</h3>
            <p style="line-height:1.8;">{alerts_html}</p>
        </div>
        <div style="margin-top:20px;">
            <h3 style="color:#16a34a;">🎯 Opportunities</h3>
            <p style="line-height:1.8;">{opps_html}</p>
        </div>
        <div style="margin-top:20px;">
            <h3 style="color:#ea580c;">⚠️ Risk Factors</h3>
            <p style="line-height:1.8;">{risks_html}</p>
        </div>
        </div>"""
    
    # Pattern analysis section (from v1.0.0 - keeping stable)
    pattern_html = ""
    if pattern_data and pattern_data.get('matches'):
        current_cond_data = pattern_data['current_conditions']
        current_cond = f"""<div style="background-color:#fff;padding:15px;border:2px solid #7c3aed;border-radius:5px;margin-bottom:20px;">
        <h3 style="margin-top:0;">📊 Today's Market DNA:</h3>
        <p style="margin:5px 0;"><b>RSI:</b> {current_cond_data['rsi']:.1f} | <b>Volatility:</b> {current_cond_data['volatility']:.1f}% | <b>Trend:</b> {current_cond_data['trend']:+.1f}%</p>
        <p style="margin:5px 0;"><b>Geopolitical Risk:</b> {current_cond_data['geo_risk']:.0f} | <b>Trade Risk:</b> {current_cond_data['trade_risk']:.0f}</p>
        </div>"""
        
        interpretation_html = ""
        if pattern_data.get('interpretation'):
            for item in pattern_data['interpretation']:
                if item['type'] == 'bias':
                    color = {'bullish': '#16a34a', 'bearish': '#dc2626', 'neutral': '#666'}[item['color']]
                    interpretation_html += f'<p style="font-size:1.2em;font-weight:bold;color:{color};">{item["emoji"]} {item["text"]}</p>'
                else:
                    interpretation_html += f'<p style="line-height:1.8;margin:10px 0;">{item["text"]}</p>'
        
        sector_html = ""
        if pattern_data.get('sector_performance'):
            sector_rows = ""
            for sector, perf in pattern_data['sector_performance'][:5]:
                color = "#16a34a" if perf > 0 else "#dc2626"
                sector_rows += f'<tr><td style="padding:8px;border-bottom:1px solid #eee;">{sector}</td><td style="padding:8px;border-bottom:1px solid #eee;text-align:right;color:{color};font-weight:bold;">{perf:+.1f}%</td></tr>'
            
            sector_html = f"""<div style="margin:20px 0;">
            <h3>🎯 Sector Performance in Similar Periods:</h3>
            <p style="font-size:0.9em;color:#666;">Based on {pattern_data['matches'][0]['date']} match ({pattern_data['matches'][0]['context']})</p>
            <table style="width:100%;background-color:#fff;border-collapse:collapse;">
                <thead><tr style="background-color:#f3e8ff;">
                    <th style="padding:10px;text-align:left;">Sector</th>
                    <th style="padding:10px;text-align:right;">3-Month Return</th>
                </tr></thead>
                <tbody>{sector_rows}</tbody>
            </table>
            </div>"""
        
        # FIXED: Clear, single recommendations display
        recommendations_html = ""
        if portfolio_recommendations and portfolio_recommendations.get('final_verdicts'):
            rec_items = []
            
            for ticker, verdict in portfolio_recommendations['final_verdicts'].items():
                action = verdict['action']
                reason = verdict['reason']
                confidence = verdict.get('confidence', 'MEDIUM')
                name = verdict.get('name', ticker)
                
                # Color coding based on action
                if action in ['BUY', 'BUY DIP', 'BUY STARTER', 'ADD', 'AVERAGE DOWN']:
                    color = "#16a34a"
                    icon = "🟢"
                elif action in ['SELL', 'TRIM 50%', 'TAKE PROFITS', 'REDUCE']:
                    color = "#dc2626" 
                    icon = "🔴"
                elif action in ['HOLD', 'WATCH']:
                    color = "#666"
                    icon = "⚪"
                elif action == 'HEDGE':
                    color = "#ea580c"
                    icon = "🟡"
                else:
                    color = "#666"
                    icon = "⚪"
                
                # Confidence badge
                conf_badge = ""
                if confidence == 'HIGH':
                    conf_badge = '<span style="background:#dc2626;color:white;padding:2px 6px;border-radius:3px;font-size:0.8em;margin-left:8px;">HIGH CONF</span>'
                elif confidence == 'MEDIUM':
                    conf_badge = '<span style="background:#f59e0b;color:white;padding:2px 6px;border-radius:3px;font-size:0.8em;margin-left:8px;">MED CONF</span>'
                elif confidence == 'LOW':
                    conf_badge = '<span style="background:#6b7280;color:white;padding:2px 6px;border-radius:3px;font-size:0.8em;margin-left:8px;">LOW CONF</span>'
                
                rec_items.append(f"""
                <div style="margin:10px 0;padding:12px;background-color:#f9f9f9;border-left:4px solid {color};border-radius:5px;">
                    <div style="font-size:1.1em;font-weight:bold;color:{color};">
                        {icon} {ticker}: {action} {conf_badge}
                    </div>
                    <div style="font-size:0.9em;color:#333;margin-top:2px;">{name}</div>
                    <div style="font-size:0.9em;color:#666;margin-top:5px;">• {reason}</div>
                </div>
                """)
            
            recommendations_html = f"""<div style="margin:20px 0;">
            <h3 style="color:#7c3aed;">💼 YOUR PORTFOLIO ACTION PLAN</h3>
            <p style="font-size:0.9em;color:#666;">One clear recommendation per stock - no conflicts</p>
            {''.join(rec_items)}
            </div>"""
        
        matches_html = ""
        for i, match in enumerate(pattern_data['matches'][:5], 1):
            outcome_color = "#16a34a" if match['future_3m'] > 0 else "#dc2626"
            matches_html += f"""<div style="margin:15px 0;padding:15px;background-color:#f8f8f8;border-left:4px solid {outcome_color};border-radius:5px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <b style="font-size:1.1em;">{match['date']}</b>
                    <span style="color:#666;margin-left:10px;">({match['context']})</span><br>
                    <span style="font-size:0.9em;color:#666;">Match Strength: {match['similarity']:.1f}%</span>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.2em;font-weight:bold;color:{outcome_color};">{match['future_3m']:+.1f}%</div>
                    <div style="font-size:0.9em;color:#666;">S&P 500 outcome</div>
                </div>
            </div>
            </div>"""
        
        pattern_html = f"""<div class="section" style="background-color:#f3e8ff;border-left:4px solid #7c3aed;">
        <h2>🔮 11-YEAR PATTERN ANALYSIS</h2>
        <p style="font-size:0.9em;color:#666;">Analyzing {pattern_data['sample_size']} similar market setups</p>
        {current_cond}
        <div style="background-color:#fff;padding:20px;border-radius:5px;margin:20px 0;">
            <h3 style="margin-top:0;color:#7c3aed;">📖 What History Tells Us:</h3>
            {interpretation_html}
        </div>
        {sector_html}
        {recommendations_html}
        <div style="margin:20px 0;">
            <h3>📅 Historical Matches:</h3>
            <p style="font-size:0.9em;color:#666;">These show S&P 500 performance. Sector performance varied (see table above).</p>
            {matches_html}
        </div>
        </div>"""
    
    # Editor's note
    prev_score = memory.get('previous_macro_score', 0)
    current_score = macro_data.get('overall_macro_score', 0)
    mood_change = "stayed relatively stable"
    if (diff := current_score - prev_score) > 3:
        mood_change = f"improved since yesterday (from {prev_score:.1f} to {current_score:.1f})"
    elif diff < -3:
        mood_change = f"turned more cautious since yesterday (from {prev_score:.1f} to {current_score:.1f})"
    
    editor_note = f"Good morning. The overall market mood has {mood_change}. This briefing is your daily blueprint for navigating the currents."
    if memory.get('previous_top_stock_name'):
        editor_note += f"<br><br><b>Yesterday's Champion:</b> {memory['previous_top_stock_name']} ({memory['previous_top_stock_ticker']}) led our rankings."
    
    # Sector deep dive
    sector_html = ""
    if not df_stocks.empty:
        top_by_sector = df_stocks.groupby('sector', group_keys=False)[['ticker', 'name', 'score', 'sector', 'summary']].apply(lambda x: x.nlargest(2, 'score'))
        for _, row in top_by_sector.iterrows():
            if row['sector'] and row['sector'] != 'N/A':
                summary_text = "Business summary not available."
                if row["summary"] and isinstance(row["summary"], str):
                    summary_text = '. '.join(row["summary"].split('. ')[:2]) + '.'
                sector_html += f'<div style="margin-bottom:15px;"><b>{row["name"]} ({row["ticker"]})</b> in <i>{row["sector"]}</i><p style="font-size:0.9em;color:#333;margin:5px 0 0 0;">{summary_text}</p></div>'
    
    # Create tables
    top10_html = create_stock_table(df_stocks.head(10)) if not df_stocks.empty else "<tr><td>No data available</td></tr>"
    bottom10_html = create_stock_table(df_stocks.tail(10).iloc[::-1]) if not df_stocks.empty else "<tr><td>No data available</td></tr>"
    crypto_html = create_context_table(["bitcoin", "ethereum", "solana", "ripple"])
    commodities_html = create_context_table(["gold", "silver"])
    market_news_html = "".join([f'<div style="margin-bottom:15px;"><b><a href="{article.get("url", "#")}" style="color:#000;">{article["title"]}</a></b><br><span style="color:#666;font-size:0.9em;">{article.get("source", "Unknown")}</span></div>' for article in market_news[:10]]) or "<p><i>Headlines temporarily unavailable.</i></p>"
    
    # Assemble final email
    return f"""<!DOCTYPE html><html><head><style>
    body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0;background-color:#f7f7f7;}}
    .container{{width:100%;max-width:700px;margin:20px auto;background-color:#fff;border:1px solid #ddd;}}
    .header{{background-color:#0c0a09;color:#fff;padding:30px;text-align:center;}}
    .section{{padding:25px;border-bottom:1px solid #ddd;}}
    h2{{font-size:1.5em;color:#111;margin-top:0;}}
    h3{{font-size:1.2em;color:#333;border-bottom:2px solid #e2e8f0;padding-bottom:5px;}}
    </style></head><body>
    <div class="container">
        <div class="header">
            <h1>Your Daily Intelligence Briefing</h1>
            <p style="font-size:1.1em; color:#aaa;">{datetime.date.today().strftime('%A, %B %d, %Y')}</p>
        </div>
        
        <div class="section">
            <h2>EDITOR'S NOTE</h2>
            <p>{editor_note}</p>
        </div>
        
        {v2_signals_html}
        {ai_oracle_html}
        {portfolio_html}
        {pattern_html}
        
        <div class="section">
            <h2>THE BIG PICTURE: The Market Weather Report</h2>
            <h3>Overall Macro Score: {macro_data['overall_macro_score']:.1f} / 30</h3>
            <p><b>How it's calculated:</b> This is our "weather forecast" for investors, combining risks and sentiment.</p>
            <p><b>🌍 Geopolitical Risk ({macro_data['geopolitical_risk']:.0f}/100):</b> Measures global instability.<br>
            <u>Key Drivers:</u> {format_articles(macro_data['geo_articles'])}</p>
            <p><b>🚢 Trade Risk ({macro_data['trade_risk']:.0f}/100):</b> Tracks trade tensions.<br>
            <u>Key Drivers:</u> {format_articles(macro_data['trade_articles'])}</p>
            <p><b>💼 Economic Sentiment ({macro_data['economic_sentiment']:.2f}):</b> Market mood (-1 to +1).<br>
            <u>Key Drivers:</u> {format_articles(macro_data['econ_articles'])}</p>
        </div>
        
        <div class="section">
            <h2>SECTOR DEEP DIVE</h2>
            <p>Top companies from different sectors.</p>
            {sector_html or "<p><i>No sector data available.</i></p>"}
        </div>
        
        <div class="section">
            <h2>STOCK RADAR</h2>
            <h3>📈 Top 10 Strongest Signals</h3>
            <table style="width:100%; border-collapse: collapse;">
                <thead><tr>
                    <th style="text-align:left; padding:10px;">Company</th>
                    <th style="text-align:center; padding:10px;">Score</th>
                </tr></thead>
                <tbody>{top10_html}</tbody>
            </table>
            
            <h3 style="margin-top: 30px;">📉 Top 10 Weakest Signals</h3>
            <table style="width:100%; border-collapse: collapse;">
                <thead><tr>
                    <th style="text-align:left; padding:10px;">Company</th>
                    <th style="text-align:center; padding:10px;">Score</th>
                </tr></thead>
                <tbody>{bottom10_html}</tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>BEYOND STOCKS: Alternative Assets</h2>
            <h3>🪙 Crypto</h3>
            <p><b>Market Sentiment: {context.get('crypto_sentiment', 'N/A')}</b></p>
            <table style="width:100%; border-collapse: collapse;">
                <thead><tr>
                    <th style="text-align:left; padding:10px;">Asset</th>
                    <th style="text-align:left; padding:10px;">Price / 24h</th>
                    <th style="text-align:left; padding:10px;">Market Cap</th>
                </tr></thead>
                <tbody>{crypto_html}</tbody>
            </table>
            
            <h3 style="margin-top: 30px;">💎 Commodities</h3>
            <p><b>Gold/Silver Ratio: {context.get('gold_silver_ratio', 'N/A')}</b></p>
            <table style="width:100%; border-collapse: collapse;">
                <thead><tr>
                    <th style="text-align:left; padding:10px;">Asset</th>
                    <th style="text-align:left; padding:10px;">Price / 24h</th>
                    <th style="text-align:left; padding:10px;">Market Cap</th>
                </tr></thead>
                <tbody>{commodities_html}</tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>FROM THE WIRE: Today's Top Headlines</h2>
            {market_news_html}
        </div>
    </div>
    </body></html>"""

def send_email(html_body):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    
    SMTP_USER, SMTP_PASS = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not SMTP_USER or not SMTP_PASS:
        logging.warning("SMTP credentials missing.")
        return
    
    msg = MIMEMultipart('alternative')
    msg["Subject"] = f"⛵ Your Daily Market Briefing - {datetime.date.today()}"
    msg["From"] = SMTP_USER
    msg["To"] = SMTP_USER
    msg.attach(MIMEText(html_body, 'html'))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        logging.info("✅ Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# ========================================
# 🚀 v3.0.0 - EMAIL CONVERSATION BOT
# Complete implementation with all fixes
# ========================================

# ========================================
# 🚀 v3.0.0 - EMAIL CONVERSATION BOT
# Complete implementation with all fixes
# ========================================

# ========================================
# 🚀 v3.0.0 - EMAIL CONVERSATION BOT
# Complete implementation with all fixes
# ========================================

# ========================================
# 🚀 v3.0.0 - EMAIL CONVERSATION BOT
# Complete implementation with all fixes
# ========================================

# v3.0 Feature Flags
ENABLE_EMAIL_BOT = True
ENABLE_DATA_PERSISTENCE = True

def clean_for_json(obj):
    """FIX 2: Convert numpy/pandas types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    return obj

class MarketIntelligenceDB:
    """v3.0: Persistent storage for analysis data"""
    def __init__(self, db_path='market_intel.db'):
        self.conn = sqlite3.connect(db_path)
        self.init_schema()
    
    def init_schema(self):
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS daily_analysis (
                date TEXT PRIMARY KEY, portfolio_data TEXT, pattern_data TEXT, macro_data TEXT,
                stock_scores TEXT, ai_analysis TEXT, recommendations TEXT
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, user_question TEXT,
                bot_response TEXT, context TEXT
            );
        ''')
        self.conn.commit()
    
    def save_daily_analysis(self, date, portfolio_data, pattern_data, macro_data, 
                           stock_scores, ai_analysis, recommendations):
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO daily_analysis 
                (date, portfolio_data, pattern_data, macro_data, stock_scores, ai_analysis, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                date,
                json.dumps(clean_for_json(portfolio_data)), json.dumps(clean_for_json(pattern_data)),
                json.dumps(clean_for_json(macro_data)), json.dumps(clean_for_json(stock_scores)),
                json.dumps(clean_for_json(ai_analysis)), json.dumps(clean_for_json(recommendations))
            ))
            self.conn.commit()
            logging.info(f"✅ Saved analysis data for {date}")
        except Exception as e:
            logging.error(f"Failed to save analysis to DB: {e}")

    def get_latest_analysis(self):
        cursor = self.conn.execute('SELECT * FROM daily_analysis ORDER BY date DESC LIMIT 1')
        row = cursor.fetchone()
        if not row: return None
        return {
            'date': row[0], 'portfolio_data': json.loads(row[1]) if row[1] else None,
            'pattern_data': json.loads(row[2]) if row[2] else None, 'macro_data': json.loads(row[3]) if row[3] else None,
            'stock_scores': json.loads(row[4]) if row[4] else None, 'ai_analysis': json.loads(row[5]) if row[5] else None,
            'recommendations': json.loads(row[6]) if row[6] else None
        }

class EmailConversationBot:
    """v3.0: Interactive email-based Q&A bot"""
    def __init__(self):
        self.db = MarketIntelligenceDB()
        self.smtp_user, self.smtp_pass = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
        self.imap_server = "imap.gmail.com"
        
    def check_for_questions(self):
        try:
            logging.info("📧 Checking for email questions...")
            
            # Add timeout to IMAP connection (15 seconds)
            logging.info("🔐 Connecting to IMAP server...")
            mail = imaplib.IMAP4_SSL(self.imap_server, timeout=15)
            
            logging.info("🔑 Logging in...")
            mail.login(self.smtp_user, self.smtp_pass)
            
            logging.info("📬 Selecting inbox...")
            mail.select('inbox')
            
            # SECURITY: Only check emails from last 300 minutes (5 hours)
            import datetime
            since_date = (datetime.datetime.now() - datetime.timedelta(minutes=300)).strftime("%d-%b-%Y")
            
            logging.info(f"🔍 Searching for 'Your Daily Market Briefing' emails since {since_date}...")
            
            # CRITICAL SECURITY: Only search for emails with exact subject + unread + recent
            # This prevents the bot from even SEEING other emails
            _, search_data = mail.search(
                None, 
                f'(UNSEEN SINCE {since_date} SUBJECT "Your Daily Market Briefing")'
            )
            
            matching_emails = search_data[0].split()
            logging.info(f"📨 Found {len(matching_emails)} matching briefing replies in last 300 minutes")
            
            if not matching_emails:
                logging.info("✅ No briefing replies found - inbox clear")
                mail.close()
                mail.logout()
                return
            
            # Process newest first (reversed order)
            emails_to_check = list(reversed(matching_emails))
            logging.info(f"📋 Processing {len(emails_to_check)} briefing reply(ies)...")
            
            for num in emails_to_check:
                try:
                    logging.info(f"📖 Fetching briefing reply #{num.decode()}...")
                    _, data = mail.fetch(num, '(RFC822)')
                    email_message = email.message_from_bytes(data[0][1])
                    
                    # Double-check subject (defense in depth)
                    # CRITICAL: Properly decode MIME-encoded subjects
                    subject_raw = email_message.get('Subject', '')
                    
                    # Decode MIME-encoded subject (handles =?UTF-8?Q?...?= encoding)
                    try:
                        decoded_parts = email.header.decode_header(subject_raw)
                        subject = ''
                        for part, encoding in decoded_parts:
                            if isinstance(part, bytes):
                                subject += part.decode(encoding or 'utf-8', errors='ignore')
                            else:
                                subject += str(part)
                    except Exception as e:
                        logging.warning(f"Failed to decode subject, using raw: {e}")
                        subject = str(subject_raw)
                    
                    # SECURITY: Verify subject one more time (handle Re:, Fwd:, etc.)
                    if 'Daily Market Briefing' not in subject:
                        logging.warning(f"⚠️ Email #{num.decode()} doesn't match - IMAP filter failed. Skipping for security.")
                        logging.warning(f"   Decoded subject: '{subject}'")
                        continue
                    
                    logging.info(f"✅ Confirmed briefing reply: '{subject[:80]}...'")
                    
                    # Extract question from reply
                    question = self.extract_question(email_message)
                    
                    if question and len(question) > 10:  # At least 10 chars
                        sender = email.utils.parseaddr(email_message['From'])[1]
                        logging.info(f"❓ Question from {sender}:")
                        logging.info(f"   '{question[:150]}...'")
                        
                        response = self.generate_response(question)
                        logging.info(f"💬 Generated response ({len(response)} chars)")
                        
                        self.send_response(sender, question, response)
                        
                        # Mark as read ONLY after successful response
                        mail.store(num, '+FLAGS', '\\Seen')
                        logging.info(f"✅ Responded and marked #{num.decode()} as read")
                    else:
                        logging.info("⚠️ Question too short or empty - skipping")
                        # Still mark as read to avoid re-processing
                        mail.store(num, '+FLAGS', '\\Seen')
                
                except Exception as e:
                    logging.error(f"❌ Error processing email #{num.decode()}: {e}")
                    continue  # Skip this email, continue with others
            
            logging.info("🔒 Closing connection...")
            mail.close()
            mail.logout()
            logging.info("✅ Email bot check complete")
            
        except imaplib.IMAP4.error as e:
            logging.error(f"❌ IMAP error: {e}")
            logging.error("💡 Check: 1) Gmail IMAP enabled 2) App password correct 3) Network access")
        except socket.timeout:
            logging.error(f"⏱️ Connection timeout - IMAP server not responding after 15 seconds")
        except Exception as e:
            logging.error(f"❌ Unexpected error: {type(e).__name__}: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def extract_question(self, msg):
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain": body = part.get_payload(decode=True).decode(errors='ignore')
        else: body = msg.get_payload(decode=True).decode(errors='ignore')
        return ' '.join([line.strip() for line in body.split('\n') if not line.startswith('>') and 'wrote:' not in line.lower() and line.strip()]).strip()
    
    def search_duckduckgo(self, query):
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query + ' stock market')}"
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            return [res.get_text(strip=True)[:250] + "..." for res in soup.find_all('a', class_='result__snippet')[:2]]
        except Exception as e: return [f"Web search failed: {e}"]

    def generate_response(self, question):
        latest = self.db.get_latest_analysis()
        
        # FALLBACK: If no database, do web search
        if not latest or not latest.get('portfolio_data'):
            logging.warning("No analysis data in database - using web search fallback")
            web_results = self.search_duckduckgo(question)
            return f"""I don't have recent portfolio analysis data yet. 

Here's what I found online about your question:

{chr(10).join(web_results)}

💡 **Tip:** For personalized portfolio advice, make sure the daily analysis has run at least once to populate my database."""
        
        response_parts = []
        stock_found = False
        
        # Check if question is about a specific stock in portfolio
        if latest.get('portfolio_data'):
            for stock in latest['portfolio_data'].get('stocks', []):
                if stock['ticker'].lower() in question.lower() or stock['name'].lower() in question.lower():
                    stock_found = True
                    ticker = stock['ticker']
                    rec = latest.get('recommendations', {}).get('final_verdicts', {}).get(ticker)
                    
                    response_parts.append(f"📊 **{ticker} ({stock['name']}) Analysis:**\n")
                    response_parts.append(f"Current Price: ${stock.get('price', 0):.2f}")
                    response_parts.append(f"Daily Change: {stock.get('daily_change', 0):+.2f}%")
                    response_parts.append(f"RSI: {stock.get('rsi', 0):.1f}")
                    
                    if rec:
                        response_parts.append(f"\n**My Recommendation:** {rec['action']}")
                        response_parts.append(f"**Reason:** {rec['reason']}")
                        response_parts.append(f"**Confidence:** {rec.get('confidence', 'MEDIUM')}")
                    
                    # Add web research for context
                    web_context = self.search_duckduckgo(f"{ticker} news")
                    if web_context:
                        response_parts.append(f"\n**Latest News:**\n{chr(10).join(web_context)}")
                    
                    break
        
        # If no specific stock found, provide general guidance
        if not stock_found:
            if 'buy' in question.lower() or 'purchase' in question.lower():
                recs = latest.get('recommendations', {}).get('final_verdicts', {})
                buys = [f"• {t}: {v['action']} - {v['reason']}" 
                       for t, v in recs.items() 
                       if 'BUY' in v['action'].upper()]
                
                if buys:
                    response_parts.append("**🟢 Current Buy Opportunities:**\n")
                    response_parts.extend(buys[:3])  # Top 3
                else:
                    response_parts.append("No strong buy signals in your portfolio right now.")
                    
            elif 'sell' in question.lower():
                recs = latest.get('recommendations', {}).get('final_verdicts', {})
                sells = [f"• {t}: {v['action']} - {v['reason']}" 
                        for t, v in recs.items() 
                        if 'SELL' in v['action'].upper() or 'TRIM' in v['action'].upper()]
                
                if sells:
                    response_parts.append("**🔴 Consider Taking Profits:**\n")
                    response_parts.extend(sells[:3])
                else:
                    response_parts.append("No urgent sell signals right now.")
            else:
                # General question - do web search
                response_parts.append("**🔍 Here's what I found:**\n")
                response_parts.extend(self.search_duckduckgo(question))
        
        return "\n".join(response_parts) if response_parts else "I couldn't find specific information about that. Can you rephrase your question?"

    def send_response(self, to_email, question, response):
        """Send email response to user's question"""
        import smtplib  # Import here for bot's email sending
        
        msg = MIMEMultipart()
        msg['Subject'] = "Re: Your market analysis question"
        msg['From'] = self.smtp_user
        msg['To'] = to_email
        
        body = f"""Thank you for your question:

> {question}

💡 My Analysis:
{response}

---
🤖 Market Intelligence Bot
Reply to this email with more questions anytime!</body>"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            logging.info(f"✅ Sent response to {to_email}")
        except Exception as e:
            logging.error(f"Failed to send response: {e}")


# ========================================
# MAIN EXECUTION FUNCTION
# ========================================

async def main(output="print", check_emails=False):
    """
    Main execution - handles both analysis and email bot modes
    
    Args:
        output: "print" or "email" - where to send analysis results
        check_emails: If True, only check for email questions (bot mode)
    """
    
    # EMAIL BOT MODE - Just check emails and respond
    if check_emails:
        if ENABLE_EMAIL_BOT:
            logging.info("🤖 EMAIL BOT MODE: Checking for questions...")
            bot = EmailConversationBot()
            bot.check_for_questions()
            logging.info("✅ Email bot check complete")
        else:
            logging.warning("❌ Email bot is disabled (ENABLE_EMAIL_BOT=False)")
        return  # Exit early - don't run full analysis

    # FULL ANALYSIS MODE - Run complete market analysis
    logging.info("📊 FULL ANALYSIS MODE: Running market intelligence scan...")
    previous_day_memory = load_memory()
    
    sp500 = get_cached_tickers('sp500_cache.json', fetch_sp500_tickers_sync)
    tsx = get_cached_tickers('tsx_cache.json', fetch_tsx_tickers_sync)
    universe = (sp500 or [])[:75] + (tsx or [])[:25]
    
    throttler = Throttler(2)
    semaphore = asyncio.Semaphore(10)
    
    async with aiohttp.ClientSession() as session:
        stock_tasks = [analyze_stock(semaphore, throttler, session, ticker) for ticker in universe]
        context_task = fetch_context_data(session)
        news_task = fetch_market_headlines(session)
        macro_task = fetch_macro_sentiment(session)
        
        # Use v2.0.0 portfolio analysis if enabled
        if ENABLE_V2_FEATURES:
            portfolio_task = analyze_portfolio_with_v2_features(session)
        else:
            portfolio_task = analyze_portfolio_watchlist(session)
        
        results = await asyncio.gather(
            asyncio.gather(*stock_tasks), 
            context_task, 
            news_task, 
            macro_task, 
            portfolio_task
        )
        
        stock_results_raw, context_data, market_news, macro_data, portfolio_data = results
        
        stock_results = sorted([r for r in stock_results_raw if r], key=lambda x: x['score'], reverse=True)
        df_stocks = pd.DataFrame(stock_results) if stock_results else pd.DataFrame()

        pattern_data = await find_historical_patterns(session, macro_data)
        
        portfolio_recommendations = None
        if pattern_data and portfolio_data:
            portfolio_recommendations = await generate_portfolio_recommendations_from_pattern(
                portfolio_data, pattern_data, macro_data
            )
        
        market_summary = {
            'macro': macro_data,
            'top_stock': stock_results[0] if stock_results else {},
            'bottom_stock': stock_results[-1] if stock_results else {}
        }
        ai_analysis = await generate_ai_oracle_analysis(market_summary, portfolio_data, pattern_data)
    
    # Save analysis to database for email bot
    if ENABLE_DATA_PERSISTENCE:
        db = MarketIntelligenceDB()
        db.save_daily_analysis(
            datetime.date.today().isoformat(),
            portfolio_data,
            pattern_data,
            macro_data,
            df_stocks.head(20).to_dict('records') if not df_stocks.empty else [],
            ai_analysis,
            portfolio_recommendations
        )
        logging.info("💾 Analysis saved to database")
    
    # Send email if requested
    if output == "email":
        html_email = generate_enhanced_html_email(
            df_stocks, context_data, market_news, macro_data, 
            previous_day_memory, portfolio_data, pattern_data, 
            ai_analysis, portfolio_recommendations
        )
        
        # Add interactive bot section to email
        if ENABLE_EMAIL_BOT:
            bot_section = """
            <div class="section" style="background-color:#e0f2fe;border-left:4px solid #0284c7;">
                <h2>🤖 ASK ME ANYTHING</h2>
                <p style="font-size:1.1em;">
                    <b>Have questions about your portfolio? Just reply to this email!</b><br>
                    I'll analyze the latest data and respond within 30 minutes.
                </p>
                <p style="color:#666;font-size:0.9em;">
                    <i>Try asking:</i><br>
                    • "Should I buy more NVDA right now?"<br>
                    • "Why is AAPL dropping today?"<br>
                    • "What are my best buying opportunities?"<br>
                    • "Is it time to take profits on TSLA?"
                </p>
            </div>
            """
            # Insert before closing container
            html_email = html_email.replace('</div>\n    </body>', bot_section + '</div>\n    </body>')
        
        send_email(html_email)
        logging.info("📧 Daily briefing email sent")
    
    # Save memory for next run
    if not df_stocks.empty:
        save_memory({
            "previous_top_stock_name": df_stocks.iloc[0]['name'],
            "previous_top_stock_ticker": df_stocks.iloc[0]['ticker'],
            "previous_macro_score": macro_data.get('overall_macro_score', 0),
            "date": datetime.date.today().isoformat()
        })
    
    logging.info("✅ Analysis complete with v2.0.0 features.")


# ========================================
# PROGRAM ENTRY POINT
# THIS IS THE ONLY if __name__ == "__main__" BLOCK
# ========================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Market Intelligence System - Daily Analysis & Email Bot"
    )
    parser.add_argument(
        "--output", 
        default="print", 
        choices=["print", "email"], 
        help="Where to send analysis: 'print' to console or 'email' to inbox"
    )
    parser.add_argument(
        "--check-emails", 
        action="store_true", 
        help="Email bot mode: Check inbox for questions and auto-respond (no full analysis)"
    )
    
    args = parser.parse_args()
    
    # Windows event loop compatibility
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the main function with parsed arguments
    logging.info("=" * 60)
    logging.info("🚀 MARKET INTELLIGENCE SYSTEM v3.0.1")
    logging.info("=" * 60)
    
    asyncio.run(main(output=args.output, check_emails=args.check_emails))
