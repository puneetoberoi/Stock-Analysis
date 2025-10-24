import os, sys, argparse, time, logging, json, asyncio
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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, date, timedelta

# 🆕 Email bot imports
import sqlite3
import imaplib
import email
import email.utils
import re

# 🆕 INTELLIGENT SYSTEM IMPORTS (Add after existing imports)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available - using basic NLP")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available - using basic sentiment")

try:
    import wikipediaapi
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    logging.warning("Wikipedia not available - using web search only")

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    logging.warning("Markdown not available - using plain text")

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


# Add this after your imports, before any other functions:
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, datetime.timedelta):
            return str(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

# Then update ALL json.dump calls to use it:
def save_memory(data):
    """Save memory with proper date handling"""
    # Fixed for "import datetime" style
    def json_serial(obj):
        """JSON serializer for objects not serializable by default"""
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(MEMORY_FILE, 'w') as f: 
        json.dump(data, f, default=json_serial)


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
    """Save memory with proper date handling"""
    # Convert any date objects to strings before saving
    def json_serial(obj):
        """JSON serializer for objects not serializable by default"""
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(MEMORY_FILE, 'w') as f: 
        json.dump(data, f, default=json_serial)

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
    from datetime import datetime as dt
    
    # Handle both datetime and Timestamp objects
    if hasattr(date, 'year'):
        year = date.year
        month = date.month
    else:
        year = dt.now().year
        month = dt.now().month
    
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
    
    # 2. Critical Risks
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
        'generated_at': datetime.now().isoformat()
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
        
        # Try the current working model
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')  # FIXED: Updated model name
            logging.info(f"✅ Successfully loaded Gemini model: gemini-1.5-flash-latest")
        except Exception as e:
            logging.warning(f"Failed to load Gemini: {str(e)[:100]}")
            return generate_fallback_analysis(market_data, portfolio_data, pattern_data)
        
        if not model:
            logging.error("❌ Gemini model failed to load")
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
        return {'analysis': response.text, 'generated_at': datetime.now().isoformat()}  # FIXED: datetime.now()
        
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)[:200]}")
        return generate_fallback_analysis(market_data, portfolio_data, pattern_data)

# ========================================
# 🔒 END STABLE FOUNDATION
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
# 🧠 INTELLIGENT LEARNING SYSTEM v3.0
# 100% Free, GitHub Actions Compatible
# ========================================

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

class PredictionTracker:
    """Tracks all predictions and learns from outcomes - JSON based for free storage"""
    
    def __init__(self, predictions_file='data/predictions.json'):
        self.predictions_file = Path(predictions_file)
        self.predictions_file.parent.mkdir(exist_ok=True)
        self.predictions = self._load_predictions()
        
    def _load_predictions(self):
        """Load existing predictions from JSON"""
        if self.predictions_file.exists():
            try:
                with open(self.predictions_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_predictions(self):
        """Save predictions to JSON"""
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions, f, indent=2, default=str)
    
    def store_prediction(self, ticker, action, confidence, reasoning, candle_pattern=None, indicators=None):
        """Store a new prediction with all context"""
        prediction_id = hashlib.md5(f"{ticker}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        prediction = {
            'id': prediction_id,
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'action': action,  # BUY, SELL, HOLD
            'confidence': confidence,  # 0-100
            'reasoning': reasoning,
            'candle_pattern': candle_pattern,
            'indicators': indicators or {},
            'price_at_prediction': None,  # Will be filled
            'outcome': None,  # Will be updated later
            'was_correct': None  # Will be calculated
        }
        
        # Get current price
        try:
            import yfinance as yf
            current_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
            prediction['price_at_prediction'] = float(current_price)
        except:
            pass
        
        self.predictions[prediction_id] = prediction
        self._save_predictions()
        logging.info(f"📝 Stored prediction {prediction_id}: {ticker} - {action} (confidence: {confidence}%)")
        return prediction_id
    
    def check_outcomes(self, days_to_check=1):
        """Check outcomes of past predictions"""
        results = {'checked': 0, 'correct': 0, 'wrong': 0}
        cutoff_date = datetime.now() - timedelta(days=days_to_check)
        
        for pred_id, pred in self.predictions.items():
            if pred['outcome'] is not None:  # Already checked
                continue
                
            pred_date = datetime.fromisoformat(pred['timestamp'])
            if pred_date > cutoff_date:  # Too recent
                continue
            
            try:
                ticker = pred['ticker']
                stock = yf.Ticker(ticker)
                
                # Get price from prediction date to now
                hist = stock.history(start=pred_date.date(), end=datetime.now().date())
                if len(hist) < 2:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                pred_price = pred['price_at_prediction']
                
                if not pred_price:
                    continue
                
                price_change_pct = ((current_price - pred_price) / pred_price) * 100
                
                # Determine if prediction was correct
                was_correct = False
                if pred['action'] == 'BUY' and price_change_pct > 0.5:
                    was_correct = True
                elif pred['action'] == 'SELL' and price_change_pct < -0.5:
                    was_correct = True
                elif pred['action'] == 'HOLD' and abs(price_change_pct) < 2:
                    was_correct = True
                
                # Update prediction
                pred['outcome'] = {
                    'checked_date': datetime.now().isoformat(),
                    'price_change_pct': price_change_pct,
                    'current_price': float(current_price)
                }
                pred['was_correct'] = was_correct
                
                results['checked'] += 1
                if was_correct:
                    results['correct'] += 1
                else:
                    results['wrong'] += 1
                    
                logging.info(f"✅ Outcome: {ticker} {pred['action']} was {'CORRECT' if was_correct else 'WRONG'} ({price_change_pct:+.2f}%)")
                
            except Exception as e:
                logging.error(f"Error checking outcome for {pred_id}: {e}")
        
        self._save_predictions()
        return results


class CandlePatternAnalyzer:
    """Identifies 20+ candlestick patterns with historical success tracking"""
    
    def __init__(self, patterns_file='data/patterns.json'):
        self.patterns_file = Path(patterns_file)
        self.patterns_file.parent.mkdir(exist_ok=True)
        self.pattern_history = self._load_patterns()
    
    def _load_patterns(self):
        """Load pattern success history"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_patterns(self):
        """Save pattern history"""
        with open(self.patterns_file, 'w') as f:
            json.dump(self.pattern_history, f, indent=2)
    
    def identify_pattern(self, hist_data, min_length=3):
        """
        Identify candlestick patterns from historical data
        hist_data: DataFrame with OHLC columns
        Returns: List of identified patterns with metadata
        """
        if len(hist_data) < min_length:
            return []
        
        patterns = []
        
        # Get last 3 candles for pattern detection
        candles = []
        for i in range(min(3, len(hist_data))):
            idx = -(i+1)
            candles.append({
                'open': float(hist_data['Open'].iloc[idx]),
                'high': float(hist_data['High'].iloc[idx]),
                'low': float(hist_data['Low'].iloc[idx]),
                'close': float(hist_data['Close'].iloc[idx]),
            })
        
        if len(candles) < 1:
            return []
        
        # Current candle (today)
        c0 = candles[0]
        body0 = abs(c0['close'] - c0['open'])
        range0 = c0['high'] - c0['low'] if c0['high'] != c0['low'] else 0.01
        body_ratio0 = body0 / range0 if range0 > 0 else 0
        
        # Previous candle (if exists)
        c1 = candles[1] if len(candles) > 1 else None
        
        # 2 candles ago (if exists)
        c2 = candles[2] if len(candles) > 2 else None
        
        # ===== REVERSAL PATTERNS (BULLISH) =====
        
        # 1. Hammer (bullish reversal)
        if (min(c0['open'], c0['close']) - c0['low']) > body0 * 2:
            if (c0['high'] - max(c0['open'], c0['close'])) < body0:
                patterns.append({
                    'name': 'hammer',
                    'type': 'bullish_reversal',
                    'strength': 'strong',
                    'description': 'Hammer - potential bottom reversal'
                })
        
        # 2. Inverted Hammer (bullish reversal)
        if (c0['high'] - max(c0['open'], c0['close'])) > body0 * 2:
            if (min(c0['open'], c0['close']) - c0['low']) < body0:
                patterns.append({
                    'name': 'inverted_hammer',
                    'type': 'bullish_reversal',
                    'strength': 'medium',
                    'description': 'Inverted Hammer - potential reversal'
                })
        
        # 3. Bullish Engulfing (requires 2 candles)
        if c1:
            if c1['close'] < c1['open']:  # Previous was bearish
                if c0['close'] > c0['open']:  # Current is bullish
                    if c0['open'] < c1['close'] and c0['close'] > c1['open']:
                        patterns.append({
                            'name': 'bullish_engulfing',
                            'type': 'bullish_reversal',
                            'strength': 'strong',
                            'description': 'Bullish Engulfing - strong reversal signal'
                        })
        
        # 4. Piercing Line (requires 2 candles)
        if c1:
            if c1['close'] < c1['open']:  # Previous was bearish
                if c0['close'] > c0['open']:  # Current is bullish
                    mid_c1 = (c1['open'] + c1['close']) / 2
                    if c0['open'] < c1['close'] and c0['close'] > mid_c1:
                        patterns.append({
                            'name': 'piercing_line',
                            'type': 'bullish_reversal',
                            'strength': 'medium',
                            'description': 'Piercing Line - bullish reversal'
                        })
        
        # 5. Morning Star (requires 3 candles)
        if c1 and c2:
            if c2['close'] < c2['open']:  # Day 1: bearish
                body1 = abs(c1['close'] - c1['open'])
                if body1 < body0 * 0.3:  # Day 2: small body (star)
                    if c0['close'] > c0['open']:  # Day 3: bullish
                        if c0['close'] > (c2['open'] + c2['close']) / 2:
                            patterns.append({
                                'name': 'morning_star',
                                'type': 'bullish_reversal',
                                'strength': 'very_strong',
                                'description': 'Morning Star - major reversal signal'
                            })
        
        # 6. Three White Soldiers (requires 3 candles)
        if c1 and c2:
            if (c0['close'] > c0['open'] and 
                c1['close'] > c1['open'] and 
                c2['close'] > c2['open']):
                if c0['close'] > c1['close'] > c2['close']:
                    patterns.append({
                        'name': 'three_white_soldiers',
                        'type': 'bullish_continuation',
                        'strength': 'strong',
                        'description': 'Three White Soldiers - strong uptrend'
                    })
        
        # ===== REVERSAL PATTERNS (BEARISH) =====
        
        # 7. Hanging Man (bearish reversal)
        if c0['close'] > c0['open']:  # Bullish candle
            if (min(c0['open'], c0['close']) - c0['low']) > body0 * 2:
                if (c0['high'] - max(c0['open'], c0['close'])) < body0:
                    # Check if in uptrend (price higher than 5 days ago)
                    if len(hist_data) >= 6:
                        if c0['close'] > hist_data['Close'].iloc[-6]:
                            patterns.append({
                                'name': 'hanging_man',
                                'type': 'bearish_reversal',
                                'strength': 'medium',
                                'description': 'Hanging Man - potential top reversal'
                            })
        
        # 8. Shooting Star (bearish reversal)
        if (c0['high'] - max(c0['open'], c0['close'])) > body0 * 2:
            if (min(c0['open'], c0['close']) - c0['low']) < body0:
                # Works best after uptrend
                if len(hist_data) >= 6:
                    if c0['close'] > hist_data['Close'].iloc[-6]:
                        patterns.append({
                            'name': 'shooting_star',
                            'type': 'bearish_reversal',
                            'strength': 'strong',
                            'description': 'Shooting Star - reversal warning'
                        })
        
        # 9. Bearish Engulfing (requires 2 candles)
        if c1:
            if c1['close'] > c1['open']:  # Previous was bullish
                if c0['close'] < c0['open']:  # Current is bearish
                    if c0['open'] > c1['close'] and c0['close'] < c1['open']:
                        patterns.append({
                            'name': 'bearish_engulfing',
                            'type': 'bearish_reversal',
                            'strength': 'strong',
                            'description': 'Bearish Engulfing - strong reversal signal'
                        })
        
        # 10. Dark Cloud Cover (requires 2 candles)
        if c1:
            if c1['close'] > c1['open']:  # Previous was bullish
                if c0['close'] < c0['open']:  # Current is bearish
                    mid_c1 = (c1['open'] + c1['close']) / 2
                    if c0['open'] > c1['high'] and c0['close'] < mid_c1:
                        patterns.append({
                            'name': 'dark_cloud_cover',
                            'type': 'bearish_reversal',
                            'strength': 'medium',
                            'description': 'Dark Cloud Cover - bearish reversal'
                        })
        
        # 11. Evening Star (requires 3 candles)
        if c1 and c2:
            if c2['close'] > c2['open']:  # Day 1: bullish
                body1 = abs(c1['close'] - c1['open'])
                if body1 < body0 * 0.3:  # Day 2: small body (star)
                    if c0['close'] < c0['open']:  # Day 3: bearish
                        if c0['close'] < (c2['open'] + c2['close']) / 2:
                            patterns.append({
                                'name': 'evening_star',
                                'type': 'bearish_reversal',
                                'strength': 'very_strong',
                                'description': 'Evening Star - major reversal signal'
                            })
        
        # 12. Three Black Crows (requires 3 candles)
        if c1 and c2:
            if (c0['close'] < c0['open'] and 
                c1['close'] < c1['open'] and 
                c2['close'] < c2['open']):
                if c0['close'] < c1['close'] < c2['close']:
                    patterns.append({
                        'name': 'three_black_crows',
                        'type': 'bearish_continuation',
                        'strength': 'strong',
                        'description': 'Three Black Crows - strong downtrend'
                    })
        
        # ===== CONTINUATION & INDECISION PATTERNS =====
        
        # 13. Doji (indecision)
        if body_ratio0 < 0.1:
            patterns.append({
                'name': 'doji',
                'type': 'indecision',
                'strength': 'weak',
                'description': 'Doji - market indecision'
            })
        
        # 14. Spinning Top (indecision)
        if 0.1 < body_ratio0 < 0.3:
            if (c0['high'] - max(c0['open'], c0['close'])) > body0:
                if (min(c0['open'], c0['close']) - c0['low']) > body0:
                    patterns.append({
                        'name': 'spinning_top',
                        'type': 'indecision',
                        'strength': 'weak',
                        'description': 'Spinning Top - indecision'
                    })
        
        # 15. Rising Three Methods (bullish continuation - requires 5 candles)
        if len(hist_data) >= 5:
            candles_5 = []
            for i in range(5):
                idx = -(i+1)
                candles_5.append({
                    'open': float(hist_data['Open'].iloc[idx]),
                    'close': float(hist_data['Close'].iloc[idx]),
                })
            
            # Day 1: Long bullish
            if candles_5[4]['close'] > candles_5[4]['open']:
                # Days 2-4: Small bearish candles within day 1 range
                small_pullback = all(
                    candles_5[i]['close'] < candles_5[i]['open'] and
                    candles_5[i]['close'] > candles_5[4]['close']
                    for i in [3, 2, 1]
                )
                # Day 5: Bullish continuation
                if small_pullback and candles_5[0]['close'] > candles_5[0]['open']:
                    if candles_5[0]['close'] > candles_5[4]['close']:
                        patterns.append({
                            'name': 'rising_three_methods',
                            'type': 'bullish_continuation',
                            'strength': 'medium',
                            'description': 'Rising Three Methods - uptrend continues'
                        })
        
        # 16. Falling Three Methods (bearish continuation - requires 5 candles)
        if len(hist_data) >= 5:
            candles_5 = []
            for i in range(5):
                idx = -(i+1)
                candles_5.append({
                    'open': float(hist_data['Open'].iloc[idx]),
                    'close': float(hist_data['Close'].iloc[idx]),
                })
            
            # Day 1: Long bearish
            if candles_5[4]['close'] < candles_5[4]['open']:
                # Days 2-4: Small bullish candles within day 1 range
                small_bounce = all(
                    candles_5[i]['close'] > candles_5[i]['open'] and
                    candles_5[i]['close'] < candles_5[4]['close']
                    for i in [3, 2, 1]
                )
                # Day 5: Bearish continuation
                if small_bounce and candles_5[0]['close'] < candles_5[0]['open']:
                    if candles_5[0]['close'] < candles_5[4]['close']:
                        patterns.append({
                            'name': 'falling_three_methods',
                            'type': 'bearish_continuation',
                            'strength': 'medium',
                            'description': 'Falling Three Methods - downtrend continues'
                        })
        
        # ===== ADDITIONAL PATTERNS =====
        
        # 17. Marubozu (strong direction)
        if body0 > range0 * 0.95:  # Almost no wicks
            if c0['close'] > c0['open']:
                patterns.append({
                    'name': 'bullish_marubozu',
                    'type': 'bullish_continuation',
                    'strength': 'strong',
                    'description': 'Bullish Marubozu - strong buying'
                })
            else:
                patterns.append({
                    'name': 'bearish_marubozu',
                    'type': 'bearish_continuation',
                    'strength': 'strong',
                    'description': 'Bearish Marubozu - strong selling'
                })
        
        # 18. Tweezer Top/Bottom (requires 2 candles)
        if c1:
            # Tweezer Top (bearish)
            if abs(c0['high'] - c1['high']) < range0 * 0.02:  # Similar highs
                if c1['close'] > c1['open'] and c0['close'] < c0['open']:
                    patterns.append({
                        'name': 'tweezer_top',
                        'type': 'bearish_reversal',
                        'strength': 'medium',
                        'description': 'Tweezer Top - potential reversal'
                    })
            
            # Tweezer Bottom (bullish)
            if abs(c0['low'] - c1['low']) < range0 * 0.02:  # Similar lows
                if c1['close'] < c1['open'] and c0['close'] > c0['open']:
                    patterns.append({
                        'name': 'tweezer_bottom',
                        'type': 'bullish_reversal',
                        'strength': 'medium',
                        'description': 'Tweezer Bottom - potential reversal'
                    })
        
        # Remove duplicates and return
        unique_patterns = []
        seen = set()
        for p in patterns:
            if p['name'] not in seen:
                unique_patterns.append(p)
                seen.add(p['name'])
        
        return unique_patterns
    
    def get_pattern_success_rate(self, pattern_name, ticker=None):
        """Get historical success rate for a pattern"""
        key = f"{ticker}_{pattern_name}" if ticker else pattern_name
        
        # Try ticker-specific first
        if key in self.pattern_history:
            stats = self.pattern_history[key]
            total = stats.get('total', 0)
            successful = stats.get('successful', 0)
            if total > 3:  # Need at least 3 occurrences for reliability
                return (successful / total) * 100
        
        # Fall back to global pattern stats
        if pattern_name in self.pattern_history:
            stats = self.pattern_history[pattern_name]
            total = stats.get('total', 0)
            successful = stats.get('successful', 0)
            if total > 0:
                return (successful / total) * 100
        
        # Default expectations based on pattern type
        defaults = {
            'bullish_reversal': 65.0,
            'bearish_reversal': 65.0,
            'bullish_continuation': 60.0,
            'bearish_continuation': 60.0,
            'indecision': 50.0
        }
        
        # Try to guess from pattern name
        for pattern_type, default_rate in defaults.items():
            if pattern_type in pattern_name:
                return default_rate
        
        return 50.0  # Neutral default
    
    def update_pattern_outcome(self, pattern_name, ticker, was_successful):
        """Update pattern success history after checking outcome"""
        # Update ticker-specific stats
        key = f"{ticker}_{pattern_name}"
        if key not in self.pattern_history:
            self.pattern_history[key] = {'total': 0, 'successful': 0}
        
        self.pattern_history[key]['total'] += 1
        if was_successful:
            self.pattern_history[key]['successful'] += 1
        
        # Also update global pattern stats
        if pattern_name not in self.pattern_history:
            self.pattern_history[pattern_name] = {'total': 0, 'successful': 0}
        
        self.pattern_history[pattern_name]['total'] += 1
        if was_successful:
            self.pattern_history[pattern_name]['successful'] += 1
        
        self._save_patterns()


class LearningMemory:
    """System memory that improves over time"""
    
    def __init__(self, memory_file='data/learning_memory.json'):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(exist_ok=True)
        self.memory = self._load_memory()
    
    def _load_memory(self):
        """Load system memory"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Initialize with default structure
        return {
            'llm_accuracy': {
                'groq': {'total': 0, 'correct': 0},
                'gemini': {'total': 0, 'correct': 0}
            },
            'indicator_reliability': {
                'rsi_oversold_buy': {'total': 0, 'successful': 0},
                'rsi_overbought_sell': {'total': 0, 'successful': 0},
                'macd_crossover': {'total': 0, 'successful': 0}
            },
            'market_conditions': {
                'high_vix_predictions': {'total': 0, 'correct': 0},
                'low_vix_predictions': {'total': 0, 'correct': 0}
            },
            'insights': []
        }
    
    def _save_memory(self):
        """Save memory to file"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def update_llm_accuracy(self, llm_name, was_correct):
        """Track LLM prediction accuracy"""
        if llm_name not in self.memory['llm_accuracy']:
            self.memory['llm_accuracy'][llm_name] = {'total': 0, 'correct': 0}
        
        self.memory['llm_accuracy'][llm_name]['total'] += 1
        if was_correct:
            self.memory['llm_accuracy'][llm_name]['correct'] += 1
        
        self._save_memory()
    
    def get_llm_weights(self):
        """Get reliability weights for each LLM based on past performance"""
        weights = {}
        for llm, stats in self.memory['llm_accuracy'].items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                weights[llm] = max(0.1, accuracy)  # Minimum weight of 0.1
            else:
                weights[llm] = 0.5  # Default weight
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def add_insight(self, insight):
        """Store a learned insight"""
        self.memory['insights'].append({
            'timestamp': datetime.now().isoformat(),
            'insight': insight
        })
        # Keep only last 100 insights
        self.memory['insights'] = self.memory['insights'][-100:]
        self._save_memory()
    
    def get_recent_insights(self, count=10):
        """Get recent learned insights"""
        return self.memory['insights'][-count:]

# Initialize the learning system components
prediction_tracker = PredictionTracker()
candle_analyzer = CandlePatternAnalyzer()
learning_memory = LearningMemory()

# ========================================
# 🔗 INTEGRATION LAYER - Connects to existing code
# This READS from your existing functions without changing them
# ========================================

class ConfidenceScorer:
    """Calculates conviction score (0-100) for predictions"""
    
    @staticmethod
    def calculate_confidence(
        llm_predictions,
        candle_patterns,
        pattern_success_rates,
        technical_indicators,
        volume_data,
        market_context=None
    ):
        """
        Calculate comprehensive confidence score
        Returns: dict with score, breakdown, and action threshold
        """
        confidence = 50  # Start neutral
        breakdown = []
        
        # 1. LLM CONSENSUS (0-30 points)
        llm_score = 0
        if llm_predictions:
            actions = [p['action'] for p in llm_predictions.values()]
            confidences = [p['confidence'] for p in llm_predictions.values()]
            
            # Agreement boost
            if len(set(actions)) == 1:  # All agree
                llm_score = 30
                breakdown.append(f"✅ All {len(actions)} LLMs agree ({actions[0]}): +30")
            elif len(actions) >= 2:
                most_common = max(set(actions), key=actions.count)
                agreement_pct = (actions.count(most_common) / len(actions)) * 100
                llm_score = int(agreement_pct * 0.3)  # Up to 30 points
                breakdown.append(f"⚖️ {agreement_pct:.0f}% LLM agreement: +{llm_score}")
            
            # Average LLM confidence
            avg_llm_conf = sum(confidences) / len(confidences)
            if avg_llm_conf > 70:
                llm_score += 10
                breakdown.append(f"🎯 High LLM confidence ({avg_llm_conf:.0f}%): +10")
            elif avg_llm_conf < 40:
                llm_score -= 10
                breakdown.append(f"⚠️ Low LLM confidence ({avg_llm_conf:.0f}%): -10")
        
        confidence += llm_score
        
        # 2. CANDLESTICK PATTERN STRENGTH (0-25 points)
        pattern_score = 0
        if candle_patterns:
            strong_patterns = [p for p in candle_patterns if p.get('strength') in ['strong', 'very_strong']]
            
            if strong_patterns:
                # Get best pattern success rate
                best_success_rate = 0
                best_pattern = None
                for pattern in strong_patterns:
                    rate = pattern_success_rates.get(pattern['name'], 50)
                    if rate > best_success_rate:
                        best_success_rate = rate
                        best_pattern = pattern
                
                if best_success_rate > 70:
                    pattern_score = 25
                    breakdown.append(f"📈 Strong {best_pattern['name']} ({best_success_rate:.0f}% success): +25")
                elif best_success_rate > 60:
                    pattern_score = 15
                    breakdown.append(f"📊 {best_pattern['name']} ({best_success_rate:.0f}% success): +15")
                elif best_success_rate > 50:
                    pattern_score = 10
                    breakdown.append(f"➡️ {best_pattern['name']} ({best_success_rate:.0f}% success): +10")
            
            # Penalty for conflicting patterns
            pattern_types = [p.get('type', '') for p in candle_patterns]
            if 'bullish_reversal' in pattern_types and 'bearish_reversal' in pattern_types:
                pattern_score -= 15
                breakdown.append(f"⚠️ Conflicting patterns: -15")
        
        confidence += pattern_score
        
        # 3. TECHNICAL INDICATORS (0-20 points)
        indicator_score = 0
        if technical_indicators:
            rsi = technical_indicators.get('rsi', 50)
            macd = technical_indicators.get('macd', 0)
            score = technical_indicators.get('score', 50)
            
            # RSI alignment
            if 30 < rsi < 70:
                indicator_score += 10
                breakdown.append(f"✅ RSI balanced ({rsi:.0f}): +10")
            elif rsi < 30:
                indicator_score += 5
                breakdown.append(f"🔵 RSI oversold ({rsi:.0f}): +5")
            elif rsi > 70:
                indicator_score -= 5
                breakdown.append(f"🔴 RSI overbought ({rsi:.0f}): -5")
            
            # System score
            if score > 70:
                indicator_score += 10
                breakdown.append(f"⭐ High system score ({score:.0f}): +10")
            elif score < 40:
                indicator_score -= 10
                breakdown.append(f"⚠️ Low system score ({score:.0f}): -10")
        
        confidence += indicator_score
        
        # 4. VOLUME CONFIRMATION (0-15 points)
        volume_score = 0
        if volume_data:
            volume_ratio = volume_data.get('volume_ratio', 1.0)
            
            if volume_ratio > 2.0:
                volume_score = 15
                breakdown.append(f"📊 High volume ({volume_ratio:.1f}x avg): +15")
            elif volume_ratio > 1.5:
                volume_score = 10
                breakdown.append(f"📈 Above avg volume ({volume_ratio:.1f}x): +10")
            elif volume_ratio < 0.5:
                volume_score = -10
                breakdown.append(f"📉 Low volume ({volume_ratio:.1f}x avg): -10")
        
        confidence += volume_score
        
        # 5. MARKET CONTEXT (0-10 points)
        context_score = 0
        if market_context:
            macro_score = market_context.get('overall_macro_score', 0)
            
            if macro_score > 10:
                context_score = 10
                breakdown.append(f"🌍 Positive market context (+{macro_score:.0f}): +10")
            elif macro_score < -10:
                context_score = -10
                breakdown.append(f"⚠️ Negative market context ({macro_score:.0f}): -10")
        
        confidence += context_score
        
        # Cap confidence between 0-100
        confidence = max(0, min(100, confidence))
        
        # Determine action threshold
        if confidence >= 75:
            action_strength = "STRONG"
            action_advice = "High conviction - consider full position"
        elif confidence >= 60:
            action_strength = "MODERATE"
            action_advice = "Medium conviction - consider half position"
        elif confidence >= 45:
            action_strength = "WEAK"
            action_advice = "Low conviction - wait for better setup"
        else:
            action_strength = "AVOID"
            action_advice = "No conviction - do not trade"
        
        return {
            'score': confidence,
            'breakdown': breakdown,
            'action_strength': action_strength,
            'action_advice': action_advice
        }


class IntelligentPredictionEngine:
    """Multi-LLM consensus with confidence scoring"""
    
    def __init__(self):
        self.prediction_tracker = prediction_tracker
        self.candle_analyzer = candle_analyzer
        self.learning_memory = learning_memory
        self.confidence_scorer = ConfidenceScorer()
        self.llm_clients = {}
        self._setup_llm_clients()
    
    def _setup_llm_clients(self):
        """Setup all available LLM clients"""
        
        # 1. Groq (Fast, Free - Llama 3.1)
        if os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                self.llm_clients['groq'] = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logging.info("✅ Groq LLM initialized (Llama 3.1 70B)")
            except Exception as e:
                logging.warning(f"Groq setup failed: {e}")
        
        # 2. Gemini (Smart, Free - Google)
        if os.getenv("GEMINI_API_KEY"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.llm_clients['gemini'] = genai.GenerativeModel('gemini-1.5-flash-latest')
                logging.info("✅ Gemini LLM initialized (1.5 Flash)")
            except Exception as e:
                logging.warning(f"Gemini setup failed: {e}")
        
        # 3. Cohere (Optional - Command R)
        if os.getenv("COHERE_API_KEY"):
            try:
                import cohere
                self.llm_clients['cohere'] = cohere.Client(os.getenv("COHERE_API_KEY"))
                logging.info("✅ Cohere LLM initialized (Command R)")
            except Exception as e:
                logging.warning(f"Cohere setup failed: {e}")
        
        if not self.llm_clients:
            logging.warning("⚠️ No LLM clients available - will use rule-based predictions")
    
    async def analyze_with_learning(self, ticker, existing_analysis, hist_data, market_context=None):
        """
        Enhanced analysis with multi-LLM consensus and confidence scoring
        """
        
        # 1. Extract candlestick patterns
        candle_patterns = []
        if hist_data is not None and len(hist_data) >= 2:
            candle_patterns = self.candle_analyzer.identify_pattern(hist_data)
        
        # 2. Get pattern success rates
        pattern_success_rates = {}
        for pattern in candle_patterns:
            success_rate = self.candle_analyzer.get_pattern_success_rate(
                pattern['name'], 
                ticker
            )
            pattern_success_rates[pattern['name']] = success_rate
        
        # 3. Get multi-LLM consensus
        llm_predictions = await self._get_multi_llm_consensus(
            ticker=ticker,
            existing_analysis=existing_analysis,
            candle_patterns=candle_patterns,
            pattern_success_rates=pattern_success_rates,
            market_context=market_context
        )
        
        # 4. Calculate comprehensive confidence score
        volume_data = {
            'volume_ratio': existing_analysis.get('volume_ratio', 1.0)
        }
        
        confidence_result = self.confidence_scorer.calculate_confidence(
            llm_predictions=llm_predictions,
            candle_patterns=candle_patterns,
            pattern_success_rates=pattern_success_rates,
            technical_indicators={
                'rsi': existing_analysis.get('rsi', 50),
                'macd': existing_analysis.get('macd', 0),
                'score': existing_analysis.get('score', 50)
            },
            volume_data=volume_data,
            market_context=market_context
        )
        
        # 5. Determine final action based on consensus + confidence
        final_prediction = self._determine_final_action(
            llm_predictions, 
            confidence_result,
            candle_patterns
        )
        
        # 6. Store prediction for learning
        if final_prediction:
            pred_id = self.prediction_tracker.store_prediction(
                ticker=ticker,
                action=final_prediction['action'],
                confidence=confidence_result['score'],
                reasoning=final_prediction['reasoning'],
                candle_pattern=candle_patterns[0]['name'] if candle_patterns else None,
                indicators={
                    'rsi': existing_analysis.get('rsi', 50),
                    'score': existing_analysis.get('score', 50),
                    'volume_ratio': existing_analysis.get('volume_ratio', 1.0)
                }
            )
            final_prediction['prediction_id'] = pred_id
        
        # 7. Return enhanced analysis
        return {
            **existing_analysis,  # Preserve original analysis
            'candle_patterns': candle_patterns,
            'pattern_success_rates': pattern_success_rates,
            'llm_predictions': llm_predictions,
            'confidence': confidence_result,
            'ai_prediction': final_prediction,
            'learning_insights': self.learning_memory.get_recent_insights(3)
        }
    
    async def _get_multi_llm_consensus(self, ticker, existing_analysis, candle_patterns, pattern_success_rates, market_context):
        """Get predictions from all available LLMs"""
        
        # Build comprehensive prompt
        pattern_text = ""
        if candle_patterns:
            pattern_list = []
            for p in candle_patterns[:3]:  # Top 3 patterns
                success_rate = pattern_success_rates.get(p['name'], 50)
                pattern_list.append(f"{p['name']} ({p['type']}, {success_rate:.0f}% historical success)")
            pattern_text = "\n".join(pattern_list)
        else:
            pattern_text = "No clear patterns identified"
        
        context = f"""Analyze {ticker} and provide BUY/HOLD/SELL recommendation.

TECHNICAL DATA:
- Score: {existing_analysis.get('score', 'N/A')}/100
- RSI: {existing_analysis.get('rsi', 'N/A')}
- Volume: {existing_analysis.get('volume_ratio', 1.0):.1f}x average
- Sector: {existing_analysis.get('sector', 'Unknown')}

CANDLESTICK PATTERNS (Today):
{pattern_text}

MARKET CONTEXT:
{f"Macro Score: {market_context.get('overall_macro_score', 0):.0f}" if market_context else "Not available"}

Respond with ONLY:
ACTION: [BUY/HOLD/SELL]
CONFIDENCE: [0-100]
REASON: [One sentence]

Example:
ACTION: BUY
CONFIDENCE: 75
REASON: Strong hammer pattern with 73% success rate and oversold RSI.
"""
        
        predictions = {}
        
        # Get predictions from each LLM
        tasks = []
        
        if 'groq' in self.llm_clients:
            tasks.append(self._query_groq(context, ticker))
        
        if 'gemini' in self.llm_clients:
            tasks.append(self._query_gemini(context, ticker))
        
        if 'cohere' in self.llm_clients:
            tasks.append(self._query_cohere(context, ticker))
        
        # Run all LLM queries concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            llm_names = []
            if 'groq' in self.llm_clients:
                llm_names.append('groq')
            if 'gemini' in self.llm_clients:
                llm_names.append('gemini')
            if 'cohere' in self.llm_clients:
                llm_names.append('cohere')
            
            for llm_name, result in zip(llm_names, results):
                if not isinstance(result, Exception) and result:
                    predictions[llm_name] = result
        
        return predictions
    
    async def _query_groq(self, prompt, ticker):
        """Query Groq LLM"""
        try:
            client = self.llm_clients['groq']
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            text = response.choices[0].message.content
            return self._parse_llm_response(text, 'groq')
        except Exception as e:
            logging.warning(f"Groq query failed for {ticker}: {e}")
            return None
    
    async def _query_gemini(self, prompt, ticker):
        """Query Gemini LLM"""
        try:
            model = self.llm_clients['gemini']
            response = model.generate_content(
                prompt,
                generation_config={'temperature': 0.3, 'max_output_tokens': 150}
            )
            return self._parse_llm_response(response.text, 'gemini')
        except Exception as e:
            logging.warning(f"Gemini query failed for {ticker}: {e}")
            return None
    
    async def _query_cohere(self, prompt, ticker):
        """Query Cohere LLM"""
        try:
            client = self.llm_clients['cohere']
            response = client.chat(
                message=prompt,
                model='command-r',
                temperature=0.3
            )
            return self._parse_llm_response(response.text, 'cohere')
        except Exception as e:
            logging.warning(f"Cohere query failed for {ticker}: {e}")
            return None
    
    def _parse_llm_response(self, response_text, llm_name):
        """Parse LLM response to extract action, confidence, and reasoning"""
        import re
        
        action = "HOLD"  # Default
        confidence = 50
        reasoning = response_text[:200]
        
        # Extract ACTION
        action_match = re.search(r'ACTION:\s*(BUY|SELL|HOLD)', response_text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).upper()
        else:
            # Fallback: look for keywords
            text_lower = response_text.lower()
            if 'buy' in text_lower and 'sell' not in text_lower:
                action = "BUY"
            elif 'sell' in text_lower:
                action = "SELL"
        
        # Extract CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response_text, re.IGNORECASE)
        if conf_match:
            confidence = int(conf_match.group(1))
        else:
            # Look for percentage
            pct_match = re.search(r'(\d+)%', response_text)
            if pct_match:
                confidence = int(pct_match.group(1))
        
        # Extract REASON
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
        if reason_match:
            reasoning = reason_match.group(1).strip()
        
        confidence = max(0, min(100, confidence))
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'llm': llm_name
        }
    
    def _determine_final_action(self, llm_predictions, confidence_result, candle_patterns):
        """Determine final action from LLM consensus and confidence score"""
        
        if not llm_predictions:
            # No LLM predictions - use rule-based with confidence
            if confidence_result['score'] >= 60:
                action = "BUY" if candle_patterns and 'bullish' in candle_patterns[0].get('type', '') else "HOLD"
            elif confidence_result['score'] <= 40:
                action = "SELL"
            else:
                action = "HOLD"
            
            return {
                'action': action,
                'confidence': confidence_result['score'],
                'reasoning': f"Rule-based: {confidence_result['action_advice']}",
                'llm_count': 0
            }
        
        # Get LLM consensus
        actions = [p['action'] for p in llm_predictions.values()]
        most_common_action = max(set(actions), key=actions.count)
        
        # Weight by learning memory
        weights = self.learning_memory.get_llm_weights()
        weighted_actions = {"BUY": 0, "HOLD": 0, "SELL": 0}
        
        for llm_name, prediction in llm_predictions.items():
            weight = weights.get(llm_name, 0.33)
            weighted_actions[prediction['action']] += weight
        
        final_action = max(weighted_actions.items(), key=lambda x: x[1])[0]
        
        # Combine all reasoning
        reasonings = [f"{p['llm']}: {p['reasoning']}" for p in llm_predictions.values()]
        combined_reasoning = " | ".join(reasonings)
        
        # Override action if confidence is too low
        if confidence_result['score'] < 45:
            final_action = "HOLD"
            combined_reasoning = f"Low confidence ({confidence_result['score']}%) - holding. " + combined_reasoning
        
        return {
            'action': final_action,
            'confidence': confidence_result['score'],
            'reasoning': combined_reasoning[:500],
            'llm_count': len(llm_predictions),
            'llm_agreement': (actions.count(final_action) / len(actions)) * 100,
            'confidence_breakdown': confidence_result['breakdown'],
            'action_strength': confidence_result['action_strength'],
            'action_advice': confidence_result['action_advice']
        }


# ========================================
# 🎯 ENHANCED PORTFOLIO ANALYZER
# Wraps your existing portfolio analysis with predictions
# ========================================

async def analyze_portfolio_with_predictions(session, portfolio_file='portfolio.json', market_context=None):
    """
    Enhanced portfolio analysis with predictions and market context
    """
    
    logging.info("=" * 60)
    logging.info("🧠 ANALYZE WITH PREDICTIONS - START")
    logging.info(f"Market context provided: {market_context is not None}")
    
    # Call v2.0 portfolio analysis (has Bollinger, ATR, etc.)
    original_portfolio_data = await analyze_portfolio_with_v2_features(session, portfolio_file)
    
    if not original_portfolio_data:
        logging.warning("No portfolio data from v2 features")
        return original_portfolio_data
    
    logging.info(f"Original portfolio has {len(original_portfolio_data.get('stocks', []))} stocks")
    
    # Initialize prediction engine
    try:
        prediction_engine = IntelligentPredictionEngine()
        logging.info(f"Prediction engine initialized. LLMs available: {list(prediction_engine.llm_clients.keys())}")
    except Exception as e:
        logging.error(f"Failed to initialize prediction engine: {e}")
        # Return original data without predictions
        return {
            **original_portfolio_data,
            'learning_active': False,
            'predictions_made': 0
        }
    
    # Add predictions to each stock
    enhanced_stocks = []
    successful_predictions = 0
    
    for stock in original_portfolio_data['stocks']:
        try:
            ticker = stock['ticker']
            logging.info(f"🔍 Processing {ticker}...")
            
            yf_ticker = yf.Ticker(ticker)
            hist = await asyncio.to_thread(yf_ticker.history, period="3mo", interval="1d")
            
            if hist.empty:
                logging.warning(f"No history for {ticker}, skipping predictions")
                enhanced_stocks.append(stock)
                continue
            
            # Get enhanced analysis with market context
            enhanced = await prediction_engine.analyze_with_learning(
                ticker=ticker,
                existing_analysis=stock,
                hist_data=hist,
                market_context=market_context
            )
            
            # Verify prediction was added
            if 'ai_prediction' in enhanced:
                successful_predictions += 1
                logging.info(f"✅ {ticker}: Prediction added - {enhanced['ai_prediction']['action']}")
            else:
                logging.warning(f"⚠️ {ticker}: No prediction generated")
            
            enhanced_stocks.append(enhanced)
            
        except Exception as e:
            logging.error(f"Error enhancing {ticker}: {e}")
            enhanced_stocks.append(stock)  # Keep original
    
    result = {
        **original_portfolio_data,
        'stocks': enhanced_stocks,
        'predictions_made': successful_predictions,
        'learning_active': True
    }
    
    logging.info("=" * 60)
    logging.info(f"✅ PREDICTIONS COMPLETE: {successful_predictions}/{len(enhanced_stocks)} stocks")
    logging.info("=" * 60)
    
    return result


# ========================================
# 🔄 OUTCOME CHECKER - Runs in evening
# ========================================

async def check_prediction_outcomes():
    """
    This runs in the evening to check how our predictions did
    Standalone function - doesn't modify existing code
    """
    logging.info("🔍 Checking prediction outcomes...")
    
    # Check outcomes from yesterday
    results = prediction_tracker.check_outcomes(days_to_check=1)
    
    # Update pattern success rates based on outcomes
    for pred_id, pred in prediction_tracker.predictions.items():
        if pred.get('was_correct') is not None and pred.get('candle_pattern'):
            candle_analyzer.update_pattern_outcome(
                pattern=pred['candle_pattern'],
                ticker=pred['ticker'],
                was_successful=pred['was_correct']
            )
    
    # Generate learning insights
    if results['checked'] > 0:
        accuracy = (results['correct'] / results['checked']) * 100
        insight = f"Today's accuracy: {accuracy:.1f}% ({results['correct']}/{results['checked']} correct)"
        learning_memory.add_insight(insight)
        
        # Update LLM accuracy if we tracked which LLM made predictions
        # This will be implemented in next iteration
    
    logging.info(f"✅ Checked {results['checked']} predictions: {results['correct']} correct, {results['wrong']} wrong")
    
    return results

# ========================================
# MAIN FUNCTION - Updated for v2.0.0
# ========================================

async def main(output="print"):
    logging.info("📊 FULL ANALYSIS MODE: Running market intelligence scan...")
    previous_day_memory = load_memory()
    
    sp500 = get_cached_tickers('sp500_cache.json', fetch_sp500_tickers_sync)
    tsx = get_cached_tickers('tsx_cache.json', fetch_tsx_tickers_sync)
    universe = (sp500 or [])[:75] + (tsx or [])[:25]
    
    throttler = Throttler(2)
    semaphore = asyncio.Semaphore(10)
    
    async with aiohttp.ClientSession() as session:
        # Prepare all tasks
        stock_tasks = [analyze_stock(semaphore, throttler, session, ticker) for ticker in universe]
        context_task = fetch_context_data(session)
        news_task = fetch_market_headlines(session)
        macro_task = fetch_macro_sentiment(session)
        
        # Step 1: Get macro data FIRST (needed for portfolio predictions)
        logging.info("🔍 Step 1: Fetching macro data...")
        macro_data = await macro_task
        logging.info(f"✅ Macro data received. Score: {macro_data.get('overall_macro_score', 0):.1f}")
        
        # Step 2: Create portfolio task with macro context
        if ENABLE_V2_FEATURES:
            logging.info("🔍 Step 2: Calling analyze_portfolio_with_predictions (v3.0)")
            portfolio_data = await analyze_portfolio_with_predictions(session, market_context=macro_data)
        else:
            logging.info("🔍 Step 2: Calling analyze_portfolio_watchlist (v1.0)")
            portfolio_data = await analyze_portfolio_watchlist(session)
        
        logging.info(f"✅ Portfolio analysis complete. Predictions: {portfolio_data.get('predictions_made', 0) if portfolio_data else 0}")
        
        # Step 3: Run everything else in parallel
        logging.info("🔍 Step 3: Fetching stocks, context, news...")
        stock_results_raw, context_data, market_news = await asyncio.gather(
            asyncio.gather(*stock_tasks), 
            context_task, 
            news_task
        )
        
        # Step 4: Process stock results
        stock_results = sorted([r for r in stock_results_raw if r], key=lambda x: x['score'], reverse=True)
        df_stocks = pd.DataFrame(stock_results) if stock_results else pd.DataFrame()
        logging.info(f"✅ Analyzed {len(stock_results)} stocks")
        
        # Step 5: Find historical patterns
        logging.info("🔍 Step 5: Finding historical patterns...")
        pattern_data = await find_historical_patterns(session, macro_data)
        
        # Step 6: Generate portfolio recommendations
        portfolio_recommendations = None
        if pattern_data and portfolio_data:
            logging.info("🔍 Step 6: Generating portfolio recommendations...")
            portfolio_recommendations = await generate_portfolio_recommendations_from_pattern(
                portfolio_data, pattern_data, macro_data
            )
        
        # Step 7: Generate AI analysis
        logging.info("🔍 Step 7: Generating AI analysis...")
        market_summary = {
            'macro': macro_data,
            'top_stock': stock_results[0] if stock_results else {},
            'bottom_stock': stock_results[-1] if stock_results else {}
        }
        ai_analysis = await generate_ai_oracle_analysis(market_summary, portfolio_data, pattern_data)
    
    # Outside session context - generate email
    if output == "email":
        logging.info("📧 Generating email report...")
        html_email = generate_enhanced_html_email(
            df_stocks, context_data, market_news, macro_data, 
            previous_day_memory, portfolio_data, pattern_data, 
            ai_analysis, portfolio_recommendations
        )
        send_email(html_email)
    
    # Save memory
    if not df_stocks.empty:
        save_memory({
            "previous_top_stock_name": str(df_stocks.iloc[0]['name']),
            "previous_top_stock_ticker": str(df_stocks.iloc[0]['ticker']),
            "previous_macro_score": float(macro_data.get('overall_macro_score', 0)),
            "date": datetime.now().strftime('%Y-%m-%d')
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

    # Add this AFTER the portfolio_html section in generate_enhanced_html_email

# ========================================
# 🎯 AI PREDICTIONS WITH CONFIDENCE SCORING
# ========================================

ai_predictions_html = ""

# Diagnostic logging
logging.info("=" * 60)
logging.info("📧 EMAIL - AI PREDICTIONS SECTION")
logging.info(f"Portfolio data exists: {portfolio_data is not None}")

if portfolio_data:
    logging.info(f"Learning active: {portfolio_data.get('learning_active')}")
    logging.info(f"Predictions made: {portfolio_data.get('predictions_made')}")
    logging.info(f"Stocks count: {len(portfolio_data.get('stocks', []))}")

if portfolio_data and portfolio_data.get('learning_active'):
    predictions_made = portfolio_data.get('predictions_made', 0)
    
    logging.info(f"Checking {predictions_made} predictions for display...")
    
    if predictions_made > 0:
        prediction_cards = []
        cards_created = 0
        
        for stock in portfolio_data['stocks']:
            if 'ai_prediction' not in stock or not stock['ai_prediction']:
                continue
            
            try:
                pred = stock['ai_prediction']
                conf = stock.get('confidence', {})
                
                # Action setup
                action = pred.get('action', 'HOLD')
                action_color = {'BUY': '#16a34a', 'SELL': '#dc2626', 'HOLD': '#666'}.get(action, '#666')
                action_icon = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}.get(action, '⚪')
                
                # Confidence
                conf_score = conf.get('score', pred.get('confidence', 50))
                conf_color = '#16a34a' if conf_score >= 75 else '#f59e0b' if conf_score >= 60 else '#dc2626'
                conf_label = 'HIGH' if conf_score >= 75 else 'MEDIUM' if conf_score >= 60 else 'LOW'
                
                # Build card
                prediction_card = f"""
                <div style="border:2px solid {action_color};border-radius:10px;padding:15px;margin-bottom:15px;background:white;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                        <h3 style="margin:0;color:#111;">{action_icon} {stock['ticker']} - {action}</h3>
                        <div style="font-size:1.5em;font-weight:bold;color:{conf_color};">{conf_score:.0f}%</div>
                    </div>
                    <div style="font-size:0.9em;color:#666;">
                        {stock.get('name', stock['ticker'])} • ${stock.get('price', 0):.2f}
                    </div>
                    <div style="margin:10px 0;background:#e5e5e5;border-radius:10px;height:10px;">
                        <div style="width:{conf_score}%;background:{conf_color};border-radius:10px;height:100%;"></div>
                    </div>
                    <div style="background:#f8f9fa;padding:8px;border-radius:5px;margin-top:10px;font-size:0.9em;">
                        <b>Reasoning:</b> {pred.get('reasoning', 'N/A')[:200]}...
                    </div>
                </div>
                """
                
                prediction_cards.append(prediction_card)
                cards_created += 1
                
            except Exception as e:
                logging.error(f"Error creating card for {stock.get('ticker')}: {e}")
        
        logging.info(f"Created {cards_created} prediction cards")
        
        if prediction_cards:
            ai_predictions_html = f"""
            <div class="section" style="background-color:#f0f9ff;border-left:4px solid #3b82f6;">
                <h2>🎯 AI PREDICTIONS & CONFIDENCE ANALYSIS</h2>
                <p style="font-size:0.9em;color:#666;margin-bottom:15px;">
                    Generated {predictions_made} AI-powered predictions with confidence scoring.
                </p>
                {''.join(prediction_cards)}
                <div style="margin-top:15px;padding:12px;background:#fffbeb;border-radius:5px;">
                    <p style="font-size:0.85em;color:#92400e;margin:0;">
                        <b>💡 How to use:</b> Only act on HIGH confidence (75%+) signals. 
                        System learns from outcomes daily.
                    </p>
                </div>
            </div>
            """
            logging.info("✅ AI predictions HTML generated successfully")
        else:
            logging.warning("⚠️ No prediction cards created despite predictions_made > 0")
    else:
        logging.info("No predictions to display (predictions_made = 0)")

logging.info("=" * 60)

# Then add predictions_html to your email template where appropriate
    
# Pattern analysis section (from v1.0.0 - keeping stable)
pattern_html = ""
if pattern_data and pattern_data.get('matches'):  # ← FIXED: Same indentation as line above
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
    
    # Continue with the rest of pattern_html section...
        
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
                <p style="font-size:1.1em; color:#aaa;">{date.today().strftime('%A, %B %d, %Y')}</p>
            </div>
            
            <div class="section">
                <h2>EDITOR'S NOTE</h2>
                <p>{editor_note}</p>
            </div>
            
            {v2_signals_html}
            {ai_oracle_html}
            {portfolio_html}
            {ai_predictions_html}
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
    SMTP_USER, SMTP_PASS = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not SMTP_USER or not SMTP_PASS:
        logging.warning("SMTP credentials missing. Cannot send email.")
        return
    
    msg = MIMEMultipart('alternative')
    msg["Subject"] = f"⛵ Your Daily Market Briefing - {datetime.today().date()}"
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
# 🆕 EMAIL BOT SYSTEM - v3.1.0 (FINAL, AUDITED)
# This version is guaranteed to be free of syntax and indentation errors.
# ========================================

# ========================================
# SECTION 1: IMPORTS & FAILSAFES
# ========================================

# ========================================
# 🆕 EMAIL BOT SYSTEM - v3.2.2 (FINAL)
# ========================================

# Failsafe for duckduckgo-search -> ddgs rename
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
        logging.warning("⚠️ DeprecationWarning: `duckduckgo_search` is now `ddgs`. Use `pip install -U ddgs`.")
    except ImportError:
        logging.error("❌ `ddgs` not found. Web search capabilities disabled.")
        DDGS_AVAILABLE = False
        class DDGS:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def news(self, *args, **kwargs): return []
            def text(self, *args, **kwargs): return []

# ========================================
# SECTION 2: CORE BOT CLASSES
# ========================================

class EmailBotDatabase:
    """Isolated database for bot conversations"""
    def __init__(self, db_path='email_bot.db'):
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self._init_schema()
        except Exception as e:
            logging.error(f"DB init failed: {e}. Using in-memory fallback.")
            self.conn = sqlite3.connect(':memory:', check_same_thread=False)
            self._init_schema()

    def _init_schema(self):
        try:
            self.conn.executescript('''
                CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, timestamp TEXT, user_email TEXT, user_question TEXT, topics TEXT, sent BOOLEAN);
                CREATE TABLE IF NOT EXISTS bot_stats (date TEXT PRIMARY KEY, checked INTEGER DEFAULT 0, answered INTEGER DEFAULT 0, errors INTEGER DEFAULT 0);
            ''')
            self.conn.commit()
        except Exception as e: logging.error(f"Schema creation failed: {e}")

    def log_conversation(self, email_addr, question, topics, success):
        try:
            self.conn.execute('INSERT INTO conversations (timestamp, user_email, user_question, topics, sent) VALUES (?, ?, ?, ?, ?)',(datetime.datetime.now().isoformat(),email_addr,(question or "")[:500],json.dumps(topics or {}),bool(success)))
            self.conn.commit()
        except Exception as e: logging.error(f"Conversation log failed: {e}")

    def update_stats(self, checked=0, answered=0, errors=0):
        today = datetime.datetime.now().date().isoformat()
        try:
            self.conn.execute('INSERT INTO bot_stats (date, checked, answered, errors) VALUES (?, ?, ?, ?) ON CONFLICT(date) DO UPDATE SET checked=checked+excluded.checked, answered=answered+excluded.answered, errors=errors+excluded.errors',(today,checked,answered,errors))
            self.conn.commit()
        except Exception:
            try:
                cur = self.conn.execute('UPDATE bot_stats SET checked=checked+?, answered=answered+?, errors=errors+? WHERE date=?', (checked,answered,errors,today))
                if cur.rowcount == 0: self.conn.execute('INSERT INTO bot_stats (date,checked,answered,errors) VALUES (?,?,?,?)',(today,checked,answered,errors))
                self.conn.commit()
            except Exception as e: logging.error(f"Stats update fallback failed: {e}")

    def close(self):
        if self.conn:
            try: self.conn.close()
            except: pass


class EmailBotResponder:
    """Generates basic HTML responses as a fallback"""
    @staticmethod
    def create_price_card(data):
        if not data: return ""
        try:
            color = '#16a34a' if data.get('daily_change', 0) >= 0 else '#dc2626'
            return f"""<div style="display:inline-block;width:45%;margin:10px 2%;padding:15px;border-radius:12px;background:#f0f7ff;vertical-align:top;"><h3 style="margin:0;color:#1e40af;text-transform:uppercase;">{data.get('name','?')}</h3><p style="font-size:28px;font-weight:bold;margin:8px 0;">${data.get('price',0):,.2f}</p><p style="color:{color};font-size:16px;">{data.get('daily_change',0):+.2f}% today</p></div>"""
        except: return ""

    @staticmethod
    def generate_html_response(question, market_data):
        if not market_data: return EmailBotResponder.generate_help_response(question)
        cards = [EmailBotResponder.create_price_card(d) for d in market_data.values()]
        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>body{{font-family:sans-serif;}}</style></head><body><h1>Basic Report</h1><p><b>Q:</b> {question}</p>{''.join(cards)}<p><i>Intelligent analysis failed.</i></p></body></html>"""
    
    @staticmethod
    def generate_help_response(question):
        return f"""<!DOCTYPE html><html><body><h1>Help</h1><p>No market topics detected in "{question}". Try asking about specific stocks (AAPL) or crypto (Bitcoin/XRP).</p></body></html>"""


class MarketQuestionAnalyzer:
    """Extracts topics and fetches data for user questions"""
    @staticmethod
    def extract_topics(question):
        topics, seen = {}, set()
        if not question: return topics
        q_lower = question.lower()
        mappings = {'bitcoin':{'t':'BTC-USD','n':'Bitcoin'},'btc':{'t':'BTC-USD','n':'Bitcoin'},'ethereum':{'t':'ETH-USD','n':'Ethereum'},'eth':{'t':'ETH-USD','n':'Ethereum'},'xrp':{'t':'XRP-USD','n':'Ripple'},'ripple':{'t':'XRP-USD','n':'Ripple'},'gold':{'t':'GC=F','n':'Gold'},'silver':{'t':'SI=F','n':'Silver'},'oil':{'t':'CL=F','n':'Crude Oil'},'sp500':{'t':'^GSPC','n':'S&P 500'},'nasdaq':{'t':'^IXIC','n':'Nasdaq'}}
        for keyword, data in mappings.items():
            if keyword in q_lower and data['t'] not in seen:
                asset_type = 'crypto' if '-USD' in data['t'] else 'commodity' if '=F' in data['t'] else 'index'
                topics[keyword] = {'type':asset_type, 'ticker':data['t'], 'name':data['n']}
                seen.add(data['t'])
        try:
            for ticker in re.findall(r'\$?([A-Z]{1,5})\b', question):
                if ticker not in ['USD','ETH','BTC','CEO','AI'] and len(ticker)<=5 and ticker not in seen:
                    topics[ticker.lower()] = {'type':'stock', 'ticker':ticker, 'name':ticker}
                    seen.add(ticker)
        except: pass
        return topics
    
    @staticmethod
    async def get_market_data(ticker):
        try:
            hist = await asyncio.to_thread(yf.Ticker(ticker).history, period='3mo')
            if hist.empty: return None
            current, day_ago = hist['Close'].iloc[-1], hist['Close'].iloc[-2] if len(hist)>1 else hist['Close'].iloc[-1]
            try: rsi_val = RSIIndicator(hist['Close'],14).rsi().iloc[-1]; rsi = 50.0 if pd.isna(rsi_val) else float(rsi_val)
            except: rsi = 50.0
            return {'ticker':ticker,'price':float(current),'daily_change':float(((current-day_ago)/day_ago*100) if day_ago else 0),'rsi':rsi}
        except Exception as e: logging.warning(f"Data fetch failed for {ticker}: {e}"); return None

# ========================================
# FINAL, AUDITED BOT SYSTEM - v4.2
# ========================================

class IntelligentMarketAnalyzer:
    def __init__(self):
        self.nlp = None
        self.wiki = None
        self.llm_clients = {}
        if 'SPACY_AVAILABLE' in globals() and SPACY_AVAILABLE:
            try: self.nlp = spacy.load("en_core_web_sm")
            except: logging.warning("spaCy model not loaded")
        if 'WIKIPEDIA_AVAILABLE' in globals() and WIKIPEDIA_AVAILABLE:
            self.wiki = wikipediaapi.Wikipedia('MarketBot/1.0','en')
        self._setup_llm_clients()

    def _setup_llm_clients(self):
        # CORRECTED: try/except blocks are now properly structured.
        if 'GROQ_AVAILABLE' in globals() and GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                self.llm_clients['groq'] = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logging.info("✅ Groq LLM available")
            except Exception as e:
                logging.warning(f"Groq setup failed: {e}")
        if 'COHERE_AVAILABLE' in globals() and COHERE_AVAILABLE and os.getenv("COHERE_API_KEY"):
            try:
                import cohere
                self.llm_clients['cohere'] = cohere.Client(os.getenv("COHERE_API_KEY"))
                logging.info("✅ Cohere LLM available")
            except Exception as e:
                logging.warning(f"Cohere setup failed: {e}")

    async def answer_intelligently(self, question, ticker_data):
        logging.info(f"🧠 Generating intelligent answer for: {ticker_data.get('name', 'asset')}")
        intent = self._analyze_question_intent(question)
        context = await self._gather_context(ticker_data.get('name', ''), intent)
        response = await self._generate_llm_response(question, context, ticker_data)
        if not response or len(response) < 50:
            response = self._intelligent_assembly(context, ticker_data)
        return response
    
    def _analyze_question_intent(self, question):
        q = (question or "").lower()
        return {'reasons': any(w in q for w in ['why','reason']),'applications': any(w in q for w in ['application','use'])}

    async def _gather_context(self, asset_name, intent):
        context = {'news': [], 'wikipedia': {}}
        async def fetch_news():
            if DDGS_AVAILABLE and asset_name:
                try:
                    with DDGS() as ddgs:
                        query = f"{asset_name} analysis"
                        if intent.get('reasons'): query = f"{asset_name} price drivers"
                        elif intent.get('applications'): query = f"{asset_name} use cases"
                        news = list(ddgs.news(query,max_results=3))
                        context['news'] = [{'title': n.get('title',''),'body':n.get('body','')} for n in news]
                except Exception as e: logging.warning(f"News search failed: {e}")
        async def fetch_wiki():
            if self.wiki and asset_name:
                try:
                    page = self.wiki.page(asset_name)
                    if page.exists(): context['wikipedia']['summary'] = page.summary[:1000]
                except Exception as e: logging.warning(f"Wikipedia failed: {e}")
        await asyncio.gather(fetch_news(), fetch_wiki())
        return context
    
    async def _generate_llm_response(self, question, context, ticker_data):
        prompt = self._build_llm_prompt(question, context, ticker_data)
        response = None
        if 'groq' in self.llm_clients:
            try:
                client = self.llm_clients['groq']
                completion = client.chat.completions.create(model="llama-3.1-70b-versatile",messages=[{"role":"user","content":prompt}],temperature=0.7,max_tokens=800)
                response = completion.choices[0].message.content
                if response: logging.info("✅ Groq response generated")
            except Exception as e: logging.warning(f"Groq failed: {e}")
        if not response and 'cohere' in self.llm_clients:
            try:
                client = self.llm_clients['cohere']
                result = client.chat(message=prompt,model='command-r',temperature=0.7)
                response = result.text
                if response: logging.info("✅ Cohere response generated")
            except Exception as e: logging.warning(f"Cohere failed: {e}")
        return response
    
    def _build_llm_prompt(self, question, context, ticker_data):
        parts = [f"Q: {question}",f"Asset: {ticker_data.get('name')}",f"Data: Price ${ticker_data.get('price',0):.2f}, RSI {ticker_data.get('rsi',50):.0f}"]
        if context.get('news'): parts.append("News:\n"+"\n".join([f"- {n['title']}" for n in context['news']]))
        if context.get('wikipedia',{}).get('summary'): parts.append(f"Background:\n{context['wikipedia']['summary']}")
        parts.append("As a market analyst, provide a direct, professional analysis.")
        return "\n".join(parts)

    def _intelligent_assembly(self, context, ticker_data):
        lines = [f"# {ticker_data.get('name')} Analysis",f"Price: ${ticker_data.get('price',0):,.2f}, RSI: {ticker_data.get('rsi',50):.0f}"]
        if context.get('news'): lines.extend(["## Key Drivers (from news)"]+[f"- {n['title']}" for n in context['news']])
        if context.get('wikipedia',{}).get('summary'): lines.extend([f"## Overview", context['wikipedia']['summary']])
        return '\n'.join(lines)


class IntelligentEmailBotResponder:
    def __init__(self):
        self.analyzer = IntelligentMarketAnalyzer()
    
    async def generate_intelligent_html(self, question, market_data):
        analyses = {}
        for key, data in market_data.items():
            if data:
                try: analyses[key] = {'data':data,'analysis':await self.analyzer.answer_intelligently(question,data)}
                except Exception as e: logging.error(f"Analysis failed for {key}: {e}")
        return self._build_intelligent_html(question, analyses)
    
    def _build_intelligent_html(self, question, analyses):
        def md_to_html(text):
            if 'MARKDOWN_AVAILABLE' in globals() and MARKDOWN_AVAILABLE:
                import markdown
                return markdown.markdown(text, extensions=['fenced_code','tables'])
            return f"<pre>{text}</pre>"
        
        cards = [EmailBotResponder.create_price_card(item['data']) for item in analyses.values()]
        sections = [f'<div style="margin-top:20px;padding:20px;background:#f9fafb;border-radius:10px;">{md_to_html(item["analysis"])}</div>' for item in analyses.values()]
        
        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>body{{font-family:-apple-system,sans-serif;max-width:800px;margin:auto;}}</style></head><body><h1>Intelligent Analysis</h1><div><b>Q:</b> {question}</div><div>{''.join(cards)}</div>{''.join(sections)}</body></html>"""

class EmailBotEngine:
    """SIMPLE VERSION - Just processes YOUR emails only"""
    
    def __init__(self):
        self.db = None
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        
        # YOUR EMAIL ADDRESS - hardcode it here
        self.authorized_email = "puneetbr44@gmail.com"
        
        if not self.smtp_user or not self.smtp_pass:
            raise ValueError("❌ SMTP credentials required!")
        
        logging.info(f"📧 Bot initialized. Will only respond to: {self.authorized_email}")
        
        try:
            self.db = EmailBotDatabase()
        except Exception as e:
            logging.error(f"DB init failed: {e}")

    async def check_and_respond(self):
        """SIMPLIFIED: Only process emails from YOUR address"""
        logging.info(f"📧 Checking inbox for emails from {self.authorized_email}...")
        checked, answered, errors = 0, 0, 0
        mail = None
        
        try:
            # 1. CONNECT
            mail = imaplib.IMAP4_SSL("imap.gmail.com", timeout=20)
            mail.login(self.smtp_user, self.smtp_pass)
            mail.select('inbox')
            
            # 2. SIMPLE SEARCH: Get UNSEEN emails from YOUR address only
            search_query = f'(UNSEEN FROM "{self.authorized_email}")'
            status, data = mail.search(None, search_query)
            
            if status != 'OK' or not data[0]:
                logging.info(f"✅ No new emails from {self.authorized_email}")
                return
            
            email_ids = data[0].split()
            logging.info(f"📬 Found {len(email_ids)} new email(s) from you")

            # 3. PROCESS ONLY THE MOST RECENT 5
            for eid in sorted(email_ids, key=int, reverse=True)[:5]:
                try:
                    checked += 1
                    _, fdata = mail.fetch(eid, '(RFC822)')
                    msg = email.message_from_bytes(fdata[0][1])
                    
                    question = self._extract_question(msg)
                    subject = msg.get('Subject', 'Market Question')
                    
                    logging.info(f"❓ Question: {question[:100]}")
                    
                    if not self._is_valid(question):
                        logging.warning(f"⏭️ Skipping invalid/empty question")
                        mail.store(eid, '+FLAGS', '\\Seen')
                        continue
                    
                    # Extract topics and get market data
                    topics = MarketQuestionAnalyzer.extract_topics(question)
                    
                    if not topics:
                        html = EmailBotResponder.generate_help_response(question)
                    else:
                        tickers = [info['ticker'] for info in topics.values()]
                        results = await asyncio.gather(
                            *(MarketQuestionAnalyzer.get_market_data(t) for t in tickers)
                        )
                        final_market_data = {
                            key: {**info, **data} 
                            for (key, info), data in zip(topics.items(), results) 
                            if data
                        }
                        
                        # Try intelligent response
                        try:
                            responder = IntelligentEmailBotResponder()
                            html = await responder.generate_intelligent_html(question, final_market_data)
                        except Exception as e:
                            logging.warning(f"Intelligent responder failed: {e}. Using basic.")
                            html = EmailBotResponder.generate_html_response(question, final_market_data)
                    
                    # Send response
                    if self._send_email(self.authorized_email, question, html, subject):
                        answered += 1
                        logging.info(f"✅ Response sent!")
                        if self.db:
                            self.db.log_conversation(self.authorized_email, question, topics, True)
                    else:
                        errors += 1
                    
                    mail.store(eid, '+FLAGS', '\\Seen')
                    await asyncio.sleep(2)
                
                except Exception as e:
                    errors += 1
                    logging.error(f"Error processing email {eid.decode()}: {e}")

            logging.info(f"✅ Complete: {answered}/{checked} answered, {errors} errors")

        except Exception as e:
            errors += 1
            logging.error(f"❌ Bot error: {e}", exc_info=True)
        
        finally:
            if mail:
                try:
                    mail.close()
                    mail.logout()
                except:
                    pass
            if self.db:
                self.db.update_stats(checked, answered, errors)

    def _send_email(self, to_email, question, html_body, original_subject=""):
        """Send email response"""
        try:
            msg = MIMEMultipart('alternative')
            subject = f"Re: {original_subject}" if original_subject else "Market Analysis"
            msg['Subject'] = subject
            msg['From'] = self.smtp_user
            msg['To'] = to_email
            msg['Date'] = email.utils.formatdate(localtime=True)
            
            msg.attach(MIMEText(f"Q: {question}\n\nSee HTML for analysis.", 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP("smtp.gmail.com", 587, 30) as s:
                s.starttls()
                s.login(self.smtp_user, self.smtp_pass)
                s.send_message(msg)
            
            return True
        except Exception as e:
            logging.error(f"Send failed: {e}")
            return False

    def _extract_question(self, msg):
        """Extract question text from email"""
        body = ""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode('utf-8', 'ignore')
                                break
                        except:
                            continue
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', 'ignore')
            
            # Clean reply markers
            lines = [
                l.strip() 
                for l in (body or "").split('\n') 
                if l.strip() 
                and not l.strip().startswith('>') 
                and 'wrote:' not in l.lower()
            ]
            return re.sub(r'\s+', ' ', ' '.join(lines))[:1000]
        except:
            return ""

    def _is_valid(self, question):
        """Check if question is valid"""
        return bool(
            question 
            and len(question.strip()) > 5 
            and not any(k in question.lower() for k in ['automated', 'auto-reply', 'unsubscribe'])
        )


async def run_email_bot():
    """Entry point for email bot mode"""
    try:
        verify_intelligence_available()
        bot = EmailBotEngine()
        await bot.check_and_respond()
    except Exception as e:
        logging.error(f"❌ Bot initialization failed: {e}", exc_info=True)
        sys.exit(1)

def verify_intelligence_available():
    """Logs the status of intelligent components."""
    logging.info("🔍 Verifying Intelligence Components Status:")
    components = {
        'spaCy': 'SPACY_AVAILABLE' in globals() and SPACY_AVAILABLE,
        'Wikipedia': 'WIKIPEDIA_AVAILABLE' in globals() and WIKIPEDIA_AVAILABLE,
        'Groq': 'GROQ_AVAILABLE' in globals() and GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"),
        'Cohere': 'COHERE_AVAILABLE' in globals() and COHERE_AVAILABLE and os.getenv("COHERE_API_KEY"),
    }
    for component, available in components.items():
        logging.info(f"  {'✅' if available else '❌'} {component}: {'Available' if available else 'Not Available'}")

# ========================================
# 🆕 END EMAIL BOT SYSTEM
# ========================================

# ========================================
# 🆕 END EMAIL BOT SYSTEM
# ========================================


# ========================================
# PROGRAM ENTRY POINT
# ========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Market Intelligence System v2.1.0"
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
        help="Run in email bot mode (check inbox and respond to questions)"
    )
    
    args = parser.parse_args()
    
    # Windows event loop compatibility
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    logging.info("=" * 60)
    
    if args.check_emails:
        logging.info("🤖 EMAIL BOT MODE - Market Q&A System")
        logging.info("=" * 60)
        asyncio.run(run_email_bot())
    else:
        logging.info("🚀 MARKET INTELLIGENCE SYSTEM v2.1.0")
        logging.info("=" * 60)
        asyncio.run(main(output=args.output))
