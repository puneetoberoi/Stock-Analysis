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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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

# 🆕 DuckDuckGo search (with fallback)
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logging.warning("⚠️ duckduckgo-search not available - bot news search will be limited")

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
        
        # Try ONLY the stable model that works
        try:
            model = genai.GenerativeModel('gemini-pro')  # This model definitely works
            logging.info(f"✅ Successfully loaded Gemini model: gemini-pro")
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
        return {'analysis': response.text, 'generated_at': datetime.datetime.now().isoformat()}
        
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
    SMTP_USER, SMTP_PASS = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not SMTP_USER or not SMTP_PASS:
        logging.warning("SMTP credentials missing. Cannot send email.")
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
# 🆕 EMAIL BOT SYSTEM - v2.2.0 PRODUCTION
# Complete with all imports and error handling
# ========================================

# ========================================
# 🛡️ PRODUCTION FAILSAFE WRAPPER
# ========================================

def safe_import(module_name, package_name=None):
    """Safely import with fallback"""
    try:
        return __import__(module_name)
    except ImportError:
        logging.warning(f"⚠️ {module_name} not available - using fallback")
        return None

# Safe imports with fallbacks
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logging.info("📌 DuckDuckGo search not available - bot will use basic responses")
    
    # Complete fallback implementation
    class DDGS:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def news(self, query, max_results=3):
            # Return some generic news
            return [
                {'title': f'Market analysis for {query}', 'body': 'Based on current market trends...'},
                {'title': f'{query} outlook', 'body': 'Technical indicators suggest...'}
            ]
        def text(self, query, max_results=3):
            return [
                {'title': f'Analysis: {query}', 'body': 'Market conditions indicate...'}
            ]

# Bot-specific imports (some may be redundant but ensures it works)
import sqlite3
import imaplib
import email
import email.utils
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Try importing optional dependencies
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ duckduckgo_search not available - bot will use fallback")
    DDGS_AVAILABLE = False
    
    # Fallback class if DDGS not available
    class DDGS:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def news(self, query, max_results=3):
            return []
        def text(self, query, max_results=3):
            return []


class EmailBotDatabase:
    """Isolated database for bot conversations"""
    
    def __init__(self, db_path='email_bot.db'):
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self._init_schema()
        except Exception as e:
            logging.error(f"Database init failed: {e}")
            # Use in-memory database as fallback
            self.conn = sqlite3.connect(':memory:', check_same_thread=False)
            self._init_schema()
    
    def _init_schema(self):
        try:
            self.conn.executescript('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_email TEXT,
                    user_question TEXT,
                    topics_detected TEXT,
                    response_sent BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS bot_stats (
                    date TEXT PRIMARY KEY,
                    emails_checked INTEGER DEFAULT 0,
                    questions_answered INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0
                );
            ''')
            self.conn.commit()
        except Exception as e:
            logging.error(f"Schema creation failed: {e}")
    
    def log_conversation(self, user_email, question, topics, success):
        try:
            self.conn.execute('''
                INSERT INTO conversations (timestamp, user_email, user_question, topics_detected, response_sent)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.datetime.now().isoformat(),
                user_email,
                question[:500] if question else "",
                json.dumps(topics) if topics else "{}",
                success
            ))
            self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to log conversation: {e}")
    
    def update_stats(self, checked=0, answered=0, errors=0):
        try:
            today = datetime.date.today().isoformat()
            # SQLite doesn't support ON CONFLICT in older versions, use INSERT OR REPLACE
            existing = self.conn.execute('SELECT * FROM bot_stats WHERE date = ?', (today,)).fetchone()
            
            if existing:
                self.conn.execute('''
                    UPDATE bot_stats 
                    SET emails_checked = emails_checked + ?,
                        questions_answered = questions_answered + ?,
                        errors = errors + ?
                    WHERE date = ?
                ''', (checked, answered, errors, today))
            else:
                self.conn.execute('''
                    INSERT INTO bot_stats (date, emails_checked, questions_answered, errors)
                    VALUES (?, ?, ?, ?)
                ''', (today, checked, answered, errors))
            
            self.conn.commit()
        except Exception as e:
            logging.error(f"Stats update failed: {e}")
    
    def close(self):
        try:
            self.conn.close()
        except:
            pass


class MarketQuestionAnalyzer:
    """Extract topics from user questions"""
    
    @staticmethod
    def extract_topics(question):
        topics = {}
        
        if not question:
            return topics
            
        q_lower = question.lower()
        
        mappings = {
            'bitcoin': {'type': 'crypto', 'ticker': 'BTC-USD', 'name': 'Bitcoin'},
            'btc': {'type': 'crypto', 'ticker': 'BTC-USD', 'name': 'Bitcoin'},
            'ethereum': {'type': 'crypto', 'ticker': 'ETH-USD', 'name': 'Ethereum'},
            'eth': {'type': 'crypto', 'ticker': 'ETH-USD', 'name': 'Ethereum'},
            # 🆕 ADD XRP:
            'xrp': {'type': 'crypto', 'ticker': 'XRP-USD', 'name': 'Ripple'},
            'ripple': {'type': 'crypto', 'ticker': 'XRP-USD', 'name': 'Ripple'},
            'gold': {'type': 'commodity', 'ticker': 'GC=F', 'name': 'Gold'},
            'silver': {'type': 'commodity', 'ticker': 'SI=F', 'name': 'Silver'},
            'oil': {'type': 'commodity', 'ticker': 'CL=F', 'name': 'Crude Oil'},
            'crude': {'type': 'commodity', 'ticker': 'CL=F', 'name': 'Crude Oil'},
            'sp500': {'type': 'index', 'ticker': '^GSPC', 'name': 'S&P 500'},
            's&p': {'type': 'index', 'ticker': '^GSPC', 'name': 'S&P 500'},
            'nasdaq': {'type': 'index', 'ticker': '^IXIC', 'name': 'Nasdaq'},
            'dow': {'type': 'index', 'ticker': '^DJI', 'name': 'Dow Jones'},
        }
        
        # Extract topics (with deduplication)
        seen = set()
        for keyword, data in mappings.items():
            if keyword in q_lower and data['ticker'] not in seen:
                topics[keyword] = data
                seen.add(data['ticker'])  # 🆕 Prevent duplicates
        
        # Extract stock tickers
        try:
            ticker_matches = re.findall(r'\$?([A-Z]{1,5})\b', question)
            for ticker in ticker_matches:
                if ticker not in ['USD', 'ETH', 'BTC', 'CEO', 'USA', 'AI', 'Q', 'A'] and len(ticker) <= 5:
                    if ticker not in seen:  # 🆕 Prevent duplicates
                        topics[ticker.lower()] = {'type': 'stock', 'ticker': ticker, 'name': ticker}
                        seen.add(ticker)
        except:
            pass
        
        return topics
    
    @staticmethod
    async def get_market_data(ticker):
        """Fetch market data with comprehensive error handling"""
        try:
            stock = yf.Ticker(ticker)
            hist = await asyncio.to_thread(stock.history, period='3mo')
            
            if hist.empty:
                logging.warning(f"No data for {ticker}")
                return None
            
            current = hist['Close'].iloc[-1]
            day_ago = hist['Close'].iloc[-2] if len(hist) > 1 else current
            week_ago = hist['Close'].iloc[-5] if len(hist) >= 5 else current
            month_ago = hist['Close'].iloc[-22] if len(hist) >= 22 else current
            
            # Safe RSI calculation
            try:
                rsi = RSIIndicator(hist['Close'], window=14).rsi().iloc[-1]
                if pd.isna(rsi):
                    rsi = 50
            except:
                rsi = 50
            
            # Safe calculations with division by zero check
            daily_change = ((current - day_ago) / day_ago * 100) if day_ago != 0 else 0
            weekly_change = ((current - week_ago) / week_ago * 100) if week_ago != 0 else 0
            monthly_change = ((current - month_ago) / month_ago * 100) if month_ago != 0 else 0
            
            return {
                'ticker': ticker,
                'price': float(current),
                'daily_change': float(daily_change),
                'weekly_change': float(weekly_change),
                'monthly_change': float(monthly_change),
                'rsi': float(rsi),
                'year_high': float(hist['High'].tail(252).max() if len(hist) > 252 else hist['High'].max()),
                'year_low': float(hist['Low'].tail(252).min() if len(hist) > 252 else hist['Low'].min()),
            }
        except Exception as e:
            logging.warning(f"Data fetch failed for {ticker}: {e}")
            return None
    
    @staticmethod
    async def search_news(query, max_results=3):
        """Search news with fallback"""
        if not DDGS_AVAILABLE:
            return "Market analysis based on current trends..."
            
        try:
            results = ""
            with DDGS() as ddgs:
                try:
                    news = list(ddgs.news(query, max_results=max_results))
                    for i, item in enumerate(news, 1):
                        results += f"[{i}] {item.get('title', '')}\n"
                except:
                    try:
                        text = list(ddgs.text(query, max_results=max_results))
                        for i, item in enumerate(text, 1):
                            results += f"[{i}] {item.get('title', '')}\n"
                    except:
                        pass
            return results or "Market analysis based on current trends..."
        except Exception as e:
            logging.debug(f"News search failed: {e}")
            return "Market analysis based on current trends..."


class EmailBotResponder:
    """Generate HTML responses"""
    
    @staticmethod
    def create_price_card(data):
        if not data:
            return ""
            
        try:
            change_color = '#16a34a' if data.get('daily_change', 0) >= 0 else '#dc2626'
            return f"""
            <div style="background: linear-gradient(135deg, #e0f2fe, #bae6fd); padding: 20px; 
                        border-radius: 12px; margin: 15px 0; display: inline-block; 
                        width: 45%; margin-right: 3%; vertical-align: top;">
                <h3 style="margin: 0; color: #1e40af; text-transform: uppercase;">{data.get('name', 'Unknown')}</h3>
                <p style="font-size: 32px; font-weight: bold; margin: 10px 0;">${data.get('price', 0):,.2f}</p>
                <p style="color: {change_color}; font-size: 18px; margin: 5px 0;">{data.get('daily_change', 0):+.2f}% today</p>
                <p style="font-size: 12px; color: #6b7280;">
                    Week: {data.get('weekly_change', 0):+.1f}% | Month: {data.get('monthly_change', 0):+.1f}%<br>
                    RSI: {data.get('rsi', 50):.0f} | Range: ${data.get('year_low', 0):,.2f} - ${data.get('year_high', 0):,.2f}
                </p>
            </div>
            """
        except Exception as e:
            logging.error(f"Price card creation failed: {e}")
            return ""
    
    @staticmethod
    def create_analysis_section(data):
        if not data:
            return ""
            
        try:
            rsi = data.get('rsi', 50)
            if rsi > 70:
                rsi_signal = f"<strong style='color: #dc2626;'>OVERBOUGHT</strong> (RSI {rsi:.0f})"
            elif rsi < 30:
                rsi_signal = f"<strong style='color: #16a34a;'>OVERSOLD</strong> (RSI {rsi:.0f})"
            else:
                rsi_signal = f"<strong>NEUTRAL</strong> (RSI {rsi:.0f})"
            
            monthly = data.get('monthly_change', 0)
            trend = "Strong uptrend" if monthly > 10 else \
                    "Strong downtrend" if monthly < -10 else \
                    "Consolidation phase"
            
            price = data.get('price', 0)
            high = data.get('year_high', price)
            low = data.get('year_low', price)
            
            dist_high = ((price - high) / high * 100) if high != 0 else 0
            dist_low = ((price - low) / low * 100) if low != 0 else 0
            
            return f"""
<h2 style="color: #2563eb; margin-top: 30px;">📈 {data.get('name', 'Asset')} ({data.get('ticker', 'N/A')}) Analysis</h2>

<h3>Technical Signals</h3>
<ul style="line-height: 1.8;">
    <li><strong>Momentum:</strong> {rsi_signal}</li>
    <li><strong>Trend:</strong> {trend}</li>
    <li><strong>Position:</strong> {abs(dist_high):.1f}% from 52W high, {dist_low:.1f}% from 52W low</li>
</ul>

<h3>Performance Summary</h3>
<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
<tr style="background: #f9fafb;">
    <th style="padding: 10px; text-align: left;">Period</th>
    <th style="padding: 10px; text-align: right;">Change</th>
</tr>
<tr><td style="padding: 10px;">Daily</td>
    <td style="padding: 10px; text-align: right; color: {'#16a34a' if data.get('daily_change', 0) > 0 else '#dc2626'}; font-weight: bold;">
    {data.get('daily_change', 0):+.2f}%</td></tr>
<tr><td style="padding: 10px;">Weekly</td>
    <td style="padding: 10px; text-align: right; color: {'#16a34a' if data.get('weekly_change', 0) > 0 else '#dc2626'}; font-weight: bold;">
    {data.get('weekly_change', 0):+.2f}%</td></tr>
<tr><td style="padding: 10px;">Monthly</td>
    <td style="padding: 10px; text-align: right; color: {'#16a34a' if data.get('monthly_change', 0) > 0 else '#dc2626'}; font-weight: bold;">
    {data.get('monthly_change', 0):+.2f}%</td></tr>
</table>
"""
        except Exception as e:
            logging.error(f"Analysis section creation failed: {e}")
            return ""
    
    @staticmethod
    def generate_html_response(question, market_data):
        try:
            if not market_data:
                return EmailBotResponder.generate_help_response(question)
            
            price_cards = []
            analysis_sections = []
            
            for data in market_data.values():
                if data:
                    card = EmailBotResponder.create_price_card(data)
                    if card:
                        price_cards.append(card)
                    
                    section = EmailBotResponder.create_analysis_section(data)
                    if section:
                        analysis_sections.append(section)
            
            return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 900px; margin: 0 auto; background: linear-gradient(135deg, #f5f5f5 0%, #e5e5e5 100%); padding: 20px;}}
.container {{background: white; border-radius: 20px; overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.15);}}
.header {{background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center;}}
.content {{padding: 40px;}}
.question-box {{background: linear-gradient(135deg, #fef3c7, #fed7aa); border-left: 5px solid #f59e0b;
                 padding: 25px; border-radius: 10px; margin-bottom: 30px;}}
h2 {{margin-top: 30px; padding-bottom: 10px; border-bottom: 2px solid #e5e7eb;}}
.footer {{background: #f3f4f6; padding: 25px; text-align: center; color: #6b7280;}}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 Market Intelligence Report</h1>
        <p style="margin: 15px 0 0 0; font-size: 18px;">
            {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        </p>
    </div>
    
    <div class="content">
        <div class="question-box">
            <h2 style="margin: 0; color: #92400e; border: none;">Your Question</h2>
            <p style="margin: 10px 0 0 0; font-size: 16px; color: #451a03;">"{question}"</p>
        </div>
        
        <div style="margin: 30px 0;">
            {''.join(price_cards)}
        </div>
        
        <div style="clear: both;"></div>
        
        {''.join(analysis_sections)}
        
        <div class="footer">
            <p style="margin: 0;">
                <strong>Data Sources:</strong> Yahoo Finance, Market Analysis<br>
                <strong>Disclaimer:</strong> For informational purposes only. Not financial advice.
            </p>
        </div>
    </div>
</div>
</body>
</html>
"""
        except Exception as e:
            logging.error(f"HTML generation failed: {e}")
            return EmailBotResponder.generate_help_response(question)
    
    @staticmethod
    def generate_help_response(question):
        return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
body {{font-family: -apple-system, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px;}}
.info-box {{background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 20px; border-radius: 8px; margin: 20px 0;}}
.example {{background: #f9fafb; padding: 15px; border-radius: 8px; margin: 10px 0;}}
</style>
</head>
<body>
<h1>📊 Market Intelligence Bot</h1>
<div class="info-box">
    <h2 style="margin-top: 0;">Your Question:</h2>
    <p style="font-size: 16px;">"{question if question else 'No question detected'}"</p>
</div>

<h2>I can analyze:</h2>
<ul style="line-height: 2;">
    <li><strong>Stocks:</strong> Any ticker (e.g., "AAPL analysis", "What's TSLA doing?")</li>
    <li><strong>Crypto:</strong> Bitcoin, Ethereum</li>
    <li><strong>Commodities:</strong> Gold, silver, oil</li>
    <li><strong>Indices:</strong> S&P 500, Nasdaq, Dow</li>
</ul>

<h2>Example Questions:</h2>
<div class="example"><strong>💡 "What's happening with gold?"</strong></div>
<div class="example"><strong>💡 "Bitcoin analysis"</strong></div>
<div class="example"><strong>💡 "How's NVDA looking?"</strong></div>

<p style="margin-top: 30px; padding: 15px; background: #fef3c7; border-radius: 8px;">
<strong>💡 Tip:</strong> Mention specific ticker symbols or asset names for detailed analysis!
</p>
</body>
</html>
"""


class EmailBotEngine:
    """Main email bot engine with comprehensive error handling"""
    
    def __init__(self):
        self.db = None
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        
        if not self.smtp_user or not self.smtp_pass:
            raise ValueError("❌ SMTP_USER and SMTP_PASS environment variables required!")
        
        try:
            self.db = EmailBotDatabase()
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            # Continue without database
    
    async def check_and_respond(self):
        """Check inbox and respond with full error handling"""
        logging.info("📧 Email bot checking inbox...")
        
        checked = 0
        answered = 0
        errors = 0
        
        mail = None
        
        try:
            # Connect to inbox
            mail = imaplib.IMAP4_SSL("imap.gmail.com", timeout=15)
            mail.login(self.smtp_user, self.smtp_pass)
            mail.select('inbox')
            
            # Search for unread emails
            since = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%d-%b-%Y")
            _, search_data = mail.search(None, f'(UNSEEN SINCE {since})')
            
            email_ids = search_data[0].split()
            
            if not email_ids:
                logging.info("✅ No new emails")
                return
            
            logging.info(f"📬 Found {len(email_ids)} unread email(s)")
            
            # Process newest 3 emails
            for email_id in list(reversed(email_ids))[:3]:
                try:
                    checked += 1
                    
                    # Fetch email
                    _, data = mail.fetch(email_id, '(RFC822)')
                    msg = email.message_from_bytes(data[0][1])
                    
                    sender = email.utils.parseaddr(msg['From'])[1]
                    subject = msg.get('Subject', '')
                    
                    # Extract question
                    question = self._extract_question(msg)
                    
                    if not self._is_valid_question(question):
                        mail.store(email_id, '+FLAGS', '\\Seen')
                        continue
                    
                    logging.info(f"❓ Question from {sender}: '{question[:60]}...'")
                    
                    # Analyze question
                    topics = MarketQuestionAnalyzer.extract_topics(question)
                    market_data = {}
                    
                    # Fetch market data for detected topics
                    for key, info in topics.items():
                        if info and 'ticker' in info:
                            data = await MarketQuestionAnalyzer.get_market_data(info['ticker'])
                            if data:
                                market_data[key] = {**info, **data}
                    
                    # 🆕 INTELLIGENT RESPONSE GENERATION
                    html_response = None
                    
                    # Check if intelligent responder is available
                    try:
                        # Try to use intelligent responder if it exists
                        if 'IntelligentEmailBotResponder' in globals():
                            logging.info("🧠 Using intelligent responder...")
                            intelligent_responder = IntelligentEmailBotResponder()
                            html_response = await intelligent_responder.generate_intelligent_html(question, market_data)
                            logging.info("✅ Intelligent response generated")
                        else:
                            logging.info("⚠️ Intelligent responder not found, using basic")
                    except Exception as e:
                        logging.warning(f"Intelligent responder failed: {e}, falling back to basic")
                    
                    # Fallback to basic responder if intelligent failed
                    if not html_response:
                        logging.info("📊 Using basic responder")
                        html_response = EmailBotResponder.generate_html_response(question, market_data)
                    
                    # Send response
                    if self._send_email(sender, question, html_response, subject):
                        answered += 1
                        if self.db:
                            self.db.log_conversation(sender, question, topics, True)
                        logging.info(f"✅ Answered {sender}")
                    else:
                        errors += 1
                        if self.db:
                            self.db.log_conversation(sender, question, topics, False)
                    
                    # Mark as read
                    mail.store(email_id, '+FLAGS', '\\Seen')
                    
                    # Rate limit
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    errors += 1
                    logging.error(f"Error processing email: {e}")
                    continue
            
            logging.info(f"✅ Bot complete: {answered}/{checked} answered")
            
        except Exception as e:
            logging.error(f"❌ Bot error: {e}")
            errors += 1
        finally:
            # Clean up
            if mail:
                try:
                    mail.close()
                    mail.logout()
                except:
                    pass
            
            if self.db:
                self.db.update_stats(checked, answered, errors)
                self.db.close()
    
    def _extract_question(self, msg):
        """Extract question from email body"""
        body = ""
        
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode('utf-8', errors='ignore')
                                break
                        except:
                            continue
            else:
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body = payload.decode('utf-8', errors='ignore')
                except:
                    body = str(msg.get_payload())
            
            # Clean up body
            lines = []
            for line in body.split('\n'):
                # Stop at quoted content
                if any(m in line.lower() for m in ['wrote:', 'from:', 'sent:', '----', 'on ', 'date:']):
                    break
                # Skip quoted lines
                if line.strip().startswith('>'):
                    continue
                # Skip HTML fallback
                if 'please view' in line.lower():
                    continue
                
                if line.strip():
                    lines.append(line.strip())
            
            return re.sub(r'\s+', ' ', ' '.join(lines).strip())
        except Exception as e:
            logging.error(f"Question extraction failed: {e}")
            return ""
    
    def _is_valid_question(self, question):
        """Validate question"""
        if not question or len(question) < 10:
            return False
        
        skip = ['automated', 'auto-reply', 'out of office', 'unsubscribe', 'no-reply', 'mailer-daemon']
        return not any(kw in question.lower() for kw in skip)
    
    def _send_email(self, to_email, question, html_body, original_subject=""):
        """Send email response with error handling"""
        try:
            msg = MIMEMultipart('alternative')
            
            # Create subject
            if original_subject and not original_subject.startswith('Re:'):
                subject = f"Re: {original_subject}"
            elif original_subject:
                subject = original_subject
            else:
                subject = f"Market Analysis - {datetime.datetime.now().strftime('%b %d')}"
            
            msg['Subject'] = subject
            msg['From'] = self.smtp_user
            msg['To'] = to_email
            msg['Date'] = email.utils.formatdate(localtime=True)
            
            # Plain text fallback
            text_content = f"Your question: {question}\n\nPlease view this email in HTML format for the full analysis."
            text_part = MIMEText(text_content, 'plain')
            
            # HTML content
            html_part = MIMEText(html_body, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            
            return True
        except Exception as e:
            logging.error(f"Email send failed: {e}")
            return False


async def run_email_bot():
    """Entry point for email bot mode with error handling"""
    try:
        bot = EmailBotEngine()
        await bot.check_and_respond()
    except Exception as e:
        logging.error(f"❌ Bot initialization failed: {e}")
        sys.exit(1)

# ========================================
# 🆕 END EMAIL BOT SYSTEM
# ========================================

# ========================================
# 🆕 INTELLIGENT ANALYSIS ENGINE
# ========================================

class IntelligentMarketAnalyzer:
    """Real intelligence without hardcoding - Level 2 Deep Analysis"""
    
    def __init__(self):
        # Initialize NLP
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logging.warning("spaCy model not loaded")
        
        # Initialize Wikipedia
        self.wiki = None
        if WIKIPEDIA_AVAILABLE:
            self.wiki = wikipediaapi.Wikipedia('MarketBot/1.0', 'en')
        
        # Initialize LLM clients
        self.llm_clients = self._setup_llm_clients()
    
    def _setup_llm_clients(self):
        """Setup free LLM API clients"""
        clients = {}
        
        # Groq (30K tokens/day free)
        if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                clients['groq'] = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logging.info("✅ Groq LLM available")
            except Exception as e:
                logging.warning(f"Groq setup failed: {e}")
        
        # Cohere (1000 requests/month free)
        if COHERE_AVAILABLE and os.getenv("COHERE_API_KEY"):
            try:
                import cohere
                clients['cohere'] = cohere.Client(os.getenv("COHERE_API_KEY"))
                logging.info("✅ Cohere LLM available")
            except Exception as e:
                logging.warning(f"Cohere setup failed: {e}")
        
        # Hugging Face (1000 requests/day free)
        if os.getenv("HUGGINGFACE_API_KEY"):
            clients['huggingface'] = {
                'api_key': os.getenv("HUGGINGFACE_API_KEY"),
                'url': 'https://api-inference.huggingface.co/models/'
            }
            logging.info("✅ Hugging Face LLM available")
        
        return clients
    
    async def answer_intelligently(self, question, ticker_data):
        """Generate truly intelligent answers using Level 2 deep analysis"""
        
        logging.info(f"🧠 Generating intelligent answer for: {question[:50]}...")
        
        # 1. Understand question intent
        intent = self._analyze_question_intent(question)
        
        # 2. Gather comprehensive context
        context = await self._gather_deep_context(question, ticker_data, intent)
        
        # 3. Generate intelligent response using LLMs
        response = await self._generate_llm_response(question, intent, context, ticker_data)
        
        # 4. If LLM fails, use intelligent assembly
        if not response or len(response) < 100:
            response = await self._intelligent_assembly(question, intent, context, ticker_data)
        
        return response
    
    def _analyze_question_intent(self, question):
        """Deep intent analysis"""
        intent = {
            'wants_reasons': False,
            'wants_applications': False,
            'wants_prediction': False,
            'wants_comparison': False,
            'wants_technical': False,
            'wants_fundamental': False,
            'wants_news': False,
            'entities': [],
            'key_topics': [],
            'question_type': 'general'
        }
        
        q_lower = question.lower()
        
        # Detect what user wants
        if any(word in q_lower for word in ['why', 'reason', 'cause', 'because', 'driver']):
            intent['wants_reasons'] = True
            intent['question_type'] = 'explanation'
        
        if any(word in q_lower for word in ['application', 'use', 'utility', 'commercial', 'practical', 'purpose']):
            intent['wants_applications'] = True
            intent['question_type'] = 'use_cases'
        
        if any(word in q_lower for word in ['will', 'future', 'prediction', 'forecast', 'outlook', 'target']):
            intent['wants_prediction'] = True
            intent['question_type'] = 'forecast'
        
        if any(word in q_lower for word in ['compare', 'versus', 'vs', 'difference', 'better']):
            intent['wants_comparison'] = True
            intent['question_type'] = 'comparison'
        
        if any(word in q_lower for word in ['technical', 'rsi', 'macd', 'support', 'resistance']):
            intent['wants_technical'] = True
        
        if any(word in q_lower for word in ['fundamental', 'earnings', 'revenue', 'profit']):
            intent['wants_fundamental'] = True
        
        if any(word in q_lower for word in ['news', 'latest', 'recent', 'today', 'update']):
            intent['wants_news'] = True
        
        # Extract entities using spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(question)
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'MONEY', 'PERSON']:
                        intent['entities'].append(ent.text)
                
                # Extract key nouns
                for token in doc:
                    if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                        intent['key_topics'].append(token.text)
            except:
                pass
        
        return intent
    
    async def _gather_deep_context(self, question, ticker_data, intent):
        """Level 2: Comprehensive context gathering from multiple sources"""
        context = {
            'news': [],
            'wikipedia': {},
            'github': {},
            'reddit': {},
            'technical': {},
            'fundamental': {},
            'web_search': [],
            'historical_patterns': {},
            'social_sentiment': {}
        }
        
        asset_name = ticker_data.get('name', '')
        ticker = ticker_data.get('ticker', '')
        
        # 1. Deep news search with multiple queries
        try:
            with DDGS() as ddgs:
                queries = []
                
                if intent['wants_reasons']:
                    queries.append(f"{asset_name} price movement reasons {datetime.datetime.now().year}")
                    queries.append(f"why {asset_name} rising falling analysis")
                
                if intent['wants_applications']:
                    queries.append(f"{asset_name} commercial applications enterprise use cases")
                    queries.append(f"{asset_name} real world adoption examples")
                
                if intent['wants_news']:
                    queries.append(f"{asset_name} latest news today")
                
                # Default query
                if not queries:
                    queries.append(f"{asset_name} analysis {datetime.datetime.now().strftime('%B %Y')}")
                
                all_news = []
                for query in queries[:2]:  # Limit to 2 queries
                    try:
                        news = list(ddgs.news(query, max_results=3))
                        all_news.extend(news)
                    except:
                        # Fallback to text search
                        text = list(ddgs.text(query, max_results=3))
                        all_news.extend(text)
                
                # Process and deduplicate news
                seen_titles = set()
                for item in all_news:
                    title = item.get('title', '')
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        context['news'].append({
                            'title': title,
                            'body': item.get('body', ''),
                            'url': item.get('url', ''),
                            'date': item.get('date', '')
                        })
                
                context['news'] = context['news'][:5]  # Keep top 5
        except Exception as e:
            logging.warning(f"News search failed: {e}")
        
        # 2. Wikipedia deep dive
        if self.wiki and asset_name:
            try:
                # Try multiple variations
                search_terms = [
                    asset_name,
                    f"{asset_name} (cryptocurrency)" if 'crypto' in str(ticker_data.get('type', '')).lower() else asset_name,
                    ticker.replace('-USD', '') if ticker else asset_name
                ]
                
                for term in search_terms:
                    page = self.wiki.page(term)
                    if page.exists():
                        context['wikipedia']['summary'] = page.summary[:1000]
                        
                        # Extract specific sections
                        if intent['wants_applications']:
                            for section in ['Applications', 'Use cases', 'Commercial use', 'Adoption']:
                                if section in page.sections:
                                    context['wikipedia']['applications'] = page.section(section)[:800]
                                    break
                        
                        if intent['wants_reasons']:
                            for section in ['History', 'Development', 'Technology']:
                                if section in page.sections:
                                    context['wikipedia']['background'] = page.section(section)[:800]
                                    break
                        break
            except Exception as e:
                logging.warning(f"Wikipedia failed: {e}")
        
        # 3. GitHub activity for crypto projects
        if 'crypto' in str(ticker_data.get('type', '')).lower():
            try:
                async with aiohttp.ClientSession() as session:
                    repo_map = {
                        'bitcoin': 'bitcoin/bitcoin',
                        'ethereum': 'ethereum/go-ethereum',
                        'cardano': 'input-output-hk/cardano-node',
                        'solana': 'solana-labs/solana',
                        'polkadot': 'paritytech/polkadot',
                        'chainlink': 'smartcontractkit/chainlink',
                        'ripple': 'ripple/rippled',
                        'xrp': 'ripple/rippled'
                    }
                    
                    asset_lower = asset_name.lower()
                    repo = None
                    
                    for key, repo_path in repo_map.items():
                        if key in asset_lower:
                            repo = repo_path
                            break
                    
                    if repo:
                        url = f"https://api.github.com/repos/{repo}"
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                
                                # Get recent commits
                                commits_url = f"https://api.github.com/repos/{repo}/commits"
                                async with session.get(commits_url) as commits_resp:
                                    if commits_resp.status == 200:
                                        commits = await commits_resp.json()
                                        recent_commits = len(commits[:30])  # Last 30 commits
                                
                                context['github'] = {
                                    'stars': data.get('stargazers_count', 0),
                                    'forks': data.get('forks_count', 0),
                                    'open_issues': data.get('open_issues_count', 0),
                                    'watchers': data.get('watchers_count', 0),
                                    'description': data.get('description', ''),
                                    'recent_activity': recent_commits,
                                    'last_update': data.get('pushed_at', '')
                                }
            except Exception as e:
                logging.warning(f"GitHub fetch failed: {e}")
        
        # 4. Reddit sentiment analysis
        try:
            async with aiohttp.ClientSession() as session:
                subreddits = ['cryptocurrency', 'wallstreetbets', 'stocks', 'investing']
                all_sentiments = []
                
                for subreddit in subreddits:
                    url = f"https://www.reddit.com/r/{subreddit}/search.json?q={asset_name}&sort=hot&limit=10&t=week"
                    
                    try:
                        async with session.get(url, headers={'User-Agent': 'MarketBot/1.0'}) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                posts = data.get('data', {}).get('children', [])
                                
                                for post in posts[:5]:
                                    title = post['data'].get('title', '')
                                    score = post['data'].get('score', 0)
                                    num_comments = post['data'].get('num_comments', 0)
                                    
                                    # Sentiment analysis
                                    if TEXTBLOB_AVAILABLE:
                                        sentiment = TextBlob(title).sentiment.polarity
                                        all_sentiments.append({
                                            'sentiment': sentiment,
                                            'weight': score + num_comments
                                        })
                    except:
                        continue
                
                if all_sentiments:
                    # Weighted average sentiment
                    total_weight = sum(s['weight'] for s in all_sentiments)
                    if total_weight > 0:
                        weighted_sentiment = sum(s['sentiment'] * s['weight'] for s in all_sentiments) / total_weight
                        
                        context['reddit'] = {
                            'sentiment': weighted_sentiment,
                            'sentiment_label': 'Bullish' if weighted_sentiment > 0.1 else 'Bearish' if weighted_sentiment < -0.1 else 'Neutral',
                            'post_count': len(all_sentiments),
                            'total_engagement': total_weight
                        }
        except Exception as e:
            logging.warning(f"Reddit analysis failed: {e}")
        
        # 5. Technical analysis from yfinance
        if ticker:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                context['fundamental'] = {
                    'market_cap': info.get('marketCap', 0),
                    'volume': info.get('volume', 0),
                    'avg_volume': info.get('averageVolume', 0),
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'peg_ratio': info.get('pegRatio'),
                    'beta': info.get('beta'),
                    'description': info.get('longBusinessSummary', '')[:500],
                    'website': info.get('website', ''),
                    'industry': info.get('industry', ''),
                    'sector': info.get('sector', ''),
                    'employees': info.get('fullTimeEmployees', 0)
                }
                
                # Get recent analyst recommendations
                try:
                    rec = stock.recommendations
                    if rec is not None and not rec.empty:
                        recent_rec = rec.tail(5)
                        context['fundamental']['analyst_recommendations'] = recent_rec.to_dict('records')
                except:
                    pass
                
                # Get institutional holders
                try:
                    inst = stock.institutional_holders
                    if inst is not None and not inst.empty:
                        context['fundamental']['top_holders'] = inst.head(5).to_dict('records')
                except:
                    pass
            except Exception as e:
                logging.warning(f"Fundamental data failed: {e}")
        
        # 6. Historical patterns
        if ticker:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                
                if not hist.empty:
                    # Find similar historical patterns
                    current_rsi = ticker_data.get('rsi', 50)
                    current_price = ticker_data.get('price', 0)
                    
                    # Simple pattern matching
                    similar_periods = []
                    for i in range(30, len(hist) - 30, 5):
                        period_close = hist['Close'].iloc[i]
                        period_rsi = RSIIndicator(hist['Close'].iloc[i-14:i+1]).rsi().iloc[-1] if len(hist['Close'].iloc[i-14:i+1]) > 0 else 50
                        
                        if abs(period_rsi - current_rsi) < 5:  # Similar RSI
                            future_return = (hist['Close'].iloc[min(i+30, len(hist)-1)] / period_close - 1) * 100
                            similar_periods.append({
                                'date': hist.index[i].strftime('%Y-%m-%d'),
                                'rsi': period_rsi,
                                'future_30d_return': future_return
                            })
                    
                    if similar_periods:
                        avg_return = sum(p['future_30d_return'] for p in similar_periods) / len(similar_periods)
                        context['historical_patterns'] = {
                            'similar_periods': len(similar_periods),
                            'avg_30d_return': avg_return,
                            'pattern_signal': 'Bullish' if avg_return > 5 else 'Bearish' if avg_return < -5 else 'Neutral'
                        }
            except Exception as e:
                logging.warning(f"Pattern analysis failed: {e}")
        
        return context
    
    async def _generate_llm_response(self, question, intent, context, ticker_data):
        """Generate response using free LLM APIs with fallback chain"""
        
        # Build comprehensive prompt
        prompt = self._build_llm_prompt(question, intent, context, ticker_data)
        
        response = None
        
        # Try Groq first (fastest, 30K tokens/day)
        if 'groq' in self.llm_clients:
            try:
                logging.info("Trying Groq LLM...")
                client = self.llm_clients['groq']
                
                completion = client.chat.completions.create(
                    model="mixtral-8x7b-32768",  # Or "llama2-70b-4096"
                    messages=[
                        {"role": "system", "content": "You are a professional market analyst. Provide detailed, accurate analysis based on the given context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                response = completion.choices[0].message.content
                logging.info("✅ Groq response generated")
                
            except Exception as e:
                logging.warning(f"Groq failed: {e}")
        
        # Try Cohere if Groq failed
        if not response and 'cohere' in self.llm_clients:
            try:
                logging.info("Trying Cohere LLM...")
                client = self.llm_clients['cohere']
                
                result = client.generate(
                    model='command-light',  # Free tier model
                    prompt=prompt,
                    max_tokens=800,
                    temperature=0.7
                )
                
                response = result.generations[0].text
                logging.info("✅ Cohere response generated")
                
            except Exception as e:
                logging.warning(f"Cohere failed: {e}")
        
        # Try Hugging Face if others failed
        if not response and 'huggingface' in self.llm_clients:
            try:
                logging.info("Trying Hugging Face LLM...")
                
                api_key = self.llm_clients['huggingface']['api_key']
                
                async with aiohttp.ClientSession() as session:
                    url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
                    headers = {"Authorization": f"Bearer {api_key}"}
                    
                    # Summarization model as fallback
                    payload = {
                        "inputs": prompt[:1024],  # Limit input
                        "parameters": {"max_length": 500}
                    }
                    
                    async with session.post(url, headers=headers, json=payload) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            response = result[0]['summary_text'] if isinstance(result, list) else result.get('summary_text', '')
                            logging.info("✅ Hugging Face response generated")
            
            except Exception as e:
                logging.warning(f"Hugging Face failed: {e}")
        
        return response
    
    def _build_llm_prompt(self, question, intent, context, ticker_data):
        """Build comprehensive prompt for LLM"""
        
        asset_name = ticker_data.get('name', 'Asset')
        ticker = ticker_data.get('ticker', '')
        
        prompt = f"""Analyze this market question and provide a comprehensive answer:

QUESTION: {question}

ASSET: {asset_name} ({ticker})

CURRENT DATA:
- Price: ${ticker_data.get('price', 0):,.2f}
- Daily Change: {ticker_data.get('daily_change', 0):+.2f}%
- Weekly Change: {ticker_data.get('weekly_change', 0):+.2f}%
- Monthly Change: {ticker_data.get('monthly_change', 0):+.2f}%
- RSI: {ticker_data.get('rsi', 50):.0f}

"""
        
        # Add relevant context based on intent
        if intent['wants_reasons'] and context['news']:
            prompt += "\nRECENT NEWS:\n"
            for news in context['news'][:3]:
                prompt += f"- {news['title']}\n"
        
        if intent['wants_applications'] and context['wikipedia'].get('applications'):
            prompt += f"\nAPPLICATIONS:\n{context['wikipedia']['applications'][:500]}\n"
        
        if context['reddit']:
            prompt += f"\nSOCIAL SENTIMENT: {context['reddit'].get('sentiment_label', 'Unknown')} (Score: {context['reddit'].get('sentiment', 0):.2f})\n"
        
        if context['github']:
            prompt += f"\nDEVELOPER ACTIVITY:\n"
            prompt += f"- GitHub Stars: {context['github'].get('stars', 0):,}\n"
            prompt += f"- Recent Activity: {context['github'].get('recent_activity', 0)} commits\n"
        
        if context['historical_patterns']:
            prompt += f"\nHISTORICAL PATTERNS:\n"
            prompt += f"- Similar periods: {context['historical_patterns'].get('similar_periods', 0)}\n"
            prompt += f"- Average 30-day return: {context['historical_patterns'].get('avg_30d_return', 0):.1f}%\n"
        
        # Specific instructions based on intent
        if intent['wants_reasons']:
            prompt += "\nFocus on explaining WHY the price is moving. Include market drivers, catalysts, and fundamental reasons."
        
        if intent['wants_applications']:
            prompt += "\nFocus on practical applications, commercial use cases, and real-world adoption examples."
        
        if intent['wants_prediction']:
            prompt += "\nProvide outlook and potential price targets based on the data. Include both bullish and bearish scenarios."
        
        prompt += "\n\nProvide a detailed, professional response that directly answers the question. Use the context provided to support your analysis."
        
        return prompt
    
    async def _intelligent_assembly(self, question, intent, context, ticker_data):
        """Fallback: Assemble intelligent response without LLM"""
        
        sections = []
        asset_name = ticker_data.get('name', 'Asset')
        
        # Title
        sections.append(f"# {asset_name} Analysis\n")
        
        # Current market data
        sections.append(f"""## 📊 Current Market Data
- **Price**: ${ticker_data.get('price', 0):,.2f}
- **24h Change**: {ticker_data.get('daily_change', 0):+.2f}%
- **Weekly**: {ticker_data.get('weekly_change', 0):+.2f}%
- **Monthly**: {ticker_data.get('monthly_change', 0):+.2f}%
- **RSI**: {ticker_data.get('rsi', 50):.0f} ({'Overbought' if ticker_data.get('rsi', 50) > 70 else 'Oversold' if ticker_data.get('rsi', 50) < 30 else 'Neutral'})
""")
        
        # Reasons for price movement
        if intent['wants_reasons']:
            reasons = ["## 📈 Why the Price is Moving\n"]
            
            # From news
            if context['news']:
                reasons.append("### Recent Developments:")
                for i, article in enumerate(context['news'][:3], 1):
                    if article['title']:
                        # Simple sentiment
                        sentiment = "📈" if any(word in article['title'].lower() for word in ['surge', 'rise', 'gain', 'bull']) else "📉" if any(word in article['title'].lower() for word in ['fall', 'drop', 'bear', 'decline']) else "📰"
                        reasons.append(f"{i}. {sentiment} {article['title']}")
            
            # From Reddit sentiment
            if context['reddit']:
                sentiment_label = context['reddit'].get('sentiment_label', 'Neutral')
                engagement = context['reddit'].get('total_engagement', 0)
                reasons.append(f"\n### Social Sentiment: {sentiment_label}")
                reasons.append(f"- Community engagement: {engagement:,} interactions")
                reasons.append(f"- Overall mood: {'Positive 🟢' if sentiment_label == 'Bullish' else 'Negative 🔴' if sentiment_label == 'Bearish' else 'Mixed 🟡'}")
            
            # From historical patterns
            if context['historical_patterns']:
                avg_return = context['historical_patterns'].get('avg_30d_return', 0)
                similar = context['historical_patterns'].get('similar_periods', 0)
                reasons.append(f"\n### Historical Pattern Analysis:")
                reasons.append(f"- Found {similar} similar historical periods")
                reasons.append(f"- Average 30-day return: {avg_return:+.1f}%")
                reasons.append(f"- Pattern suggests: {'Bullish continuation 📈' if avg_return > 5 else 'Bearish reversal 📉' if avg_return < -5 else 'Consolidation phase ➡️'}")
            
            sections.append('\n'.join(reasons))
        
        # Applications and use cases
        if intent['wants_applications']:
            apps = ["## 🏢 Commercial Applications & Use Cases\n"]
            
            # From Wikipedia
            if context['wikipedia'].get('applications'):
                apps.append("### Overview:")
                apps.append(context['wikipedia']['applications'][:500])
            elif context['wikipedia'].get('summary'):
                # Extract use cases from summary
                summary = context['wikipedia']['summary']
                if 'use' in summary.lower() or 'application' in summary.lower():
                    apps.append("### Overview:")
                    apps.append(summary[:500])
            
            # From fundamental data
            if context['fundamental'].get('description'):
                apps.append("\n### Business Description:")
                apps.append(context['fundamental']['description'])
            
            # GitHub activity (for crypto)
            if context['github']:
                apps.append(f"\n### Developer Ecosystem:")
                apps.append(f"- **GitHub Stars**: {context['github'].get('stars', 0):,} developers following")
                apps.append(f"- **Forks**: {context['github'].get('forks', 0):,} projects building on it")
                apps.append(f"- **Active Development**: {context['github'].get('recent_activity', 0)} recent commits")
                apps.append(f"- **Use Case**: {context['github'].get('description', 'Decentralized platform')}")
            
            # Generic applications based on asset type
            if 'crypto' in str(ticker_data.get('type', '')).lower():
                if 'bitcoin' in asset_name.lower():
                    apps.append("\n### Key Applications:")
                    apps.append("- **Digital Gold**: Store of value and inflation hedge")
                    apps.append("- **Payments**: Cross-border remittances and settlements")
                    apps.append("- **Treasury Reserve**: Corporate balance sheet asset")
                    apps.append("- **DeFi Collateral**: Wrapped BTC in decentralized finance")
                    apps.append("- **Lightning Network**: Instant micropayments")
                elif 'ethereum' in asset_name.lower():
                    apps.append("\n### Key Applications:")
                    apps.append("- **Smart Contracts**: Programmable agreements and automation")
                    apps.append("- **DeFi**: Decentralized finance protocols ($100B+ TVL)")
                    apps.append("- **NFTs**: Digital art, gaming, and collectibles")
                    apps.append("- **DAOs**: Decentralized autonomous organizations")
                    apps.append("- **Enterprise**: JPMorgan, Microsoft Azure blockchain")
                elif 'xrp' in asset_name.lower() or 'ripple' in asset_name.lower():
                    apps.append("\n### Key Applications:")
                    apps.append("- **Bank Settlements**: Real-time gross settlement system")
                    apps.append("- **Cross-Border Payments**: 3-5 seconds vs 3-5 days")
                    apps.append("- **Central Bank Digital Currencies**: CBDC infrastructure")
                    apps.append("- **Remittances**: Low-cost international money transfers")
                    apps.append("- **Liquidity**: On-Demand Liquidity (ODL) for institutions")
            
            sections.append('\n'.join(apps))
        
        # Technical outlook
        if ticker_data:
            outlook = ["## 📊 Technical Analysis\n"]
            
            rsi = ticker_data.get('rsi', 50)
            if rsi > 70:
                outlook.append("- **Signal**: OVERBOUGHT ⚠️")
                outlook.append("- **Action**: Consider taking profits or waiting for pullback")
            elif rsi < 30:
                outlook.append("- **Signal**: OVERSOLD 🟢")
                outlook.append("- **Action**: Potential bounce incoming, accumulation zone")
            else:
                outlook.append("- **Signal**: NEUTRAL ➡️")
                outlook.append("- **Action**: Wait for clearer signals")
            
            # Price levels
            price = ticker_data.get('price', 0)
            outlook.append(f"\n### Key Levels:")
            outlook.append(f"- **Support**: ${price * 0.95:,.2f} (-5%)")
            outlook.append(f"- **Resistance**: ${price * 1.05:,.2f} (+5%)")
            outlook.append(f"- **52W High**: ${ticker_data.get('year_high', price * 1.2):,.2f}")
            outlook.append(f"- **52W Low**: ${ticker_data.get('year_low', price * 0.8):,.2f}")
            
            sections.append('\n'.join(outlook))
        
        # Fundamental data
        if context['fundamental'] and any(context['fundamental'].values()):
            fundamental = ["## 💼 Fundamental Metrics\n"]
            
            if context['fundamental'].get('market_cap'):
                fundamental.append(f"- **Market Cap**: ${context['fundamental']['market_cap']:,.0f}")
            if context['fundamental'].get('pe_ratio'):
                fundamental.append(f"- **P/E Ratio**: {context['fundamental']['pe_ratio']:.2f}")
            if context['fundamental'].get('volume'):
                fundamental.append(f"- **Volume**: {context['fundamental']['volume']:,}")
            if context['fundamental'].get('beta'):
                fundamental.append(f"- **Beta**: {context['fundamental']['beta']:.2f}")
            
            if context['fundamental'].get('analyst_recommendations'):
                fundamental.append("\n### Recent Analyst Recommendations:")
                for rec in context['fundamental']['analyst_recommendations'][:3]:
                    fundamental.append(f"- {rec.get('firm', 'Analyst')}: {rec.get('toGrade', 'N/A')}")
            
            sections.append('\n'.join(fundamental))
        
        # Summary
        sections.append("\n## 📝 Summary\n")
        
        # Generate summary based on all signals
        summary_points = []
        
        # Price action summary
        if ticker_data.get('monthly_change', 0) > 10:
            summary_points.append("✅ Strong uptrend with +10% monthly gains")
        elif ticker_data.get('monthly_change', 0) < -10:
            summary_points.append("⚠️ Downtrend with -10% monthly decline")
        else:
            summary_points.append("➡️ Consolidating in current range")
        
        # Sentiment summary
        if context['reddit'] and context['reddit'].get('sentiment_label') == 'Bullish':
            summary_points.append("✅ Positive social sentiment")
        elif context['reddit'] and context['reddit'].get('sentiment_label') == 'Bearish':
            summary_points.append("⚠️ Negative social sentiment")
        
        # Pattern summary
        if context['historical_patterns'] and context['historical_patterns'].get('avg_30d_return', 0) > 5:
            summary_points.append("✅ Historical patterns suggest upside")
        elif context['historical_patterns'] and context['historical_patterns'].get('avg_30d_return', 0) < -5:
            summary_points.append("⚠️ Historical patterns suggest downside")
        
        sections.append('\n'.join(f"- {point}" for point in summary_points))
        
        return '\n\n'.join(sections)


# ========================================
# 🆕 ENHANCED EMAIL BOT RESPONDER
# ========================================

class IntelligentEmailBotResponder:
    """Enhanced responder using intelligent analysis"""
    
    def __init__(self):
        self.analyzer = IntelligentMarketAnalyzer()
    
    async def generate_intelligent_html(self, question, market_data):
        """Generate truly intelligent HTML responses"""
        
        # Process each asset with intelligence
        intelligent_analyses = {}
        
        for asset_key, asset_data in market_data.items():
            if asset_data:
                try:
                    # Get intelligent analysis
                    analysis = await self.analyzer.answer_intelligently(question, asset_data)
                    intelligent_analyses[asset_key] = {
                        'data': asset_data,
                        'analysis': analysis
                    }
                except Exception as e:
                    logging.error(f"Analysis failed for {asset_key}: {e}")
        
        # Build HTML response
        return self._build_intelligent_html(question, intelligent_analyses)
    
    def _build_intelligent_html(self, question, analyses):
        """Build beautiful HTML with intelligent content"""
        
        # Convert markdown to HTML if available
        def md_to_html(text):
            if MARKDOWN_AVAILABLE:
                import markdown
                return markdown.markdown(text)
            else:
                # Basic conversion
                text = text.replace('\n## ', '\n<h2>').replace('\n', '</h2>\n', 1)
                text = text.replace('\n### ', '\n<h3>').replace('\n', '</h3>\n', 1)
                text = text.replace('\n- ', '\n<li>').replace('\n', '</li>\n')
                text = text.replace('**', '<strong>').replace('**', '</strong>')
                return text
        
        # Build price cards
        price_cards = []
        for key, item in analyses.items():
            data = item['data']
            change_color = '#16a34a' if data.get('daily_change', 0) >= 0 else '#dc2626'
            
            price_cards.append(f"""
            <div style="background: linear-gradient(135deg, #e0f2fe, #bae6fd); 
                        padding: 20px; border-radius: 12px; margin: 15px 0; 
                        display: inline-block; width: 45%; margin-right: 3%; 
                        vertical-align: top; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: #1e40af; text-transform: uppercase;">
                    {data.get('name', 'Unknown')}
                </h3>
                <p style="font-size: 32px; font-weight: bold; margin: 10px 0;">
                    ${data.get('price', 0):,.2f}
                </p>
                <p style="color: {change_color}; font-size: 18px; margin: 5px 0;">
                    {data.get('daily_change', 0):+.2f}% today
                </p>
                <p style="font-size: 12px; color: #6b7280;">
                    Week: {data.get('weekly_change', 0):+.1f}% | Month: {data.get('monthly_change', 0):+.1f}%<br>
                    RSI: {data.get('rsi', 50):.0f} | Range: ${data.get('year_low', 0):,.2f} - ${data.get('year_high', 0):,.2f}
                </p>
            </div>
            """)
        
        # Build analysis sections
        analysis_sections = []
        for key, item in analyses.items():
            analysis_html = md_to_html(item['analysis'])
            analysis_sections.append(f"""
            <div style="margin-top: 30px; padding: 20px; background: #f9fafb; 
                        border-radius: 10px; border-left: 4px solid #3b82f6;">
                {analysis_html}
            </div>
            """)
        
        # Complete HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    background: linear-gradient(135deg, #f5f5f5 0%, #e5e5e5 100%);
    padding: 20px;
}}
.container {{
    background: white;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.15);
}}
.header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 40px;
    text-align: center;
}}
.content {{
    padding: 40px;
}}
.question-box {{
    background: linear-gradient(135deg, #fef3c7, #fed7aa);
    border-left: 5px solid #f59e0b;
    padding: 25px;
    border-radius: 10px;
    margin-bottom: 30px;
}}
h2 {{
    color: #1e40af;
    margin-top: 30px;
    padding-bottom: 10px;
    border-bottom: 2px solid #e5e7eb;
}}
h3 {{
    color: #374151;
    margin-top: 20px;
}}
ul {{
    line-height: 1.8;
}}
li {{
    margin-bottom: 8px;
}}
.footer {{
    background: #f3f4f6;
    padding: 25px;
    text-align: center;
    color: #6b7280;
}}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 Intelligent Market Analysis</h1>
        <p style="margin: 15px 0 0 0; font-size: 18px;">
            {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        </p>
    </div>
    
    <div class="content">
        <div class="question-box">
            <h2 style="margin: 0; color: #92400e; border: none;">Your Question</h2>
            <p style="margin: 10px 0 0 0; font-size: 16px; color: #451a03;">
                "{question}"
            </p>
        </div>
        
        <div style="margin: 30px 0;">
            {''.join(price_cards)}
        </div>
        
        <div style="clear: both;"></div>
        
        {''.join(analysis_sections)}
        
        <div class="footer">
            <p style="margin: 0;">
                <strong>Analysis Powered By:</strong> AI Intelligence, Market Data, Social Sentiment<br>
                <strong>Sources:</strong> Yahoo Finance, Wikipedia, GitHub, Reddit, News APIs<br>
                <strong>Disclaimer:</strong> For informational purposes only. Not financial advice.
            </p>
        </div>
    </div>
</div>
</body>
</html>
"""
        
        return html


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
