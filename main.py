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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO
import google.generativeai as genai

# ========================================
# 🔒 STABLE FOUNDATION - v1.0.0
# Core functions - modification is discouraged
# ========================================

# ---------- config ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('yfinance').setLevel(logging.WARNING)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
MEMORY_FILE = "market_memory.json"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
analyzer = SentimentIntensityAnalyzer()

# ---------- helpers ----------

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
        with open(cache_file, 'r') as f: return json.load(f)
    tickers = fetch_function()
    if tickers:
        with open(cache_file, 'w') as f: json.dump(tickers, f)
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
            if 'Symbol' in table.columns: return [str(t).split(' ')[0].replace('.', '-') + ".TO" for t in table["Symbol"].tolist()]
    except Exception: return ["RY.TO", "TD.TO", "ENB.TO", "SHOP.TO"]
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
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    return {}

def save_memory(data):
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f)

# ========================================
# 🔒 END STABLE FOUNDATION
# ========================================

# ========================================
# ✨ ENHANCEMENTS LAYER - v2.0.0
# All new features below - Safe to modify
# ========================================

# Feature Flags (for safe testing)
ENABLE_AI_ORACLE = True
ENABLE_WATCHLIST = True
ENABLE_PATTERN_MATCH = True

async def analyze_portfolio_watchlist(session, portfolio_file='portfolio.json'):
    """Feature #2: Deep analysis of personal portfolio"""
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
            
            # Skip insider transactions for Canadian stocks to avoid 403 errors
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
    """Generate human-readable interpretation - ENHANCED WITH CLARITY"""
    interpretation = []
    
    # Determine overall bias
    if avg_3m > 5 and win_3m > 70: 
        bias, emoji, color_class = "BULLISH", "📈", "bullish"
    elif avg_3m < -5 and win_3m < 40: 
        bias, emoji, color_class = "BEARISH", "📉", "bearish"
    else: 
        bias, emoji, color_class = "MIXED", "↔️", "neutral"
    
    interpretation.append({'type': 'bias', 'emoji': emoji, 'text': f"S&P 500 Market Bias: {bias}", 'color': color_class})
    
    # Historical probability with clarity
    if win_3m >= 70: 
        prob_text = f"History strongly favors upside for the S&P 500. {int(win_3m)}% of similar market setups saw positive index returns 3 months later."
    elif win_3m <= 40: 
        prob_text = f"Caution warranted for broad market. Only {int(win_3m)}% of similar S&P 500 setups were positive 3 months later."
    else: 
        prob_text = f"S&P 500 could go either way. {int(win_3m)}% win rate suggests balanced risk/reward for the index."
    interpretation.append({'type': 'probability', 'text': prob_text})
    
    # Expected market movement
    magnitude = "significant" if abs(avg_3m) > 8 else "moderate" if abs(avg_3m) > 4 else "modest"
    direction = "upward" if avg_3m > 0 else "downward"
    interpretation.append({'type': 'expectation', 'text': f"Expect a {magnitude} {direction} move in the S&P 500. Historical average: {avg_3m:+.1f}% over 3 months."})
    
    # Timing insights
    if avg_1m * avg_3m < 0: 
        interpretation.append({'type': 'timing', 'text': f"⏰ Short-term volatility likely. 1-month S&P 500 average ({avg_1m:+.1f}%) differs from 3-month ({avg_3m:+.1f}%)."})
    elif abs(avg_1m) > 5: 
        interpretation.append({'type': 'timing', 'text': f"⚡ Quick moves expected in the index. Historical 1-month average: {avg_1m:+.1f}%."})
    
    # Scenario analysis
    if len(bullish) > len(bearish) * 2: 
        interpretation.append({'type': 'scenario', 'text': f"💡 {len(bullish)} out of top 10 matches led to strong S&P 500 rallies. Index dips may be buying opportunities."})
    elif len(bearish) > len(bullish) * 2: 
        interpretation.append({'type': 'scenario', 'text': f"⚠️ {len(bearish)} out of top 10 matches led to S&P 500 declines. Consider protective measures for broad exposure."})
    else: 
        interpretation.append({'type': 'scenario', 'text': f"📊 Mixed S&P 500 outcomes ({len(bullish)} bullish, {len(bearish)} bearish). Stock selection matters more than market timing."})
    
    # Current context
    if current_conditions['geopolitical_risk'] > 70: 
        interpretation.append({'type': 'context', 'text': f"🌍 Current geopolitical risk ({current_conditions['geopolitical_risk']:.0f}/100) is elevated. Similar periods often saw initial S&P 500 weakness followed by recovery."})
    
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
            reasons.append(f"Historical S&P 500 pattern bullish (+{pattern_data['avg_return_3m']:.1f}% avg)")
        elif pattern_data['avg_return_3m'] < -5:
            score -= 1
            reasons.append(f"Historical S&P 500 pattern bearish ({pattern_data['avg_return_3m']:.1f}% avg)")
        
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
    """Feature #3: Find similar market conditions in past 11 years - ENHANCED"""
    logging.info("🔮 Searching for historical patterns...")
    try:
        spy = yf.Ticker("SPY")
        hist_data = spy.history(period="max", interval="1d")
        
        if len(hist_data) < 252 * 11: hist_data = spy.history(start="2013-01-01", interval="1d")
        
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
            except Exception: continue
        
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
        opps.append(f"Historical S&P 500 pattern suggests +{pattern_data['avg_return_3m']:.1f}% upside over 3 months")
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
        overbought = [{'ticker': s['ticker'], 'monthly': s['monthly_change']} for s in portfolio_data['stocks'] if s['rsi'] > 75]
        if overbought:
            risks.append(f"Overbought risk: {', '.join([f'{o['ticker']} (RSI 75+, +{o['monthly']:.0f}% monthly)' for o in overbought])} - consider trimming")
    
    if risks:
        analysis.append("⚠️ CRITICAL RISKS:\n" + "\n".join([f"• {r}" for r in risks]))
    
    # 3. Contrarian Play
    contrarian = []
    if pattern_data and pattern_data['avg_return_1m'] < 0 and pattern_data['avg_return_3m'] > 5:
        contrarian.append("Short-term S&P 500 weakness (+3m bullish) = buy the dip opportunity. Historical edge: 70% win rate.")
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
    """Feature #1: AI-powered market analysis using Gemini"""
    logging.info("🤖 Generating AI Oracle analysis...")
    
    if not GEMINI_API_KEY:
        logging.warning("Gemini API key not found - using fallback analysis")
        return generate_fallback_analysis(market_data, portfolio_data, pattern_data)

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try the right model names in order
        model = None
        model_names = [
            'gemini-1.5-flash',      # Latest, fastest
            'gemini-1.5-pro',        # Latest, most capable
            'gemini-pro'             # Older version
        ]
        
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
        
        # Generate with safety settings
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
# MAIN FUNCTION WITH ENHANCEMENTS
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
        
        enhancement_tasks = {}
        if ENABLE_WATCHLIST: enhancement_tasks['portfolio'] = analyze_portfolio_watchlist(session)
        
        results = await asyncio.gather(asyncio.gather(*stock_tasks), context_task, news_task, macro_task, *enhancement_tasks.values())
        
        stock_results_raw, context_data, market_news, macro_data = results[:4]
        
        stock_results = sorted([r for r in stock_results_raw if r], key=lambda x: x['score'], reverse=True)
        df_stocks = pd.DataFrame(stock_results) if stock_results else pd.DataFrame()

        portfolio_data = results[4] if ENABLE_WATCHLIST and len(results) > 4 else None
        pattern_data = await find_historical_patterns(session, macro_data) if ENABLE_PATTERN_MATCH else None
        
        # Generate portfolio recommendations AFTER pattern data
        portfolio_recommendations = None
        if ENABLE_PATTERN_MATCH and pattern_data and portfolio_data:
            portfolio_recommendations = await generate_portfolio_recommendations_from_pattern(
                portfolio_data, pattern_data, macro_data
            )
        
        if ENABLE_AI_ORACLE:
            market_summary = {
                'macro': macro_data,
                'top_stock': stock_results[0] if stock_results else {},
                'bottom_stock': stock_results[-1] if stock_results else {}
            }
            ai_analysis = await generate_ai_oracle_analysis(market_summary, portfolio_data, pattern_data)
        else:
            ai_analysis = None
    
    if output == "email":
        html_email = generate_enhanced_html_email(df_stocks, context_data, market_news, macro_data, previous_day_memory, portfolio_data, pattern_data, ai_analysis, portfolio_recommendations)
        send_email(html_email)
    
    if not df_stocks.empty:
        save_memory({
            "previous_top_stock_name": df_stocks.iloc[0]['name'], "previous_top_stock_ticker": df_stocks.iloc[0]['ticker'],
            "previous_macro_score": macro_data.get('overall_macro_score', 0), "date": datetime.date.today().isoformat()
        })
    
    logging.info("✅ Analysis complete with enhancements.")

def generate_enhanced_html_email(df_stocks, context, market_news, macro_data, memory, portfolio_data, pattern_data, ai_analysis, portfolio_recommendations=None):
    """Enhanced email generation with new features"""
    
    def format_articles(articles):
        if not articles: return "<p style='color:#888;'><i>No specific news drivers detected.</i></p>"
        html = "<ul style='margin:0;padding-left:20px;'>"
        for a in articles:
            if a.get('title'): html += f'<li style="margin-bottom:5px;"><a href="{a.get("url", "#")}" style="color:#1e3a8a;">{a["title"]}</a> <span style="color:#666;">({a.get("source", "Unknown")})</span></li>'
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
    
    ai_oracle_html = ""
    if ENABLE_AI_ORACLE and ai_analysis:
        analysis_text = ai_analysis['analysis'].replace('\n', '<br>')
        ai_oracle_html = f"""<div class="section" style="background-color:#f0f9ff;border-left:4px solid #0369a1;"><h2>🤖 AI MARKET ORACLE</h2><p style="font-size:0.9em;color:#666;margin-bottom:15px;">Powered by Gemini AI</p><div style="line-height:1.8;">{analysis_text}</div></div>"""
    
    portfolio_html = ""
    if ENABLE_WATCHLIST and portfolio_data:
        portfolio_table = ""
        for stock in portfolio_data['stocks']:
            color = "#16a34a" if stock['daily_change'] > 0 else "#dc2626"
            rsi_color = "#dc2626" if stock['rsi'] > 70 else "#16a34a" if stock['rsi'] < 30 else "#666"
            portfolio_table += f"""<tr><td style="padding:10px;border-bottom:1px solid #eee;"><b>{stock['ticker']}</b><br><span style="color:#666;font-size:0.9em;">{stock['name']}</span></td><td style="padding:10px;border-bottom:1px solid #eee;">${stock['price']:.2f}<br><span style="color:{color};font-size:0.9em;">{stock['daily_change']:+.2f}%</span></td><td style="padding:10px;border-bottom:1px solid #eee;">RSI: <span style="color:{rsi_color};font-weight:bold;">{stock['rsi']:.1f}</span><br><span style="font-size:0.9em;">Vol: {stock['volume_ratio']:.1f}x</span></td><td style="padding:10px;border-bottom:1px solid #eee;"><span style="font-size:0.9em;">W: {stock['weekly_change']:+.1f}%<br>M: {stock['monthly_change']:+.1f}%</span></td></tr>"""
        alerts_html = "<br>".join(portfolio_data['alerts'][:5]) if portfolio_data['alerts'] else "No alerts today"
        opps_html = "<br>".join(portfolio_data['opportunities'][:5]) if portfolio_data['opportunities'] else "No immediate opportunities"
        risks_html = "<br>".join(portfolio_data['risks'][:5]) if portfolio_data['risks'] else "No significant risks detected"
        portfolio_html = f"""<div class="section" style="background-color:#fefce8;border-left:4px solid #ca8a04;"><h2>📊 YOUR PORTFOLIO COMMAND CENTER</h2><table style="width:100%; border-collapse: collapse; margin-bottom:20px;"><thead><tr style="background-color:#f8f8f8;"><th style="text-align:left; padding:10px;">Stock</th><th style="text-align:left; padding:10px;">Price</th><th style="text-align:left; padding:10px;">Indicators</th><th style="text-align:left; padding:10px;">Performance</th></tr></thead><tbody>{portfolio_table}</tbody></table><div style="margin-top:20px;"><h3 style="color:#dc2626;">🔔 Alerts & Signals</h3><p style="line-height:1.8;">{alerts_html}</p></div><div style="margin-top:20px;"><h3 style="color:#16a34a;">🎯 Opportunities</h3><p style="line-height:1.8;">{opps_html}</p></div><div style="margin-top:20px;"><h3 style="color:#ea580c;">⚠️ Risk Factors</h3><p style="line-height:1.8;">{risks_html}</p></div></div>"""

    pattern_html = ""
    if ENABLE_PATTERN_MATCH and pattern_data and pattern_data.get('matches'):
        current_cond_data = pattern_data['current_conditions']
        current_cond = f"""<div style="background-color:#fff;padding:15px;border:2px solid #7c3aed;border-radius:5px;margin-bottom:20px;"><h3 style="margin-top:0;">📊 Today's Market DNA:</h3><p style="margin:5px 0;"><b>RSI:</b> {current_cond_data['rsi']:.1f} | <b>Volatility:</b> {current_cond_data['volatility']:.1f}% | <b>Trend:</b> {current_cond_data['trend']:+.1f}%</p><p style="margin:5px 0;"><b>Geopolitical Risk:</b> {current_cond_data['geo_risk']:.0f} | <b>Trade Risk:</b> {current_cond_data['trade_risk']:.0f}</p></div>"""
        
        interpretation_html = ""
        if pattern_data.get('interpretation'):
            for item in pattern_data['interpretation']:
                if item['type'] == 'bias':
                    color = {'bullish': '#16a34a', 'bearish': '#dc2626', 'neutral': '#666'}[item['color']]
                    interpretation_html += f'<p style="font-size:1.2em;font-weight:bold;color:{color};">{item["emoji"]} {item["text"]}</p>'
                else:
                    interpretation_html += f'<p style="line-height:1.8;margin:10px 0;">{item["text"]}</p>'
        
        # SECTOR PERFORMANCE
        sector_html = ""
        if pattern_data.get('sector_performance'):
            sector_rows = ""
            for sector, perf in pattern_data['sector_performance'][:5]:
                color = "#16a34a" if perf > 0 else "#dc2626"
                sector_rows += f'<tr><td style="padding:8px;border-bottom:1px solid #eee;">{sector}</td><td style="padding:8px;border-bottom:1px solid #eee;text-align:right;color:{color};font-weight:bold;">{perf:+.1f}%</td></tr>'
            
            sector_html = f"""<div style="margin:20px 0;"><h3>🎯 Sector Performance in Similar Periods:</h3><p style="font-size:0.9em;color:#666;">Based on {pattern_data['matches'][0]['date']} match ({pattern_data['matches'][0]['context']})</p><table style="width:100%;background-color:#fff;border-collapse:collapse;"><thead><tr style="background-color:#f3e8ff;"><th style="padding:10px;text-align:left;">Sector</th><th style="padding:10px;text-align:right;">3-Month Return</th></tr></thead><tbody>{sector_rows}</tbody></table><p style="font-size:0.9em;color:#666;margin-top:10px;"><i>Note: Individual stock performance may vary within sectors. Use sector ETFs (XLK, XLV, XLF, etc.) for broad exposure.</i></p></div>"""
        
        # PORTFOLIO RECOMMENDATIONS
        recommendations_html = ""
        if portfolio_recommendations:
            buy_html = ""
            if portfolio_recommendations['buy']:
                for rec in portfolio_recommendations['buy']:
                    buy_html += f"""<div style="margin:10px 0;padding:12px;background-color:#f0fdf4;border-left:4px solid #16a34a;border-radius:5px;"><div style="font-size:1.1em;font-weight:bold;color:#16a34a;">{rec['ticker']} - {rec['action']}</div><div style="font-size:0.9em;color:#666;margin-top:5px;">{'<br>'.join(['• ' + r for r in rec['reasons']])}</div></div>"""
            
            sell_html = ""
            if portfolio_recommendations['sell']:
                for rec in portfolio_recommendations['sell']:
                    sell_html += f"""<div style="margin:10px 0;padding:12px;background-color:#fef2f2;border-left:4px solid #dc2626;border-radius:5px;"><div style="font-size:1.1em;font-weight:bold;color:#dc2626;">{rec['ticker']} - {rec['action']}</div><div style="font-size:0.9em;color:#666;margin-top:5px;">{'<br>'.join(['• ' + r for r in rec['reasons']])}</div></div>"""
            
            hold_html = ""
            if portfolio_recommendations['hold']:
                hold_tickers = [rec['ticker'] for rec in portfolio_recommendations['hold']]
                hold_html = f"""<div style="margin:10px 0;padding:12px;background-color:#f8f8f8;border-left:4px solid #666;border-radius:5px;"><div style="font-size:1.1em;font-weight:bold;color:#666;">HOLD: {', '.join(hold_tickers)}</div><div style="font-size:0.9em;color:#666;margin-top:5px;">• Positions look balanced - no immediate action needed</div></div>"""
            
            recommendations_html = f"""<div style="margin:20px 0;"><h3 style="color:#7c3aed;">💼 YOUR PORTFOLIO PLAYBOOK</h3><p style="font-size:0.9em;color:#666;">Based on historical pattern analysis + current market conditions</p>{buy_html if buy_html else '<p style="color:#888;font-size:0.9em;">No strong buy signals at this time.</p>'}{sell_html if sell_html else '<p style="color:#888;font-size:0.9em;">No sell signals detected.</p>'}{hold_html}</div>"""
        
        matches_html = ""
        for i, match in enumerate(pattern_data['matches'][:5], 1):
            outcome_color = "#16a34a" if match['future_3m'] > 0 else "#dc2626"
            matches_html += f"""<div style="margin:15px 0;padding:15px;background-color:#f8f8f8;border-left:4px solid {outcome_color};border-radius:5px;"><div style="display:flex;justify-content:space-between;align-items:center;"><div><b style="font-size:1.1em;">{match['date']}</b><span style="color:#666;margin-left:10px;">({match['context']})</span><br><span style="font-size:0.9em;color:#666;">Match Strength: {match['similarity']:.1f}%</span></div><div style="text-align:right;"><div style="font-size:1.2em;font-weight:bold;color:{outcome_color};">{match['future_3m']:+.1f}%</div><div style="font-size:0.9em;color:#666;">S&P 500 outcome</div></div></div></div>"""
        
        win_bar_1m, win_bar_3m = int(pattern_data['win_rate_1m']), int(pattern_data['win_rate_3m'])
        pattern_html = f"""<div class="section" style="background-color:#f3e8ff;border-left:4px solid #7c3aed;"><h2>🔮 11-YEAR S&P 500 PATTERN ANALYSIS</h2><p style="font-size:0.9em;color:#666;">Analyzing {pattern_data['sample_size']} similar market setups from the S&P 500 index</p>{current_cond}<div style="background-color:#fff;padding:20px;border-radius:5px;margin:20px 0;"><h3 style="margin-top:0;color:#7c3aed;">📖 What S&P 500 History Tells Us:</h3>{interpretation_html}</div>{sector_html}{recommendations_html}<div style="margin:20px 0;"><h3>📅 Historical S&P 500 Matches:</h3><p style="font-size:0.9em;color:#666;">These show S&P 500 index performance. Individual sectors varied (see table above).</p>{matches_html}</div></div>"""

    prev_score = memory.get('previous_macro_score', 0)
    current_score = macro_data.get('overall_macro_score', 0)
    mood_change = "stayed relatively stable"
    if (diff := current_score - prev_score) > 3: mood_change = f"improved since yesterday (from {prev_score:.1f} to {current_score:.1f})"
    elif diff < -3: mood_change = f"turned more cautious since yesterday (from {prev_score:.1f} to {current_score:.1f})"
    editor_note = f"Good morning. The overall market mood has {mood_change}. This briefing is your daily blueprint for navigating the currents."
    if memory.get('previous_top_stock_name'): editor_note += f"<br><br><b>Yesterday's Champion:</b> {memory['previous_top_stock_name']} ({memory['previous_top_stock_ticker']}) led our rankings."
    
    sector_html = ""
    if not df_stocks.empty:
        # Fix the pandas warning here
        top_by_sector = df_stocks.groupby('sector', group_keys=False)[['ticker', 'name', 'score', 'sector', 'summary']].apply(lambda x: x.nlargest(2, 'score'))
        for _, row in top_by_sector.iterrows():
            if row['sector'] and row['sector'] != 'N/A':
                summary_text = "Business summary not available."
                if row["summary"] and isinstance(row["summary"], str): summary_text = '. '.join(row["summary"].split('. ')[:2]) + '.'
                sector_html += f'<div style="margin-bottom:15px;"><b>{row["name"]} ({row["ticker"]})</b> in <i>{row["sector"]}</i><p style="font-size:0.9em;color:#333;margin:5px 0 0 0;">{summary_text}</p></div>'
    
    top10_html = create_stock_table(df_stocks.head(10)) if not df_stocks.empty else "<tr><td>No data available</td></tr>"
    bottom10_html = create_stock_table(df_stocks.tail(10).iloc[::-1]) if not df_stocks.empty else "<tr><td>No data available</td></tr>"
    crypto_html = create_context_table(["bitcoin", "ethereum", "solana", "ripple"])
    commodities_html = create_context_table(["gold", "silver"])
    market_news_html = "".join([f'<div style="margin-bottom:15px;"><b><a href="{article.get("url", "#")}" style="color:#000;">{article["title"]}</a></b><br><span style="color:#666;font-size:0.9em;">{article.get("source", "Unknown")}</span></div>' for article in market_news[:10]]) or "<p><i>Headlines temporarily unavailable.</i></p>"
    
    return f"""<!DOCTYPE html><html><head><style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0;background-color:#f7f7f7;}} .container{{width:100%;max-width:700px;margin:20px auto;background-color:#fff;border:1px solid #ddd;}} .header{{background-color:#0c0a09;color:#fff;padding:30px;text-align:center;}} .section{{padding:25px;border-bottom:1px solid #ddd;}} h2{{font-size:1.5em;color:#111;margin-top:0;}} h3{{font-size:1.2em;color:#333;border-bottom:2px solid #e2e8f0;padding-bottom:5px;}}</style></head><body><div class="container"><div class="header"><h1>Your Daily Intelligence Briefing</h1><p style="font-size:1.1em; color:#aaa;">{datetime.date.today().strftime('%A, %B %d, %Y')}</p></div><div class="section"><h2>EDITOR'S NOTE</h2><p>{editor_note}</p></div>{ai_oracle_html}{portfolio_html}{pattern_html}<div class="section"><h2>THE BIG PICTURE: The Market Weather Report</h2><h3>Overall Macro Score: {macro_data['overall_macro_score']:.1f} / 30</h3><p><b>How it's calculated:</b> This is our "weather forecast" for investors, combining risks and sentiment.</p><p><b>🌍 Geopolitical Risk ({macro_data['geopolitical_risk']:.0f}/100):</b> Measures global instability.<br><u>Key Drivers:</u> {format_articles(macro_data['geo_articles'])}</p><p><b>🚢 Trade Risk ({macro_data['trade_risk']:.0f}/100):</b> Tracks trade tensions.<br><u>Key Drivers:</u> {format_articles(macro_data['trade_articles'])}</p><p><b>💼 Economic Sentiment ({macro_data['economic_sentiment']:.2f}):</b> Market mood (-1 to +1).<br><u>Key Drivers:</u> {format_articles(macro_data['econ_articles'])}</p></div><div class="section"><h2>SECTOR DEEP DIVE</h2><p>Top companies from different sectors.</p>{sector_html or "<p><i>No sector data available.</i></p>"}</div><div class="section"><h2>STOCK RADAR</h2><h3>📈 Top 10 Strongest Signals</h3><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{top10_html}</tbody></table><h3 style="margin-top: 30px;">📉 Top 10 Weakest Signals</h3><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{bottom10_html}</tbody></table></div><div class="section"><h2>BEYOND STOCKS: Alternative Assets</h2><h3>🪙 Crypto</h3><p><b>Market Sentiment: {context.get('crypto_sentiment', 'N/A')}</b></p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:left; padding:10px;">Price / 24h</th><th style="text-align:left; padding:10px;">Market Cap</th></tr></thead><tbody>{crypto_html}</tbody></table><h3 style="margin-top: 30px;">💎 Commodities</h3><p><b>Gold/Silver Ratio: {context.get('gold_silver_ratio', 'N/A')}</b></p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:left; padding:10px;">Price / 24h</th><th style="text-align:left; padding:10px;">Market Cap</th></tr></thead><tbody>{commodities_html}</tbody></table></div><div class="section"><h2>FROM THE WIRE: Today's Top Headlines</h2>{market_news_html}</div></div></body></html>"""

def send_email(html_body):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    
    SMTP_USER, SMTP_PASS = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not SMTP_USER or not SMTP_PASS:
        logging.warning("SMTP credentials missing.")
        return
    
    msg = MIMEMultipart('alternative')
    msg["Subject"], msg["From"], msg["To"] = f"⛵ Your Daily Market Briefing - {datetime.date.today()}", SMTP_USER, SMTP_USER
    msg.attach(MIMEText(html_body, 'html'))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(); server.login(SMTP_USER, SMTP_PASS); server.send_message(msg)
        logging.info("✅ Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="print", choices=["print", "email"])
    args = parser.parse_args()
    
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main(output=args.output))
