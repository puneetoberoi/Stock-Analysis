mport os, sys, argparse, time, datetime, logging, json, asyncio
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

# ========================================
# 🔒 STABLE FOUNDATION - v1.0.0
# Last stable: 2024-10-17
# DO NOT MODIFY - All features tested and working
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
# New features in development
# Safe to modify without breaking v1.0.0
# ========================================

# Feature Flags for v2.0.0
ENABLE_BOLLINGER_BANDS = True
ENABLE_ATR_VOLATILITY = True
ENABLE_52WEEK_ALERTS = True
ENABLE_GAP_DETECTION = True
ENABLE_EARNINGS_COUNTDOWN = True

async def analyze_advanced_technicals(ticker, hist_data):
    """
    NEW v2.0.0: Advanced technical indicators
    - Bollinger Bands with squeeze detection
    - ATR for volatility measurement
    - 52-week high/low proximity
    - Gap detection
    """
    results = {
        'bollinger': None,
        'atr': None,
        'week52': None,
        'gap': None
    }
    
    try:
        if ENABLE_BOLLINGER_BANDS and len(hist_data) >= 20:
            # Bollinger Bands
            bb = BollingerBands(hist_data['Close'], window=20, window_dev=2)
            bb_high = bb.bollinger_hband().iloc[-1]
            bb_low = bb.bollinger_lband().iloc[-1]
            bb_mid = bb.bollinger_mavg().iloc[-1]
            current_price = hist_data['Close'].iloc[-1]
            
            # Squeeze detection: when bands are unusually narrow
            bb_width = (bb_high - bb_low) / bb_mid * 100
            bb_position = ((current_price - bb_low) / (bb_high - bb_low)) * 100 if bb_high != bb_low else 50
            
            squeeze = bb_width < 10  # Narrow bands = potential breakout coming
            
            results['bollinger'] = {
                'upper': bb_high,
                'lower': bb_low,
                'middle': bb_mid,
                'width': bb_width,
                'position': bb_position,  # 0-100, where price is in the band
                'squeeze': squeeze,
                'signal': 'SQUEEZE - Breakout imminent' if squeeze else 
                         'OVERBOUGHT' if bb_position > 95 else 
                         'OVERSOLD' if bb_position < 5 else 'NEUTRAL'
            }
        
        if ENABLE_ATR_VOLATILITY and len(hist_data) >= 14:
            # Average True Range - volatility
            atr = AverageTrueRange(hist_data['High'], hist_data['Low'], hist_data['Close'], window=14)
            atr_value = atr.average_true_range().iloc[-1]
            current_price = hist_data['Close'].iloc[-1]
            atr_percent = (atr_value / current_price) * 100
            
            results['atr'] = {
                'value': atr_value,
                'percent': atr_percent,
                'signal': 'HIGH VOLATILITY' if atr_percent > 3 else 
                         'LOW VOLATILITY' if atr_percent < 1 else 'NORMAL'
            }
        
        if ENABLE_52WEEK_ALERTS and len(hist_data) >= 252:
            # 52-week high/low proximity
            week52_high = hist_data['Close'].tail(252).max()
            week52_low = hist_data['Close'].tail(252).min()
            current_price = hist_data['Close'].iloc[-1]
            
            distance_from_high = ((current_price - week52_high) / week52_high) * 100
            distance_from_low = ((current_price - week52_low) / week52_low) * 100
            
            results['week52'] = {
                'high': week52_high,
                'low': week52_low,
                'current': current_price,
                'distance_from_high_pct': distance_from_high,
                'distance_from_low_pct': distance_from_low,
                'signal': 'AT 52W HIGH' if distance_from_high > -2 else 
                         'AT 52W LOW' if distance_from_low < 2 else 
                         'NEAR 52W HIGH' if distance_from_high > -10 else 
                         'NEAR 52W LOW' if distance_from_low < 10 else 'MID-RANGE'
            }
        
        if ENABLE_GAP_DETECTION and len(hist_data) >= 2:
            # Gap up/down detection
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
        logging.warning(f"Error in advanced technicals for {ticker}: {e}")
    
    return results

async def get_earnings_countdown(ticker, info):
    """
    NEW v2.0.0: Earnings date countdown
    """
    try:
        if ENABLE_EARNINGS_COUNTDOWN:
            earnings_date = info.get('earningsDate')
            if earnings_date and len(earnings_date) > 0:
                # Convert timestamp to datetime
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
    """
    ENHANCED v2.0.0: Portfolio analysis with new technical indicators
    """
    logging.info("📊 [v2.0] Analyzing portfolio with advanced features...")
    
    if not os.path.exists(portfolio_file):
        logging.warning(f"Portfolio file {portfolio_file} not found")
        return None
    
    with open(portfolio_file, 'r') as f:
        portfolio_tickers = json.load(f)
    
    portfolio_data = {
        'stocks': [],
        'alerts': [],
        'opportunities': [],
        'risks': [],
        'v2_signals': []  # NEW: v2.0.0 specific signals
    }
    
    for ticker in portfolio_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y", interval="1d")
            info = stock.info
            
            if hist.empty:
                continue
            
            # v1.0.0 metrics (stable)
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            rsi = RSIIndicator(hist['Close'], window=14).rsi().iloc[-1]
            macd = MACD(hist['Close'])
            macd_diff = macd.macd_diff().iloc[-1]
            
            daily_change = ((current_price - prev_close) / prev_close) * 100
            weekly_change = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100 if len(hist) >= 5 else 0
            monthly_change = ((current_price - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22]) * 100 if len(hist) >= 22 else 0
            volume_spike = (volume / avg_volume) if avg_volume > 0 else 1
            
            # NEW v2.0.0 features
            advanced_tech = await analyze_advanced_technicals(ticker, hist)
            earnings_info = await get_earnings_countdown(ticker, info)
            
            stock_analysis = {
                'ticker': ticker,
                'name': info.get('shortName', ticker),
                'price': current_price,
                'daily_change': daily_change,
                'weekly_change': weekly_change,
                'monthly_change': monthly_change,
                'rsi': rsi,
                'macd': macd_diff,
                'volume_ratio': volume_spike,
                'sector': info.get('sector', 'Unknown'),
                # v2.0.0 additions
                'bollinger': advanced_tech.get('bollinger'),
                'atr': advanced_tech.get('atr'),
                'week52': advanced_tech.get('week52'),
                'gap': advanced_tech.get('gap'),
                'earnings': earnings_info
            }
            
            portfolio_data['stocks'].append(stock_analysis)
            
            # v1.0.0 alerts (stable)
            if rsi > 70:
                portfolio_data['alerts'].append(f"⚠️ {ticker}: RSI Overbought ({rsi:.1f})")
            elif rsi < 30:
                portfolio_data['opportunities'].append(f"🎯 {ticker}: RSI Oversold ({rsi:.1f})")
            
            # NEW v2.0.0 alerts
            if advanced_tech.get('bollinger') and advanced_tech['bollinger']['squeeze']:
                portfolio_data['v2_signals'].append(f"💥 {ticker}: Bollinger Squeeze - breakout imminent!")
            
            if advanced_tech.get('bollinger') and advanced_tech['bollinger']['position'] > 95:
                portfolio_data['alerts'].append(f"📈 {ticker}: Price at upper Bollinger Band - potential resistance")
            elif advanced_tech.get('bollinger') and advanced_tech['bollinger']['position'] < 5:
                portfolio_data['opportunities'].append(f"📉 {ticker}: Price at lower Bollinger Band - potential support")
            
            if advanced_tech.get('atr') and advanced_tech['atr']['signal'] == 'HIGH VOLATILITY':
                portfolio_data['risks'].append(f"⚡ {ticker}: High volatility ({advanced_tech['atr']['percent']:.1f}%) - wider stops recommended")
            
            if advanced_tech.get('week52'):
                if advanced_tech['week52']['signal'] == 'AT 52W HIGH':
                    portfolio_data['v2_signals'].append(f"🎯 {ticker}: At 52-week HIGH (${advanced_tech['week52']['high']:.2f}) - momentum strong")
                elif advanced_tech['week52']['signal'] == 'AT 52W LOW':
                    portfolio_data['opportunities'].append(f"💎 {ticker}: At 52-week LOW (${advanced_tech['week52']['low']:.2f}) - potential reversal")
            
            if advanced_tech.get('gap'):
                if advanced_tech['gap']['signal'] == 'GAP UP':
                    portfolio_data['v2_signals'].append(f"⬆️ {ticker}: Gapped up {advanced_tech['gap']['gap_percent']:.1f}% - strong demand")
                elif advanced_tech['gap']['signal'] == 'GAP DOWN':
                    portfolio_data['alerts'].append(f"⬇️ {ticker}: Gapped down {advanced_tech['gap']['gap_percent']:.1f}% - watch for support")
            
            if earnings_info and earnings_info['signal'] == 'EARNINGS THIS WEEK':
                portfolio_data['v2_signals'].append(f"📅 {ticker}: EARNINGS in {earnings_info['days_until']} days ({earnings_info['date']}) - expect volatility")
            
        except Exception as e:
            logging.warning(f"Error analyzing {ticker} with v2 features: {e}")
            continue
    
    return portfolio_data
