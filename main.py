import os, sys, argparse, time, datetime, logging, json, asyncio
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import aiohttp
from bs4 import BeautifulSoup
from asyncio_throttle import Throttler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO

# ---------- config ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('yfinance').setLevel(logging.WARNING)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
MEMORY_FILE = "market_memory.json"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")
analyzer = SentimentIntensityAnalyzer()

# ---------- helpers ----------

async def make_robust_request(session, url, params=None, retries=3, delay=2, timeout=20):
    """Enhanced request with better error logging"""
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=REQUEST_HEADERS, timeout=timeout) as response:
                text = await response.text()
                if response.status == 200:
                    return text
                elif response.status == 426:  # NewsAPI specific error
                    logging.error(f"NewsAPI requires paid plan for this endpoint")
                    return None
                elif response.status == 429:
                    logging.warning(f"Rate limit hit, waiting longer...")
                    await asyncio.sleep(delay * 3)
                else:
                    logging.warning(f"Request failed with status {response.status}: {text[:200]}")
        except asyncio.TimeoutError:
            logging.warning(f"Request timeout for {url}")
        except Exception as e:
            logging.warning(f"Request attempt {attempt + 1} failed: {str(e)[:100]}")
        
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
            if 'Symbol' in table.columns: 
                return [str(t).split(' ')[0].replace('.', '-') + ".TO" for t in table["Symbol"].tolist()]
    except Exception: 
        pass
    return ["RY.TO", "TD.TO", "ENB.TO", "SHOP.TO", "CNR.TO"]

async def fetch_finviz_news_throttled(throttler, session, ticker):
    async with throttler:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        content = await make_robust_request(session, url)
        if not content: return []
        soup = BeautifulSoup(content, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return []
        return [{"title": row.a.text, "url": row.a['href']} for row in news_table.find_all('tr')[:5] if row.a]

async def fetch_market_headlines():
    """Completely rewritten news fetching with multiple fallbacks"""
    logging.info("Fetching market headlines...")
    all_headlines = []
    
    async with aiohttp.ClientSession() as session:
        # Priority 1: NewsAPI Top Headlines (most reliable)
        if NEWSAPI_KEY:
            try:
                # Try business category first
                url = "https://newsapi.org/v2/top-headlines"
                params = {
                    'category': 'business',
                    'country': 'us',
                    'apiKey': NEWSAPI_KEY,
                    'pageSize': 10
                }
                
                content = await make_robust_request(session, url, params=params)
                if content:
                    data = json.loads(content)
                    if data.get('status') == 'ok' and data.get('articles'):
                        for article in data['articles'][:5]:
                            if article.get('title') and article['title'] != '[Removed]':
                                all_headlines.append({
                                    "title": article['title'],
                                    "url": article.get('url', '#'),
                                    "source": article.get('source', {}).get('name', 'NewsAPI')
                                })
                        logging.info(f"✅ Got {len(all_headlines)} headlines from NewsAPI")
            except Exception as e:
                logging.error(f"NewsAPI headlines failed: {e}")
        
        # Priority 2: Finnhub Market News
        if FINNHUB_KEY and len(all_headlines) < 5:
            try:
                url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
                content = await make_robust_request(session, url)
                if content:
                    news = json.loads(content)
                    for article in news[:10]:
                        if len(all_headlines) >= 10:
                            break
                        headline = article.get('headline', '')
                        if headline:
                            all_headlines.append({
                                "title": headline,
                                "url": article.get('url', '#'),
                                "source": article.get('source', 'Finnhub')
                            })
                    logging.info(f"✅ Added Finnhub headlines, total: {len(all_headlines)}")
            except Exception as e:
                logging.error(f"Finnhub news failed: {e}")
        
        # Priority 3: Direct RSS/JSON feeds (no scraping needed)
        if len(all_headlines) < 5:
            try:
                # Try CNN Money RSS converted to JSON
                cnn_url = "https://api.rss2json.com/v1/api.json?rss_url=http://rss.cnn.com/rss/money_markets.rss"
                content = await make_robust_request(session, cnn_url)
                if content:
                    data = json.loads(content)
                    for item in data.get('items', [])[:5]:
                        if len(all_headlines) >= 10:
                            break
                        all_headlines.append({
                            "title": item.get('title', ''),
                            "url": item.get('link', '#'),
                            "source": "CNN Money"
                        })
                    logging.info(f"✅ Added CNN headlines, total: {len(all_headlines)}")
            except Exception as e:
                logging.warning(f"CNN RSS failed: {e}")
    
    # Remove duplicates
    seen = set()
    unique_headlines = []
    for h in all_headlines:
        if h['title'] and h['title'] not in seen:
            seen.add(h['title'])
            unique_headlines.append(h)
    
    if not unique_headlines:
        logging.error("❌ All headline sources failed!")
        return [{"title": "Unable to fetch headlines - check API keys", "url": "#", "source": "System"}]
    
    logging.info(f"✅ Returning {len(unique_headlines)} unique headlines")
    return unique_headlines[:10]

async def fetch_macro_sentiment(session):
    """Enhanced macro sentiment analysis with better NewsAPI usage"""
    result = {
        "geopolitical_risk": 0,
        "trade_risk": 0,
        "economic_sentiment": 0,
        "overall_macro_score": 0,
        "geo_articles": [],
        "trade_articles": [],
        "econ_articles": []
    }
    
    if not NEWSAPI_KEY:
        logging.warning("No NewsAPI key for macro sentiment")
        return result
    
    try:
        # Get date range for last 7 days
        to_date = datetime.date.today()
        from_date = to_date - datetime.timedelta(days=7)
        
        async def search_news(query, category_name):
            """Search news with specific query"""
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': query,
                    'from': from_date.isoformat(),
                    'to': to_date.isoformat(),
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': 20,
                    'apiKey': NEWSAPI_KEY
                }
                
                content = await make_robust_request(session, url, params=params, retries=2)
                if content:
                    data = json.loads(content)
                    if data.get('status') == 'ok':
                        articles = data.get('articles', [])
                        logging.info(f"Found {len(articles)} articles for {category_name}")
                        return articles
                    elif data.get('status') == 'error':
                        logging.error(f"NewsAPI error for {category_name}: {data.get('message')}")
            except Exception as e:
                logging.error(f"Failed to search {category_name}: {e}")
            return []
        
        # Fetch different categories
        geo_task = search_news("war OR conflict OR military OR geopolitical", "geopolitical")
        trade_task = search_news("tariff OR \"trade war\" OR sanctions", "trade")
        econ_task = search_news("\"federal reserve\" OR inflation OR \"interest rates\" OR recession", "economic")
        
        geo_articles, trade_articles, econ_articles = await asyncio.gather(
            geo_task, trade_task, econ_task
        )
        
        # Process geopolitical
        if geo_articles:
            result['geopolitical_risk'] = min(len(geo_articles) / 10 * 100, 100)
            result['geo_articles'] = [
                {
                    "title": a.get('title', ''),
                    "url": a.get('url', '#'),
                    "source": a.get('source', {}).get('name', 'Unknown')
                }
                for a in geo_articles[:3]
                if a.get('title') and a['title'] != '[Removed]'
            ]
        
        # Process trade
        if trade_articles:
            result['trade_risk'] = min(len(trade_articles) / 10 * 100, 100)
            result['trade_articles'] = [
                {
                    "title": a.get('title', ''),
                    "url": a.get('url', '#'), 
                    "source": a.get('source', {}).get('name', 'Unknown')
                }
                for a in trade_articles[:3]
                if a.get('title') and a['title'] != '[Removed]'
            ]
        
        # Process economic with sentiment
        if econ_articles:
            sentiments = []
            for article in econ_articles:
                title = article.get('title', '')
                if title and title != '[Removed]':
                    sentiment = analyzer.polarity_scores(title).get('compound', 0)
                    sentiments.append(sentiment)
            
            if sentiments:
                result['economic_sentiment'] = sum(sentiments) / len(sentiments)
            
            result['econ_articles'] = [
                {
                    "title": a.get('title', ''),
                    "url": a.get('url', '#'),
                    "source": a.get('source', {}).get('name', 'Unknown')
                }
                for a in econ_articles[:3]
                if a.get('title') and a['title'] != '[Removed]'
            ]
        
        # Calculate overall score
        result['overall_macro_score'] = (
            -(result['geopolitical_risk'] / 100 * 15)
            - (result['trade_risk'] / 100 * 10)
            + (result['economic_sentiment'] * 15)
        )
        
        logging.info(f"✅ Macro scores - Geo: {result['geopolitical_risk']:.0f}, Trade: {result['trade_risk']:.0f}, Econ: {result['economic_sentiment']:.2f}")
        
    except Exception as e:
        logging.error(f"Macro sentiment failed: {e}")
    
    return result

async def fetch_context_data(session):
    context_data = {}
    
    # Crypto data
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
    
    # Commodities
    try:
        gold_ticker = yf.Ticker('GC=F')
        silver_ticker = yf.Ticker('SI=F')
        
        gold_info = await asyncio.to_thread(getattr, gold_ticker, 'info')
        silver_info = await asyncio.to_thread(getattr, silver_ticker, 'info')
        
        gold_price = gold_info.get('regularMarketPrice', gold_info.get('previousClose'))
        silver_price = silver_info.get('regularMarketPrice', silver_info.get('previousClose'))
        
        context_data['gold'] = {
            'name': 'Gold',
            'symbol': 'GC=F',
            'current_price': gold_price
        }
        context_data['silver'] = {
            'name': 'Silver',
            'symbol': 'SI=F',
            'current_price': silver_price
        }
        
        if gold_price and silver_price:
            context_data['gold_silver_ratio'] = f"{gold_price/silver_price:.1f}:1"
    except Exception as e:
        logging.warning(f"Commodities data fetch failed: {e}")
    
    # Fear & Greed
    try:
        fg_content = await make_robust_request(session, "https://api.alternative.me/fng/?limit=1")
        if fg_content:
            fg_data = json.loads(fg_content)
            context_data['crypto_sentiment'] = fg_data['data'][0]['value_classification']
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
            
            # Get news
            articles = await fetch_finviz_news_throttled(throttler, session, ticker)
            
            # Sentiment analysis
            avg_sent = 0
            if articles:
                sentiments = [analyzer.polarity_scores(a["title"]).get("compound", 0) for a in articles[:5]]
                avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Score calculation
            score = 50 + (avg_sent * 20)
            
            if (tech := compute_technical_indicators(data["Close"])):
                if 40 < tech.get("rsi", 50) < 65:
                    score += 15
            
            if info.get('trailingPE') and 0 < info.get('trailingPE') < 35:
                score += 15
            
            return {
                "ticker": ticker,
                "score": min(score, 100),
                "name": info.get('shortName', ticker),
                "sector": info.get('sector', 'N/A'),
                "summary": info.get('longBusinessSummary', None)
            }
            
        except Exception as e:
            if '$' not in str(e):
                logging.debug(f"Error analyzing {ticker}: {e}")
            return None

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_memory(data):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(data, f)

async def main(output="print"):
    previous_day_memory = load_memory()
    
    # Test API keys
    if NEWSAPI_KEY:
        logging.info(f"NewsAPI key detected: {NEWSAPI_KEY[:8]}...")
    else:
        logging.warning("NewsAPI key not found!")
    
    if FINNHUB_KEY:
        logging.info(f"Finnhub key detected: {FINNHUB_KEY[:8]}...")
    else:
        logging.warning("Finnhub key not found!")
    
    sp500 = get_cached_tickers('sp500_cache.json', fetch_sp500_tickers_sync)
    tsx = get_cached_tickers('tsx_cache.json', fetch_tsx_tickers_sync)
    universe = (sp500 or [])[:75] + (tsx or [])[:25]
    
    throttler = Throttler(2)
    semaphore = asyncio.Semaphore(10)
    
    async with aiohttp.ClientSession() as session:
        stock_tasks = [analyze_stock(semaphore, throttler, session, ticker) for ticker in universe]
        context_task = fetch_context_data(session)
        news_task = fetch_market_headlines()
        macro_task = fetch_macro_sentiment(session)
        
        results, context_data, market_news, macro_data = await asyncio.gather(
            asyncio.gather(*stock_tasks),
            context_task,
            news_task,
            macro_task
        )
    
    stock_results = [r for r in results if r]
    df_stocks = pd.DataFrame(stock_results).sort_values("score", ascending=False) if stock_results else pd.DataFrame()
    
    if output == "email":
        html_email = generate_html_email(df_stocks, context_data, market_news, macro_data, previous_day_memory)
        send_email(html_email)
    else:
        # Print summary for debugging
        print(f"Headlines fetched: {len(market_news)}")
        print(f"Macro scores - Geo: {macro_data['geopolitical_risk']}, Trade: {macro_data['trade_risk']}")
        print(f"Stocks analyzed: {len(stock_results)}")
    
    if not df_stocks.empty:
        save_memory({
            "previous_top_stock_name": df_stocks.iloc[0]['name'],
            "previous_top_stock_ticker": df_stocks.iloc[0]['ticker'],
            "previous_macro_score": macro_data.get('overall_macro_score', 0),
            "date": datetime.date.today().isoformat()
        })
    
    logging.info("✅ Analysis complete.")

def generate_html_email(df_stocks, context, market_news, macro_data, memory):
    def format_articles(articles):
        if not articles: 
            return "<p style='color:#888;'><i>No specific news drivers detected.</i></p>"
        html = "<ul style='margin:0;padding-left:20px;'>"
        for a in articles:
            if a.get('title'):
                html += f'<li style="margin-bottom:5px;"><a href="{a.get("url", "#")}" style="color:#1e3a8a;">{a["title"]}</a> <span style="color:#666;">({a.get("source", "Unknown")})</span></li>'
        html += "</ul>"
        return html
    
    def create_stock_table(df):
        return "".join([
            f'<tr><td style="padding:10px;border-bottom:1px solid #eee;"><b>{row["ticker"]}</b><br><span style="color:#666;font-size:0.9em;">{row["name"]}</span></td><td style="padding:10px;border-bottom:1px solid #eee;text-align:center;font-weight:bold;font-size:1.1em;">{row["score"]:.0f}</td></tr>' 
            for _, row in df.iterrows()
        ])
    
    def create_context_table(ids):
        rows = ""
        for asset_id in ids:
            if asset := context.get(asset_id):
                price = f"${asset.get('current_price', 0):,.2f}"
                change_24h = asset.get('price_change_percentage_24h', 0) or 0
                mcap = f"${asset.get('market_cap', 0) / 1_000_000_000:.1f}B" if asset.get('market_cap') else "N/A"
                color_24h = "#16a34a" if change_24h >= 0 else "#dc2626"
                rows += f'<tr><td style="padding:10px;border-bottom:1px solid #eee;"><b>{asset.get("name", "")}</b><br><span style="color:#666;font-size:0.9em;">{asset.get("symbol","").upper()}</span></td><td style="padding:10px;border-bottom:1px solid #eee;">{price}<br><span style="color:{color_24h};font-size:0.9em;">{change_24h:.2f}% (24h)</span></td><td style="padding:10px;border-bottom:1px solid #eee;">{mcap}</td></tr>'
        return rows

    # Generate content sections
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

    # Sector analysis
    sector_html = ""
    if not df_stocks.empty:
        top_by_sector = df_stocks.groupby('sector', group_keys=False).apply(lambda x: x.nlargest(2, 'score'))
        for _, row in top_by_sector.iterrows():
            if row['sector'] and row['sector'] != 'N/A':
                summary_text = "Business summary not available."
                if row["summary"] and isinstance(row["summary"], str):
                    summary_text = '. '.join(row["summary"].split('. ')[:2]) + '.'
                sector_html += f'<div style="margin-bottom:15px;"><b>{row["name"]} ({row["ticker"]})</b> in <i>{row["sector"]}</i><p style="font-size:0.9em;color:#333;margin:5px 0 0 0;">{summary_text}</p></div>'

    # Tables
    top10_html = create_stock_table(df_stocks.head(10)) if not df_stocks.empty else "<tr><td>No data available</td></tr>"
    bottom10_html = create_stock_table(df_stocks.tail(10).iloc[::-1]) if not df_stocks.empty else "<tr><td>No data available</td></tr>"
    crypto_html = create_context_table(["bitcoin", "ethereum", "solana", "ripple"])
    commodities_html = create_context_table(["gold", "silver"])
    
    # Headlines
    market_news_html = ""
    for article in market_news[:10]:
        if article.get('title'):
            market_news_html += f'<div style="margin-bottom:15px;"><b><a href="{article.get("url", "#")}" style="color:#000;">{article["title"]}</a></b><br><span style="color:#666;font-size:0.9em;">{article.get("source", "Unknown")}</span></div>'
    
    if not market_news_html:
        market_news_html = "<p><i>Headlines temporarily unavailable.</i></p>"

    return f"""
    <!DOCTYPE html><html><head><style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0;background-color:#f7f7f7;}} .container{{width:100%;max-width:700px;margin:20px auto;background-color:#fff;border:1px solid #ddd;}} .header{{background-color:#0c0a09;color:#fff;padding:30px;text-align:center;}} .section{{padding:25px;border-bottom:1px solid #ddd;}} h2{{font-size:1.5em;color:#111;margin-top:0;}} h3{{font-size:1.2em;color:#333;border-bottom:2px solid #e2e8f0;padding-bottom:5px;}}</style></head><body><div class="container">
    <div class="header"><h1>Your Daily Intelligence Briefing</h1><p style="font-size:1.1em; color:#aaa;">{datetime.date.today().strftime('%A, %B %d, %Y')}</p></div>
    <div class="section"><h2>EDITOR'S NOTE</h2><p>{editor_note}</p></div>
    <div class="section"><h2>THE BIG PICTURE: The Market Weather Report</h2>
        <h3>Overall Macro Score: {macro_data['overall_macro_score']:.1f} / 30</h3>
        <p><b>How it's calculated:</b> This is our "weather forecast" for investors, combining risks and sentiment. A positive score suggests optimism ("risk-on"), while negative signals caution ("risk-off").</p>
        <p><b>🌍 Geopolitical Risk ({macro_data['geopolitical_risk']:.0f}/100):</b> Measures global instability by scanning news for conflict keywords.<br><u>Key Drivers:</u> {format_articles(macro_data['geo_articles'])}</p>
        <p><b>🚢 Trade Risk ({macro_data['trade_risk']:.0f}/100):</b> Tracks mentions of 'trade war', 'tariffs', etc.<br><u>Key Drivers:</u> {format_articles(macro_data['trade_articles'])}</p>
        <p><b>💼 Economic Sentiment ({macro_data['economic_sentiment']:.2f}):</b> Analyzes the tone of news about inflation, rates, and growth (-1 to +1).<br><u>Key Drivers:</u> {format_articles(macro_data['econ_articles'])}</p>
    </div>
    <div class="section"><h2>SECTOR DEEP DIVE</h2><p>Top-scoring companies from different sectors.</p>{sector_html or "<p><i>No sector data available.</i></p>"}</div>
    <div class="section"><h2>STOCK RADAR</h2>
        <h3>📈 Top 10 Strongest Signals</h3><p>Stocks scored (0-100) on valuation, momentum, and news sentiment.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{top10_html}</tbody></table>
        <h3 style="margin-top: 30px;">📉 Top 10 Weakest Signals</h3><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{bottom10_html}</tbody></table>
    </div>
    <div class="section"><h2>BEYOND STOCKS: Alternative Assets</h2>
        <h3>🪙 Crypto: The Digital Frontier</h3><p><b>Market Sentiment: {context.get('crypto_sentiment', 'N/A')}</b></p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:left; padding:10px;">Price / 24h</th><th style="text-align:left; padding:10px;">Market Cap</th></tr></thead><tbody>{crypto_html}</tbody></table>
        <h3 style="margin-top: 30px;">💎 Commodities</h3><p><b>Gold/Silver Ratio: {context.get('gold_silver_ratio', 'N/A')}</b></p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:left; padding:10px;">Price / 24h</th><th style="text-align:left; padding:10px;">Market Cap</th></tr></thead><tbody>{commodities_html}</tbody></table>
    </div>
    <div class="section"><h2>FROM THE WIRE: Today's Top Headlines</h2>{market_news_html}</div>
    </div></body></html>
    """

def send_email(html_body):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="print", choices=["print", "email"])
    args = parser.parse_args()
    
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main(output=args.output))
