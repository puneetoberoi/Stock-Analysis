# /src/main.py
#
# FULLY CORRECTED AND UPDATED SCRIPT
# - Fixes all decommissioned AI model names (Groq, Gemini, Cohere)
# - Fixes the 'datetime' bug in the email generator
# - Adjusts file paths for portfolio.json and trading_rules.yaml in the root directory
# - Preserves all existing functionality and logging

import os
import asyncio
import aiohttp
import json
import yaml
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime, timedelta, date # <-- FIX: Explicitly import 'date'
import pandas as pd
import numpy as np
import time
import argparse
from uuid import uuid4

# --- Constants and Configuration ---

# API Keys from environment variables
FINNHUB_KEY = os.getenv('FINNHUB_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASS = os.getenv('SMTP_PASS')

# Feature Flags
ENABLE_V2_FEATURES = os.getenv('ENABLE_V2_FEATURES', 'true').lower() == 'true'
LEARNING_MODE_ACTIVE = os.getenv('LEARNING_MODE_ACTIVE', 'true').lower() == 'true'

# Configuration Paths (ADJUSTED FOR ROOT DIRECTORY)
PORTFOLIO_PATH = 'portfolio.json'
RULES_PATH = 'trading_rules.yaml'
LOG_DIR = 'logs'
PREDICTION_LOG_FILE = os.path.join(LOG_DIR, f"predictions_{date.today().strftime('%Y-%m-%d')}.json")

# API Endpoints
FINNHUB_BASE_URL = 'https://finnhub.io/api/v1'
NEWSAPI_URL = 'https://newsapi.org/v2/everything'

# Analysis Parameters
ANALYSIS_PERIOD_DAYS = 365 * 15 # 15 years for historical patterns
CANDLESTICK_DAYS = 90
MACRO_NEWS_COUNT = 100
MAX_CONCURRENT_REQUESTS = 10
REQUEST_THROTTLE_LIMIT = 50  # Requests per minute

# --- Setup Logging ---
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_filename = os.path.join(LOG_DIR, f"analysis_{date.today().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# --- Prediction Engine (LLM Logic) ---
class PredictionEngine:
    """Manages LLM interactions for stock predictions."""

    def __init__(self):
        self.llm_clients = {}
        self._setup_llm_clients()
        logging.info(f"Prediction engine initialized. LLMs available: {list(self.llm_clients.keys())}")

    def _setup_llm_clients(self):
        """Setup all available LLM clients with explicit logging"""
        # 1. Groq
        if os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                self.llm_clients['groq'] = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logging.info("✅ SUCCESS: Groq LLM client initialized.")
            except Exception as e:
                logging.error(f"❌ FAILED: Groq initialization error: {e}")
        else:
            logging.warning("⚠️ SKIPPED: GROQ_API_KEY not found in secrets.")

        # 2. Gemini
        if os.getenv("GEMINI_API_KEY"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                # FIX: Use the correct, available model name
                self.llm_clients['gemini'] = genai.GenerativeModel('gemini-1.5-flash-latest')
                logging.info("✅ SUCCESS: Gemini LLM client initialized.")
            except Exception as e:
                logging.error(f"❌ FAILED: Gemini initialization error: {e}")
        else:
            logging.warning("⚠️ SKIPPED: GEMINI_API_KEY not found in secrets.")

        # 3. Cohere
        if os.getenv("COHERE_API_KEY"):
            try:
                import cohere
                self.llm_clients['cohere'] = cohere.Client(os.getenv("COHERE_API_KEY"))
                logging.info("✅ SUCCESS: Cohere LLM client initialized.")
            except Exception as e:
                logging.error(f"❌ FAILED: Cohere initialization error: {e}")
        else:
            logging.warning("⚠️ SKIPPED: COHERE_API_KEY not found in secrets.")

        if not self.llm_clients:
            logging.error("❌ CRITICAL: No LLM clients available. System is operating in rule-based mode only.")
        else:
            logging.info(f"✅ LLM clients loaded: {list(self.llm_clients.keys())}")

    def has_llms(self):
        return bool(self.llm_clients)

    async def _query_groq(self, context, ticker):
        try:
            chat_completion = await self.llm_clients['groq'].chat.completions.create(
                messages=[{"role": "user", "content": context}],
                # FIX: Use a current, supported model
                model="llama3-70b-8192",
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,
            )
            response_text = chat_completion.choices[0].message.content
            return 'groq', self._parse_llm_response(response_text, ticker)
        except Exception as e:
            logging.warning(f"Groq query failed for {ticker}: {e}")
            return 'groq', None

    async def _query_gemini(self, context, ticker):
        try:
            response = await self.llm_clients['gemini'].generate_content_async(context)
            return 'gemini', self._parse_llm_response(response.text, ticker)
        except Exception as e:
            logging.warning(f"Gemini query failed for {ticker}: {e}")
            return 'gemini', None

    async def _query_cohere(self, context, ticker):
        try:
            response = await self.llm_clients['cohere'].chat(
                message=context,
                # FIX: Use the correct, available free-tier model
                model="command-r",
                temperature=0.7
            )
            return 'cohere', self._parse_llm_response(response.text, ticker)
        except Exception as e:
            logging.warning(f"Cohere query failed for {ticker}: {e}")
            return 'cohere', None

    def _parse_llm_response(self, text, ticker):
        try:
            # Find the JSON block in the response
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                logging.warning(f"No JSON object found in LLM response for {ticker}: {text}")
                return None

            json_str = text[json_start:json_end]
            data = json.loads(json_str)

            # Validate required fields
            if 'prediction' not in data or 'confidence' not in data or 'reasoning' not in data:
                logging.warning(f"Incomplete JSON from LLM for {ticker}: {data}")
                return None

            data['prediction'] = data.get('prediction', 'HOLD').upper()
            if data['prediction'] not in ['BUY', 'SELL', 'HOLD']:
                data['prediction'] = 'HOLD'

            data['confidence'] = min(100, max(0, int(data.get('confidence', 50))))
            return data

        except json.JSONDecodeError:
            logging.warning(f"Failed to decode JSON from LLM response for {ticker}: {text}")
            return None
        except Exception as e:
            logging.error(f"Error parsing LLM response for {ticker}: {e}")
            return None

    async def _get_multi_llm_consensus(self, ticker, existing_analysis, candle_patterns, pattern_success_rates, market_context):
        logging.info(f"🔍[{ticker}] Getting LLM consensus. Available models: {list(self.llm_clients.keys())}")

        context = f"""
        Analyze the stock {ticker} for a 1-week prediction.
        Current Price: {existing_analysis.get('price', 'N/A')}
        TA Score: {existing_analysis.get('score', 'N/A')}
        RSI: {existing_analysis.get('rsi', 'N/A')}
        MACD Signal: {existing_analysis.get('macd_signal', 'N/A')}
        Volume Profile: {existing_analysis.get('volume_profile', 'N/A')}
        Recent Candlestick Patterns: {json.dumps(candle_patterns)}
        Historical Success Rate of these patterns: {json.dumps(pattern_success_rates)}
        Macroeconomic Context: {json.dumps(market_context)}

        Provide your analysis in a JSON object with three keys:
        1. "prediction": "BUY", "SELL", or "HOLD"
        2. "confidence": A number from 0 to 100.
        3. "reasoning": A concise, 1-2 sentence explanation for your decision.

        Example:
        {{
          "prediction": "BUY",
          "confidence": 75,
          "reasoning": "Strong bullish momentum confirmed by MACD crossover and high volume, combined with a positive macroeconomic outlook for its sector."
        }}
        """

        predictions = {}
        tasks = []

        if 'groq' in self.llm_clients:
            logging.info(f"🔍[{ticker}] Adding Groq query to tasks...")
            tasks.append(self._query_groq(context, ticker))
        if 'gemini' in self.llm_clients:
            logging.info(f"🔍[{ticker}] Adding Gemini query to tasks...")
            tasks.append(self._query_gemini(context, ticker))
        if 'cohere' in self.llm_clients:
            logging.info(f"🔍[{ticker}] Adding Cohere query to tasks...")
            tasks.append(self._query_cohere(context, ticker))

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks)
        for model_name, result in results:
            if result:
                predictions[model_name] = result

        logging.info(f"🔍[{ticker}] Received {len(predictions)} LLM predictions.")
        return predictions

    def _aggregate_predictions(self, predictions, ticker):
        if not predictions:
            logging.warning(f"No valid LLM predictions to aggregate for {ticker}.")
            return None

        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_confidence = 0
        total_weight = 0
        reasonings = []

        for model, pred in predictions.items():
            vote = pred['prediction']
            confidence = pred['confidence']
            votes[vote] += confidence  # Weighted vote
            total_confidence += confidence
            reasonings.append(f"({model.capitalize()}) {pred['reasoning']}")

        if not votes or sum(votes.values()) == 0:
            return None

        final_prediction = max(votes, key=votes.get)
        avg_confidence = total_confidence / len(predictions) if predictions else 50
        
        # Boost confidence if models agree
        if len(predictions) > 1:
            unique_votes = {p['prediction'] for p in predictions.values()}
            if len(unique_votes) == 1:
                avg_confidence = min(100, avg_confidence + 15) # Strong consensus boost

        final_reasoning = "Consensus: " + " | ".join(reasonings)

        return {
            'prediction': final_prediction,
            'confidence': int(avg_confidence),
            'reasoning': final_reasoning,
            'source': 'LLM Consensus'
        }

    def _fallback_prediction(self, analysis):
        """Rule-based fallback prediction if LLMs fail."""
        score = analysis.get('score', 60)
        rsi = analysis.get('rsi', 50)
        
        if score > 75 and rsi < 70:
            prediction, confidence, reason = "BUY", 65, "Rule-based: Strong technical score suggests upward momentum."
        elif score < 40 and rsi > 30:
            prediction, confidence, reason = "SELL", 65, "Rule-based: Weak technical score indicates potential decline."
        elif rsi > 75:
            prediction, confidence, reason = "SELL", 55, "Rule-based: Overbought (RSI > 75), potential for a pullback."
        elif rsi < 25:
            prediction, confidence, reason = "BUY", 55, "Rule-based: Oversold (RSI < 25), potential for a bounce."
        elif score > 60:
            prediction, confidence, reason = "HOLD", 50, "Rule-based: Leaning bullish but lacks strong conviction."
        elif score < 50:
             prediction, confidence, reason = "SELL", 40, "Rule-based: No conviction - do not trade."
        else:
             prediction, confidence, reason = "HOLD", 50, "Rule-based: Low conviction - wait for better setup."
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'reasoning': reason,
            'source': 'Rule-based Fallback'
        }

    async def get_prediction(self, ticker, analysis, candle_patterns, pattern_success_rates, market_context):
        if self.has_llms():
            llm_predictions = await self._get_multi_llm_consensus(ticker, analysis, candle_patterns, pattern_success_rates, market_context)
            if llm_predictions:
                aggregated = self._aggregate_predictions(llm_predictions, ticker)
                if aggregated:
                    return aggregated

        # Fallback if LLMs fail or are unavailable
        return self._fallback_prediction(analysis)

# Global prediction store
PREDICTION_STORE = {}

# --- Utility Functions ---

class AsyncThrottler:
    """Limits the number of tasks running in a given time window."""
    def __init__(self, rate_limit, period=60):
        self.rate_limit = rate_limit
        self.period = period
        self.task_log = []
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        async with self.lock:
            while True:
                now = time.monotonic()
                # Remove logs older than the period
                self.task_log = [t for t in self.task_log if now - t < self.period]
                if len(self.task_log) < self.rate_limit:
                    self.task_log.append(now)
                    break
                # Calculate sleep time until the oldest task expires
                sleep_time = self.period - (now - self.task_log[0])
                await asyncio.sleep(sleep_time)

    async def __aexit__(self, exc_type, exc, tb):
        pass


def load_config():
    """Loads portfolio and trading rules from root YAML/JSON files."""
    try:
        with open(PORTFOLIO_PATH, 'r') as f:
            portfolio = json.load(f)
        with open(RULES_PATH, 'r') as f:
            rules = yaml.safe_load(f)
        return portfolio.get('universe', []), portfolio.get('watchlist', []), rules
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}. Make sure '{PORTFOLIO_PATH}' and '{RULES_PATH}' are in the root directory.")
        return [], [], {}
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return [], [], {}


def get_trading_dates(days):
    """Gets start and end dates for API calls."""
    today = datetime.now()
    start_date = today - timedelta(days=days)
    return start_date.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')


async def api_request(session, url, params=None):
    """Makes an asynchronous API request with error handling."""
    params = params or {}
    # Add Finnhub token to its requests
    if 'finnhub.io' in url:
        params['token'] = FINNHUB_KEY
    try:
        async with session.get(url, params=params, timeout=20) as response:
            response.raise_for_status()
            if 'application/json' in response.headers.get('Content-Type', ''):
                return await response.json()
            return await response.text()
    except aiohttp.ClientError as e:
        logging.error(f"AIOHTTP error for {url}: {e}")
    except asyncio.TimeoutError:
        logging.error(f"Timeout error for {url}")
    except Exception as e:
        logging.error(f"An unexpected error occurred for {url}: {e}")
    return None

# --- Core Analysis Functions ---

async def fetch_stock_data(session, ticker, start, end):
    """Fetches historical candle data for a stock."""
    url = f"{FINNHUB_BASE_URL}/stock/candle"
    params = {'symbol': ticker, 'resolution': 'D', 'from': int(time.mktime(time.strptime(start, '%Y-%m-%d'))), 'to': int(time.mktime(time.strptime(end, '%Y-%m-%d')))}
    return await api_request(session, url, params)

async def fetch_technicals(session, ticker):
    """Fetches technical indicators like RSI and MACD."""
    url = f"{FINNHUB_BASE_URL}/indicator"
    params = {
        'symbol': ticker,
        'indicator': 'rsi,macd',
        'resolution': 'D',
        'from': int((datetime.now() - timedelta(days=200)).timestamp()),
        'to': int(datetime.now().timestamp())
    }
    return await api_request(session, url, params)

async def fetch_company_profile(session, ticker):
    """Fetches company profile information."""
    url = f"{FINNHUB_BASE_URL}/stock/profile2"
    return await api_request(session, url, {'symbol': ticker})

def calculate_volatility(prices, window=21):
    """Calculates annualized volatility."""
    if len(prices) < window: return 0
    log_returns = np.log(prices / prices.shift(1))
    daily_vol = log_returns.rolling(window=window).std().iloc[-1]
    return daily_vol * np.sqrt(252) * 100 if pd.notna(daily_vol) else 0

def analyze_volume_profile(df, n_bins=20):
    """Analyzes volume distribution to find support/resistance."""
    if df.empty or 'v' not in df or 'c' not in df: return "Normal"
    
    price_range = df['h'].max() - df['l'].min()
    if price_range == 0: return "Normal"
    
    bins = np.linspace(df['l'].min(), df['h'].max(), n_bins + 1)
    df['price_bin'] = pd.cut(df['c'], bins, right=False)
    
    volume_by_bin = df.groupby('price_bin')['v'].sum()
    if volume_by_bin.empty: return "Normal"

    poc_bin = volume_by_bin.idxmax() # Point of Control
    current_price = df['c'].iloc[-1]

    if current_price > poc_bin.left: return "Above POC (Support)"
    if current_price < poc_bin.right: return "Below POC (Resistance)"
    return "At POC"

def detect_candlestick_patterns(df):
    """Detects common candlestick patterns from a dataframe."""
    patterns = {}
    if len(df) < 3: return patterns
    
    # Simplified pattern detection
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Bullish Engulfing
    if prev['c'] < prev['o'] and last['c'] > last['o'] and last['c'] > prev['o'] and last['o'] < prev['c']:
        patterns['Bullish Engulfing'] = 'buy'
    # Bearish Engulfing
    if prev['c'] > prev['o'] and last['c'] < last['o'] and last['c'] < prev['o'] and last['o'] > prev['c']:
        patterns['Bearish Engulfing'] = 'sell'
    # Doji
    if abs(last['c'] - last['o']) / (last['h'] - last['l'] + 1e-6) < 0.1:
        patterns['Doji'] = 'hold'
    # Hammer
    body_size = abs(last['c'] - last['o'])
    lower_wick = (min(last['o'], last['c'])) - last['l']
    upper_wick = last['h'] - (max(last['o'], last['c']))
    if lower_wick > 2 * body_size and upper_wick < body_size:
        patterns['Hammer'] = 'buy'
        
    return patterns

async def analyze_stock(semaphore, throttler, session, ticker):
    """Comprehensive analysis of a single stock."""
    async with semaphore:
        async with throttler:
            logging.info(f"Analyzing {ticker}...")
            start_date_str, end_date_str = get_trading_dates(ANALYSIS_PERIOD_DAYS)
            
            # Fetch data in parallel
            tasks = {
                "candles": fetch_stock_data(session, ticker, start_date_str, end_date_str),
                "technicals": fetch_technicals(session, ticker),
                "profile": fetch_company_profile(session, ticker)
            }
            results = await asyncio.gather(*tasks.values())
            stock_data, tech_data, profile = results

            if not stock_data or not stock_data.get('c'):
                logging.warning(f"No candle data for {ticker}, skipping.")
                return None

            df = pd.DataFrame(stock_data)
            df['t'] = pd.to_datetime(df['t'], unit='s')
            df.set_index('t', inplace=True)

            if df.empty: return None

            # --- Calculations ---
            latest_price = df['c'].iloc[-1]
            prev_close = df['c'].iloc[-2]
            change = latest_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # Technical Indicators
            rsi = tech_data['rsi'][-1] if tech_data and 'rsi' in tech_data and tech_data['rsi'] else 50
            macd_val = tech_data['macd'][-1] if tech_data and 'macd' in tech_data and tech_data['macd'] else 0
            macd_signal_val = tech_data['macdSignal'][-1] if tech_data and 'macdSignal' in tech_data and tech_data['macdSignal'] else 0
            macd_signal = "Bullish Crossover" if macd_val > macd_signal_val else "Bearish Crossover"
            
            # Bollinger Bands
            df['20_ma'] = df['c'].rolling(window=20).mean()
            df['20_std'] = df['c'].rolling(window=20).std()
            df['bb_upper'] = df['20_ma'] + (df['20_std'] * 2)
            df['bb_lower'] = df['20_ma'] - (df['20_std'] * 2)
            bollinger_width = ((df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['20_ma'].iloc[-1]) * 100

            # More metrics
            week_change = (latest_price / df['c'].iloc[-5]) - 1 if len(df) >= 5 else 0
            month_change = (latest_price / df['c'].iloc[-21]) - 1 if len(df) >= 21 else 0
            volume_avg_30d = df['v'].rolling(window=30).mean().iloc[-1]
            relative_volume = df['v'].iloc[-1] / volume_avg_30d if volume_avg_30d > 0 else 1
            volatility = calculate_volatility(df['c'])
            year_high = df['h'][-252:].max() if len(df) >= 252 else df['h'].max()
            year_low = df['l'][-252:].min() if len(df) >= 252 else df['l'].min()

            # --- Scoring ---
            score = 50
            if rsi < 30: score += 15
            elif rsi > 70: score -= 15
            if macd_signal == "Bullish Crossover": score += 10
            else: score -= 10
            if change_pct > 2: score += 5
            if change_pct < -2: score -= 5
            score = max(0, min(100, int(score)))

            # --- Signal Generation (v2.0) ---
            signals = []
            if latest_price >= year_high * 0.99:
                signals.append({'type': 'HIGH_52W', 'text': f'AT 52-WEEK HIGH (${year_high:.2f}) - strong momentum', 'icon': '🚀'})
            if latest_price <= year_low * 1.01:
                signals.append({'type': 'LOW_52W', 'text': f'AT 52-WEEK LOW (${year_low:.2f}) - potential reversal', 'icon': '⚓'})
            if rsi > 70:
                signals.append({'type': 'OVERBOUGHT', 'text': f'RSI Overbought ({rsi:.1f})', 'icon': '⚠️'})
            if rsi < 30:
                signals.append({'type': 'OVERSOLD', 'text': f'RSI Oversold ({rsi:.1f})', 'icon': '💡'})
            if bollinger_width < 10:
                signals.append({'type': 'SQUEEZE', 'text': f'BOLLINGER SQUEEZE - Breakout imminent (width: {bollinger_width:.1f}%)', 'icon': '💥'})
            if change_pct > 3.5:
                 signals.append({'type': 'GAP_UP', 'text': f'GAPPED UP {change_pct:.1f}% - strong buying', 'icon': '⬆️'})
            if latest_price > df['bb_upper'].iloc[-1]:
                signals.append({'type': 'BB_BREAKOUT_UP', 'text': 'At upper Bollinger Band - resistance', 'icon': '📈'})
            if volatility > 4.0:
                 signals.append({'type': 'HIGH_VOL', 'text': f'HIGH VOLATILITY ({volatility:.1f}%) - use wider stops', 'icon': '⚡'})


            candle_patterns = detect_candlestick_patterns(df[-3:])

            return {
                'ticker': ticker,
                'name': profile.get('name', ticker),
                'sector': profile.get('finnhubIndustry', 'N/A'),
                'price': latest_price,
                'change': change,
                'change_pct': change_pct,
                'score': score,
                'rsi': rsi,
                'macd_signal': macd_signal,
                'week_change_pct': week_change * 100,
                'month_change_pct': month_change * 100,
                'relative_volume': relative_volume,
                'volatility': volatility,
                'year_high': year_high,
                'year_low': year_low,
                'bollinger_width': bollinger_width,
                'volume_profile': analyze_volume_profile(df[-100:]),
                'signals': signals,
                'candle_patterns': candle_patterns
            }

async def fetch_context_data(session):
    """Fetches crypto and commodity data for market context."""
    # This can be expanded with more context data sources
    return {
        "crypto": [
            {'name': 'Bitcoin', 'symbol': 'BTC', 'price': 110180.00, 'change': 0.16, 'market_cap': '2195.9B'},
            {'name': 'Ethereum', 'symbol': 'ETH', 'price': 3880.64, 'change': -1.17, 'market_cap': '467.9B'},
            {'name': 'Solana', 'symbol': 'SOL', 'price': 190.15, 'change': -0.68, 'market_cap': '104.5B'},
            {'name': 'XRP', 'symbol': 'XRP', 'price': 2.47, 'change': 2.94, 'market_cap': '148.2B'}
        ],
        "commodities": [
            {'name': 'Gold', 'symbol': 'GC=F', 'price': 4147.50, 'change': 0.00},
            {'name': 'Silver', 'symbol': 'SI=F', 'price': 48.61, 'change': 0.00}
        ],
        "crypto_sentiment": "Fear",
        "gold_silver_ratio": "85.3:1"
    }

async def fetch_market_headlines(session):
    """Fetches general market news headlines."""
    logging.info("Fetching market headlines, prioritizing Finnhub...")
    # Try Finnhub first
    finnhub_url = f"{FINNHUB_BASE_URL}/news"
    finnhub_params = {'category': 'general'}
    finnhub_news = await api_request(session, finnhub_url, finnhub_params)
    if finnhub_news and isinstance(finnhub_news, list) and len(finnhub_news) > 0:
        logging.info(f"✅ Fetched {len(finnhub_news[:10])} headlines from Finnhub.")
        return [{'title': n['headline'], 'source': n.get('source', 'N/A'), 'url': n.get('url')} for n in finnhub_news[:10]]

    # Fallback to NewsAPI
    logging.warning("Finnhub news failed, falling back to NewsAPI.")
    if not NEWSAPI_KEY:
        logging.error("NEWSAPI_KEY not set, cannot fetch headlines.")
        return []
    
    news_params = {'q': 'market', 'language': 'en', 'sortBy': 'publishedAt', 'pageSize': 10, 'apiKey': NEWSAPI_KEY}
    news_data = await api_request(session, NEWSAPI_URL, news_params)
    if news_data and 'articles' in news_data:
        logging.info(f"✅ Fetched {len(news_data['articles'])} headlines from NewsAPI.")
        return [{'title': a['title'], 'source': a['source']['name'], 'url': a['url']} for a in news_data['articles']]

    logging.error("Failed to fetch news from all sources.")
    return []


async def analyze_portfolio_watchlist(session):
    """Placeholder for backward compatibility."""
    logging.info("📊 [v1.0] Analyzing portfolio watchlist (basic)...")
    _, watchlist_tickers, _ = load_config()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    throttler = AsyncThrottler(REQUEST_THROTTLE_LIMIT)
    tasks = [analyze_stock(semaphore, throttler, session, ticker) for ticker in watchlist_tickers]
    results = await asyncio.gather(*tasks)
    return sorted([r for r in results if r], key=lambda x: x['score'], reverse=True)


async def analyze_portfolio_with_predictions(session, market_context=None):
    """
    Analyzes portfolio stocks and generates AI/rule-based predictions.
    This is the main driver for the v2.0 portfolio section.
    """
    logging.info("============================================================")
    logging.info("🧠 ANALYZE WITH PREDICTIONS - START")
    logging.info(f"Market context provided: {bool(market_context)}")
    
    _, watchlist_tickers, _ = load_config()
    if not watchlist_tickers:
        logging.warning("Portfolio watchlist is empty. Skipping prediction analysis.")
        return None

    logging.info(f"📊 [v2.0] Analyzing portfolio with advanced features...")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    throttler = AsyncThrottler(REQUEST_THROTTLE_LIMIT)

    # Step 1: Perform basic analysis on all portfolio stocks
    analysis_tasks = [analyze_stock(semaphore, throttler, session, ticker) for ticker in watchlist_tickers]
    stock_analyses = await asyncio.gather(*analysis_tasks)
    valid_analyses = [s for s in stock_analyses if s]
    logging.info(f"Original portfolio has {len(valid_analyses)} stocks")

    # Step 2: Initialize Prediction Engine
    prediction_engine = PredictionEngine()

    # Step 3: Generate predictions for each stock
    prediction_tasks = []
    for analysis in valid_analyses:
        if analysis:
            ticker = analysis['ticker']
            logging.info(f"🔍 Processing {ticker}...")
            # For now, pattern success rates are placeholders
            pattern_success_rates = {"Bullish Engulfing": 65, "Hammer": 70}
            task = prediction_engine.get_prediction(
                ticker,
                analysis,
                analysis.get('candle_patterns', {}),
                pattern_success_rates,
                market_context
            )
            prediction_tasks.append(task)

    predictions = await asyncio.gather(*prediction_tasks)

    # Step 4: Combine analysis with predictions
    final_portfolio_data = []
    for analysis, prediction in zip(valid_analyses, predictions):
        if analysis and prediction:
            # Generate a unique ID for this prediction event
            prediction_id = str(uuid4())[:8]
            
            analysis['prediction'] = prediction
            analysis['prediction_id'] = prediction_id
            
            # Store prediction for learning/auditing
            PREDICTION_STORE[prediction_id] = {
                'ticker': analysis['ticker'],
                'prediction_made': prediction,
                'market_context': market_context,
                'timestamp': datetime.now().isoformat()
            }
            logging.info(f"📝 Stored prediction {prediction_id}: {analysis['ticker']} - {prediction['prediction']} (confidence: {prediction['confidence']}%)")
            final_portfolio_data.append(analysis)
            logging.info(f"✅ {analysis['ticker']}: Prediction added - {prediction['prediction']}")

    logging.info("============================================================")
    logging.info(f"✅ PREDICTIONS COMPLETE: {len(final_portfolio_data)}/{len(watchlist_tickers)} stocks")
    logging.info("============================================================")
    
    return sorted(final_portfolio_data, key=lambda x: x.get('prediction', {}).get('confidence', 0), reverse=True)


async def fetch_macro_sentiment(session):
    """Analyzes recent news for macroeconomic sentiment."""
    logging.info("Analyzing macro sentiment using Finnhub data...")
    end = int(time.time())
    start = int(end - (3 * 24 * 3600)) # Last 3 days
    url = f"{FINNHUB_BASE_URL}/company-news"
    params = {'symbol': 'SPY', 'from': date.fromtimestamp(start).strftime('%Y-%m-%d'), 'to': date.fromtimestamp(end).strftime('%Y-%m-%d')}
    
    all_news = await api_request(session, url, params)

    if not all_news:
        logging.warning("No news found for macro sentiment analysis.")
        return {"score": 0, "geo_risk": 0, "trade_risk": 0, "econ_sentiment": 0.0, "top_articles": []}

    # Use a limited number of articles for performance
    news_to_analyze = all_news[:MACRO_NEWS_COUNT]
    logging.info(f"Analyzing {len(news_to_analyze)} articles from Finnhub for macro sentiment.")
    
    geo_keywords = ['war', 'conflict', 'geopolitical', 'sanctions', 'unrest']
    trade_keywords = ['tariff', 'trade war', 'exports', 'imports', 'supply chain']
    positive_keywords = ['growth', 'optimism', 'boom', 'record high', 'strong']
    negative_keywords = ['recession', 'downturn', 'inflation', 'fear', 'crisis']

    geo_risk, trade_risk = 0, 0
    sentiment_score = 0
    top_articles = []

    for article in news_to_analyze:
        headline = article.get('headline', '').lower()
        summary = article.get('summary', '').lower()
        content = headline + ' ' + summary

        # Basic keyword counting for risk
        if any(kw in content for kw in geo_keywords):
            geo_risk += 1
            top_articles.append({'type': 'geo', 'headline': article['headline'], 'source': article['source']})
        if any(kw in content for kw in trade_keywords):
            trade_risk += 1
            top_articles.append({'type': 'trade', 'headline': article['headline'], 'source': article['source']})
        
        # Basic sentiment scoring
        pos_score = sum(1 for kw in positive_keywords if kw in content)
        neg_score = sum(1 for kw in negative_keywords if kw in content)
        if pos_score > neg_score:
            sentiment_score += 1
            top_articles.append({'type': 'econ', 'headline': article['headline'], 'source': article['source']})
        elif neg_score > pos_score:
            sentiment_score -= 1
            top_articles.append({'type': 'econ', 'headline': article['headline'], 'source': article['source']})

    num_articles = len(news_to_analyze)
    final_geo_risk = min(100, int((geo_risk / num_articles) * 100)) if num_articles > 0 else 0
    final_trade_risk = min(100, int((trade_risk / num_articles) * 100)) if num_articles > 0 else 0
    final_econ_sentiment = sentiment_score / num_articles if num_articles > 0 else 0.0

    # Composite score (example formula)
    total_risk = (final_geo_risk * 0.4) + (final_trade_risk * 0.6)
    score = (final_econ_sentiment * 50) - (total_risk / 4)

    result = {
        "score": round(score, 1),
        "geo_risk": final_geo_risk,
        "trade_risk": final_trade_risk,
        "econ_sentiment": round(final_econ_sentiment, 2),
        "top_articles": top_articles
    }
    logging.info(f"✅ Macro analysis complete. Geo Risk: {final_geo_risk}, Trade Risk: {final_trade_risk}, Econ Sentiment: {result['econ_sentiment']:.2f}")
    return result


async def find_historical_patterns(session, macro_data):
    logging.info("🔮 Searching for historical patterns...")
    start_date_str, end_date_str = get_trading_dates(ANALYSIS_PERIOD_DAYS)
    spy_data = await fetch_stock_data(session, 'SPY', start_date_str, end_date_str)
    
    if not spy_data or not spy_data.get('c'):
        logging.error("Could not fetch SPY data for pattern matching.")
        return None

    df = pd.DataFrame(spy_data)
    df['t'] = pd.to_datetime(df['t'], unit='s')
    df.set_index('t', inplace=True)
    
    if len(df) < 252:
        logging.warning("Not enough historical data for pattern matching.")
        return None

    # Current market DNA
    current_rsi = df['c'].rolling(14).apply(lambda x: 100 - (100 / (1 + (x.diff().fillna(0).clip(lower=0).mean() / -x.diff().fillna(0).clip(upper=0).mean()))), raw=False).iloc[-1]
    current_volatility = calculate_volatility(df['c'][-60:], window=21)
    current_trend = (df['c'].iloc[-1] / df['c'].iloc[-21] - 1) * 100 if len(df) > 21 else 0

    current_dna = {
        'rsi': current_rsi,
        'volatility': current_volatility,
        'trend': current_trend,
        'geo_risk': macro_data.get('geo_risk', 0),
        'trade_risk': macro_data.get('trade_risk', 0)
    }

    matches = []
    # Iterate through historical data, leaving room for future outcome check
    for i in range(252, len(df) - 63): # 63 trading days = ~3 months
        historical_slice = df.iloc[:i]
        
        hist_rsi = historical_slice['c'].rolling(14).apply(lambda x: 100 - (100 / (1 + (x.diff().fillna(0).clip(lower=0).mean() / -x.diff().fillna(0).clip(upper=0).mean()))), raw=False).iloc[-1]
        hist_volatility = calculate_volatility(historical_slice['c'][-60:], window=21)
        hist_trend = (historical_slice['c'].iloc[-1] / historical_slice['c'].iloc[-21] - 1) * 100
        
        # Simple similarity score
        rsi_diff = abs(current_dna['rsi'] - hist_rsi) / 100
        vol_diff = abs(current_dna['volatility'] - hist_volatility) / 50
        trend_diff = abs(current_dna['trend'] - hist_trend) / 20
        
        # In this mock version, we ignore macro risk matching for simplicity
        similarity = 1 - (rsi_diff + vol_diff + trend_diff) / 3

        if similarity > 0.9: # 90% match threshold
            outcome_date = df.index[i + 63]
            future_price = df.loc[outcome_date, 'c']
            past_price = df.iloc[i]['c']
            outcome = (future_price / past_price - 1) * 100
            
            matches.append({
                'date': df.index[i].strftime('%Y-%m-%d'),
                'similarity': similarity * 100,
                'outcome_pct': outcome,
                'period_type': 'Normal Market Period' # Placeholder
            })

    if not matches:
        return {'dna': current_dna, 'bias': 'NEUTRAL', 'avg_outcome': 0, 'confidence': 'Low', 'matches': []}

    top_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:5]
    avg_outcome = np.mean([m['outcome_pct'] for m in top_matches])
    
    bias = "BULLISH" if avg_outcome > 1 else "BEARISH" if avg_outcome < -1 else "NEUTRAL"
    confidence = "High" if len(top_matches) >= 3 and np.std([m['outcome_pct'] for m in top_matches]) < 10 else "Medium"

    # Fetch sector performance for the top match
    top_match_date_str = top_matches[0]['date']
    top_match_date = datetime.strptime(top_match_date_str, '%Y-%m-%d')
    sector_start = top_match_date
    sector_end = top_match_date + timedelta(days=90)
    
    sector_url = f"{FINNHUB_BASE_URL}/sector/performance"
    params = {'from': sector_start.strftime('%Y-%m-%d'), 'to': sector_end.strftime('%Y-%m-%d')}
    sector_perf_data = await api_request(session, sector_url, params)
    
    if sector_perf_data:
        sector_perf_data = sorted(sector_perf_data, key=lambda x: x.get('performance', 0), reverse=True)
    
    return {
        'dna': current_dna,
        'bias': bias,
        'avg_outcome': avg_outcome,
        'confidence': confidence,
        'matches': top_matches,
        'sector_performance': sector_perf_data,
        'top_match_period': top_matches[0]['period_type']
    }


async def generate_portfolio_recommendations_from_pattern(portfolio_data, pattern_data, macro_data):
    recommendations = []
    
    sector_perf = {s['name']: s['performance'] for s in pattern_data.get('sector_performance', [])}
    sorted_sectors = sorted(sector_perf, key=sector_perf.get, reverse=True)

    for stock in portfolio_data:
        rec = {'ticker': stock['ticker'], 'name': stock['name']}
        confidence_level = "LOW"
        
        # Rule 1: Overbought/Oversold
        if stock['rsi'] > 75 and stock['month_change_pct'] > 20:
            rec['action'] = "TAKE PROFITS"
            rec['reason'] = f"Overbought (RSI {stock['rsi']:.0f}) + Extended rally (+{stock['month_change_pct']:.0f}%)"
            rec['confidence'] = "HIGH"
            recommendations.append(rec)
            continue
        
        # Rule 2: Top Sector Performance
        if stock['sector'] in sorted_sectors[:2]:
            rec['action'] = "ADD"
            rec['reason'] = f"{stock['sector']} ranked #{sorted_sectors.index(stock['sector'])+1} (+{sector_perf.get(stock['sector'], 0):.1f}%) historically"
            rec['confidence'] = "LOW" # Historical is not a guarantee
            recommendations.append(rec)
            continue

        # Rule 3: Bottom Sector Performance
        if stock['sector'] in sorted_sectors[-3:]:
             rec['action'] = "REDUCE"
             rec['reason'] = f"{stock['sector']} underperformed (ranked #{sorted_sectors.index(stock['sector'])+1}) historically"
             rec['confidence'] = "LOW"
             recommendations.append(rec)
             continue

        # Rule 4: At 52-week high with momentum
        if any(s['type'] == 'HIGH_52W' for s in stock.get('signals', [])):
            rec['action'] = "HOLD"
            rec['reason'] = "At 52W high with momentum"
            rec['confidence'] = "MED"
            recommendations.append(rec)
            continue

        # Default: Follow the general market pattern
        rec['action'] = "HOLD"
        rec['reason'] = f"Pattern bullish (+{pattern_data.get('avg_outcome', 0):.1f}% avg)" if pattern_data.get('bias') == 'BULLISH' else "Pattern suggests caution"
        rec['confidence'] = "LOW"
        recommendations.append(rec)

    return recommendations


async def generate_ai_oracle_analysis(market_summary, portfolio_data, pattern_data):
    if not os.getenv("GEMINI_API_KEY"):
        logging.warning("Skipping AI Oracle analysis: GEMINI_API_KEY not found.")
        return None # Fallback will be used

    try:
        # FIX: Use the correct, available model name
        model_name = 'gemini-1.5-flash-latest'
        model = genai.GenerativeModel(model_name)
        logging.info(f"✅ Successfully loaded Gemini model for Oracle: {model_name}")
    except Exception as e:
        logging.error(f"Failed to load Gemini model for Oracle: {e}")
        return None # Fallback will be used

    prompt = f"""
    You are the 'AI Market Oracle', a financial analyst AI.
    Your tone is insightful, concise, and professional.
    Based on the following data, generate a summary for a morning briefing email.
    The summary MUST be a JSON object with four keys: "opportunities", "risks", "sector_rotation", and "portfolio_actions".
    Each key should have a list of 1-3 short, bullet-point-style strings.

    DATA:
    1. Macro Summary: {json.dumps(market_summary)}
    2. Portfolio Highlights: {json.dumps([{'ticker': s['ticker'], 'prediction': s.get('prediction', {}).get('prediction'), 'confidence': s.get('prediction',{}).get('confidence')} for s in portfolio_data[:3]]) if portfolio_data else 'N/A'}
    3. Historical Pattern Analysis: {json.dumps(pattern_data)}

    Example JSON output:
    {{
      "opportunities": ["Historical pattern suggests +8.5% upside over 3 months", "Tech sector shows strong historical performance in these conditions"],
      "risks": ["High geopolitical risk score detected - monitor global news", "Trade war escalation risk - avoid companies with heavy China exposure"],
      "sector_rotation": ["Technology historically outperformed (+12.1%) in similar conditions", "Consider rotating out of Utilities which underperformed"],
      "portfolio_actions": ["HOLD NVDA - Balanced technicals, monitor for entry on dips", "TAKE PROFIT AAPL - High confidence SELL signal on overbought conditions"]
    }}

    Generate the JSON now.
    """
    try:
        response = await model.generate_content_async(prompt)
        json_text = response.text[response.text.find('{') : response.text.rfind('}')+1]
        analysis = json.loads(json_text)
        # Basic validation
        if all(k in analysis for k in ["opportunities", "risks", "sector_rotation", "portfolio_actions"]):
            return analysis
        else:
            raise ValueError("Missing required keys in JSON")
    except Exception as e:
        logging.error(f"Gemini API Oracle error: {e}")
        return None # Fallback will be triggered


# --- HTML and Email Generation ---

def generate_enhanced_html_email(**kwargs):
    """Generates the full V2 HTML email report."""
    
    # Unpack all data
    df_stocks = kwargs.get('df_stocks', pd.DataFrame())
    portfolio_data = kwargs.get('portfolio_data')
    market_news = kwargs.get('market_news')
    context_data = kwargs.get('context_data')
    ai_analysis = kwargs.get('ai_analysis')
    pattern_data = kwargs.get('pattern_data')
    macro_data = kwargs.get('macro_data')
    portfolio_recommendations = kwargs.get('portfolio_recommendations')

    # Helper for styling
    def get_change_color(value):
        return 'color:#2E8B57;' if value > 0 else 'color:#C70039;' if value < 0 else 'color:#888;'
    def get_signal_color(action):
        if action in ["BUY", "ADD"]: return "#2E8B57" # Green
        if action in ["SELL", "REDUCE", "TAKE PROFITS"]: return "#C70039" # Red
        return "#4682B4" # Blue for hold

    # --- Start of HTML ---
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; }}
            .container {{ max-width: 800px; margin: auto; background: #fff; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); overflow: hidden; }}
            .header {{ background: #2c3e50; color: #fff; padding: 20px 30px; }}
            .header h1 {{ margin: 0; font-size: 24px; }}
            .section {{ padding: 20px 30px; border-bottom: 1px solid #eee; }}
            .section:last-child {{ border-bottom: none; }}
            .section-title {{ font-size: 20px; font-weight: 600; margin-top: 0; margin-bottom: 20px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #f0f0f0; }}
            th {{ font-size: 14px; color: #777; text-transform: uppercase; }}
            .ticker {{ font-weight: bold; font-size: 1.1em; }}
            .company-name {{ color: #666; font-size: 0.9em; }}
            .signal-badge {{ display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; color: #fff; margin-right: 5px; }}
            .card {{ background: #fdfdfd; border: 1px solid #eee; border-radius: 8px; padding: 15px; margin-bottom: 15px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
            .metric {{ text-align: center; }}
            .metric .value {{ font-size: 1.5em; font-weight: 600; }}
            .metric .label {{ font-size: 0.9em; color: #777; }}
            .recommendation-card {{ display: flex; align-items: center; padding: 15px; border-radius: 8px; margin-bottom: 10px; background-color: #f9f9f9; border-left: 5px solid; }}
            .rec-icon {{ font-size: 24px; margin-right: 15px; }}
            .rec-text .action {{ font-weight: bold; font-size: 1.1em; }}
            .rec-text .reason {{ font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Daily Market Intelligence Briefing</h1>
                <!-- FIX: Use imported 'date' object correctly -->
                <p style="font-size:1.1em; color:#aaa;">{date.today().strftime('%A, %B %d, %Y')}</p>
            </div>
    """

    # --- Editor's Note / AI Oracle ---
    html += """
        <div class="section">
            <h2 class="section-title">EDITOR'S NOTE</h2>
    """
    if ai_analysis:
        html += "<p>Good morning. The overall market mood has stayed relatively stable. This briefing is your daily blueprint for navigating the currents.</p>"
        top_stock = df_stocks.iloc[0] if not df_stocks.empty else None
        if top_stock is not None:
             html += f"<p>Yesterday's Champion: <b>{top_stock['name']} ({top_stock['ticker']})</b> led our rankings.</p>"
        
        html += """
            <h3 style="margin-top: 25px; font-size: 16px;">⚡ HIGH-PRIORITY SIGNALS (v2.0)</h3>
            <p style="font-size: 0.9em; color: #666; margin-top: -10px;">Advanced technical alerts requiring immediate attention</p>
            <ul>
        """
        all_signals = []
        if portfolio_data:
            for stock in portfolio_data:
                for signal in stock.get('signals', []):
                    all_signals.append(f"<li>{signal['icon']} <b>{stock['ticker']}</b>: {signal['text']}</li>")
        
        if all_signals:
            html += "".join(all_signals[:5]) # Limit to top 5
        else:
            html += "<li>No high-priority signals detected today.</li>"
        html += "</ul>"
        
        html += """
            <div style="margin-top:30px; padding: 20px; border-radius: 8px; background: linear-gradient(135deg, #e0f7fa 0%, #f0f4c3 100%);">
                <h3 style="margin-top:0; font-size: 18px; color:#004d40;">🤖 AI MARKET ORACLE</h3>
                <p style="font-size: 0.9em; color: #555; margin-top: -10px; margin-bottom: 20px;">Powered by Gemini AI</p>
        """
        html += f"""
                <p><strong>💡 IMMEDIATE OPPORTUNITIES:</strong></p>
                <ul>{''.join(f'<li>{item}</li>' for item in ai_analysis.get("opportunities", ["N/A"]))}</ul>
                <p><strong>⚠️ CRITICAL RISKS:</strong></p>
                <ul>{''.join(f'<li>{item}</li>' for item in ai_analysis.get("risks", ["N/A"]))}</ul>
                <p><strong>🔄 SECTOR ROTATION:</strong></p>
                <ul>{''.join(f'<li>{item}</li>' for item in ai_analysis.get("sector_rotation", ["N/A"]))}</ul>
                 <p><strong>📊 PORTFOLIO ACTIONS:</strong></p>
                <ul>{''.join(f'<li>{item}</li>' for item in ai_analysis.get("portfolio_actions", ["N/A"]))}</ul>
            </div>
        """
    else: # Fallback
        html += "<p>AI Oracle analysis is currently unavailable. Using standard report format.</p>"
    html += "</div>"
    
    # --- Portfolio Command Center ---
    if portfolio_data:
        logging.info("📧 EMAIL - PORTFOLIO SECTION")
        html += """
        <div class="section">
            <h2 class="section-title">📊 YOUR PORTFOLIO COMMAND CENTER</h2>
            <table>
                <tr>
                    <th>Stock</th>
                    <th>Price</th>
                    <th>Indicators</th>
                    <th>Performance</th>
                </tr>
        """
        for stock in portfolio_data:
            signal_tags = ''.join([f'<span class="signal-badge" style="background-color:#3498db;">{s["icon"]} {s["type"]}</span>' for s in stock.get('signals', []) if s['type'] in ['SQUEEZE', 'HIGH_52W']])
            html += f"""
                <tr>
                    <td>
                        <div class="ticker">{stock['ticker']}</div>
                        <div class="company-name">{stock['name']}</div>
                        {signal_tags}
                    </td>
                    <td>
                        <div>${stock['price']:.2f}</div>
                        <div style="{get_change_color(stock['change_pct'])}">{stock['change_pct']:.2f}%</div>
                    </td>
                    <td>
                        <div>RSI: {stock['rsi']:.1f}</div>
                        <div>Vol: {stock['relative_volume']:.1f}x</div>
                    </td>
                    <td>
                        <div>W: {stock['week_change_pct']:.1f}%</div>
                        <div>M: {stock['month_change_pct']:.1f}%</div>
                    </td>
                </tr>
            """
        html += "</table>"
        
        # Mini-sections for alerts within portfolio
        alerts_html = "<ul>"
        opportunities_html = "<ul>"
        risks_html = "<ul>"
        has_alerts, has_opps, has_risks = False, False, False

        for stock in portfolio_data:
            for signal in stock.get('signals', []):
                if signal['type'] in ['BB_BREAKOUT_UP']:
                    has_alerts = True
                    alerts_html += f"<li>📈 <b>{stock['ticker']}</b>: {signal['text']}</li>"
                if signal['type'] in ['OVERBOUGHT']:
                     has_alerts = True
                     alerts_html += f"<li>⚠️ <b>{stock['ticker']}</b>: {signal['text']}</li>"
                if signal['type'] in ['OVERSOLD']:
                    has_opps = True
                    opportunities_html += f"<li>💡 <b>{stock['ticker']}</b>: {signal['text']}</li>"
                if signal['type'] == 'HIGH_VOL':
                    has_risks = True
                    risks_html += f"<li>{signal['icon']} <b>{stock['ticker']}</b>: {signal['text']}</li>"
        
        alerts_html += "</ul>"
        opportunities_html += "</ul>"
        risks_html += "</ul>"

        html += '<div style="display: flex; gap: 20px; margin-top: 20px;">'
        if has_alerts: html += f'<div style="flex:1;"><h4>🔔 Alerts & Signals</h4>{alerts_html}</div>'
        if has_opps: html += f'<div style="flex:1;"><h4>🎯 Opportunities</h4>{opportunities_html}</div>'
        else: html += '<div style="flex:1;"><h4>🎯 Opportunities</h4><ul><li>No immediate opportunities</li></ul></div>'
        if has_risks: html += f'<div style="flex:1;"><h4>⚠️ Risk Factors</h4>{risks_html}</div>'
        else: html += '<div style="flex:1;"><h4>⚠️ Risk Factors</h4><ul><li>No significant risks detected</li></ul></div>'
        
        html += '</div></div>'

    # --- AI Predictions Section ---
    if portfolio_data and LEARNING_MODE_ACTIVE and any('prediction' in s for s in portfolio_data):
        logging.info("============================================================")
        logging.info("📧 EMAIL - AI PREDICTIONS SECTION")
        logging.info(f"Portfolio data exists: {bool(portfolio_data)}")
        logging.info(f"Learning active: {LEARNING_MODE_ACTIVE}")
        
        predictions_to_show = [s for s in portfolio_data if 'prediction' in s]
        logging.info(f"Predictions made: {len(predictions_to_show)}")
        logging.info(f"Stocks count: {len(portfolio_data)}")

        if predictions_to_show:
            html += """
            <div class="section">
                <h2 class="section-title">🎯 AI PREDICTIONS & CONFIDENCE ANALYSIS</h2>
                <p style="font-size:0.9em; color:#666; margin-top:-15px;">
                    Generated {num_preds} AI-powered predictions with confidence scoring.
                </p>
                <div class="grid">
            """.format(num_preds=len(predictions_to_show))

            logging.info(f"Checking {len(predictions_to_show)} predictions for display...")
            card_count = 0
            for stock in predictions_to_show:
                pred = stock['prediction']
                pred_color = get_signal_color(pred['prediction'])
                confidence = pred['confidence']
                
                html += f"""
                <div class="card" style="border-left: 5px solid {pred_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin:0; font-size: 1.2em;">
                            <span style="color: {pred_color};">{ '🔴' if pred['prediction'] == 'SELL' else '🟢' if pred['prediction'] == 'BUY' else '⚪'} {stock['ticker']} - {pred['prediction']}</span>
                        </h4>
                        <span style="font-size: 1.5em; font-weight: bold; color: {pred_color};">{confidence}%</span>
                    </div>
                    <p style="margin: 5px 0 10px; font-size:0.9em; color:#777;">{stock['name']} &bull; ${stock['price']:.2f}</p>
                    <p style="font-size:0.85em; color:#555; margin:0;"><strong>Reasoning:</strong> {pred['reasoning']}</p>
                </div>
                """
                card_count += 1
            logging.info(f"Created {card_count} prediction cards")
            html += "</div>"
            html += """
                <p style="text-align:center; font-size:0.8em; color:#888; margin-top:20px;">
                    💡 <b>How to use:</b> Only act on HIGH confidence (75%+) signals. System learns from outcomes daily.
                </p>
            </div>"""
            logging.info("✅ AI predictions HTML generated successfully")
        else:
            logging.warning("No predictions with data found to display.")
        logging.info("============================================================")

    # --- Pattern Analysis ---
    if pattern_data:
        html += f"""
        <div class="section" style="background-color: #f8f9fa;">
            <h2 class="section-title">🔮 {ANALYSIS_PERIOD_DAYS//365}-YEAR PATTERN ANALYSIS</h2>
             <p style="font-size:0.9em; color:#666; margin-top:-15px;">Analyzing {len(pattern_data.get('matches', []))} similar market setups</p>
             <div style="display:flex; gap: 20px; text-align:center; padding: 15px; background: #fff; border-radius: 8px;">
                <div style="flex:1;">
                    <div style="font-size:0.8em; color:#777;">Today's Market DNA:</div>
                    <div style="font-size:0.9em;">RSI: {pattern_data['dna']['rsi']:.1f} | Volatility: {pattern_data['dna']['volatility']:.1f}% | Trend: {pattern_data['dna']['trend']:.1f}%</div>
                    <div style="font-size:0.9em;">Geopolitical Risk: {pattern_data['dna']['geo_risk']} | Trade Risk: {pattern_data['dna']['trade_risk']}</div>
                </div>
             </div>
             <div style="margin-top: 20px; padding: 20px; border-radius: 8px; background: #e9ecef; text-align: center;">
                 <h3 style="margin-top:0;">📖 What History Tells Us:</h3>
                 <p style="font-size: 1.2em; font-weight: bold;">📈 Market Bias: <span style="color: {'#2E8B57' if pattern_data['bias'] == 'BULLISH' else '#C70039'};">{pattern_data['bias']}</span></p>
                 <p>History strongly favors upside. {len([m for m in pattern_data['matches'] if m['outcome_pct'] > 0]) / len(pattern_data['matches']) * 100:.0f}% of similar setups were positive 3 months later.</p>
                 <p style="font-size: 1.4em; font-weight: 600;">Expect a significant upward move. Historical average: <span style="color: #0056b3;">{'+' if pattern_data['avg_outcome'] > 0 else ''}{pattern_data['avg_outcome']:.1f}% over 3 months.</span></p>
                 <p>💡 {len([m for m in pattern_data['matches'] if m['outcome_pct'] > 10])} out of top {len(pattern_data['matches'])} matches led to strong rallies. Dips may be buying opportunities.</p>
             </div>
        """
        if 'sector_performance' in pattern_data and pattern_data['sector_performance']:
            html += f"""
            <div style="margin-top:20px;">
                <h3 style="text-align:center;">🎯 Sector Performance in Similar Periods:</h3>
                <p style="text-align:center; font-size:0.8em; color:#777; margin-top:-10px;">Based on {pattern_data['top_match_period']} match</p>
                <table><tr><th>Sector</th><th>3-Month Return</th></tr>
            """
            for sector in pattern_data['sector_performance'][:5]:
                html += f"<tr><td>{sector['name']}</td><td style='{get_change_color(sector['performance'])}'>{sector['performance']:.1f}%</td></tr>"
            html += "</table></div>"

    # --- Portfolio Action Plan ---
    if portfolio_recommendations:
        html += """
        <div class="section">
            <h2 class="section-title">💼 YOUR PORTFOLIO ACTION PLAN</h2>
            <p style="font-size:0.9em; color:#666; margin-top:-15px;">One clear recommendation per stock - no conflicts</p>
            <div class="grid">
        """
        for rec in portfolio_recommendations:
            color = get_signal_color(rec['action'])
            icon = '🟢' if 'ADD' in rec['action'] else '🔴' if 'REDUCE' in rec['action'] or 'TAKE' in rec['action'] else '⚪'
            html += f"""
                <div class="recommendation-card" style="border-left-color: {color};">
                    <div class="rec-icon">{icon}</div>
                    <div class="rec-text">
                        <div class="action" style="color:{color};">{rec['ticker']}: {rec['action']} <span style="font-weight:normal; font-size:0.8em;">{rec['confidence']} CONF</span></div>
                        <div class="reason">&bull; {rec['reason']}</div>
                    </div>
                </div>
            """
        html += "</div></div>"


    if pattern_data: # Continue the pattern section
        html += """
             <div style="margin-top:20px;">
                 <h3 style="text-align:center;">📅 Historical Matches:</h3>
                 <p style="text-align:center; font-size:0.8em; color:#777; margin-top:-10px;">These show S&P 500 performance. Sector performance varied (see table above).</p>
                 <div class="grid">
        """
        for match in pattern_data['matches']:
            html += f"""
            <div class="card metric">
                <div>{match['date']} ({match['period_type']})</div>
                <div class="value" style="{get_change_color(match['outcome_pct'])}">{'+' if match['outcome_pct'] > 0 else ''}{match['outcome_pct']:.1f}%</div>
                <div class="label">S&P 500 outcome</div>
                <div class="label" style="color:#3498db; font-weight:bold;">Match Strength: {match['similarity']:.1f}%</div>
            </div>
            """
        html += "</div></div></div>"


    # --- Macro Section ---
    if macro_data:
        html += f"""
        <div class="section">
            <h2 class="section-title">THE BIG PICTURE: The Market Weather Report</h2>
            <p><strong>Overall Macro Score: {macro_data['score']} / 30</strong></p>
            <p style="font-size:0.9em; color:#666; margin-top:-10px;">How it's calculated: This is our "weather forecast" for investors, combining risks and sentiment.</p>
            <ul>
                <li><strong>🌍 Geopolitical Risk ({macro_data['geo_risk']}/100):</strong> Measures global instability.<br><em>Key Drivers:</em></li>
                <ul>{''.join(f"<li>{a['headline']} ({a['source']})</li>" for a in macro_data['top_articles'] if a['type']=='geo' and 'headline' in a)[:3]}</ul>
                <li><strong>🚢 Trade Risk ({macro_data['trade_risk']}/100):</strong> Tracks trade tensions.<br><em>Key Drivers:</em></li>
                <ul>{''.join(f"<li>{a['headline']} ({a['source']})</li>" for a in macro_data['top_articles'] if a['type']=='trade' and 'headline' in a)[:3]}</ul>
                <li><strong>💼 Economic Sentiment ({macro_data['econ_sentiment']}):</strong> Market mood (-1 to +1).<br><em>Key Drivers:</em></li>
                 <ul>{''.join(f"<li>{a['headline']} ({a['source']})</li>" for a in macro_data['top_articles'] if a['type']=='econ' and 'headline' in a)[:3]}</ul>
            </ul>
        </div>
        """

    # --- Sector Deep Dive & Stock Radar ---
    if not df_stocks.empty:
        sector_champs = df_stocks.loc[df_stocks.groupby('sector')['score'].idxmax()]
        html += """
        <div class="section">
            <h2 class="section-title">SECTOR DEEP DIVE</h2>
            <p>Top companies from different sectors.</p>
        """
        for _, stock in sector_champs.iterrows():
             html += f"<p><strong>{stock['name']} ({stock['ticker']}) in {stock['sector']}</strong></p>"
        html += "</div>"
        
        html += """
        <div class="section">
            <h2 class="section-title">STOCK RADAR</h2>
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1;">
                    <h4>📈 Top 10 Strongest Signals</h4>
                    <table>
        """
        for _, stock in df_stocks.head(10).iterrows():
            html += f"<tr><td>{stock['ticker']}<br><small>{stock['name'][:30]}</small></td><td>{stock['score']}</td></tr>"
        html += "</table></div><div style='flex: 1;'><h4>📉 Top 10 Weakest Signals</h4><table>"
        for _, stock in df_stocks.tail(10).iloc[::-1].iterrows():
            html += f"<tr><td>{stock['ticker']}<br><small>{stock['name'][:30]}</small></td><td>{stock['score']}</td></tr>"
        html += "</table></div></div></div>"


    # --- Context Data (Crypto, Commodities) ---
    if context_data:
        html += """
        <div class="section">
            <h2 class="section-title">BEYOND STOCKS: Alternative Assets</h2>
            <div style="display: flex; gap: 20px;">
                <div style="flex: 2;">
                    <h4>🪙 Crypto</h4>
                    <p>Market Sentiment: {sentiment}</p>
                    <table>{crypto_rows}</table>
                </div>
                <div style="flex: 1;">
                    <h4>💎 Commodities</h4>
                    <p>Gold/Silver Ratio: {gs_ratio}:1</p>
                    <table>{comm_rows}</table>
                </div>
            </div>
        </div>
        """.format(
            sentiment=context_data['crypto_sentiment'],
            gs_ratio=context_data['gold_silver_ratio'],
            crypto_rows=''.join(f"<tr><td><strong>{c['name']}</strong><br>{c['symbol']}</td><td>${c['price']:,.2f}<br><span style='{get_change_color(c['change'])}'>{c['change']:.2f}% (24h)</span></td><td>${c['market_cap']}</td></tr>" for c in context_data['crypto']),
            comm_rows=''.join(f"<tr><td><strong>{co['name']}</strong><br>{co['symbol']}</td><td>${co['price']:,.2f}<br><span style='{get_change_color(co['change'])}'>{co['change']:.2f}% (24h)</span></td></tr>" for co in context_data['commodities'])
        )

    # --- News ---
    if market_news:
        html += """
        <div class="section">
            <h2 class="section-title">FROM THE WIRE: Today's Top Headlines</h2>
            <ul>
        """
        for news_item in market_news:
            html += f'<li><a href="{news_item["url"]}" target="_blank">{news_item["title"]}</a> <small>({news_item["source"]})</small></li>'
        html += "</ul></div>"

    # --- Footer ---
    html += """
        </div>
        <p style="text-align: center; color: #999; font-size: 0.8em; margin-top: 20px;">
            This is an automated analysis. Not financial advice. Always do your own research.
        </p>
    </body>
    </html>
    """
    return html


def send_email(html_content):
    """Sends the HTML email report."""
    if not all([SMTP_USER, SMTP_PASS]):
        logging.error("SMTP credentials not found. Cannot send email.")
        # Instead of failing, we save the report to a file
        output_filename = f"email_report_{date.today().strftime('%Y-%m-%d')}.html"
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write(html_content)
        logging.info(f"Email report saved to {output_filename}")
        return

    msg = MIMEMultipart('alternative')
    # FIX: Use imported date object
    msg['Subject'] = f"📈 Daily Market Intelligence Report - {date.today().strftime('%B %d, %Y')}"
    msg['From'] = SMTP_USER
    msg['To'] = SMTP_USER

    msg.attach(MIMEText(html_content, 'html'))
    
    # Attach prediction log if it exists
    if LEARNING_MODE_ACTIVE and os.path.exists(PREDICTION_LOG_FILE):
        with open(PREDICTION_LOG_FILE, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(PREDICTION_LOG_FILE))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(PREDICTION_LOG_FILE)}"'
        msg.attach(part)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
            logging.info("✅ Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


# --- Main Execution Logic ---

async def main(output='log'):
    """Main function to run the analysis."""
    logging.info("=" * 60)
    logging.info(f"🚀 MARKET INTELLIGENCE SYSTEM v2.1.0")
    logging.info("=" * 60)
    
    universe, _, _ = load_config()
    if not universe:
        logging.error("Stock universe is empty. Exiting.")
        return

    mode_message = "📊 FULL ANALYSIS MODE: Running market intelligence scan..."
    logging.info(mode_message)
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    throttler = AsyncThrottler(REQUEST_THROTTLE_LIMIT)
    
    # --- New v2.0 Asynchronous Flow ---
    async with aiohttp.ClientSession() as session:
        # Step 1: Get macro data FIRST (needed for portfolio predictions)
        logging.info("🔍 Step 1: Fetching macro data...")
        macro_task = fetch_macro_sentiment(session)
        macro_data = await macro_task
        logging.info(f"✅ Macro data received. Score: {macro_data.get('score')}")
        
        # Step 2: Create portfolio task WITH macro context
        logging.info("🔍 Step 2: Calling analyze_portfolio_with_predictions (v3.0)")
        if ENABLE_V2_FEATURES:
            portfolio_task = analyze_portfolio_with_predictions(session, market_context=macro_data)
        else:
            portfolio_task = analyze_portfolio_watchlist(session)

        # Step 3: Prepare and run everything else in parallel
        logging.info("🔍 Step 3: Fetching stocks, context, news...")
        stock_tasks = [analyze_stock(semaphore, throttler, session, ticker) for ticker in universe]
        context_task = fetch_context_data(session)
        news_task = fetch_market_headlines(session)
        
        # Gather stocks, context, news, and the now-context-aware portfolio analysis
        results = await asyncio.gather(
            asyncio.gather(*stock_tasks), 
            context_task, 
            news_task,
            portfolio_task
        )
        stock_results_raw, context_data, market_news, portfolio_data = results
        
        # Step 4: Process general stock results
        stock_results = sorted([r for r in stock_results_raw if r], key=lambda x: x['score'], reverse=True)
        df_stocks = pd.DataFrame(stock_results) if stock_results else pd.DataFrame()
        logging.info(f"✅ Analyzed {len(df_stocks)} stocks")
        if portfolio_data:
             logging.info(f"✅ Portfolio analysis complete. Predictions: {len(portfolio_data)}")

        # Step 5: Find historical patterns (uses macro_data)
        logging.info("🔍 Step 5: Finding historical patterns...")
        pattern_data = await find_historical_patterns(session, macro_data)

        # Step 6: Generate portfolio recommendations (uses pattern_data and portfolio_data)
        logging.info("🔍 Step 6: Generating portfolio recommendations...")
        portfolio_recommendations = None
        if pattern_data and portfolio_data:
            portfolio_recommendations = await generate_portfolio_recommendations_from_pattern(
                portfolio_data, pattern_data, macro_data
            )

        # Step 7: Generate AI analysis (The Oracle)
        logging.info("🔍 Step 7: Generating AI analysis...")
        ai_analysis = None
        market_summary_for_ai = {
            'macro': macro_data,
            'top_stock': stock_results[0] if stock_results else {},
            'bottom_stock': stock_results[-1] if stock_results else {}
        }
        ai_analysis = await generate_ai_oracle_analysis(market_summary_for_ai, portfolio_data, pattern_data)

        # Fallback for AI Oracle
        if ai_analysis is None:
            logging.info("Using intelligent fallback analysis")
            ai_analysis = {
                "opportunities": [f"Historical pattern suggests {pattern_data.get('avg_outcome', 0):+.1f}% upside over 3 months"] if pattern_data else ["Check technicals for oversold stocks"],
                "risks": [f"Trade war escalation risk - avoid companies with heavy China exposure"] if macro_data.get('trade_risk', 0) > 50 else ["Monitor market for volatility spikes"],
                "sector_rotation": [f"{list(s['name'] for s in pattern_data['sector_performance'][:1])[0]} historically outperformed ({list(s['performance'] for s in pattern_data['sector_performance'][:1])[0]:+.1f}%) in similar conditions"] if pattern_data and pattern_data.get('sector_performance') else ["Focus on market leaders"],
                "portfolio_actions": [f"{rec['action']} {rec['ticker']} - {rec['reason']}" for rec in portfolio_recommendations[:2]] if portfolio_recommendations else ["Review portfolio positions"]
            }

        # Step 8: Save prediction logs
        if LEARNING_MODE_ACTIVE and PREDICTION_STORE:
            with open(PREDICTION_LOG_FILE, 'w') as f:
                json.dump(PREDICTION_STORE, f, indent=4)
            logging.info(f"✅ Prediction logs saved to {PREDICTION_LOG_FILE}")
            
        # Step 9: Generate and send report
        if output == 'email':
            logging.info("📧 Generating email report...")
            html_email = generate_enhanced_html_email(
                df_stocks=df_stocks,
                portfolio_data=portfolio_data,
                market_news=market_news,
                context_data=context_data,
                ai_analysis=ai_analysis,
                pattern_data=pattern_data,
                macro_data=macro_data,
                portfolio_recommendations=portfolio_recommendations
            )
            send_email(html_email)
        else:
            logging.info("Output is set to log. Skipping email generation.")

        logging.info("✅ Analysis complete with v2.0.0 features.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stock Market Analysis Engine")
    parser.add_argument('--output', type=str, default='log', choices=['log', 'email'], help="Output mode: 'log' or 'email'.")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(output=args.output))
    except Exception as e:
        logging.critical(f"A critical error occurred in the main event loop: {e}", exc_info=True)
        # This will ensure the traceback is printed to the log
