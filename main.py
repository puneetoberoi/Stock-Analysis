import os, sys, argparse, time, datetime
import requests, math
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- config ----------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")

analyzer = SentimentIntensityAnalyzer()

# ---------- helpers ----------
def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, timeout=20)
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    df = pd.read_html(str(table))[0]
    return df.Symbol.tolist()

def fetch_tsx_tickers_local():
    local = "tsx_tickers.csv"
    if os.path.exists(local):
        df = pd.read_csv(local)
        return df["symbol"].astype(str).tolist()
    else:
        return ["RY.TO", "TD.TO", "ENB.TO", "BNS.TO", "CNQ.TO", "TRP.TO", "SHOP.TO", "BCE.TO"]

def compute_technical_indicators(series):
    """Compute RSI, MACD, and EMAs using ta library"""
    series = series.dropna()
    if len(series) < 50:
        return None
    df = pd.DataFrame({"close": series})

    # RSI
    rsi = RSIIndicator(df["close"], window=14).rsi()
    df["rsi_14"] = rsi

    # MACD
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()

    # EMAs
    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = EMAIndicator(df["close"], window=200).ema_indicator()

    # Simple swing detection
    df["swing_high"] = df["close"][(df["close"].shift(1) < df["close"]) & (df["close"].shift(-1) < df["close"])]
    df["swing_low"] = df["close"][(df["close"].shift(1) > df["close"]) & (df["close"].shift(-1) > df["close"])]

    out = {
        "rsi": float(df["rsi_14"].iloc[-1]),
        "macd": float(df["macd"].iloc[-1]),
        "macd_signal": float(df["macd_signal"].iloc[-1]),
        "ema20_vs_close": float(df["ema20"].iloc[-1]) - float(df["close"].iloc[-1]),
        "ema50_vs_close": float(df["ema50"].iloc[-1]) - float(df["close"].iloc[-1]),
        "ema200_vs_close": float(df["ema200"].iloc[-1]) - float(df["close"].iloc[-1]),
    }
    return out

def news_headlines(query, max_results=10):
    if not NEWSAPI_KEY:
        return []
    url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&pageSize={max_results}&apiKey={NEWSAPI_KEY}"
    try:
        r = requests.get(url, timeout=10).json()
        return r.get("articles", [])
    except Exception:
        return []

def headline_sentiment(headline):
    s = analyzer.polarity_scores(headline)
    return s

# ---------- scoring ----------
def score_stock(fund, tech, sentiment_score):
    score = 0.0
    # fundamentals
    if fund.get("pe") and 0 < fund["pe"] < 40:
        score += 20
    if fund.get("de_ratio") and fund["de_ratio"] < 1.5:
        score += 10
    if fund.get("revenue_3y_growth") and fund["revenue_3y_growth"] > 0.1:
        score += 10
    # technicals
    if tech:
        if tech.get("rsi") and 40 < tech["rsi"] < 70:
            score += 10
        if tech.get("macd") and tech["macd"] > tech.get("macd_signal", 0):
            score += 10
    # growth
    if fund.get("eps_growth_3y", 0) > 0.1:
        score += 15
    # sentiment
    score += max(min((sentiment_score * 10), 10), -10)
    # macro/sector placeholder
    score += 5
    return score

# ---------- main ----------
def main(output="email"):
    sp500 = fetch_sp500_tickers()
    tsx = fetch_tsx_tickers_local()
    universe = sp500[:200] + tsx[:100]
    print(f"Running for {len(u
