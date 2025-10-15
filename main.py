# main.py
import os, sys, argparse, time, math
import requests, datetime, yaml
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
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
    table = soup.find("table", {"id":"constituents"})
    df = pd.read_html(str(table))[0]
    return df.Symbol.tolist()

def fetch_tsx_tickers_local():
    # For TSX, shipping a static list is safer. You can maintain tsx_tickers.csv in repo.
    local = "tsx_tickers.csv"
    if os.path.exists(local):
        df = pd.read_csv(local)
        return df['symbol'].astype(str).tolist()
    else:
        # fallback: common large TSX names (useful to start)
        return ["RY.TO","TD.TO","ENB.TO","BNS.TO","CNQ.TO","TRP.TO","SHOP.TO","BCE.TO"]

def get_price_df(tickers):
    data = yf.download(tickers, period="6mo", interval="1d", threads=True, progress=False)
    # When multiple tickers, structure data accordingly
    if isinstance(tickers, str):
        ohlc = data[['Open','High','Low','Close','Volume']].copy()
        ohlc.columns = pd.MultiIndex.from_product([[tickers], ohlc.columns])
    return data

def compute_technical_indicators(df_close):
    # df_close: pd.Series
    out = {}
    series = df_close.dropna()
    if len(series) < 35:
        return None
    df = pd.DataFrame({"close": series})
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df = pd.concat([df, macd], axis=1)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)
    # swing high/low simplified
    df['swing_high'] = df['close'][(df['close'].shift(1) < df['close']) & (df['close'].shift(-1) < df['close'])]
    df['swing_low'] = df['close'][(df['close'].shift(1) > df['close']) & (df['close'].shift(-1) > df['close'])]
    out['latest'] = df.iloc[-1].to_dict()
    out['rsi'] = float(df['rsi_14'].iloc[-1])
    out['macd'] = float(df['MACD_12_26_9'].iloc[-1]) if 'MACD_12_26_9' in df.columns else None
    out['ema20_vs_close'] = float(df['ema20'].iloc[-1]) - float(df['close'].iloc[-1])
    return out

def news_headlines(query, max_results=10):
    if not NEWSAPI_KEY:
        return []
    url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&pageSize={max_results}&apiKey={NEWSAPI_KEY}"
    r = requests.get(url, timeout=10).json()
    return r.get('articles', [])

def headline_sentiment(headline):
    s = analyzer.polarity_scores(headline)
    return s

# ---------- scoring ----------
def score_stock(fund, tech, sentiment_score):
    # fund: dict with keys: pe, debt_to_equity, revenue_3yr, eps_growth
    # tech: dict from compute_technical_indicators
    score = 0.0
    # fundamentals (40)
    if fund.get('pe') and fund['pe']>0 and fund['pe']<40:
        score += 20
    if fund.get('de_ratio') and fund['de_ratio'] < 1.5:
        score += 10
    if fund.get('revenue_3y_growth') and fund['revenue_3y_growth']>0.1:
        score += 10
    # technical (20)
    if tech:
        if tech.get('rsi') and tech['rsi'] < 70 and tech['rsi'] > 40:
            score += 10
        if tech.get('macd') and tech['macd'] > 0:
            score += 10
    # growth (20)
    if fund.get('eps_growth_3y',0) > 0.1:
        score += 15
    # sentiment (10)
    score += max(min((sentiment_score*10), 10), -10)  # scale vader compound [-1,1] to [-10,10]
    # macro/sector (10) -> neutral now
    score += 5
    return score

# ---------- main orchestration ----------
def main(output='email'):
    sp500 = fetch_sp500_tickers()
    tsx = fetch_tsx_tickers_local()
    universe = (sp500[:200]) + tsx[:100]   # to keep below API limits; you can expand
    print(f"Running for {len(universe)} tickers (sampled).")

    results = []
    for ticker in universe:
        try:
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period="1y", interval="1d")
            if hist is None or hist.empty:
                continue
            close = hist['Close']
            tech = compute_technical_indicators(close)
            info = yf_ticker.info
            fund = {
                'pe': info.get('trailingPE'),
                'de_ratio': info.get('debtToEquity') or info.get('debtToEquity'),
                'marketCap': info.get('marketCap'),
                'eps': info.get('trailingEps'),
                'eps_growth_3y': info.get('earningsQuarterlyGrowth') or 0,
                'revenue_3y_growth': None
            }
            # sentiment via headlines
            articles = news_headlines(ticker, max_results=5)
            comp = 0
            for a in articles:
                s = headline_sentiment(a.get('title','') + ' ' + (a.get('description') or ''))
                comp += s['compound']
            avg_sent = (comp/len(articles)) if articles else 0
            score = score_stock(fund, tech, avg_sent)
            results.append({
                'ticker': ticker,
                'score': score,
                'fund': fund,
                'tech': tech,
                'sentiment': avg_sent
            })
            time.sleep(0.25)
        except Exception as e:
            print("err", ticker, e)

    df = pd.DataFrame(results).sort_values('score', ascending=False)
    top15 = df.head(15)
    bottom15 = df.tail(15)

    # sector summary (simple)
    sectors = {'strength': ['Industrials','Financials'], 'weakness': ['Information Technology']}
    # news summary (top market headlines)
    market_news = []
    mnews = news_headlines("stock market OR equities OR S&P 500 OR TSX", max_results=6)
    for a in mnews:
        market_news.append({'title': a.get('title'), 'source': a.get('source',{}).get('name'), 'url': a.get('url')})

    # portfolio suggestion: moderate risk
    portfolio = {
        'equities': 0.6,
        'bonds_or_cash': 0.2,
        'gold_silver_crypto_mini': 0.1,
        'high_quality_cyclicals': 0.1
    }

    # create markdown
    md = []
    md.append(f"# Daily Market Report — {datetime.datetime.utcnow().date()}\n")
    md.append("## Top 15 (by composite score)\n")
    md.append(top15[['ticker','score']].to_markdown(index=False))
    md.append("\n## Bottom 15 (by composite score)\n")
    md.append(bottom15[['ticker','score']].to_markdown(index=False))
    md.append("\n## Sector snapshot\n")
    md.append(str(sectors))
    md.append("\n## Market headlines\n")
    for n in market_news:
        md.append(f"- **{n['title']}** — {n['source']}")
    md.append("\n## Portfolio (moderate risk) suggestion\n")
    md.append(str(portfolio))
    report_md = "\n\n".join(md)

    # Output options
    if output == 'email':
        send_email(report_md)
    elif output == 'slack':
        send_slack(report_md)
    else:
        print(report_md)
    # also save
    with open("daily_report.md","w") as f:
        f.write(report_md)
    print("Done.")

def send_email(body):
    import smtplib
    from email.message import EmailMessage
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    if not SMTP_USER or not SMTP_PASS:
        print("SMTP creds missing; printing instead.\n")
        print(body)
        return
    msg = EmailMessage()
    msg['Subject'] = f"Daily Market Report {datetime.date.today()}"
    msg['From'] = SMTP_USER
    msg['To'] = SMTP_USER
    msg.set_content(body)
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(SMTP_USER, SMTP_PASS)
    s.send_message(msg)
    s.quit()
    print("Email sent.")

def send_slack(body):
    url = os.getenv("SLACK_WEBHOOK")
    if not url:
        print(body)
        return
    requests.post(url, json={"text": body[:3000]})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='print', choices=['print','email','slack'])
    args = parser.parse_args()
    main(output=args.output)
