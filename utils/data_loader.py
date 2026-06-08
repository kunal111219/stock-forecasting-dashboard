# import yfinance as yf
# import pandas as pd
# import streamlit as st


# @st.cache_data(ttl=3600)
# def search_ticker(query):
#     """
#     Dynamically search for any stock ticker worldwide.
#     No hardcoded list — works for any exchange.
#     Returns: (ticker_symbol, company_name)
#     """
#     try:
#         # Try direct ticker (US stocks — AAPL, TSLA, MSFT)
#         ticker = yf.Ticker(query.upper())
#         info = ticker.info
#         if info.get('regularMarketPrice') or info.get('currentPrice'):
#             return query.upper(), info.get('longName', query.upper())

#         # Try with .NS suffix (NSE Indian stocks — TCS.NS, INFY.NS)
#         ticker_ns = yf.Ticker(query.upper() + ".NS")
#         info_ns = ticker_ns.info
#         if info_ns.get('regularMarketPrice') or info_ns.get('currentPrice'):
#             return query.upper() + ".NS", info_ns.get('longName', query.upper())

#         # Try with .BO suffix (BSE Indian stocks)
#         ticker_bo = yf.Ticker(query.upper() + ".BO")
#         info_bo = ticker_bo.info
#         if info_bo.get('regularMarketPrice') or info_bo.get('currentPrice'):
#             return query.upper() + ".BO", info_bo.get('longName', query.upper())

#         return None, None

#     except Exception:
#         return None, None


# @st.cache_data(ttl=300)  # Refresh every 5 minutes
# def fetch_stock_data(ticker, period):
#     """
#     Fetch live OHLCV data from Yahoo Finance.
#     All metrics calculated dynamically — nothing hardcoded.
#     Returns: (dataframe, stock_info_dict)
#     """
#     try:
#         stock = yf.Ticker(ticker)
#         df = stock.history(period=period)

#         if df.empty:
#             return None, None

#         # Keep only OHLCV columns
#         df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
#         df.index = pd.to_datetime(df.index).tz_localize(None)

#         info = stock.info
#         return df, info

#     except Exception:
#         return None, None


import os
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
from requests import Session

# ─────────────────────────────────────────────────────────────
# Load API keys from .env — NEVER hardcode keys in code!
# ─────────────────────────────────────────────────────────────
load_dotenv()
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
FINNHUB_BASE = "https://finnhub.io/api/v1"

# ─────────────────────────────────────────────────────────────
# STRATEGY:
# yfinance   → Historical OHLCV (free, unlimited, no API key)
# Finnhub    → Real-time price + News sentiment (60 req/min)
# ─────────────────────────────────────────────────────────────

INDIAN_STOCKS = {
    "TCS": "TCS.NS", "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS", "WIPRO": "WIPRO.NS",
    "HDFCBANK": "HDFCBANK.NS", "ICICIBANK": "ICICIBANK.NS",
    "TATAMOTORS": "TATAMOTORS.NS", "BAJFINANCE": "BAJFINANCE.NS",
    "HINDUNILVR": "HINDUNILVR.NS", "SBIN": "SBIN.NS",
    "AXISBANK": "AXISBANK.NS", "MARUTI": "MARUTI.NS",
    "SUNPHARMA": "SUNPHARMA.NS", "HCLTECH": "HCLTECH.NS",
    "TECHM": "TECHM.NS", "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS", "ONGC": "ONGC.NS",
    "NESTLEIND": "NESTLEIND.NS", "POWERGRID": "POWERGRID.NS",
}

# Bypass Yahoo Finance bot detection
def _get_yf_session():
    session = Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session

def resolve_ticker(query):
    """
    Resolve any stock query to correct yfinance ticker.
    Handles Indian stocks (.NS), US stocks, and common names.
    """
    query = query.strip().upper()

    # Already has exchange suffix
    if query.endswith('.NS') or query.endswith('.BO'):
        return query

    # Known Indian stock
    if query in INDIAN_STOCKS:
        return INDIAN_STOCKS[query]

    # Default — try as-is (works for US stocks)
    return query


def is_indian(ticker):
    """Check if ticker is an Indian stock."""
    return '.NS' in ticker or '.BO' in ticker or \
           ticker in INDIAN_STOCKS


@st.cache_data(ttl=3600)
def search_ticker(query):
    """
    Search and validate any stock ticker.
    Uses yfinance fast_info to verify ticker exists.
    Returns: (ticker, company_name, source)
    """
    query = query.strip().upper()
    ticker = resolve_ticker(query)

    # Try resolving with yfinance
    suffixes_to_try = [ticker]
    if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
        suffixes_to_try += [ticker + '.NS', ticker + '.BO']

    for symbol in suffixes_to_try:
        try:
            stock = yf.Ticker(symbol)
            # Use fast_info — much less likely to hit rate limits
            fast = stock.fast_info
            price = getattr(fast, 'last_price', None)

            if price and float(price) > 0:
                # Get company name
                try:
                    name = stock.info.get('longName') or \
                           stock.info.get('shortName') or symbol
                except Exception:
                    name = symbol

                # Determine source for real-time
                source = "finnhub" if not is_indian(symbol) \
                         else "yfinance"
                return symbol, name, source

            time.sleep(0.3)

        except Exception:
            time.sleep(0.5)
            continue

    return None, None, None

@st.cache_data(ttl=300)
def fetch_historical_yfinance(ticker, period):
    try:
        session = _get_yf_session()
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period=period)

        if df.empty:
            return None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        return df

    except Exception as e:
        st.error(f"❌ History fetch error: {e}")
        return None
# @st.cache_data(ttl=300)
# def fetch_historical_yfinance(ticker, period):
#     """
#     Fetch historical OHLCV using yfinance.
#     Works for both Indian and US stocks.
#     Free, no API key, no daily limits.
#     """
#     try:
#         stock = yf.Ticker(ticker)
#         df = stock.history(period=period)

#         if df.empty:
#             return None

#         df = df[['Open', 'High', 'Low',
#                  'Close', 'Volume']].dropna()
#         df.index = pd.to_datetime(df.index).tz_localize(None)
#         df = df.sort_index()
#         return df

#     except Exception as e:
#         st.error(f"❌ History fetch error: {e}")
#         return None


@st.cache_data(ttl=30)  # Real-time — refresh every 30 seconds
def fetch_realtime_finnhub(ticker):
    """
    Get real-time quote from Finnhub.
    Only for US stocks — Indian stocks use yfinance price.
    60 requests/minute free.
    """
    if not FINNHUB_KEY:
        return None

    # Finnhub doesn't support Indian stocks
    if is_indian(ticker):
        return None

    try:
        r = requests.get(
            f"{FINNHUB_BASE}/quote",
            params={"symbol": ticker, "token": FINNHUB_KEY},
            timeout=10
        )
        data = r.json()

        # Validate response
        if not data.get("c") or data["c"] == 0:
            return None

        return {
            "current":    data.get("c", 0),
            "change":     data.get("d", 0),
            "pct_change": data.get("dp", 0),
            "high":       data.get("h", 0),
            "low":        data.get("l", 0),
            "open":       data.get("o", 0),
            "prev_close": data.get("pc", 0)
        }

    except Exception:
        return None


@st.cache_data(ttl=1800)  # Refresh every 30 minutes
def fetch_news_sentiment(ticker):
    """
    Fetch company news + sentiment from Finnhub.
    Only works reliably for US stocks.
    Returns: (news_list, sentiment_dict)
    """
    if not FINNHUB_KEY or is_indian(ticker):
        return [], {}

    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() -
                timedelta(days=7)).strftime("%Y-%m-%d")

    # Clean ticker (remove any suffix)
    clean = ticker.split('.')[0]

    try:
        # News articles
        r1 = requests.get(
            f"{FINNHUB_BASE}/company-news",
            params={
                "symbol": clean,
                "from": week_ago,
                "to": today,
                "token": FINNHUB_KEY
            },
            timeout=10
        )
        news = r1.json() if r1.status_code == 200 else []

        # Sentiment score
        r2 = requests.get(
            f"{FINNHUB_BASE}/news-sentiment",
            params={"symbol": clean, "token": FINNHUB_KEY},
            timeout=10
        )
        sentiment = r2.json() if r2.status_code == 200 else {}

        return news[:10], sentiment

    except Exception:
        return [], {}


def fetch_stock_data(ticker, period, source):
    """
    Unified data fetcher.
    Historical: always yfinance (free, reliable)
    Real-time: Finnhub for US, yfinance for Indian
    Returns: (df, info, realtime)
    """
    # Always use yfinance for historical data
    df = fetch_historical_yfinance(ticker, period)

    if df is None or df.empty:
        return None, None, None

    # Real-time price
    realtime = None
    if source == "finnhub":
        realtime = fetch_realtime_finnhub(ticker)

    # Build info dict
    currency = "INR" if is_indian(ticker) else "USD"
    current_price = (realtime["current"]
                     if realtime and realtime.get("current")
                     else df['Close'].iloc[-1])

    info = {
        "longName": ticker,
        "currency": currency,
        "currentPrice": current_price,
        "source": "Finnhub + yfinance" if source == "finnhub"
                  else "yfinance"
    }

    # Try to get company name
    try:
        stock = yf.Ticker(ticker)
        name = (stock.info.get('longName') or
                stock.info.get('shortName') or ticker)
        info["longName"] = name
    except Exception:
        pass

    return df, info, realtime