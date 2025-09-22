\
import os, time, math, json, datetime as dt
import pandas as pd
import numpy as np
import requests
import ccxt
import yfinance as yf
import feedparser
from dateutil import tz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .utils import get_logger, retry

UTC = tz.gettz("UTC")

def to_utc(ts):
    if isinstance(ts, (int, float, np.integer, np.floating)):
        return dt.datetime.utcfromtimestamp(ts/1000 if ts > 1e12 else ts).replace(tzinfo=UTC)
    if isinstance(ts, dt.datetime):
        return ts.astimezone(UTC).replace(tzinfo=UTC)
    return pd.to_datetime(ts, utc=True).to_pydatetime()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# --- Binance via ccxt: OHLCV ---
@retry((ccxt.NetworkError, ccxt.ExchangeError, requests.RequestException), tries=5, delay=1.0)
def fetch_ohlcv_ccxt(symbol="BTC/USDT", timeframe="1h", since_ms=None, limit=1000):
    ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    data = []
    fetch_since = since_ms
    while True:
        chunk = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
        if not chunk:
            break
        data.extend(chunk)
        if len(chunk) < limit:
            break
        fetch_since = chunk[-1][0] + 1
        time.sleep(ex.rateLimit / 1000)
    cols = ["timestamp","open","high","low","close","volume"]
    df = pd.DataFrame(data, columns=cols)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("time").drop(columns=["timestamp"])
    return df

# --- Binance Futures endpoints: funding & OI ---
BINANCE_FAPI = "https://fapi.binance.com"

def _time_chunks(start_ms, end_ms, step_ms):
    cur = start_ms
    while cur < end_ms:
        yield cur, min(end_ms, cur + step_ms - 1)
        cur += step_ms

@retry((requests.RequestException,), tries=5, delay=1.0)
def fetch_funding(symbol="BTCUSDT", start_ms=None, end_ms=None, limit=1000):
    # /fapi/v1/fundingRate ? symbol & startTime & endTime & limit
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    out = []
    if start_ms is None or end_ms is None:
        # last n records
        r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=20)
        r.raise_for_status()
        out.extend(r.json())
    else:
        # chunk by ~1 month
        step = 30 * 24 * 3600 * 1000
        for s, e in _time_chunks(start_ms, end_ms, step):
            r = requests.get(url, params={"symbol": symbol, "startTime": s, "endTime": e, "limit": limit}, timeout=20)
            if r.status_code == 429:
                time.sleep(1.0)
                r = requests.get(url, params={"symbol": symbol, "startTime": s, "endTime": e, "limit": limit}, timeout=20)
            r.raise_for_status()
            arr = r.json()
            if not arr:
                continue
            out.extend(arr)
            time.sleep(0.2)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df.set_index("time")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["fundingRate"]].sort_index()

@retry((requests.RequestException,), tries=5, delay=1.0)
def fetch_oi(symbol="BTCUSDT", period="4h", start_ms=None, end_ms=None, limit=500):
    # /futures/data/openInterestHist ? symbol & period(5m,15m,30m,1h,2h,4h,6h,12h,1d) & startTime & endTime & limit
    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    if period not in ["5m","15m","30m","1h","2h","4h","6h","12h","1d"]:
        raise ValueError("invalid period")
    out = []
    if start_ms is None or end_ms is None:
        r = requests.get(url, params={"symbol": symbol, "period": period, "limit": limit}, timeout=20)
        r.raise_for_status()
        out.extend(r.json())
    else:
        # chunk by at most 30 days for safety
        step = 30 * 24 * 3600 * 1000
        for s, e in _time_chunks(start_ms, end_ms, step):
            params = {"symbol": symbol, "period": period, "limit": limit, "startTime": s, "endTime": e}
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 400:
                # try tighter window if server complains
                step = 7 * 24 * 3600 * 1000
                for s2, e2 in _time_chunks(s, e, step):
                    r2 = requests.get(url, params={"symbol": symbol, "period": period, "limit": limit, "startTime": s2, "endTime": e2}, timeout=20)
                    r2.raise_for_status()
                    out.extend(r2.json())
                    time.sleep(0.25)
                continue
            r.raise_for_status()
            out.extend(r.json())
            time.sleep(0.25)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("time")
    df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    return df[["sumOpenInterest"]].sort_index()

# --- Yahoo Finance: QQQ, DXY ---
def fetch_yf(ticker="QQQ", period="120d", interval="1h", auto_adjust=True):
    df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=auto_adjust, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=str.lower)
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close":"close"})
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["open","high","low","close","volume"]]

# --- Sentiment via RSS + VADER ---
def fetch_rss_sentiment(feeds, hours=24, logger=None):
    sid = SentimentIntensityAnalyzer()
    now = pd.Timestamp.utcnow()
    rows = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:200]:
                title = e.get("title","")
                published = e.get("published_parsed") or e.get("updated_parsed")
                if published:
                    t = pd.Timestamp.fromtimestamp(time.mktime(published), tz="UTC")
                else:
                    t = now
                if now - t > pd.Timedelta(hours=hours):
                    continue
                s = sid.polarity_scores(title).get("compound", 0.0)
                rows.append({"time": t, "source": url, "title": title, "compound": s})
        except Exception as ex:
            if logger: logger.warning(f"RSS error {url}: {ex}")
    if not rows:
        return pd.DataFrame(columns=["compound"])
    df = pd.DataFrame(rows).set_index("time").sort_index()
    # hourly resample mean
    return df["compound"].resample("1H").mean().to_frame("sentiment")

# --- Fear & Greed (Alternative.me) ---
def fetch_fng(days=90):
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=15)
        r.raise_for_status()
        arr = r.json().get("data", [])
        rows = []
        for a in arr:
            t = pd.to_datetime(a["timestamp"], unit="s", utc=True)
            rows.append({"time": t, "fng": float(a["value"])})
        df = pd.DataFrame(rows).set_index("time").sort_index()
        df = df.iloc[-days:]
        return df
    except Exception:
        return pd.DataFrame(columns=["fng"])

def merge_align(dfs, how="outer"):
    out = None
    for d in dfs:
        if d is None or d.empty: 
            continue
        if out is None:
            out = d.copy()
        else:
            out = out.join(d, how=how)
    if out is None:
        return pd.DataFrame()
    out = out.sort_index().drop_duplicates()
    out = out.ffill().bfill()
    return out

