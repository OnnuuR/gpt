\
import numpy as np
import pandas as pd

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def sma(series, n):
    return series.rolling(n).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def returns(series):
    # pandas FutureWarning'u için:
    return series.pct_change(fill_method=None)

def rolling_corr(a, b, window=72):
    ar = a.pct_change(fill_method=None)
    br = b.pct_change(fill_method=None)
    return ar.rolling(window).corr(br)

def atr(high, low, close, n=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def add_core_features(df, rsi_period=14, sma_fast=20, sma_slow=50, corr_window=72, qqq_col="qqq_close"):
    out = df.copy()
    out["rsi"] = rsi(out["close"], rsi_period)
    out["sma_fast"] = sma(out["close"], sma_fast)
    out["sma_slow"] = sma(out["close"], sma_slow)
    if qqq_col in out.columns:
        out["qqq_ema"] = ema(out[qqq_col], 50)
        out["corr_qqq"] = rolling_corr(out["close"], out[qqq_col], corr_window)
        out["qqq_trend_up"] = (out[qqq_col] > out["qqq_ema"]).astype(int)
    else:
        out["corr_qqq"] = 0.0
        out["qqq_trend_up"] = 0
    if "sentiment" in out.columns:
        out["sentiment_smooth"] = out["sentiment"].rolling(6).mean()
    if "fng" in out.columns:
        out["fng_z"] = (out["fng"] - out["fng"].rolling(30).mean()) / (out["fng"].rolling(30).std() + 1e-9)
    out["atr"] = atr(out["high"], out["low"], out["close"], n=14)
    out["atr_pct"] = out["atr"] / out["close"]
    out["ret1"] = returns(out["close"])
    return out
