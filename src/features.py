import pandas as pd
import numpy as np

def returns(series):
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

    # Basit RSI
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-12)
    out["rsi"] = 100 - (100 / (1 + rs))

    # SMA’lar ve QQQ trend
    out["sma_fast"] = out["close"].rolling(sma_fast).mean()
    out["sma_slow"] = out["close"].rolling(sma_slow).mean()
    out["qqq_trend_up"] = ((out[qqq_col].rolling(3).mean() - out[qqq_col].rolling(10).mean()) > 0).astype(int) if qqq_col in out else 0

    # BTC <-> QQQ korelasyon
    if qqq_col in out:
        out["corr_qqq"] = rolling_corr(out["close"], out[qqq_col], window=corr_window)
    else:
        out["corr_qqq"] = np.nan

    # Sentiment smoothing (varsa)
    if "sentiment" in out:
        out["sentiment_smooth"] = out["sentiment"].rolling(24, min_periods=1).mean()

    # ATR & ATR yüzdesi
    out["atr"] = atr(out["high"], out["low"], out["close"], n=14)
    out["atr_pct"] = out["atr"] / out["close"]

    out["ret1"] = returns(out["close"])
    return out
