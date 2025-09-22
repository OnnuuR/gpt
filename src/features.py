import pandas as pd
import numpy as np

def returns(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.pct_change(fill_method=None)

def rolling_corr(a: pd.Series, b: pd.Series, window: int = 72) -> pd.Series:
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    ar = a_num.pct_change(fill_method=None)
    br = b_num.pct_change(fill_method=None)
    # min_periods'i çok küçük tutmayalım ki ilk değerlerde NaN normal olsun
    return ar.rolling(window, min_periods=max(5, window // 3)).corr(br)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_core_features(
    df: pd.DataFrame,
    rsi_period: int = 14,
    sma_fast: int = 20,
    sma_slow: int = 50,
    corr_window: int = 72,
    qqq_col: str = "qqq_close",
) -> pd.DataFrame:
    out = df.copy()

    # Temel fiyat kolonlarını numeriğe zorla
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # RSI
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-12)
    out["rsi"] = 100 - (100 / (1 + rs))

    # SMA'lar
    out["sma_fast"] = out["close"].rolling(sma_fast).mean()
    out["sma_slow"] = out["close"].rolling(sma_slow).mean()

    # QQQ trend & korelasyon (varsa)
    if qqq_col in out.columns:
        out[qqq_col] = pd.to_numeric(out[qqq_col], errors="coerce")
        fast_q = out[qqq_col].rolling(3).mean()
        slow_q = out[qqq_col].rolling(10).mean()
        trend = (fast_q - slow_q) > 0
        out["qqq_trend_up"] = trend.astype(float).fillna(0).astype(int)
        out["corr_qqq"] = rolling_corr(out["close"], out[qqq_col], window=corr_window)
    else:
        out["qqq_trend_up"] = 0
        out["corr_qqq"] = np.nan

    # Sentiment smoothing (varsa)
    if "sentiment" in out.columns:
        out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce")
        out["sentiment_smooth"] = out["sentiment"].rolling(24, min_periods=1).mean()

    # ATR & ATR%
    out["atr"] = atr(out["high"], out["low"], out["close"], n=14)
    out["atr_pct"] = out["atr"] / out["close"]

    # Basit getiriler
    out["ret1"] = returns(out["close"])
    return out
