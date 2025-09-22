from __future__ import annotations

import math
from typing import Tuple, Any, Optional, Sequence, Dict

import numpy as np
import pandas as pd


def _get(row: pd.Series, key: str, default: Any = None) -> Any:
    """Safe getter for row values with NaN handling."""
    if key in row and pd.notna(row[key]):
        return row[key]
    return default


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _score_long(row: pd.Series, rules: Dict[str, Any]) -> float:
    """
    Uzun (long) yön puanı — trend yukarı, momentum ve risk iştahıyla artar,
    aşırı/çok düşük volatiliteyle azalır.
    """
    rsi_buy = _as_float(rules.get("rsi_buy", 52.0))
    require_qqq_up = bool(rules.get("require_qqq_up", False))
    min_corr_qqq = _as_float(rules.get("min_corr_qqq", -1.0))
    atr_min_pct = _as_float(rules.get("atr_min_pct", 0.0008))

    rsi = _as_float(_get(row, "rsi", 50.0), 50.0)
    close = _as_float(_get(row, "close", None), 0.0)
    sma_fast = _as_float(_get(row, "sma_fast", close), close)
    sma_slow = _as_float(_get(row, "sma_slow", close), close)
    atr_pct = _as_float(_get(row, "atr_pct", None), 0.0)

    qqq_up = bool(_get(row, "qqq_trend_up", False))
    corr_qqq = _get(row, "corr_qqq", None)
    corr_qqq = _as_float(corr_qqq, np.nan) if corr_qqq is not None else np.nan
    fng = _get(row, "fng", None)
    fng = _as_float(fng, np.nan) if fng is not None else np.nan

    regime_up = sma_fast > sma_slow
    score = 0.7 if regime_up else 0.2  # trend baz puanı

    # Momentum
    if rsi >= rsi_buy:
        score += 0.25

    # QQQ etkisi
    if require_qqq_up and not qqq_up:
        score -= 0.20
    else:
        if qqq_up:
            score += 0.05

    # Korelasyon: pozitif korelasyon + QQQ yukarı ise yardımcı; çok düşük korelasyon ceza değil
    if not np.isnan(corr_qqq):
        if corr_qqq >= max(min_corr_qqq, 0.0) and qqq_up:
            score += 0.05
        elif corr_qqq < min_corr_qqq:
            score -= 0.05

    # Duygu: yüksek FNG long'u destekler
    if not np.isnan(fng):
        if fng >= 60:
            score += 0.05
        elif fng <= 20:
            score -= 0.05

    # Volatilite: aşırı yüksek/düşükte ufak kırpma
    if atr_pct >= 0.015:  # ~%1.5+
        score -= 0.08
    elif atr_pct < atr_min_pct:
        score -= 0.03

    return _clip01(score)


def _score_short(row: pd.Series, rules: Dict[str, Any]) -> float:
    """
    Kısa (short) yön puanı — trend aşağı, düşük momentum ve risk iştahı ile artar,
    aşırı/çok düşük volatiliteyle azalır.
    """
    rsi_sell = _as_float(rules.get("rsi_sell", 48.0))
    require_qqq_down = bool(rules.get("require_qqq_down", False))
    min_corr_qqq = _as_float(rules.get("min_corr_qqq", -1.0))
    atr_min_pct = _as_float(rules.get("atr_min_pct", 0.0008))

    rsi = _as_float(_get(row, "rsi", 50.0), 50.0)
    close = _as_float(_get(row, "close", None), 0.0)
    sma_fast = _as_float(_get(row, "sma_fast", close), close)
    sma_slow = _as_float(_get(row, "sma_slow", close), close)
    atr_pct = _as_float(_get(row, "atr_pct", None), 0.0)

    qqq_up = bool(_get(row, "qqq_trend_up", False))
    qqq_down = not qqq_up
    corr_qqq = _get(row, "corr_qqq", None)
    corr_qqq = _as_float(corr_qqq, np.nan) if corr_qqq is not None else np.nan
    fng = _get(row, "fng", None)
    fng = _as_float(fng, np.nan) if fng is not None else np.nan

    regime_dn = sma_fast < sma_slow
    score = 0.7 if regime_dn else 0.2  # trend baz puanı

    # Momentum tersine: RSI düşükse short desteklenir
    if rsi <= rsi_sell:
        score += 0.25

    # QQQ etkisi (short için aşağı trend destek)
    if require_qqq_down and not qqq_down:
        score -= 0.20
    else:
        if qqq_down:
            score += 0.05

    # Korelasyon: pozitif korelasyon + QQQ aşağı ise short'u destekler
    if not np.isnan(corr_qqq):
        if corr_qqq >= max(min_corr_qqq, 0.0) and qqq_down:
            score += 0.05
        elif corr_qqq < min_corr_qqq:
            score -= 0.05

    # Duygu: aşırı korku (düşük FNG) short'u az da olsa destekler
    if not np.isnan(fng):
        if fng <= 20:
            score += 0.05
        elif fng >= 60:
            score -= 0.05

    # Volatilite: aşırı yüksek/düşükte ufak kırpma (short tarafında da simetrik)
    if atr_pct >= 0.015:
        score -= 0.08
    elif atr_pct < atr_min_pct:
        score -= 0.03

    return _clip01(score)


def rule_signal(row: pd.Series, rules: Dict[str, Any]) -> Tuple[int, float]:
    """
    Long/Short kararını aynı anda puanlar, hangi taraf daha güçlüyse onu seçer.
    Çıktı:
      signal   : -1 (short), 0 (flat), +1 (long)
      confidence: [0..1] — göreli hedef (event_backtest target'a çevirecek)
    """
    # Taraf puanları
    long_score = _score_long(row, rules)
    short_score = _score_short(row, rules)

    # Minimum güven eşiği (taraflardan biri bariz değilse flat kal)
    min_take = float(rules.get("min_take_conf", 0.15))

    if long_score < min_take and short_score < min_take:
        return 0, 0.0

    if long_score > short_score:
        return 1, _clip01(long_score)
    elif short_score > long_score:
        return -1, _clip01(short_score)
    else:
        # Beraberlikte: trend tarafını seç (daha “doğal” akış)
        sma_fast = _as_float(_get(row, "sma_fast", _get(row, "close", 0.0)))
        sma_slow = _as_float(_get(row, "sma_slow", _get(row, "close", 0.0)))
        if sma_fast > sma_slow and long_score >= min_take:
            return 1, _clip01(long_score)
        if sma_fast < sma_slow and short_score >= min_take:
            return -1, _clip01(short_score)
        return 0, 0.0


def _infer_probs(proba: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Çeşitli proba formatlarını p_up / p_down'a indirger.
    Dönen: (p_up, p_down) — bilinmiyorsa (None, None)
    Kabul edilenler:
      - float/int → p_up
      - dict → anahtarlar: 'up','down' veya {1, -1} vb.
      - list/ndarray → [p_down, p_up] ya da [p_up, p_down]; toplamı ~1 ise çıkarım yapılır
    """
    try:
        # Skaler: p_up
        if isinstance(proba, (int, float)) and 0.0 <= float(proba) <= 1.0:
            p_up = float(proba)
            return p_up, 1.0 - p_up

        # Dict
        if isinstance(proba, dict):
            keys = {str(k).lower(): v for k, v in proba.items()}
            if "up" in keys or "1" in keys or "+1" in keys:
                p_up = float(keys.get("up", keys.get("1", keys.get("+1"))))
                p_down = float(keys.get("down", keys.get("-1", 1.0 - p_up)))
                return p_up, p_down

        # Liste / ndarray
        if isinstance(proba, (list, tuple, np.ndarray)):
            arr = np.array(proba, dtype=float).flatten()
            if arr.size == 2 and 0.99 <= arr.sum() <= 1.01:
                # Heuristik: daha büyük olanı "p_up" kabul et ve indekslerini sinyale göre uyumla
                # ama uzun-vade içgüdü: çoğu sklearn binary [p_neg, p_pos] döner
                # o yüzden varsayılan: [p_down, p_up]
                p_down, p_up = float(arr[0]), float(arr[1])
                return p_up, p_down
            # 3 sınıf (down/flat/up) gibi durumlar için: en büyük iki olasılıktan up/down yakala
            if arr.size >= 3:
                # dene: [p_down, p_flat, p_up]
                p_down = float(arr[0])
                p_up = float(arr[-1])
                if 0.0 <= p_up <= 1.0 and 0.0 <= p_down <= 1.0:
                    return p_up, p_down
    except Exception:
        pass
    return None, None


def ensemble_with_ml(signal: int, confidence: float, proba: Any) -> Tuple[int, float]:
    """
    Kural sinyali ile ML olasılıklarını harmanlar.
    - signal==+1 ise p_up ile, signal==-1 ise p_down ile karışım yap.
    - ML yoksa olduğu gibi döner.
    """
    if proba is None:
        return int(signal), float(confidence)

    p_up, p_down = _infer_probs(proba)
    conf = float(confidence)

    if signal > 0 and p_up is not None:
        conf = 0.5 * conf + 0.5 * float(p_up)
    elif signal < 0 and p_down is not None:
        conf = 0.5 * conf + 0.5 * float(p_down)
    # signal==0 ise ML tek başına yön vermesin; belirsizlikte flat kal.

    return int(signal), _clip01(conf)
