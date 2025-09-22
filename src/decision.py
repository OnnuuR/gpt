from __future__ import annotations

import math
from typing import Tuple, Any, Optional, Dict, List

import numpy as np
import pandas as pd


# --------- yardımcılar ---------
def _get(row: pd.Series, key: str, default: Any = None) -> Any:
    if key in row and pd.notna(row[key]):
        return row[key]
    return default

def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))

def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def _quantize(x: float, levels: List[float]) -> float:
    if not levels:
        return x
    # en yakın seviyeye yuvarla
    return float(min(levels, key=lambda l: abs(x - l)))


# --------- taraf puanlayıcıları ---------
def _score_long(row: pd.Series, rules: Dict[str, Any]) -> float:
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
    score = 0.7 if regime_up else 0.15  # trend baz

    if rsi >= rsi_buy:
        score += 0.25

    if require_qqq_up and not qqq_up:
        score -= 0.20
    else:
        if qqq_up:
            score += 0.05

    if not np.isnan(corr_qqq):
        if corr_qqq >= max(min_corr_qqq, 0.0) and qqq_up:
            score += 0.05
        elif corr_qqq < min_corr_qqq:
            score -= 0.05

    if not np.isnan(fng):
        if fng >= 60:
            score += 0.05
        elif fng <= 20:
            score -= 0.05

    if atr_pct >= 0.015:
        score -= 0.08
    elif atr_pct < atr_min_pct:
        score -= 0.03

    return _clip01(score)


def _score_short(row: pd.Series, rules: Dict[str, Any]) -> float:
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
    score = 0.7 if regime_dn else 0.15  # trend baz

    if rsi <= rsi_sell:
        score += 0.25

    if require_qqq_down and not qqq_down:
        score -= 0.20
    else:
        if qqq_down:
            score += 0.05

    if not np.isnan(corr_qqq):
        if corr_qqq >= max(min_corr_qqq, 0.0) and qqq_down:
            score += 0.05
        elif corr_qqq < min_corr_qqq:
            score -= 0.05

    if not np.isnan(fng):
        if fng <= 20:
            score += 0.05
        elif fng >= 60:
            score -= 0.05

    if atr_pct >= 0.015:
        score -= 0.08
    elif atr_pct < atr_min_pct:
        score -= 0.03

    return _clip01(score)


# --------- ana sinyal ---------
def rule_signal(row: pd.Series, rules: Dict[str, Any]) -> Tuple[int, float]:
    """
    Çıkış:
      signal: -1 (short), 0 (flat), +1 (long)
      confidence: [0..1]
    """
    long_score = _score_long(row, rules)
    short_score = _score_short(row, rules)

    min_take = _as_float(rules.get("min_take_conf", 0.15), 0.15)
    side_margin = _as_float(rules.get("side_margin", 0.10), 0.10)
    conf_min_on = _as_float(rules.get("conf_min_on", 0.60), 0.60)

    # taraf seçimi: marj yoksa flat kal
    diff = long_score - short_score
    if diff > side_margin and long_score >= min_take:
        signal = 1
        conf = long_score
    elif diff < -side_margin and short_score >= min_take:
        signal = -1
        conf = short_score
    else:
        return 0, 0.0

    # güven kademelendirme
    levels = rules.get("quantize_levels", [0.0, 0.25, 0.5, 0.75, 1.0])
    try:
        levels = [float(x) for x in levels]
    except Exception:
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    conf = _clip01(conf)
    conf = max(conf, conf_min_on)  # sinyal açıldıysa en az bu kadar
    conf = _quantize(conf, levels)

    return int(signal), float(conf)


# --------- ML entegrasyonu (opsiyonel) ---------
def _infer_probs(proba: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Çeşitli formatları (p_up, p_down) a indirger.
    """
    try:
        if isinstance(proba, (int, float)) and 0.0 <= float(proba) <= 1.0:
            p_up = float(proba)
            return p_up, 1.0 - p_up
        if isinstance(proba, dict):
            keys = {str(k).lower(): v for k, v in proba.items()}
            if "up" in keys or "1" in keys or "+1" in keys:
                p_up = float(keys.get("up", keys.get("1", keys.get("+1"))))
                p_down = float(keys.get("down", keys.get("-1", 1.0 - p_up)))
                return p_up, p_down
        if isinstance(proba, (list, tuple, np.ndarray)):
            arr = np.array(proba, dtype=float).flatten()
            if arr.size == 2 and 0.99 <= arr.sum() <= 1.01:
                p_down, p_up = float(arr[0]), float(arr[1])  # sklearn: [neg, pos]
                return p_up, p_down
            if arr.size >= 3:
                p_down = float(arr[0])
                p_up = float(arr[-1])
                return p_up, p_down
    except Exception:
        pass
    return None, None


def ensemble_with_ml(signal: int, confidence: float, proba: Any) -> Tuple[int, float]:
    """
    Kural sinyali ile ML olasılıklarını harmanlar.
    """
    if proba is None:
        return int(signal), float(confidence)

    p_up, p_down = _infer_probs(proba)
    conf = float(confidence)

    if signal > 0 and p_up is not None:
        conf = 0.5 * conf + 0.5 * float(p_up)
    elif signal < 0 and p_down is not None:
        conf = 0.5 * conf + 0.5 * float(p_down)

    return int(signal), _clip01(conf)
