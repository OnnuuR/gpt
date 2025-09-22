import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def _get(row, key, default=np.nan):
    try:
        v = row.get(key, default)
    except Exception:
        v = default
    try:
        return float(v)
    except Exception:
        return np.nan


def _rolling_slope(arr: np.ndarray) -> float:
    # basit 2-nokta eğim (baş-son)
    if arr.size < 2 or not np.isfinite(arr).all():
        return 0.0
    ref = float(np.nanmean(arr))
    if not np.isfinite(ref) or ref == 0.0:
        return 0.0
    return (arr[-1] - arr[0]) / (len(arr) * ref)


def rule_signal(feats: pd.DataFrame, i: int, cfg: Dict[str, Any]) -> Tuple[int, float]:
    """
    Rejim + trend-takip (boğada long-bias, ayıda short sınırlı):
      - Rejim: SMA fast > SMA slow => bull, < => bear
      - Boğa: breakout (48h), RSI yeniden-yükseliş, QQQ pozitif eğim -> long
      - Ayı: breakdown + RSI zayıf -> short (short güven tavanlı veya kapalı)
      - ATR yüksekse güven azalt
      - Sig ∈ {-1,0,1}, conf ∈ [0,1]
    """
    rules = cfg.get("rules", {}) or {}

    # --- rejim tabanı / bantlar ---
    base_pos_bull = float(rules.get("base_pos_bull", 0.90))
    base_pos_bear = float(rules.get("base_pos_bear", -0.10))
    neutral_band  = float(rules.get("neutral_band", 0.10))

    # RSI / momentum
    rsi_low    = float(rules.get("rsi_low", 45.0))
    rsi_high   = float(rules.get("rsi_high", 55.0))
    mom_look   = int(rules.get("mom_look", 24))
    mom_thr    = float(rules.get("mom_bias_thr", 0.010))  # ~%1

    # Breakout
    breakout_look = int(rules.get("breakout_look", 48))
    breakout_conf = float(rules.get("breakout_conf", 0.85))

    # QQQ overlay
    qqq_look      = int(rules.get("qqq_look", 120))  # ~5 gün
    qqq_boost     = float(rules.get("qqq_boost", 0.10))
    qqq_drag      = float(rules.get("qqq_drag", 0.05))

    # FNG
    use_fng      = bool(rules.get("use_fng", True))
    fng_bias_thr = float(rules.get("fng_bias_thr", 55.0))
    fng_boost    = float(rules.get("fng_overlay", 0.05))

    # Short kısıtları
    no_short_in_bull = bool(rules.get("no_short_in_bull", True))
    short_conf_cap   = float(rules.get("short_conf_cap", 0.30))

    row = feats.iloc[i]
    close    = _get(row, "close")
    sma_fast = _get(row, "sma_fast")
    sma_slow = _get(row, "sma_slow")
    rsi      = _get(row, "rsi_14")
    fng      = _get(row, "fng")
    atr      = _get(row, "atr")         # varsa features'ta
    atr_pct  = _get(row, "atr_pct")     # yoksa aşağıda tahmin edeceğiz

    if not np.isfinite(close) or not np.isfinite(sma_fast) or not np.isfinite(sma_slow):
        return 0, 0.0

    bull = sma_fast > sma_slow
    bear = sma_fast < sma_slow

    # ATR pct yoksa ~14 bar stdev ile kabaca türet
    if not np.isfinite(atr_pct):
        win = 14
        if i - win >= 0:
            seg = feats["close"].iloc[i - win : i + 1].pct_change().dropna().values
            if seg.size > 0 and np.isfinite(seg).all():
                atr_pct = float(np.clip(np.nanstd(seg), 1e-4, 0.2))
            else:
                atr_pct = 0.01
        else:
            atr_pct = 0.01

    # --- ham base ---
    base = base_pos_bull if bull else (base_pos_bear if bear else 0.0)

    # --- overlay: QQQ eğimi ---
    overlay = 0.0
    if "qqq_close" in feats.columns and qqq_look >= 8 and i - qqq_look >= 0:
        qseg = feats["qqq_close"].iloc[i - qqq_look : i + 1].values
        qslope = _rolling_slope(qseg)
        if qslope > 0:
            overlay += qqq_boost
        elif qslope < 0:
            overlay -= qqq_drag

    # --- overlay: momentum (BTC) ---
    if mom_look >= 8 and i - mom_look >= 0:
        prev = feats["close"].iloc[i - mom_look]
        if np.isfinite(prev) and prev != 0.0:
            mret = close / prev - 1.0
            if bull and mret > mom_thr:
                overlay += 0.10
            if bear and mret < -mom_thr:
                overlay -= 0.10

    # --- overlay: RSI re-entry ---
    if np.isfinite(rsi):
        if bull and rsi >= rsi_high and close >= sma_fast:
            overlay += 0.10
        if bear and rsi <= rsi_low and close <= sma_fast:
            overlay -= 0.10

    # --- overlay: FNG ---
    if use_fng and np.isfinite(fng):
        if bull and fng >= fng_bias_thr:
            overlay += fng_boost
        if bear and fng <= 100.0 - fng_bias_thr:
            overlay -= fng_boost

    raw = float(np.clip(base + overlay, -1.0, 1.0))

    # boğada short’u kapat
    if bull and no_short_in_bull:
        raw = max(raw, 0.0)

    # nötr bant
    if raw >  neutral_band:
        sig = 1
    elif raw < -neutral_band:
        sig = -1
    else:
        sig = 0

    # --- breakout/breakdown ---
    brk = False
    if breakout_look >= 24 and i - breakout_look >= 0:
        window = feats["close"].iloc[i - breakout_look : i + 1].values
        if np.isfinite(window).all():
            hh = float(np.max(window))
            ll = float(np.min(window))
            if bull and close >= hh * (1.0 - 1e-6) and close >= sma_fast:
                sig = 1
                brk = True
            if bear and close <= ll * (1.0 + 1e-6) and close <= sma_fast:
                sig = -1
                brk = True

    # --- conf: |raw| + trend eğimi ---
    conf_raw = min(1.0, abs(raw))

    slope_look = int(rules.get("slope_look", 48))
    slope_norm = float(rules.get("slope_conf_norm", 0.0035))
    conf_slope = 0.5
    if slope_look >= 8 and i - slope_look >= 0:
        seg = feats["sma_fast"].iloc[i - slope_look : i + 1].values
        s = _rolling_slope(seg)
        conf_slope = min(1.0, 0.5 + 0.5 * (abs(s) / max(1e-6, slope_norm)))

    conf = float(np.clip(max(conf_raw, conf_slope), 0.0, 1.0))

    # breakout minimum conf
    if brk:
        if sig > 0:
            conf = max(conf, breakout_conf)
        elif sig < 0:
            conf = max(conf, breakout_conf * 0.6)

    # ATR yüksekse güven azalt (whipsaw önleme)
    # ~% ATR 2% üstünde ise çarpan düşüyor
    if atr_pct > 0.02:
        factor = np.interp(atr_pct, [0.02, 0.05], [0.85, 0.60])
        factor = float(np.clip(factor, 0.60, 1.0))
        conf *= factor

    # short güven tavanı
    if sig < 0:
        conf = min(conf, short_conf_cap)

    return sig, float(np.clip(conf, 0.0, 1.0))


def ensemble_with_ml(feats: pd.DataFrame, i: int, cfg: Dict[str, Any], model=None) -> Tuple[int, float]:
    # ML kapalıysa doğrudan kural
    sig, conf = rule_signal(feats, i, cfg)

    ml_cfg = cfg.get("ml", {}) or {}
    if not ml_cfg.get("enable", False) or model is None:
        return sig, conf

    try:
        cols = ml_cfg.get("feature_cols") or []
        if not cols or any(c not in feats.columns for c in cols):
            return sig, conf

        x = feats.iloc[[i]][cols].astype(float).to_numpy()
        proba = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x)[0]
            proba = float(p[1]) if len(p) > 1 else float(p[0])
        elif hasattr(model, "predict"):
            z = float(model.predict(x)[0])
            proba = 1.0 / (1.0 + np.exp(-np.clip(z, -8, 8)))

        if proba is None or not np.isfinite(proba):
            return sig, conf

        ml_edge = float(np.clip((proba - 0.5) * 2.0, -1.0, 1.0))
        w = float(ml_cfg.get("blend_weight", 0.3))
        raw = np.clip(sig * conf + w * ml_edge, -1.0, 1.0)
        sig = 1 if raw > 0 else (-1 if raw < 0 else 0)
        conf = float(np.clip(max(abs(raw), conf), 0.0, 1.0))
        return sig, conf
    except Exception:
        return sig, conf
