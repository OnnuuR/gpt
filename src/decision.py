import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def _get(row, key, default=np.nan):
    """Row içinden güvenli şekilde float çek."""
    try:
        v = row.get(key, default)
    except Exception:
        v = default
    try:
        return float(v)
    except Exception:
        return np.nan


def rule_signal(feats: pd.DataFrame, i: int, cfg: Dict[str, Any]) -> Tuple[int, float]:
    """
    Rejim + overlay yaklaşımı (long-short simetrik değil, boğada long-bias):
      - Rejim: SMA fast > SMA slow (boğa) / < (ayı)
      - Boğa: base_pos_bull, ayı: base_pos_bear
      - Overlay: momentum + mean-reversion (+ opsiyonel FNG) + breakout/breakdown
      - raw ∈ [-1, 1] -> nötr bant ile sinyale (sig ∈ {-1,0,1}), conf ∈ [0,1]
    """
    rules = cfg.get("rules", {}) or {}

    # --- temel parametreler ---
    base_pos_bull = float(rules.get("base_pos_bull", 0.80))
    base_pos_bear = float(rules.get("base_pos_bear", -0.10))
    neutral_band  = float(rules.get("neutral_band", 0.08))

    # Momentum
    mom_look     = int(rules.get("mom_look", 24))
    mom_bias_thr = float(rules.get("mom_bias_thr", 0.010))  # ~%1
    overlay_mom  = float(rules.get("overlay_mom", 0.25))

    # Mean-reversion (RSI)
    rsi_low    = float(rules.get("rsi_low", 40.0))
    rsi_high   = float(rules.get("rsi_high", 60.0))
    overlay_mr = float(rules.get("overlay_mr", 0.15))

    # Slope (confidence güçlendirme)
    slope_look      = int(rules.get("slope_look", 48))
    slope_conf_norm = float(rules.get("slope_conf_norm", 0.004))

    # FNG (opsiyonel)
    use_fng      = bool(rules.get("use_fng", True))
    fng_bias_thr = float(rules.get("fng_bias_thr", 60.0))
    fng_overlay  = float(rules.get("fng_overlay", 0.05))

    # Breakout/Down
    breakout_look  = int(rules.get("breakout_look", 72))
    breakout_conf  = float(rules.get("breakout_conf", 0.80))

    # Short’u sınırlama
    no_short_in_bull = bool(rules.get("no_short_in_bull", True))
    no_long_in_bear  = bool(rules.get("no_long_in_bear", False))
    short_conf_cap   = float(rules.get("short_conf_cap", 0.35))  # short güven tavanı

    row = feats.iloc[i]
    close    = _get(row, "close")
    sma_fast = _get(row, "sma_fast")
    sma_slow = _get(row, "sma_slow")
    rsi      = _get(row, "rsi_14")
    fng      = _get(row, "fng")

    if not np.isfinite(close) or not np.isfinite(sma_fast) or not np.isfinite(sma_slow):
        return 0, 0.0

    # --- rejim ---
    bull = sma_fast > sma_slow
    bear = sma_fast < sma_slow

    # participation tabanı
    base = base_pos_bull if bull else (base_pos_bear if bear else 0.0)

    # --- overlay: momentum ---
    overlay = 0.0
    if i - mom_look >= 0:
        prev = feats["close"].iloc[i - mom_look]
        if np.isfinite(prev) and prev != 0.0:
            ret = close / prev - 1.0
            if bull and ret >  mom_bias_thr:
                overlay += overlay_mom
            elif bear and ret < -mom_bias_thr:
                overlay -= overlay_mom

    # --- overlay: mean-reversion (RSI) ---
    if np.isfinite(rsi):
        if bull and rsi <= rsi_low:
            overlay += overlay_mr
        if bear and rsi >= rsi_high:
            overlay -= overlay_mr

    # --- overlay: FNG ---
    if use_fng and np.isfinite(fng):
        if bull and fng >= fng_bias_thr:
            overlay += fng_overlay
        if bear and fng <= 100.0 - fng_bias_thr:
            overlay -= fng_overlay

    # ham hedef
    raw = float(np.clip(base + overlay, -1.0, 1.0))

    # boğada short kapatma / ayıda long kapatma (ayarlarla)
    if bull and no_short_in_bull:
        raw = max(raw, 0.0)
    if bear and no_long_in_bear:
        raw = min(raw, 0.0)

    # nötr bant
    if raw >  neutral_band:
        sig = 1
    elif raw < -neutral_band:
        sig = -1
    else:
        sig = 0

    # --- breakout/breakdown tetikleyicisi ---
    if breakout_look > 4 and i - breakout_look >= 0:
        window = feats["close"].iloc[i - breakout_look : i + 1]
        if np.isfinite(window).all():
            hh = float(np.max(window))
            ll = float(np.min(window))
            # boğada yeni tepe ve RSI makul → long’u zorla
            if bull and close >= hh * (1.0 - 1e-6) and (not np.isfinite(rsi) or rsi >= 55):
                sig = 1
                # mevcut conf üzerine çıkar
                # conf hesaplandıktan sonra max(...) ile yükselteceğiz
                brk_long = True
            else:
                brk_long = False
            # ayıda yeni dip ve RSI düşük → short’u zorla (conf short cap ile)
            if bear and close <= ll * (1.0 + 1e-6) and (not np.isfinite(rsi) or rsi <= 45):
                sig = -1
                brk_short = True
            else:
                brk_short = False
        else:
            brk_long = brk_short = False
    else:
        brk_long = brk_short = False

    # --- güven (conf) ---
    # 1) |raw| taban
    conf_raw = min(1.0, abs(raw))
    # 2) kısa SMA eğimi ile güçlendirme
    slope = 0.0
    if i - slope_look >= 0 and "sma_fast" in feats.columns:
        seg = feats["sma_fast"].iloc[i - slope_look : i + 1].values
        if np.isfinite(seg).all():
            ref = float(np.nanmean(seg))
            if np.isfinite(ref) and ref != 0.0:
                slope = (seg[-1] - seg[0]) / (slope_look * ref)
    conf_slope = min(1.0, 0.5 + 0.5 * (abs(slope) / max(1e-6, slope_conf_norm)))

    conf = float(np.clip(max(conf_raw, conf_slope), 0.0, 1.0))

    # breakout’la minimum güven eşiği
    if brk_long and sig == 1:
        conf = max(conf, breakout_conf)
    if brk_short and sig == -1:
        conf = max(conf, breakout_conf * short_conf_cap)

    # short güveni tavanla
    if sig < 0:
        conf = min(conf, short_conf_cap)

    return sig, conf


def ensemble_with_ml(feats: pd.DataFrame, i: int, cfg: Dict[str, Any], model=None) -> Tuple[int, float]:
    """
    Geriye dönük uyumluluk için sağlanan sarmalayıcı.
    - ML devre dışıysa veya model yoksa: rule_signal(...) döndürür.
    - ML etkinse: rule + ML basit blend.
    """
    sig, conf = rule_signal(feats, i, cfg)

    try:
        ml_cfg = cfg.get("ml", {}) or {}
        if not ml_cfg.get("enable", False) or model is None:
            return sig, conf

        feature_cols = ml_cfg.get("feature_cols") or []
        if not feature_cols or any(c not in feats.columns for c in feature_cols):
            return sig, conf

        row = feats.iloc[i]
        x = row[feature_cols].astype(float).to_numpy()[None, :]

        proba = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x)[0]
            proba = float(p[1]) if len(p) > 1 else float(p[0])
        elif hasattr(model, "predict"):
            pred = float(model.predict(x)[0])
            proba = 1.0 / (1.0 + np.exp(-np.clip(pred, -8, 8)))

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
