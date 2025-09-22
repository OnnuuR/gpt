import numpy as np

def _isfinite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def rule_signal(row, params):
    # Esnek varsayılanlar (settings.yaml override edebilir)
    rsi_buy = params.get("rsi_buy", 30)
    rsi_sell = params.get("rsi_sell", 70)
    sentiment_min = params.get("sentiment_min", 0.05)      # daha esnek
    exit_sent_thresh = params.get("exit_sent_thresh", -0.5)
    atr_min_pct = params.get("atr_min_pct", 0.0015)        # 0.15%
    require_qqq_up = params.get("require_qqq_up", False)   # default False
    min_corr_qqq = params.get("min_corr_qqq", -0.05)       # nötre izin ver

    # Girdiler
    rsi = row.get("rsi", np.nan)
    sent = row.get("sentiment_smooth", row.get("sentiment", np.nan))
    q_up = row.get("qqq_trend_up", np.nan)
    corrq = row.get("corr_qqq", np.nan)
    sma_fast = row.get("sma_fast", np.nan)
    sma_slow = row.get("sma_slow", np.nan)
    close = row.get("close", np.nan)
    atrp = row.get("atr_pct", np.nan)

    # Rejim filtresi
    ok_regime = (_isfinite(sma_fast) and _isfinite(sma_slow) and sma_fast > sma_slow)
    if require_qqq_up and _isfinite(q_up):
        ok_regime = ok_regime and int(q_up) == 1

    # Volatilite & korelasyon: NaN ise bloke etme
    ok_vol = (not _isfinite(atrp)) or (atrp >= atr_min_pct)
    ok_corr = (not _isfinite(corrq)) or (corrq >= min_corr_qqq)

    # Duygu: veri yoksa nötr kabul et
    if not _isfinite(sent) or abs(float(sent)) < 1e-6:
        ok_sent = True
        sent_val = 0.0
    else:
        sent_val = float(sent)
        ok_sent = sent_val >= sentiment_min

    # Giriş/Çıkış
    allow_long = (_isfinite(rsi) and rsi < rsi_buy) and ok_regime and ok_vol and ok_corr and ok_sent
    allow_exit = ((_isfinite(rsi) and rsi > rsi_sell and _isfinite(close) and _isfinite(sma_fast) and close < sma_fast)
                  or (sent_val < exit_sent_thresh))

    signal = 1 if allow_long else (-1 if allow_exit else 0)

    # Güven skoru
    conf = 0.0
    if signal == 1:
        rsi_term = min(1.0, (rsi_buy - rsi) / max(rsi_buy, 1))
        sent_term = max(0.0, sent_val)
        vol_term = 1.0 if not _isfinite(atrp) else min(1.0, (atrp / max(atr_min_pct, 1e-9)))
        conf = float(min(1.0, 0.5 * rsi_term + 0.4 * sent_term + 0.1 * vol_term))
    elif signal == -1:
        conf = 0.6

    return int(signal), float(conf)

def ensemble_with_ml(rule_sig, rule_conf, proba, weight_rule=0.6):
    if proba is None:
        return rule_sig, rule_conf
    buy_p = proba.get(1, 0.0)
    sell_p = proba.get(-1, 0.0)
    ml_sig = 1 if buy_p > 0.55 else (-1 if sell_p > 0.55 else 0)
    ml_conf = max(abs(buy_p - 0.5), abs(sell_p - 0.5)) * 2
    if ml_sig == 0:
        return rule_sig, rule_conf
    if rule_sig == 0:
        return ml_sig, ml_conf
    if ml_sig == rule_sig:
        return rule_sig, min(1.0, weight_rule * rule_conf + (1 - weight_rule) * ml_conf)
    return rule_sig, max(0.0, weight_rule * rule_conf - (1 - weight_rule) * ml_conf)
