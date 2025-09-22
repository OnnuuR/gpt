\
import numpy as np

import numpy as np

def rule_signal(row, params):
    # Parametreler
    rsi_buy = params.get("rsi_buy", 30)
    rsi_sell = params.get("rsi_sell", 70)
    sentiment_min = params.get("sentiment_min", 0.2)      # bir tık artırdık
    exit_sent_thresh = params.get("exit_sent_thresh", -0.5)
    atr_min_pct = params.get("atr_min_pct", 0.0025)       # ~0.25%
    require_qqq_up = params.get("require_qqq_up", True)
    min_corr_qqq = params.get("min_corr_qqq", 0.05)       # en az hafif pozitif korelasyon

    # Inputlar
    rsi = row.get("rsi", np.nan)
    sent = row.get("sentiment_smooth", row.get("sentiment", 0.0))
    q_up = int(row.get("qqq_trend_up", 0))
    corrq = float(row.get("corr_qqq", 0.0))
    sma_fast = row.get("sma_fast", np.nan)
    sma_slow = row.get("sma_slow", np.nan)
    close = row.get("close", np.nan)
    atrp = float(row.get("atr_pct", 0.0))

    # Rejim filtresi: trend + (opsiyonel) QQQ trendi
    ok_regime = (np.isfinite(sma_fast) and np.isfinite(sma_slow) and sma_fast > sma_slow)
    if require_qqq_up:
        ok_regime = ok_regime and (q_up == 1)

    ok_vol = atrp >= atr_min_pct        # volatilite çok düşükse sinyal alma
    ok_corr = corrq >= min_corr_qqq     # QQQ ile küçük de olsa pozitif ilişki
    ok_sent = sent >= sentiment_min

    allow_long = (np.isfinite(rsi) and rsi < rsi_buy) and ok_regime and ok_vol and ok_corr and ok_sent
    # Çıkış: RSI aşırı ve momentum zayıfladıysa veya duygu sert negatife döndüyse
    allow_exit = ((np.isfinite(rsi) and rsi > rsi_sell and np.isfinite(close) and np.isfinite(sma_fast) and close < sma_fast)
                  or (sent < exit_sent_thresh))

    signal = 1 if allow_long else (-1 if allow_exit else 0)

    # güven skoru: girişte RSI + duygu + volatilite birleşimi
    conf = 0.0
    if signal == 1:
        # rsi 0-1, sent 0-1 normalize gibi; atr katkısı küçük
        rsi_term = min(1.0, (rsi_buy - rsi) / max(rsi_buy, 1))
        sent_term = max(0.0, sent)
        vol_term = min(1.0, (atrp / max(atr_min_pct, 1e-9)))
        conf = float(min(1.0, 0.5 * rsi_term + 0.4 * sent_term + 0.1 * vol_term))
    elif signal == -1:
        # çıkışta çok agresif olmayalım; rebalance/exit’te güven yüksek gelmesin
        conf = 0.6

    return int(signal), float(conf)

def ensemble_with_ml(rule_sig, rule_conf, proba, weight_rule=0.6):
    if proba is None:
        return rule_sig, rule_conf
    buy_p = proba.get(1, 0.0)
    sell_p = proba.get(-1, 0.0)
    # normalized ML vote
    ml_sig = 1 if buy_p > 0.55 else (-1 if sell_p > 0.55 else 0)
    ml_conf = max(abs(buy_p - 0.5), abs(sell_p - 0.5)) * 2
    # blend
    if ml_sig == 0:
        return rule_sig, rule_conf
    if rule_sig == 0:
        return ml_sig, ml_conf
    # same direction
    if ml_sig == rule_sig:
        return rule_sig, min(1.0, weight_rule*rule_conf + (1-weight_rule)*ml_conf)
    # conflict -> reduce confidence
    return rule_sig, max(0.0, weight_rule*rule_conf - (1-weight_rule)*ml_conf)
