import os, argparse, yaml
import pandas as pd
import numpy as np
from dateutil import tz
from joblib import load
from datetime import datetime, timezone

from src.utils import get_logger
from src import data_aggregator as da
from src.features import add_core_features
from src.decision import rule_signal, ensemble_with_ml
from src.ml_model import train_model, predict_proba_row
from src.backtest import event_backtest
from src.macro_calendar import fetch_ics_events, in_risk_window

UTC = tz.gettz("UTC")

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def download_all(cfg, logger):
    sym = cfg["market"]["symbols"][0]
    tf = cfg["market"]["timeframe"]
    lookback_days = cfg["market"]["lookback_days"]
    since = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).value // 10**6

    logger.info(f"Downloading OHLCV for {sym} {tf} {lookback_days}d")
    ohlcv = da.fetch_ohlcv_ccxt(symbol=sym, timeframe=tf, since_ms=since)
    ohlcv.to_csv(os.path.join(cfg["general"]["data_dir"], "btc_ohlcv.csv"))
    logger.info(f"Saved btc_ohlcv.csv ({len(ohlcv)} rows)")

    yft = cfg["market"]["yfinance"]
    qqq = da.fetch_yf(yft["qqq"], period=f"{lookback_days}d", interval=tf if tf in ["1h","4h","1d"] else "1h")
    dxy = da.fetch_yf(yft["dxy"], period=f"{lookback_days}d", interval=tf if tf in ["1h","4h","1d"] else "1h")
    if not qqq.empty:
        qqq = qqq.rename(columns={"close": "qqq_close"})
        qqq.to_csv(os.path.join(cfg["general"]["data_dir"], "qqq.csv"))
    if not dxy.empty:
        dxy = dxy.rename(columns={"close": "dxy_close"})
        dxy.to_csv(os.path.join(cfg["general"]["data_dir"], "dxy.csv"))
    logger.info(f"QQQ rows={len(qqq)} DXY rows={len(dxy)}")

    sent = da.fetch_rss_sentiment(cfg["sentiment"]["rss_feeds"], hours=cfg["sentiment"]["rolling_hours"], logger=logger)
    if not sent.empty:
        sent.to_csv(os.path.join(cfg["general"]["data_dir"], "sentiment.csv"))
        logger.info(f"Saved sentiment.csv ({len(sent)} rows)")
    fng = da.fetch_fng(days=min(lookback_days, 365 * 2))
    if not fng.empty:
        fng.to_csv(os.path.join(cfg["general"]["data_dir"], "fng.csv"))
        logger.info(f"Saved fng.csv ({len(fng)} rows)")

import os
import pandas as pd
from typing import Dict, Any

def build_dataset(cfg: Dict[str, Any], logger) -> pd.DataFrame:
    """
    Tüm kaynakları 1 saatlik banda indirip kesişim (inner join) ile tamamen
    eş-zamanlı (synchronous) bir veri seti oluşturur; ardından özellikleri üretir.
    """
    data_dir = cfg["general"]["data_dir"]
    tf = cfg.get("market", {}).get("timeframe", "1h").lower()
    if tf != "1h":
        raise ValueError("Bu senkron sürüm şu an '1h' için tasarlandı.")

    lookback_days = int(cfg.get("market", {}).get("lookback_days", 120))  # 3-4 ay önerildi
    use_sentiment = bool(cfg.get("sync", {}).get("use_sentiment", False)) # eş-zamanlılık için kapalı

    # --- 1) BTC (zorunlu) ---
    logger.info("Ana BTC OHLCV verisi yükleniyor...")
    base_path = os.path.join(data_dir, "btc_ohlcv.csv")
    base = pd.read_csv(base_path, index_col=0)
    base.index = pd.to_datetime(base.index, utc=True, errors="coerce")
    base = base[["open", "high", "low", "close", "volume"]].sort_index()

    # Pencereyi son N güne indir
    end = base.index.max()
    start = end - pd.Timedelta(days=lookback_days)
    base = base.loc[(base.index >= start) & (base.index <= end)]

    # BTC zaten 1h; yinede garanti olsun:
    base = base.resample("1h").last()

    # Küçük yardımcı
    def _read_csv(name, cols):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()
        if cols:
            df = df[cols]
        return df

    # --- 2) Opsiyonel kaynaklar (1h’e resample + pencere kırp) ---
    qqq = _read_csv("qqq.csv", ["qqq_close"])
    if not qqq.empty:
        qqq = qqq.loc[(qqq.index >= start) & (qqq.index <= end)]
        qqq = qqq.resample("1h").last()  # doldurma yok
        logger.info("-> Başarıyla yüklendi ve 1h'e indirildi: QQQ")

    dxy = _read_csv("dxy.csv", ["dxy_close"])
    if not dxy.empty:
        dxy = dxy.loc[(dxy.index >= start) & (dxy.index <= end)]
        dxy = dxy.resample("1h").last()
        logger.info("-> Başarıyla yüklendi ve 1h'e indirildi: DXY")

    fng = _read_csv("fng.csv", ["fng"])
    if not fng.empty:
        fng = fng.loc[(fng.index >= start) & (fng.index <= end)]
        fng = fng.resample("1h").ffill()  # günlük → 1h (ileri taşıma doğal)
        logger.info("-> Başarıyla yüklendi ve 1h'e indirildi: FNG")

    sent = pd.DataFrame()
    if use_sentiment:
        sent_raw = _read_csv("sentiment.csv", ["sentiment"])
        if not sent_raw.empty:
            sent = sent_raw.loc[(sent_raw.index >= start) & (sent_raw.index <= end)]
            sent = sent.resample("1h").mean()  # yayma yok; yoksa NaN kalsın
            logger.info("-> Başarıyla yüklendi ve 1h'e indirildi: SENTIMENT (aktif)")
    else:
        logger.info("SENTIMENT devre dışı (sync.use_sentiment=false).")

    # --- 3) Sıkı EŞ-ZAMANLILIK: INNER JOIN (kesişim) ---
    # Sadece aynı saat damgasında hepsi varsa tutulur.
    merged = base.copy()
    for df in [qqq, dxy, fng]:
        if not df.empty:
            merged = merged.join(df, how="inner")  # kesişim
    if use_sentiment and not sent.empty:
        merged = merged.join(sent, how="inner")

    # Sayısala dönüştür (korelasyon/ATR için kritik)
    for col in ["open", "high", "low", "close", "volume", "qqq_close", "dxy_close", "fng", "sentiment"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # ÖZET (yüklenen & kesişim)
    logger.info("--- Veri Yükleme Özeti ---")
    def _span(df, name):
        if df.empty: return
        logger.info(f"[{name}] Zaman aralığı: {df.index.min()} -> {df.index.max()} | Satır sayısı: {len(df)}")
    _span(base, "BTC")
    _span(qqq, "QQQ")
    _span(dxy, "DXY")
    if use_sentiment: _span(sent, "SENTIMENT")
    _span(fng, "FNG")
    logger.info(f"[SYNC] Kesişim satır sayısı: {len(merged)} (1h)")

    # --- 4) Çekirdek özellikler ---
    logger.info("Çekirdek özellikler (feature) hesaplanıyor...")
    feats = add_core_features(
        merged,
        rsi_period=cfg["features"]["rsi_period"],
        sma_fast=cfg["features"]["sma_fast"],
        sma_slow=cfg["features"]["sma_slow"],
        corr_window=cfg["features"]["corr_window"],
        qqq_col="qqq_close",
    )

    # Rolling ısınma dönemlerinden doğan NaN'ları at (ör. corr_window, SMA)
    drop_cols = ["sma_fast", "sma_slow", "atr", "atr_pct"]
    if "qqq_close" in feats.columns:
        drop_cols += ["corr_qqq", "qqq_trend_up"]
    feats = feats.dropna(subset=[c for c in drop_cols if c in feats.columns])

    return feats

def run_backtest(cfg, logger):
    feats = build_dataset(cfg, logger)
    # optional ML
    model_path = os.path.join(cfg["general"]["models_dir"], "btc_gb.joblib")
    ml_cfg = cfg.get("ml", {})
    model = load(model_path) if (ml_cfg.get("enable", False) and os.path.exists(model_path)) else None


    rows = []
    # macro gating
    events = None
    if cfg["macro"]["use_calendar"] and cfg["macro"]["ics_links"]:
        events = fetch_ics_events(cfg["macro"]["ics_links"])

    for t, row in feats.iterrows():
        # macro risk: block new positions around events
        block = False
        if events is not None and not events.empty:
            now_utc = pd.Timestamp(t, tz="UTC").to_pydatetime()
            block, _ = in_risk_window(
                now_utc, events, cfg["macro"]["pre_event_hours"], cfg["macro"]["post_event_minutes"]
            )
        sig, conf = rule_signal(row, cfg["rules"])
        if block and sig == 1:
            sig = 0
            conf = 0.0
        proba = None
        if model is not None and str(t.minute) in ["0", "15", "30", "45"]:
            # reduce predict frequency a bit
            proba = predict_proba_row(model, row)
        esig, econf = ensemble_with_ml(sig, conf, proba)
        rows.append({"time": t, "close": row["close"], "signal": esig, "confidence": econf})

    sigdf = pd.DataFrame(rows).set_index("time")

    # --- Hızlı sağlık kontrolü ---
    try:
        sigdf = sigdf.sort_index()
        cmin, cmax = float(sigdf["close"].min()), float(sigdf["close"].max())
        npos = int((sigdf["signal"] == 1).sum())
        nneg = int((sigdf["signal"] == -1).sum())
        print(f"[CHECK] close min/max: {cmin:.2f} / {cmax:.2f} | signals +1:{npos}  -1:{nneg}")
        if abs(cmax - cmin) < 1e-6:
            print("[WARN] close min==max → fiyat oynaksız görünüyor (yanlış format/parse?)")
    except Exception as e:
        print(f"[CHECK] Sağlık kontrolü atlandı: {e}")

    sigdf["close"] = pd.to_numeric(sigdf["close"], errors="coerce").ffill().bfill()



    # Olay odaklı backtest
    out, stats, trades = event_backtest(
        sigdf,
        fee=cfg["risk"].get("taker_fee", 0.0004),
        slip=cfg["risk"].get("slippage", 0.0005),
        max_pos=cfg["risk"].get("max_position_pct", 0.5),
        min_hold_bars=cfg["risk"].get("min_hold_bars", 6),
        cooldown_bars=cfg["risk"].get("cooldown_bars", 4),
        rebalance_threshold=cfg["risk"].get("rebalance_threshold", 0.25),
    )

    print("Backtest stats:", stats)



    # --- İNSANİ ÖZET ---
    # Dönem (ay cinsinden)
    start_ts = sigdf.index[0]
    end_ts = sigdf.index[-1]
    months = (end_ts - start_ts).days / 30.44

    # Buy & Hold getiri (aynı dönem)
    bh_ret_pct = (sigdf["close"].iloc[-1] / sigdf["close"].iloc[0] - 1.0) * 100.0
    strat_ret_pct = float(stats.get("total_return_pct", 0.0))
    diff_pct = strat_ret_pct - bh_ret_pct

    print("\n================= ÖZET =================")
    print(f"Dönem              : {start_ts}  →  {end_ts}  (~{months:.1f} ay)")
    print(f"Buy & Hold Getirisi: {bh_ret_pct:+.2f}%")
    print(f"Strateji Getirisi  : {strat_ret_pct:+.2f}%")
    print(f"Fark (Strat - B&H) : {diff_pct:+.2f}%")
    print(f"Maks. DD           : {stats.get('max_dd_pct', 0):.2f}%")
    print(f"Sharpe-benzeri     : {stats.get('sharpe_like', 0):.2f}")
    print(f"İşlem sayısı       : {stats.get('num_trades', 0)}")

    # Kullanılan metrikler (bu stratejinin karar mantığı)
    print("\nKullanılan metrikler / kaynak veriler:")
    print("- BTC fiyat serisi (OHLCV), RSI(14), SMA(20/50), MA Cross (sma_fast vs sma_slow)")
    print("- Volatilite: ATR(14) ve ATR yüzdesi (atr_pct)")
    print("- Geleneksel piyasa: QQQ (Nasdaq 100 ETF) → trend & BTC-QQQ korelasyonu")
    print("- Duygu (haber): CoinDesk & Cointelegraph RSS başlıkları → VADER sentiment (opsiyonel)")
    print("- (Opsiyonel) Makro ICS takvimi kapalıysa etkisizdir. Altın (XAU) şu anda çekilmiyor; istersek ekleriz.")
    print("  Not: Altın verisini dahil etmek için 'src/data_aggregator.py' ve 'build_dataset' tarafına XAUUSD ekleyebiliriz.")

    # Son 10 işlemi tablo gibi göster
    if len(trades) > 0:
        print("\nSon işlemler (en yeni 10):")
        head = f"{'time':<20} {'action':<10} {'price':>10} {'old→new':>14} {'equity':>10}"
        print(head)
        print("-" * len(head))
        for tr in trades[-10:]:
            tm = tr.get("time", "")
            action = tr.get("action", "")
            price = tr.get("price", 0.0)
            old_t = tr.get("old_target", 0.0)
            new_t = tr.get("new_target", 0.0)
            eq = tr.get("equity", 0.0)
            print(f"{tm[:19]:<20} {action:<10} {price:>10.2f} {old_t:>5.2f}→{new_t:<5.2f} {eq:>10.4f}")
    else:
        print("\nİşlem bulunmadı.")

    # Dosyaları yine de kaydet
    out.to_csv(os.path.join(cfg["general"]["data_dir"], "backtest_equity.csv"))
    try:
        trades_path = os.path.join(cfg["general"]["data_dir"], "trades.csv")
        pd.DataFrame(trades).to_csv(trades_path, index=False)
        print(f"\nKaydedildi: {trades_path}")
    except Exception:
        pass

    return stats


def run_train(cfg, logger):
    feats = build_dataset(cfg, logger)
    model_path = os.path.join(cfg["general"]["models_dir"], "btc_gb.joblib")
    clf = train_model(feats, cfg, model_path)
    print(f"Model saved to {model_path}")

def main():
    ap = argparse.ArgumentParser(description="BTC Machine")
    ap.add_argument("cmd", choices=["download", "train", "backtest"], help="Action to run")
    ap.add_argument("--config", default="./config/settings.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    logger = get_logger(level=cfg.get("general", {}).get("log_level", "INFO"))

    # Varsayılanları güvenle doldur
    cfg.setdefault("general", {})
    cfg["general"].setdefault("data_dir", "./data")
    cfg["general"].setdefault("models_dir", "./models")
    cfg["general"].setdefault("state_dir", "./state")
    cfg["general"].setdefault("log_level", "INFO")
    cfg.setdefault("ml", {})
    cfg["ml"].setdefault("enable", False)

    # Dizinleri oluştur
    os.makedirs(cfg["general"]["data_dir"], exist_ok=True)
    os.makedirs(cfg["general"]["models_dir"], exist_ok=True)
    os.makedirs(cfg["general"]["state_dir"], exist_ok=True)


    if args.cmd == "download":
        download_all(cfg, logger)
    elif args.cmd == "train":
        run_train(cfg, logger)
    elif args.cmd == "backtest":
        run_backtest(cfg, logger)

if __name__ == "__main__":
    main()
