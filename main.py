# main.py
import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from dateutil import tz
from joblib import load
from typing import Dict, Any

from src.utils import get_logger
from src import data_aggregator as da
from src.features import add_core_features
from src.decision import rule_signal, ensemble_with_ml
from src.ml_model import train_model, predict_proba_row
from src.backtest import event_backtest
from src.macro_calendar import fetch_ics_events, in_risk_window

UTC = tz.gettz("UTC")


def load_cfg(path: str):
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
    qqq = da.fetch_yf(yft["qqq"], period=f"{lookback_days}d",
                      interval=tf if tf in ["1h", "4h", "1d"] else "1h")
    dxy = da.fetch_yf(yft["dxy"], period=f"{lookback_days}d",
                      interval=tf if tf in ["1h", "4h", "1d"] else "1h")
    if not qqq.empty:
        qqq = qqq.rename(columns={"close": "qqq_close"})
        qqq.to_csv(os.path.join(cfg["general"]["data_dir"], "qqq.csv"))
    if not dxy.empty:
        dxy = dxy.rename(columns={"close": "dxy_close"})
        dxy.to_csv(os.path.join(cfg["general"]["data_dir"], "dxy.csv"))
    logger.info(f"QQQ rows={len(qqq)} DXY rows={len(dxy)}")

    sent = da.fetch_rss_sentiment(cfg["sentiment"]["rss_feeds"],
                                  hours=cfg["sentiment"]["rolling_hours"],
                                  logger=logger)
    if not sent.empty:
        sent.to_csv(os.path.join(cfg["general"]["data_dir"], "sentiment.csv"))
        logger.info(f"Saved sentiment.csv ({len(sent)} rows)")
    fng = da.fetch_fng(days=min(lookback_days, 365 * 2))
    if not fng.empty:
        fng.to_csv(os.path.join(cfg["general"]["data_dir"], "fng.csv"))
        logger.info(f"Saved fng.csv ({len(fng)} rows)")


def build_dataset(cfg: Dict[str, Any], logger) -> pd.DataFrame:
    """
    Tüm kaynakları 1 saatlik banda indirip kesişim (inner join) ile tamamen
    eş-zamanlı bir veri seti oluşturur; ardından temel özellikleri üretir.
    """
    data_dir = cfg["general"]["data_dir"]
    tf = cfg.get("market", {}).get("timeframe", "1h").lower()
    if tf != "1h":
        raise ValueError("Bu senkron sürüm şu an '1h' için tasarlandı.")

    lookback_days = int(cfg.get("market", {}).get("lookback_days", 120))
    use_sentiment = bool(cfg.get("sync", {}).get("use_sentiment", False))

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

    # Saatlik garanti
    base = base.resample("1h").last()

    # Yardımcı CSV okuyucu
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

    # --- 2) Opsiyoneller (1h + pencere) ---
    qqq = _read_csv("qqq.csv", ["qqq_close"])
    if not qqq.empty:
        qqq = qqq.loc[(qqq.index >= start) & (qqq.index <= end)].resample("1h").last()
        logger.info("-> Başarıyla yüklendi ve 1h'e indirildi: QQQ")

    dxy = _read_csv("dxy.csv", ["dxy_close"])
    if not dxy.empty:
        dxy = dxy.loc[(dxy.index >= start) & (dxy.index <= end)].resample("1h").last()
        logger.info("-> Başarıyla yüklendi ve 1h'e indirildi: DXY")

    fng = _read_csv("fng.csv", ["fng"])
    if not fng.empty:
        fng = fng.loc[(fng.index >= start) & (fng.index <= end)].resample("1h").ffill()
        logger.info("-> Başarıyla yüklendi ve 1h'e indirildi: FNG")

    sent = pd.DataFrame()
    if use_sentiment:
        sent_raw = _read_csv("sentiment.csv", ["sentiment"])
        if not sent_raw.empty:
            sent = sent_raw.loc[(sent_raw.index >= start) & (sent_raw.index <= end)].resample("1h").mean()
            logger.info("-> Başarıyla yüklendi ve 1h'e indirildi: SENTIMENT (aktif)")
    else:
        logger.info("SENTIMENT devre dışı (sync.use_sentiment=false).")

    # --- 3) INNER JOIN (kesişim) ---
    merged = base.copy()
    for df in [qqq, dxy, fng]:
        if not df.empty:
            merged = merged.join(df, how="inner")
    if use_sentiment and not sent.empty:
        merged = merged.join(sent, how="inner")

    # Numerik tipler
    for col in ["open", "high", "low", "close", "volume", "qqq_close", "dxy_close", "fng", "sentiment"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # ÖZET
    logger.info("--- Veri Yükleme Özeti ---")
    def _span(df, name):
        if df.empty:
            return
        logger.info(f"[{name}] Zaman aralığı: {df.index.min()} -> {df.index.max()} | Satır sayısı: {len(df)}")
    _span(base, "BTC")
    _span(qqq, "QQQ")
    _span(dxy, "DXY")
    if use_sentiment:
        _span(sent, "SENTIMENT")
    _span(fng, "FNG")
    logger.info(f"[SYNC] Kesişim satır sayısı: {len(merged)} (1h)")

    # --- 4) Özellikler ---
    logger.info("Çekirdek özellikler (feature) hesaplanıyor...")
    feats = add_core_features(
        merged,
        rsi_period=cfg["features"]["rsi_period"],
        sma_fast=cfg["features"]["sma_fast"],
        sma_slow=cfg["features"]["sma_slow"],
        corr_window=cfg["features"]["corr_window"],
        qqq_col="qqq_close",
    )

    # SADECE çekirdek warmup kolonlarını zorunlu tut
    warm_cols = ["sma_fast", "sma_slow", "atr", "atr_pct"]
    feats = feats.dropna(subset=[c for c in warm_cols if c in feats.columns])

    # close güvence
    if feats["close"].isna().any():
        feats["close"] = pd.to_numeric(feats["close"], errors="coerce").ffill().bfill()

    return feats


def run_backtest(cfg, logger):
    feats = build_dataset(cfg, logger)

    # Opsiyonel ML
    model_path = os.path.join(cfg["general"]["models_dir"], "btc_gb.joblib")
    ml_cfg = cfg.get("ml", {})
    model = load(model_path) if (ml_cfg.get("enable", False) and os.path.exists(model_path)) else None

    # Makro takvim (opsiyonel)
    macro_cfg = cfg.get("macro", {})
    events = None
    if bool(macro_cfg.get("use_calendar")) and macro_cfg.get("ics_links"):
        events = fetch_ics_events(macro_cfg.get("ics_links"))

    # --- SİNYAL ÜRETİMİ ---
    rows = []
    risk_cfg = cfg.get("risk", {})
    dec_n = int(risk_cfg.get("decision_every_n_bars", 12))
    max_pos = float(risk_cfg.get("max_position", risk_cfg.get("max_position_pct", 0.60)))
    allow_short = bool(risk_cfg.get("allow_short", True))

    for i in range(1, len(feats)):
        ts = feats.index[i]
        if dec_n > 1 and (i % dec_n != 0):
            continue

        # Kurallar
        sig, conf = rule_signal(feats, i, cfg)  # imza: (feats, i, cfg)
        if not allow_short and sig < 0:
            sig, conf = 0, 0.0

        # Makro riske göre block (sadece long'u kesiyoruz)
        if events is not None and not events.empty and sig > 0:
            now_utc = pd.Timestamp(ts, tz="UTC").to_pydatetime()
            block, _ = in_risk_window(
                now_utc,
                events,
                macro_cfg.get("pre_event_hours", 2),
                macro_cfg.get("post_event_minutes", 60),
            )
            if block:
                sig, conf = 0, 0.0

        # Opsiyonel ML ensemble (sadece saat başlarında örnek)
        proba = None
        if model is not None and ts.minute in [0, 15, 30, 45]:
            # row olarak feats.iloc[i] verelim
            proba = predict_proba_row(model, feats.iloc[i])
            s2, c2 = ensemble_with_ml(sig, conf, proba)
            sig, conf = s2, c2

        raw = float(np.clip(sig * conf, -1.0, 1.0))
        target = float(np.clip(raw, -1.0, 1.0)) * max_pos

        rows.append({
            "time": ts,
            "signal": int(np.sign(sig)),  # -1,0,+1
            "confidence": float(conf),    # [0..1]
            "target": float(target),      # [-max_pos..+max_pos]
            "close": float(feats["close"].iloc[i]),
        })

    # Sinyal tablosu
    sigdf = pd.DataFrame(rows)
    if not sigdf.empty and "time" in sigdf.columns:
        sigdf["time"] = pd.to_datetime(sigdf["time"], utc=True, errors="coerce")
        sigdf = sigdf.dropna(subset=["time"]).set_index("time").sort_index()
    else:
        print("[BT] Uyarı: Sinyal DataFrame'i boş! Kadansı düşürmeyi/kuralları gevşetmeyi deneyin.")
        return

    # Tip/limit düzeltmeleri
    sigdf["signal"] = pd.to_numeric(sigdf.get("signal", 0), errors="coerce").fillna(0).astype(int)
    sigdf["confidence"] = (
        pd.to_numeric(sigdf.get("confidence", 0.0), errors="coerce")
        .fillna(0.0).clip(0.0, 1.0)
    )
    sigdf["target"] = (
        pd.to_numeric(sigdf.get("target", 0.0), errors="coerce")
        .fillna(0.0).clip(-max_pos, max_pos)
    )
    sigdf["close"] = pd.to_numeric(sigdf["close"], errors="coerce").ffill().bfill()

    # Sağlık kontrolü
    try:
        cmin, cmax = float(sigdf["close"].min()), float(sigdf["close"].max())
        npos = int((sigdf["signal"] == 1).sum())
        nneg = int((sigdf["signal"] == -1).sum())
        print(f"[CHECK] close min/max: {cmin:.2f} / {cmax:.2f} | signals +1:{npos}  -1:{nneg}")
    except Exception as e:
        print(f"[CHECK] Sağlık kontrolü atlandı: {e}")

    # --- Backtest ---
    fee = risk_cfg.get("fee", risk_cfg.get("taker_fee", 0.0004))
    slip = risk_cfg.get("slippage", 0.0005)
    max_pos_bt = risk_cfg.get("max_position", risk_cfg.get("max_position_pct", max_pos))
    kwargs_bt = dict(
        fee=fee,
        slip=slip,
        max_pos=max_pos_bt,
        min_hold_bars=risk_cfg.get("min_hold_bars", 12),
        cooldown_bars=risk_cfg.get("cooldown_bars", 4),
        rebalance_threshold=risk_cfg.get("rebalance_threshold", 0.40),
        allow_short=allow_short,
        decision_every_n_bars=dec_n,
        stop_loss_pct=risk_cfg.get("stop_loss_pct", None),
        take_profit_pct=risk_cfg.get("take_profit_pct", None),
        trailing_stop_pct=risk_cfg.get("trailing_stop_pct", None),
    )

    out, stats, trades = event_backtest(sigdf, **kwargs_bt)
    print("Backtest stats:", stats)

    # --- Özet ---
    start_ts = sigdf.index[0]
    end_ts = sigdf.index[-1]
    months = (end_ts - start_ts).days / 30.44

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

    print("\nKullanılan metrikler / kaynak veriler:")
    print("- BTC fiyat serisi (OHLCV), RSI(14), SMA(20/50), MA Cross (sma_fast vs sma_slow)")
    print("- Volatilite: ATR(14) ve ATR yüzdesi (atr_pct)")
    print("- Geleneksel piyasa: QQQ (Nasdaq 100 ETF) → trend & BTC-QQQ korelasyonu")
    print("- Duygu (haber): CoinDesk & Cointelegraph RSS başlıkları → VADER sentiment (opsiyonel)")
    print("- (Opsiyonel) Makro ICS takvimi kapalıysa etkisizdir. Altın (XAU) şu anda çekilmiyor; istersek ekleriz.")
    print("  Not: Altın verisini dahil etmek için 'src/data_aggregator.py' ve 'build_dataset' tarafına XAUUSD ekleyebiliriz.")

    # --- Rapor: Son işlemler & kayıt (DataFrame veya list güvenli) ---
    print("\nSon işlemler (en yeni 10):")
    print("time                 action          price        old→new     equity")
    print("--------------------------------------------------------------------")

    last10_records = []
    try:
        if isinstance(trades, pd.DataFrame):
            last10_records = trades.tail(10).to_dict("records")
        elif isinstance(trades, list):
            last10_records = trades[-10:]
        else:
            last10_records = []
    except Exception:
        try:
            last10_records = trades[-10:] if isinstance(trades, list) else []
        except Exception:
            last10_records = []

    if not last10_records:
        print("(yok)")
    else:
        for tr in last10_records:
            t = tr.get("time", "")
            t_str = t.strftime("%Y-%m-%d %H:%M:%S") if hasattr(t, "strftime") else str(t)[:19]
            action = str(tr.get("action", ""))

            try:
                price_str = f"{float(tr.get('price', float('nan'))):>10.2f}"
            except Exception:
                price_str = f"{str(tr.get('price', '')):>10}"
            try:
                eq_str = f"{float(tr.get('equity', float('nan'))):>8.4f}"
            except Exception:
                eq_str = f"{str(tr.get('equity', '')):>8}"

            if "old→new" in tr:
                old_new = str(tr["old→new"])
            else:
                old_t = tr.get("old_target", None)
                new_t = tr.get("new_target", None)
                if old_t is not None and new_t is not None:
                    try:
                        old_new = f"{float(old_t):.2f}→{float(new_t):.2f}"
                    except Exception:
                        old_new = f"{old_t}→{new_t}"
                else:
                    old_new = ""

            print(f"{t_str:<19}  {action:<11}  {price_str}  {old_new:>8}  {eq_str}")

    # CSV kayıtları
    try:
        bt_eq_path = os.path.join(cfg["general"]["data_dir"], "backtest_equity.csv")
        out.to_csv(bt_eq_path)
    except Exception as e:
        print(f"\n[WARN] Equity CSV kaydı başarısız: {e}")

    try:
        trades_path = os.path.join(cfg["general"]["data_dir"], "trades.csv")
        if isinstance(trades, pd.DataFrame):
            trades.to_csv(trades_path, index=False)
        else:
            pd.DataFrame(trades).to_csv(trades_path, index=False)
        print(f"\nKaydedildi: {trades_path}")
    except Exception as e:
        print(f"\n[WARN] İşlemler CSV kaydı başarısız: {e}")

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

    # Varsayılanlar
    cfg.setdefault("general", {})
    cfg["general"].setdefault("data_dir", "./data")
    cfg["general"].setdefault("models_dir", "./models")
    cfg["general"].setdefault("state_dir", "./state")
    cfg["general"].setdefault("log_level", "INFO")

    cfg.setdefault("ml", {})
    cfg["ml"].setdefault("enable", False)

    cfg.setdefault("macro", {})
    cfg["macro"].setdefault("use_calendar", False)
    cfg["macro"].setdefault("ics_links", [])
    cfg["macro"].setdefault("pre_event_hours", 2)
    cfg["macro"].setdefault("post_event_minutes", 60)

    # Dizinler
    os.makedirs(cfg["general"]["data_dir"], exist_ok=True)
    os.makedirs(cfg["general"]["models_dir"], exist_ok=True)
    os.makedirs(cfg["general"]["state_dir"], exist_ok=True)

    # Komut çalıştırma
    if args.cmd == "download":
        download_all(cfg, logger)
    elif args.cmd == "train":
        run_train(cfg, logger)
    elif args.cmd == "backtest":
        run_backtest(cfg, logger)
    else:
        logger.error(f"Bilinmeyen komut: {args.cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
