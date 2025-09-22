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

def build_dataset(cfg, logger):
    # BTC OHLCV
    base = pd.read_csv(os.path.join(cfg["general"]["data_dir"], "btc_ohlcv.csv"), index_col=0)
    base.index = pd.to_datetime(base.index, utc=True, errors="coerce")
    base = base[["open", "high", "low", "close", "volume"]]

    # Optional inputs
    qqq_path = os.path.join(cfg["general"]["data_dir"], "qqq.csv")
    dxy_path = os.path.join(cfg["general"]["data_dir"], "dxy.csv")
    sent_path = os.path.join(cfg["general"]["data_dir"], "sentiment.csv")
    fng_path = os.path.join(cfg["general"]["data_dir"], "fng.csv")

    qqq = pd.read_csv(qqq_path, index_col=0) if os.path.exists(qqq_path) else pd.DataFrame()
    if not qqq.empty:
        qqq.index = pd.to_datetime(qqq.index, utc=True, errors="coerce")

    dxy = pd.read_csv(dxy_path, index_col=0) if os.path.exists(dxy_path) else pd.DataFrame()
    if not dxy.empty:
        dxy.index = pd.to_datetime(dxy.index, utc=True, errors="coerce")

    sent = pd.read_csv(sent_path, index_col=0) if os.path.exists(sent_path) else pd.DataFrame()
    if not sent.empty:
        sent.index = pd.to_datetime(sent.index, utc=True, errors="coerce")

    fng = pd.read_csv(fng_path, index_col=0) if os.path.exists(fng_path) else pd.DataFrame()
    if not fng.empty:
        fng.index = pd.to_datetime(fng.index, utc=True, errors="coerce")

    merged = base.copy()
    if not qqq.empty:
        merged = merged.join(qqq[["qqq_close"]], how="left")
    if not dxy.empty:
        merged = merged.join(dxy[["dxy_close"]], how="left")
    if not sent.empty:
        merged = merged.join(sent[["sentiment"]], how="left")
    if not fng.empty:
        merged = merged.join(fng[["fng"]].resample("1h").ffill(), how="left")

    merged = merged.ffill().bfill().infer_objects(copy=False)

    feats = add_core_features(
        merged,
        rsi_period=cfg["features"]["rsi_period"],
        sma_fast=cfg["features"]["sma_fast"],
        sma_slow=cfg["features"]["sma_slow"],
        corr_window=cfg["features"]["corr_window"],
        qqq_col="qqq_close",
    )
    return feats

def run_backtest(cfg, logger):
    feats = build_dataset(cfg, logger)
    # optional ML
    model_path = os.path.join(cfg["general"]["models_dir"], "btc_gb.joblib")
    model = load(model_path) if (cfg["ml"]["enable"] and os.path.exists(model_path)) else None

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
    out.to_csv(os.path.join(cfg["general"]["data_dir"], "backtest_equity.csv"))
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
    logger = get_logger(level=cfg["general"]["log_level"])

    os.makedirs(cfg["general"]["data_dir"], exist_ok=True)
    os.makedirs(cfg["general"]["state_dir"], exist_ok=True)
    os.makedirs(cfg["general"]["models_dir"], exist_ok=True)

    if args.cmd == "download":
        download_all(cfg, logger)
    elif args.cmd == "train":
        run_train(cfg, logger)
    elif args.cmd == "backtest":
        run_backtest(cfg, logger)

if __name__ == "__main__":
    main()
