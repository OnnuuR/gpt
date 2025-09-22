from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class BTParams:
    fee: float = 0.0004
    slippage: float = 0.0005
    max_position: float = 1.0         # |position| üst sınırı
    target_floor: float = 0.0         # allow flat
    rebalance_threshold: float = 0.25 # hedefe yakınsama eşiği
    min_hold_bars: int = 6
    cooldown_bars: int = 2
    allow_short: bool = True
    decision_every_n_bars: int = 3

    # risk kesiciler (opsiyonel; 0 ise devre dışı)
    stop_loss_pct: float = 0.0        # örn 0.03 (=%3)
    take_profit_pct: float = 0.0      # örn 0.04
    trailing_stop_pct: float = 0.0    # örn 0.05


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _apply_cost(equity: float, delta_pos: float, p: BTParams) -> float:
    """Pozisyon değişiminde işlem maliyeti uygula."""
    if delta_pos == 0.0:
        return equity
    cost = (abs(delta_pos)) * (p.fee + p.slippage)
    return equity * (1.0 - cost)


def _price_ret(cur: float, prev: float) -> float:
    if prev <= 0 or not np.isfinite(prev) or not np.isfinite(cur):
        return 0.0
    return (cur / prev) - 1.0


def _update_trailing(extreme: float, price: float, pos_sign: int) -> float:
    """Long için tepeyi, short için dibi günceller."""
    if pos_sign > 0:
        return max(extreme, price)
    elif pos_sign < 0:
        return min(extreme, price)
    return extreme


def _trail_hit(entry_extreme: float, price: float, pos_sign: int, trail_pct: float) -> bool:
    if trail_pct <= 0.0 or pos_sign == 0:
        return False
    if pos_sign > 0:
        # long: tepeye göre düşüş
        return (entry_extreme - price) / entry_extreme >= trail_pct if entry_extreme > 0 else False
    else:
        # short: dibe göre yükseliş
        return (price - entry_extreme) / entry_extreme >= trail_pct if entry_extreme > 0 else False


def event_backtest(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Beklenen sütunlar:
      - close (float)
      - signal (int in {-1,0,1})
      - confidence (float in [0,1])
    Dönüş:
      - out_df (equity zaman serisi)
      - stats (sözlük)
      - trades_df (işlem günlüğü)
    """
    # Parametreleri yükle
    p = BTParams(**{
        k: v for k, v in kwargs.items()
        if k in BTParams().__dict__.keys()
    })

    # Giriş doğrulamaları
    need_cols = {"close", "signal", "confidence"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"event_backtest: eksik sütunlar: {missing}")
    if len(df) < 3:
        raise ValueError("event_backtest: veri çok kısa")

    # Numpy view
    close = df["close"].to_numpy(dtype=float)
    signal = df["signal"].to_numpy(dtype=float)
    conf = df["confidence"].to_numpy(dtype=float)

    n = len(df)
    out_time = df.index

    # Başlangıç durumları
    equity = 1.0
    pos = 0.0                # signed exposure [-1..1]
    prev_price = close[0]
    entry_price = None       # aktif pozisyon giriş fiyatı
    trail_extreme = None     # trailing için tepe/dip
    hold_bars = 0
    cooldown = 0

    # Kayıtlar
    eq_series = [equity]
    trades: List[Dict[str, Any]] = []

    # Long/short pnL ayrımı
    pnl_long = 0.0
    pnl_short = 0.0

    # Karar seyreltme sayacı
    decide_tick = 0

    # İstatistik
    peak_equity = equity
    max_dd = 0.0

    print(f"[BT] n_bars={n}, first={close[0]:.2f} last={close[-1]:.2f}")

    for i in range(1, n):
        price = close[i]
        r = _price_ret(price, prev_price)

        # 1) Fiyat hareketinden equity güncelle (mevcut pozisyonla)
        equity *= (1.0 + pos * r)
        eq_series.append(equity)

        # PnL kırılımı (işlemsel olmayan, yaklaşık)
        if pos > 0:
            pnl_long += pos * r
        elif pos < 0:
            pnl_short += pos * r  # negatif pos * r pozitif olabilir (short kârı)

        # 2) Risk kesiciler (stop/tp/trailing) — sadece aktif pozisyonda
        exit_reason: Optional[str] = None
        if pos != 0.0 and entry_price is not None:
            pos_sign = _sign(pos)
            # TP / SL
            ret_from_entry = (price / entry_price) - 1.0
            if p.take_profit_pct > 0.0:
                if (pos_sign > 0 and ret_from_entry >= p.take_profit_pct) or \
                   (pos_sign < 0 and -ret_from_entry >= p.take_profit_pct):
                    exit_reason = "TAKE_PROFIT"
            if exit_reason is None and p.stop_loss_pct > 0.0:
                if (pos_sign > 0 and ret_from_entry <= -p.stop_loss_pct) or \
                   (pos_sign < 0 and -ret_from_entry <= -p.stop_loss_pct):
                    exit_reason = "STOP_LOSS"
            # Trailing
            if exit_reason is None and p.trailing_stop_pct > 0.0 and trail_extreme is not None:
                if _trail_hit(trail_extreme, price, pos_sign, p.trailing_stop_pct):
                    exit_reason = "TRAIL_STOP"

            if exit_reason is not None:
                # Pozisyonu kapat
                delta = -pos
                equity = _apply_cost(equity, delta, p)
                trades.append({
                    "time": out_time[i], "action": exit_reason,
                    "price": price, "old→new": f"{pos:.2f}→0.00", "equity": round(equity, 4)
                })
                pos = 0.0
                entry_price = None
                trail_extreme = None
                hold_bars = 0
                cooldown = max(cooldown, p.cooldown_bars)
                # equity serisine bu işlemin maliyet etkisi zaten yansıdı

        # 3) Karar sıklığı / cooldown / min_hold
        decide_tick += 1
        can_decide = (
            (decide_tick % max(1, p.decision_every_n_bars) == 0) and
            (cooldown <= 0) and
            (hold_bars >= p.min_hold_bars or pos == 0.0)
        )

        if can_decide:
            # hedef maruziyet
            raw_target = float(signal[i]) * float(conf[i])
            if not p.allow_short:
                raw_target = max(0.0, raw_target)

            # sınırlar
            target = float(np.clip(raw_target, -p.max_position, p.max_position))

            # floor (sadece pozitif floor; short için simetrik kullanmıyoruz)
            if target >= 0:
                target = max(target, p.target_floor)
            else:
                target = min(target, -p.target_floor)

            if abs(target - pos) >= p.rebalance_threshold:
                # trade
                delta = target - pos
                # masraf uygula
                equity = _apply_cost(equity, delta, p)

                # aksiyon tipi
                old_pos = pos
                pos = target
                if old_pos == 0 and pos > 0:
                    action = "ENTER_LONG"
                elif old_pos == 0 and pos < 0:
                    action = "ENTER_SHORT"
                elif pos == 0:
                    action = "EXIT"
                elif _sign(old_pos) != _sign(pos):
                    action = "FLIP"
                else:
                    action = "REBALANCE"

                trades.append({
                    "time": out_time[i], "action": action,
                    "price": price, "old→new": f"{old_pos:.2f}→{pos:.2f}",
                    "equity": round(equity, 4)
                })

                # giriş bilgileri
                if pos != 0.0:
                    entry_price = price
                    trail_extreme = price
                    hold_bars = 0
                else:
                    entry_price = None
                    trail_extreme = None
                    hold_bars = 0
                    cooldown = max(cooldown, p.cooldown_bars)
            else:
                # pozisyon korunuyor → trailing extreme güncelle
                if pos != 0.0 and trail_extreme is not None:
                    trail_extreme = _update_trailing(trail_extreme, price, _sign(pos))

        # sayaçlar
        if pos != 0.0:
            hold_bars += 1
        if cooldown > 0:
            cooldown -= 1

        # istatistik: max drawdown
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_dd = max(max_dd, dd)

        prev_price = price

    # Çıkış dataframe
    out = pd.DataFrame({"equity": eq_series}, index=df.index)

    # İstatistikler
    total_return = (out["equity"].iloc[-1] / out["equity"].iloc[0]) - 1.0
    rets = out["equity"].pct_change().fillna(0.0).to_numpy()
    vol = np.std(rets) * np.sqrt(24 * 365) if len(rets) > 1 else 0.0
    sharpe_like = (np.mean(rets) / np.std(rets)) * np.sqrt(24 * 365) if np.std(rets) > 1e-12 else 0.0

    # B&H karşılaştırması
    bh = (close[-1] / close[0]) - 1.0

    stats = {
        "total_return_pct": total_return * 100.0,
        "sharpe_like": sharpe_like,
        "max_dd_pct": max_dd * 100.0,
        "num_trades": len(trades),
        "bh_return_pct": bh * 100.0,
        "pnl_long_rel": pnl_long,   # bilgilendirici
        "pnl_short_rel": pnl_short, # bilgilendirici
    }

    # Konsol çıktısı
    print(f"[BT] final_target={float(signal[-1]) * float(conf[-1]):.2f}  "
          f"equity_start={out['equity'].iloc[0]:.4f}  equity_end={out['equity'].iloc[-1]:.4f}  steps={n}")

    # İşlem günlüğü dataframe
    trades_df = pd.DataFrame(trades, columns=["time", "action", "price", "old→new", "equity"])

    return out, stats, trades_df
