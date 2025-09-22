import pandas as pd
import numpy as np

def _stats_from_equity(equity):
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    total_return_pct = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
    sharpe_like = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(24 * 365) if len(rets) > 3 else 0.0
    max_dd_pct = (1 - (eq / eq.cummax()).min()) * 100.0
    return {
        "total_return_pct": float(total_return_pct),
        "sharpe_like": float(sharpe_like),
        "max_dd_pct": float(max_dd_pct),
    }

def event_backtest(
    df,
    fee=0.0004,
    slip=0.0005,
    max_pos=0.5,
    min_hold_bars=12,
    cooldown_bars=8,
    rebalance_threshold=0.5,
    exit_confirm_bars=2,
):
    # --- Güvenli fiyat/sinyal okuma + sıra garantisi ---
    df = df.sort_index()
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    sig   = pd.to_numeric(df["signal"], errors="coerce").fillna(0).to_numpy(dtype=int)
    conf  = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    n = len(close)
    print(f"[BT] n_bars={n}, first={close[0]:.2f} last={close[-1]:.2f}")  # teşhis

    # Eğer yeterli bar yoksa boş metrik döndür
    if n < 2:
        out = df.copy().iloc[:1]
        out["equity"] = [1.0]
        stats = {"total_return_pct": 0.0, "sharpe_like": 0.0, "max_dd_pct": 0.0, "num_trades": 0}
        return out, stats, []

    equity = [1.0]   # listeyi hiç geriye dönüp değiştirmeyelim
    eq = 1.0         # scalar equity; tüm hesaplar bunun üstünde, sonra append
    target = 0.0
    entry_i = -10**9
    last_trade_i = -10**9
    trades = []
    exit_streak = 0
    cost = fee + slip

    for i in range(1, n):
        price_prev, price = close[i - 1], close[i]
        s = int(sig[i])
        c = float(np.clip(conf[i], 0.0, 1.0))
        new_target = target

        # exit streak
        exit_streak = (exit_streak + 1) if s == -1 else 0

        # ÇIKIŞ
        if s == -1 and target > 0 and (i - entry_i) >= min_hold_bars and exit_streak >= exit_confirm_bars:
            new_target = 0.0

        # GİRİŞ / REBALANCE
        elif s == 1:
            if (i - last_trade_i) >= cooldown_bars:
                if target <= 0:
                    new_target = float(np.clip(max_pos * c, 0.0, max_pos))
                else:
                    desired = float(np.clip(max_pos * c, 0.0, max_pos))
                    if abs(desired - target) >= rebalance_threshold * max_pos:
                        new_target = desired

        # Hedef değiştiyse maliyeti UYGULA (scalar eq üzerinde), listeyi değil
        if abs(new_target - target) > 1e-9:
            trade_cost = abs(new_target - target) * cost
            eq *= (1.0 - trade_cost)

            old_target = target
            target = new_target
            last_trade_i = i
            entry_i = i if target > 0.0 else -10**9

            tr = {
                "i": i,
                "time": str(df.index[i]),
                "action": "EXIT" if (target == 0.0 and old_target > 0.0)
                          else ("ENTER" if (old_target <= 0.0 and target > 0.0) else "REBALANCE"),
                "price": float(price),
                "old_target": float(old_target),
                "new_target": float(target),
                "equity": float(eq),
            }
            trades.append(tr)

        # Fiyat hareketinden PnL → her barda ekle
        ret = (price / (price_prev + 1e-12)) - 1.0
        eq *= (1.0 + target * ret)
        equity.append(eq)

    out = df.copy().iloc[: len(equity)]
    out["equity"] = equity

    stats = _stats_from_equity(equity)
    stats["num_trades"] = len(trades)

    print(f"[BT] final_target={target:.2f}  equity_start={equity[0]:.4f}  equity_end={equity[-1]:.4f}  steps={len(equity)}")
    return out, stats, trades
