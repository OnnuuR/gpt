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
    close = df["close"].astype(float).to_numpy()
    sig = df["signal"].astype(int).to_numpy()
    conf = df["confidence"].astype(float).to_numpy()

    equity = [1.0]
    target = 0.0
    entry_i = -10**9
    last_trade_i = -10**9
    trades = []
    exit_streak = 0
    cost = fee + slip

    for i in range(1, len(df)):
        price_prev, price = close[i - 1], close[i]
        s = int(sig[i])
        c = float(np.clip(conf[i], 0.0, 1.0))
        new_target = target

        # exit streak takibi
        if s == -1:
            exit_streak += 1
        else:
            exit_streak = 0

        # ÇIKIŞ (onay barı + min_hold)
        if s == -1 and target > 0 and (i - entry_i) >= min_hold_bars and exit_streak >= exit_confirm_bars:
            new_target = 0.0

        # GİRİŞ / RE-BALANCE (cooldown sonrası)
        elif s == 1:
            if (i - last_trade_i) >= cooldown_bars:
                if target <= 0:
                    new_target = float(np.clip(max_pos * c, 0.0, max_pos))
                else:
                    desired = float(np.clip(max_pos * c, 0.0, max_pos))
                    if abs(desired - target) >= rebalance_threshold * max_pos:
                        new_target = desired

        # Hedef değiştiyse tek seferde uygula (maliyet yedir)
        if abs(new_target - target) > 1e-9:
            trade_cost = abs(new_target - target) * cost
            equity[-1] *= (1.0 - trade_cost)
            trades.append({"i": i, "price": price, "delta": new_target - target, "equity": equity[-1]})
            target = new_target
            last_trade_i = i
            entry_i = i if target > 0.0 else -10**9

        # Fiyat hareketi
        ret = (price / (price_prev + 1e-12)) - 1.0
        equity.append(equity[-1] * (1.0 + target * ret))

    out = df.copy().iloc[: len(equity)]
    out["equity"] = equity
    stats = _stats_from_equity(equity)
    stats["num_trades"] = len(trades)
    return out, stats, trades
