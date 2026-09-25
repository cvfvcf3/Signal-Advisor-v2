"""
Evaluator: checks PENDING journal signals against subsequent CLOSED
candles and resolves them to CORRECT / INCORRECT (a third status,
INVALIDATED, is applied separately by advisor_engine.py when the live
signal reverses against an open position — that's a risk-management
exit, not a price-target timeout, so it stays distinct).

NO "EXPIRED" STATUS: a real trade is never left in limbo — if the
evaluation window runs out without hitting SL or TP/trailing-exit, the
position is treated as closed at the last available price, and the
outcome is decided by the sign of the resulting pnl_pct: positive (or
zero) closes CORRECT, negative closes INCORRECT. This mirrors how an
actual trade works — you're always either up or down when you close,
there's no third option.

PATH-AWARE, CANDLE-BY-CANDLE WALK, chronological order.

HARD STOP-LOSS: the original stop_loss level always wins if touched,
trailing armed or not.

TRAILING (when the mode's trailing.enabled is set):
  - Arms ONLY once price reaches the full original take_profit level —
    not a fraction of it. Trailing exists to capture MORE than the
    original target when the trend keeps going, never less.
  - Once armed, the trailing-stop price is
        BUY:  max(peak_high * (1 - trail_distance_pct), take_profit)
        SELL: min(peak_low  * (1 + trail_distance_pct), take_profit)
    — floored at the original TP, so a trailing exit is never worse
    than the plain fixed-TP outcome would have been.
  - If price reaches TP but the window ends before any retracement
    trigger, the trade closes at the last candle's close — since that
    close is at/above the original TP (for a BUY), pnl_pct is still
    positive and it resolves CORRECT, same conclusion as before, just
    reached via the timeout path instead of an explicit trigger.

WHEN TRAILING IS DISABLED for a mode: simple fixed TP/SL — exit CORRECT
the instant the original take_profit is touched.

PNL: every resolution computes pnl_pct — the actual signed % price move
from entry to exit_price, uniformly for both CORRECT and INCORRECT.

This only ever evaluates against CLOSED candles the caller supplies —
it never estimates or assumes future price action.
"""


def pnl_pct(action, entry, exit_price):
    """Signed % price move from entry to exit_price, trade-direction aware."""
    if action == "BUY":
        return ((exit_price - entry) / entry) * 100
    else:  # SELL
        return ((entry - exit_price) / entry) * 100


def is_ready_to_evaluate(signal, candles_since_entry):
    return len(candles_since_entry) >= signal["evaluate_after_candles"]


def evaluate_signal(signal, candles_since_entry, trailing_cfg=None):
    """
    Returns None if not enough candles yet and no exit condition met.
    Otherwise dict: status ("CORRECT"|"INCORRECT"), exit_price, pnl_pct,
    mae_pct, mfe_pct.
    """
    entry = signal["entry_price"]
    action = signal["action"]
    tp = signal["take_profit"]
    sl = signal["stop_loss"]
    max_candles = signal["evaluate_after_candles"]

    window = candles_since_entry[:max_candles]
    mae_pct = 0.0
    mfe_pct = 0.0

    trailing_enabled = bool(trailing_cfg and trailing_cfg.get("enabled"))
    trail_distance_pct = trailing_cfg.get("trail_distance_pct", 0.3) if trailing_cfg else 0.3
    armed = False
    peak = entry

    for candle in window:
        high, low = candle[2], candle[3]

        if action == "BUY":
            mfe_pct = max(mfe_pct, ((high - entry) / entry) * 100)
            mae_pct = max(mae_pct, ((entry - low) / entry) * 100)
            hit_sl = low <= sl
        else:
            mfe_pct = max(mfe_pct, ((entry - low) / entry) * 100)
            mae_pct = max(mae_pct, ((high - entry) / entry) * 100)
            hit_sl = high >= sl

        if hit_sl:
            return {"status": "INCORRECT", "exit_price": round(sl, 8),
                    "pnl_pct": round(pnl_pct(action, entry, sl), 4),
                    "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}

        if not trailing_enabled:
            hit_tp = (high >= tp) if action == "BUY" else (low <= tp)
            if hit_tp:
                return {"status": "CORRECT", "exit_price": round(tp, 8),
                        "pnl_pct": round(pnl_pct(action, entry, tp), 4),
                        "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}
            continue

        if action == "BUY":
            if not armed:
                if high >= tp:
                    armed = True
                    peak = high
            if armed:
                peak = max(peak, high)
                trail_stop_price = max(peak * (1 - trail_distance_pct), tp)
                if low <= trail_stop_price:
                    return {"status": "CORRECT", "exit_price": round(trail_stop_price, 8),
                            "pnl_pct": round(pnl_pct(action, entry, trail_stop_price), 4),
                            "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}
        else:
            if not armed:
                if low <= tp:
                    armed = True
                    peak = low
            if armed:
                peak = min(peak, low)
                trail_stop_price = min(peak * (1 + trail_distance_pct), tp)
                if high >= trail_stop_price:
                    return {"status": "CORRECT", "exit_price": round(trail_stop_price, 8),
                            "pnl_pct": round(pnl_pct(action, entry, trail_stop_price), 4),
                            "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}

    # Window ran out with no explicit trigger — close at last price, decide
    # by the sign of the actual resulting P&L. No limbo "EXPIRED" bucket.
    if len(window) >= max_candles:
        last_close = window[-1][4]
        final_pnl = pnl_pct(action, entry, last_close)
        status = "CORRECT" if final_pnl >= 0 else "INCORRECT"
        return {"status": status, "exit_price": round(last_close, 8),
                "pnl_pct": round(final_pnl, 4),
                "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}

    return None


def candles_since(all_closed_candles, entry_timestamp_ms):
    return [c for c in all_closed_candles if c[0] > entry_timestamp_ms]
