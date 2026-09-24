"""
Evaluator: checks PENDING journal signals against subsequent CLOSED
candles and resolves them to CORRECT / INCORRECT / EXPIRED (a fourth
status, INVALIDATED, is applied separately by advisor_engine.py when a
signal's underlying thesis reverses — see that file).

PATH-AWARE, CANDLE-BY-CANDLE WALK: each candle since entry is processed
in chronological order.

HARD STOP-LOSS: the original stop_loss level (computed at signal
creation) is always checked first and always wins if touched — even
once trailing has armed. This keeps a firm risk floor under every trade
regardless of trailing state.

TRAILING (when signal's mode has trailing.enabled in config):
  - "Armed" once price has moved in the trade's favor by
    activation_pct * target_pct from entry (target_pct is derived from
    the original take_profit distance, which is itself the volatility-
    adaptive target computed at signal creation).
  - Once armed, a trailing stop is tracked at
    peak_price -/+ (trail_distance_pct * target_pct), and the trade
    exits CORRECT the moment price retraces to that level.
  - If price never arms trailing and the stop-loss is never hit either,
    or if it arms but never retraces far enough, the trade EXPIRES at
    the last candle's close when the window runs out — the real P&L at
    that point (which can still be positive) is recorded via pnl_pct.
  - Trailing effectively supersedes the original fixed take_profit as
    an immediate-exit level once armed; before arming, a very sharp
    single-candle move straight through the full original target is
    still treated as reaching "armed" state (since activation_pct < 1
    means the original target is always beyond the activation point).

WHEN TRAILING IS DISABLED for a mode (or no trailing_cfg passed): falls
back to the simple fixed TP/SL behavior — exit CORRECT the instant the
original take_profit is touched.

EXPIRED vs INCORRECT: a timeout with neither SL nor a trailing/TP exit
triggered resolves EXPIRED, not INCORRECT — this is tracked as its own
bucket (see journal.get_accuracy) so "ran out of time, still felt fine"
isn't conflated with "got stopped out".

PNL: every resolution computes pnl_pct — the actual signed % price move
from entry to exit_price — uniformly across CORRECT / INCORRECT /
EXPIRED, since even a timeout has a real (if usually small) profit or
loss at its exit price.

This only ever evaluates against CLOSED candles the caller supplies —
it never estimates or assumes future price action.
"""


def pnl_pct(action, entry, exit_price):
    """Signed % price move from entry to exit_price, trade-direction aware."""
    if action == "BUY":
        return ((exit_price - entry) / entry) * 100
    else:  # SELL
        return ((entry - exit_price) / entry) * 100


def _derive_target_and_stop_pct(signal):
    """Recovers the original target%/stop% from the absolute TP/SL prices
    stored on the signal — avoids needing extra schema columns."""
    entry = signal["entry_price"]
    action = signal["action"]
    tp = signal["take_profit"]
    sl = signal["stop_loss"]

    if action == "BUY":
        target_pct = (tp - entry) / entry
        stop_pct = (entry - sl) / entry
    else:
        target_pct = (entry - tp) / entry
        stop_pct = (sl - entry) / entry

    return max(target_pct, 0.0), max(stop_pct, 0.0)


def is_ready_to_evaluate(signal, candles_since_entry):
    return len(candles_since_entry) >= signal["evaluate_after_candles"]


def evaluate_signal(signal, candles_since_entry, trailing_cfg=None):
    """
    signal: journal row (dict) — needs action, entry_price, take_profit,
            stop_loss, evaluate_after_candles.
    candles_since_entry: CLOSED candles after entry, chronological,
                          [ts, open, high, low, close, volume].
    trailing_cfg: the mode's `trailing` config dict, e.g.
                  {"enabled": True, "activation_pct": 0.5, "trail_distance_pct": 0.3}
                  or None/disabled to use simple fixed TP/SL behavior.

    Returns None if not enough candles yet and no exit condition met.
    Otherwise dict: status, exit_price, pnl_pct, mae_pct, mfe_pct.
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
    target_pct = stop_pct = 0.0
    activation_gain_pct = trail_distance_price_pct = 0.0
    armed = False
    peak = entry  # highest high (BUY) or lowest low (SELL) seen since entry

    if trailing_enabled:
        target_pct, stop_pct = _derive_target_and_stop_pct(signal)
        activation_gain_pct = trailing_cfg.get("activation_pct", 0.5) * target_pct
        trail_distance_price_pct = trailing_cfg.get("trail_distance_pct", 0.3) * target_pct

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

        # Hard stop-loss always wins if touched, trailing or not.
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

        # --- trailing path ---
        if action == "BUY":
            peak = max(peak, high)
            gain_pct = (peak - entry) / entry
            if not armed and gain_pct >= activation_gain_pct:
                armed = True
            if armed:
                trail_stop_price = peak * (1 - trail_distance_price_pct)
                if low <= trail_stop_price:
                    return {"status": "CORRECT", "exit_price": round(trail_stop_price, 8),
                            "pnl_pct": round(pnl_pct(action, entry, trail_stop_price), 4),
                            "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}
        else:  # SELL
            peak = min(peak, low)
            gain_pct = (entry - peak) / entry
            if not armed and gain_pct >= activation_gain_pct:
                armed = True
            if armed:
                trail_stop_price = peak * (1 + trail_distance_price_pct)
                if high >= trail_stop_price:
                    return {"status": "CORRECT", "exit_price": round(trail_stop_price, 8),
                            "pnl_pct": round(pnl_pct(action, entry, trail_stop_price), 4),
                            "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}

    # Window elapsed with no exit condition triggered.
    if len(window) >= max_candles:
        last_close = window[-1][4]
        return {"status": "EXPIRED", "exit_price": round(last_close, 8),
                "pnl_pct": round(pnl_pct(action, entry, last_close), 4),
                "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}

    return None  # still pending


def candles_since(all_closed_candles, entry_timestamp_ms):
    return [c for c in all_closed_candles if c[0] > entry_timestamp_ms]
