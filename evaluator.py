"""
Evaluator: checks PENDING journal signals against subsequent CLOSED
candles and resolves them to CORRECT / INCORRECT.

RESOLUTION RULE (fixed, not ambiguous):
    BUY  is CORRECT if, within `evaluate_after_candles` candles after
         entry, future_high >= entry * (1 + success_move_pct).
    SELL is CORRECT if, within the same window,
         future_low <= entry * (1 - success_move_pct).
    If the window elapses without the target being hit, the signal is
    INCORRECT — not left ambiguous, and not silently dropped.

MAE / MFE are recorded for every resolved signal (Maximum Adverse/
Favorable Excursion, as a % of entry price) so weights/thresholds can
later be tuned from more than just win/loss — e.g. a BUY that was
correct but had deep MAE first is a noisier win than a clean one.

This only ever evaluates against CLOSED candles the caller supplies —
it never estimates or assumes future price action.
"""


def _window_high_low(candles):
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    return max(highs), min(lows)


def is_ready_to_evaluate(signal, candles_since_entry):
    """
    signal: a journal row (dict) with 'evaluate_after_candles'.
    candles_since_entry: list of CLOSED candles that occurred after the
                          signal's entry candle, in chronological order.

    Returns True once enough candles have closed to apply the resolution
    rule; False if we should keep waiting.
    """
    return len(candles_since_entry) >= signal["evaluate_after_candles"]


def evaluate_signal(signal, candles_since_entry):
    """
    signal: journal row (dict) — needs action, entry_price,
            success_move_pct, evaluate_after_candles.
    candles_since_entry: CLOSED candles after entry, chronological,
                          [ts, open, high, low, close, volume].

    Returns None if not enough candles yet (still PENDING) — caller
    should skip and re-check next tick.

    Otherwise returns dict:
        status: "CORRECT" | "INCORRECT"
        exit_price: the price the outcome was determined at
        mae_pct: max adverse excursion, % of entry
        mfe_pct: max favorable excursion, % of entry
    """
    if not is_ready_to_evaluate(signal, candles_since_entry):
        return None

    window = candles_since_entry[: signal["evaluate_after_candles"]]
    entry = signal["entry_price"]
    action = signal["action"]
    success_pct = signal["success_move_pct"]

    window_high, window_low = _window_high_low(window)
    last_close = window[-1][4]

    if action == "BUY":
        target = entry * (1 + success_pct)
        hit = window_high >= target
        mfe_pct = ((window_high - entry) / entry) * 100
        mae_pct = ((entry - window_low) / entry) * 100
        exit_price = target if hit else last_close
        status = "CORRECT" if hit else "INCORRECT"

    elif action == "SELL":
        target = entry * (1 - success_pct)
        hit = window_low <= target
        mfe_pct = ((entry - window_low) / entry) * 100
        mae_pct = ((window_high - entry) / entry) * 100
        exit_price = target if hit else last_close
        status = "CORRECT" if hit else "INCORRECT"

    else:
        # WAIT signals are never journaled in the first place, but guard
        # against being called on one anyway.
        return None

    return {
        "status": status,
        "exit_price": round(exit_price, 8),
        "mae_pct": round(mae_pct, 4),
        "mfe_pct": round(mfe_pct, 4),
    }


def candles_since(all_closed_candles, entry_timestamp_ms):
    """
    Filters a full closed-candle series down to only those that occurred
    strictly after entry_timestamp_ms (the entry signal's candle).
    """
    return [c for c in all_closed_candles if c[0] > entry_timestamp_ms]
