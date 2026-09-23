"""
Evaluator: checks PENDING journal signals against subsequent CLOSED
candles and resolves them to CORRECT / INCORRECT / EXPIRED.

PATH-AWARE RESOLUTION: each candle since entry is walked in chronological
order. Whichever level — take_profit or stop_loss — is touched FIRST
determines the outcome:
    BUY:  TP touched when candle high >= take_profit
          SL touched when candle low  <= stop_loss
    SELL: TP touched when candle low  <= take_profit
          SL touched when candle high >= stop_loss
If a single candle's range touches BOTH levels (a large/volatile candle),
the conservative assumption is that STOP-LOSS was hit first — this avoids
overstating accuracy on wide-range candles where the true intra-candle
order is unknown from OHLC data alone.

EXPIRED vs INCORRECT: if neither TP nor SL is touched within
`evaluate_after_candles` candles, the signal resolves EXPIRED (NOT
INCORRECT) at the last available close. A setup that never got stopped
out but also never reached its target is a different failure mode than
one that reversed hard against the position. EXPIRED is excluded from
the accuracy percentage (journal.get_accuracy only divides CORRECT /
(CORRECT + INCORRECT)), but still shown as its own count.

PNL: every resolution (CORRECT, INCORRECT, or EXPIRED) also computes
pnl_pct — the actual % price move from entry to exit_price, signed for
the trade direction. This is deliberately uniform across all three
statuses: an EXPIRED trade's exit is just whatever the last candle's
close was, and that can still be a real (small) profit or loss even
though neither target nor stop was touched. Treating EXPIRED as "no
result" would hide that a real position, had one been taken, would have
closed somewhere — usually near flat, but not exactly zero.

MAE / MFE are accumulated across all candles actually walked (up to the
resolution point or the end of the window), as a % of entry price.

This only ever evaluates against CLOSED candles the caller supplies —
it never estimates or assumes future price action.
"""


def _pnl_pct(action, entry, exit_price):
    if action == "BUY":
        return ((exit_price - entry) / entry) * 100
    else:  # SELL
        return ((entry - exit_price) / entry) * 100


def is_ready_to_evaluate(signal, candles_since_entry):
    """True once either enough candles have closed to apply the timeout
    rule, or (checked by evaluate_signal itself) TP/SL was already hit."""
    return len(candles_since_entry) >= signal["evaluate_after_candles"]


def evaluate_signal(signal, candles_since_entry):
    """
    signal: journal row (dict) — needs action, entry_price, take_profit,
            stop_loss, evaluate_after_candles.
    candles_since_entry: CLOSED candles after entry, chronological,
                          [ts, open, high, low, close, volume].

    Returns None if not enough candles yet AND neither TP nor SL has been
    touched — caller should keep this signal PENDING and re-check next
    tick.

    Otherwise returns dict:
        status: "CORRECT" | "INCORRECT" | "EXPIRED"
        exit_price: the price the outcome was determined at
        pnl_pct: signed % move from entry to exit_price (trade-direction aware)
        mae_pct: max adverse excursion, % of entry, over candles walked
        mfe_pct: max favorable excursion, % of entry, over candles walked
    """
    entry = signal["entry_price"]
    action = signal["action"]
    tp = signal["take_profit"]
    sl = signal["stop_loss"]
    max_candles = signal["evaluate_after_candles"]

    window = candles_since_entry[:max_candles]
    mae_pct = 0.0
    mfe_pct = 0.0

    for candle in window:
        high, low = candle[2], candle[3]

        if action == "BUY":
            mfe_pct = max(mfe_pct, ((high - entry) / entry) * 100)
            mae_pct = max(mae_pct, ((entry - low) / entry) * 100)
            hit_tp = high >= tp
            hit_sl = low <= sl
        else:  # SELL
            mfe_pct = max(mfe_pct, ((entry - low) / entry) * 100)
            mae_pct = max(mae_pct, ((high - entry) / entry) * 100)
            hit_tp = low <= tp
            hit_sl = high >= sl

        if hit_sl:
            return {"status": "INCORRECT", "exit_price": round(sl, 8),
                    "pnl_pct": round(_pnl_pct(action, entry, sl), 4),
                    "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}
        if hit_tp:
            return {"status": "CORRECT", "exit_price": round(tp, 8),
                    "pnl_pct": round(_pnl_pct(action, entry, tp), 4),
                    "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}

    # Neither level touched in the candles seen so far.
    if len(window) >= max_candles:
        last_close = window[-1][4]
        return {"status": "EXPIRED", "exit_price": round(last_close, 8),
                "pnl_pct": round(_pnl_pct(action, entry, last_close), 4),
                "mae_pct": round(mae_pct, 4), "mfe_pct": round(mfe_pct, 4)}

    return None  # still pending — not enough candles yet, no hit yet


def candles_since(all_closed_candles, entry_timestamp_ms):
    """
    Filters a full closed-candle series down to only those that occurred
    strictly after entry_timestamp_ms (the entry signal's candle).
    """
    return [c for c in all_closed_candles if c[0] > entry_timestamp_ms]
