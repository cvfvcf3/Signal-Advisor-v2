"""
Smart Money Concepts (SMC) layer.

Detects: fractal-confirmed swing highs/lows, BOS (Break of Structure),
CHoCH (Change of Character), Order Blocks, Fair Value Gaps (FVG), and
liquidity sweeps. Combines them into a single bullish_score/bearish_score,
weighted so the maximum achievable single-direction score is 100 — same
scale as every other layer.

LOOK-AHEAD BIAS: a swing high/low at candle i is only "confirmed" once
`fractal_right` candles AFTER it are known. find_confirmed_fractals()
never returns a pivot unless enough follow-through candles already exist
in the data passed to it — the system can never "know" a pivot existed
before it was actually confirmable in real time. Callers must pass a
CLOSED-candle OHLCV series (no forming candle), same rule as every other
layer.

NOISE CONTROL: on low timeframes (scalp mode), this layer's weight should
be capped by the caller (advisor_engine.py) unless the higher-timeframe
macro trend agrees with the SMC signal direction — see config.yaml
modes.scalp.smc_weight_cap_if_htf_disagrees. Low-timeframe wicks otherwise
produce frequent false BOS/CHoCH triggers.
"""

import pandas as pd


# Sub-signal weights — calibrated so the maximum simultaneously-achievable
# single-direction score is 100 (BOS-or-CHoCH is mutually exclusive with
# itself, but can co-occur with order_block + fvg + liquidity_sweep):
# 40 + 10 + 30 + 20 = 100
W_BOS = 40
W_CHOCH = 40
W_ORDER_BLOCK = 10
W_FVG = 30
W_LIQUIDITY_SWEEP = 20


def _to_dataframe(ohlcv):
    return pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])


def find_confirmed_fractals(df, left=5, right=5):
    """
    Returns confirmed swing points in chronological order:
        [{"index": i, "price": float, "type": "high"|"low"}, ...]

    A candle at index i is only included once `right` candles after it
    exist in df — i.e. this never looks further than the data actually
    available, and never claims a pivot was known before its confirmation
    window closed.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    fractals = []

    last_confirmable = n - right - 1
    for i in range(left, last_confirmable + 1):
        window_high = highs[i - left:i + right + 1]
        window_low = lows[i - left:i + right + 1]

        if highs[i] == window_high.max() and (window_high == highs[i]).sum() == 1:
            fractals.append({"index": i, "price": float(highs[i]), "type": "high"})

        if lows[i] == window_low.min() and (window_low == lows[i]).sum() == 1:
            fractals.append({"index": i, "price": float(lows[i]), "type": "low"})

    fractals.sort(key=lambda f: f["index"])
    return fractals


def _detect_structure(fractals):
    """
    From the confirmed fractal sequence, determine current structure bias
    (bullish/bearish/ranging) and return the most recent swing high/low
    to use as BOS/CHoCH reference levels.
    """
    highs = [f for f in fractals if f["type"] == "high"]
    lows = [f for f in fractals if f["type"] == "low"]

    if len(highs) < 2 or len(lows) < 2:
        return {"bias": "unknown", "last_high": None, "last_low": None}

    last_high, prev_high = highs[-1], highs[-2]
    last_low, prev_low = lows[-1], lows[-2]

    higher_highs = last_high["price"] > prev_high["price"]
    higher_lows = last_low["price"] > prev_low["price"]
    lower_highs = last_high["price"] < prev_high["price"]
    lower_lows = last_low["price"] < prev_low["price"]

    if higher_highs and higher_lows:
        bias = "bullish"
    elif lower_highs and lower_lows:
        bias = "bearish"
    else:
        bias = "ranging"

    return {"bias": bias, "last_high": last_high, "last_low": last_low}


def _detect_fvg(df, lookback=20):
    """
    3-candle Fair Value Gap detection over the last `lookback` closed
    candles. Returns the most recent UNFILLED bullish and bearish FVG
    (unfilled = price hasn't traded back into the gap since it formed).
    """
    n = len(df)
    start = max(2, n - lookback)
    bullish_fvg = None
    bearish_fvg = None

    for i in range(start, n):
        c1_high = df["high"].iloc[i - 2]
        c1_low = df["low"].iloc[i - 2]
        c3_high = df["high"].iloc[i]
        c3_low = df["low"].iloc[i]

        if c1_high < c3_low:  # bullish gap up
            gap_low, gap_high = c1_high, c3_low
            filled = (df["low"].iloc[i + 1:] <= gap_high).any() if i + 1 < n else False
            if not filled:
                bullish_fvg = {"index": i, "gap_low": float(gap_low), "gap_high": float(gap_high)}

        if c1_low > c3_high:  # bearish gap down
            gap_low, gap_high = c3_high, c1_low
            filled = (df["high"].iloc[i + 1:] >= gap_low).any() if i + 1 < n else False
            if not filled:
                bearish_fvg = {"index": i, "gap_low": float(gap_low), "gap_high": float(gap_high)}

    return bullish_fvg, bearish_fvg


def _detect_order_block(df, break_index, direction, search_back=10):
    """
    Order block = the last opposite-direction candle before a BOS/CHoCH
    break. direction="bullish" -> last bearish (down) candle before the
    break; direction="bearish" -> last bullish (up) candle before the break.
    """
    search_start = max(0, break_index - search_back)
    for i in range(break_index - 1, search_start - 1, -1):
        is_bear_candle = df["close"].iloc[i] < df["open"].iloc[i]
        is_bull_candle = df["close"].iloc[i] > df["open"].iloc[i]
        if direction == "bullish" and is_bear_candle:
            return {"index": i, "low": float(df["low"].iloc[i]), "high": float(df["high"].iloc[i])}
        if direction == "bearish" and is_bull_candle:
            return {"index": i, "low": float(df["low"].iloc[i]), "high": float(df["high"].iloc[i])}
    return None


def _detect_liquidity_sweep(df, fractals, lookback=10):
    """
    Liquidity sweep: a recent candle wicks beyond a confirmed swing
    low/high but CLOSES back on the other side of it — a stop-hunt
    followed by reversal.
    """
    if not fractals:
        return None

    recent = df.iloc[-lookback:]
    lows = [f for f in fractals if f["type"] == "low"]
    highs = [f for f in fractals if f["type"] == "high"]

    if lows:
        level = lows[-1]["price"]
        wick_below = (recent["low"] < level) & (recent["close"] > level)
        if wick_below.any():
            return {"type": "bullish_sweep", "level": level}

    if highs:
        level = highs[-1]["price"]
        wick_above = (recent["high"] > level) & (recent["close"] < level)
        if wick_above.any():
            return {"type": "bearish_sweep", "level": level}

    return None


def analyze(ohlcv, fractal_left=5, fractal_right=5):
    """
    ohlcv: list of CLOSED candles [ts, open, high, low, close, volume],
           most recent candle last. Needs ~ (fractal_left+fractal_right+40)
           candles minimum for meaningful structure + FVG/sweep lookbacks.

    Returns dict: bullish_score, bearish_score (0-100), details
    """
    min_candles = fractal_left + fractal_right + 40
    if len(ohlcv) < min_candles:
        return {"bullish_score": 0, "bearish_score": 0, "details": {"error": "not_enough_candles"}}

    df = _to_dataframe(ohlcv)
    price = df["close"].iloc[-1]

    fractals = find_confirmed_fractals(df, left=fractal_left, right=fractal_right)
    structure = _detect_structure(fractals)

    bullish = 0.0
    bearish = 0.0
    details = {"structure_bias": structure["bias"], "structure_event": "none"}

    last_high = structure["last_high"]
    last_low = structure["last_low"]

    # --- BOS / CHoCH ---
    if last_high and price > last_high["price"]:
        if structure["bias"] == "bullish":
            bullish += W_BOS
            details["structure_event"] = "BOS_bullish_continuation"
        else:
            bullish += W_CHOCH
            details["structure_event"] = "CHoCH_bullish_reversal"
        ob = _detect_order_block(df, last_high["index"], "bullish")
        if ob:
            bullish += W_ORDER_BLOCK
            details["order_block"] = ob

    elif last_low and price < last_low["price"]:
        if structure["bias"] == "bearish":
            bearish += W_BOS
            details["structure_event"] = "BOS_bearish_continuation"
        else:
            bearish += W_CHOCH
            details["structure_event"] = "CHoCH_bearish_reversal"
        ob = _detect_order_block(df, last_low["index"], "bearish")
        if ob:
            bearish += W_ORDER_BLOCK
            details["order_block"] = ob

    # --- FVG (price currently sitting inside/near an unfilled gap) ---
    bullish_fvg, bearish_fvg = _detect_fvg(df)
    if bullish_fvg and bullish_fvg["gap_low"] <= price <= bullish_fvg["gap_high"] * 1.02:
        bullish += W_FVG
        details["fvg"] = bullish_fvg
    elif bearish_fvg and bearish_fvg["gap_low"] * 0.98 <= price <= bearish_fvg["gap_high"]:
        bearish += W_FVG
        details["fvg"] = bearish_fvg

    # --- Liquidity sweep ---
    sweep = _detect_liquidity_sweep(df, fractals)
    if sweep:
        if sweep["type"] == "bullish_sweep":
            bullish += W_LIQUIDITY_SWEEP
        else:
            bearish += W_LIQUIDITY_SWEEP
        details["liquidity_sweep"] = sweep

    return {
        "bullish_score": round(min(bullish, 100), 2),
        "bearish_score": round(min(bearish, 100), 2),
        "details": details,
    }
