"""
Multi-timeframe confirmation layer.

Checks whether the trend on each "confirm" timeframe (e.g. 5m/15m for
scalp mode, 1h/4h for day mode) agrees with a bullish or bearish bias.
Score = (number of timeframes agreeing) / (total timeframes checked),
scaled to 0-100 — so this layer is naturally on the same scale as the
others without extra normalization.

Trend on each timeframe is judged by EMA(fast) vs EMA(slow) on CLOSED
candles only (same closed-candle rule as technical.py, to avoid
signal flicker from the still-forming candle).
"""

import pandas as pd


def _to_dataframe(ohlcv):
    return pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def analyze(ohlcv_by_timeframe, ema_fast_period=9, ema_slow_period=21, min_candles=30):
    """
    ohlcv_by_timeframe: dict like {"5m": [...], "15m": [...]}, each value a
                         list of CLOSED candles [ts, open, high, low, close, volume].

    Returns dict:
        bullish_score: 0-100
        bearish_score: 0-100
        details: per-timeframe verdict, for dashboard/debugging
    """
    details = {}
    bull_count = 0
    bear_count = 0
    usable = 0

    for tf, ohlcv in ohlcv_by_timeframe.items():
        if not ohlcv or len(ohlcv) < min_candles:
            details[tf] = "not_enough_data"
            continue

        df = _to_dataframe(ohlcv)
        fast = _ema(df["close"], ema_fast_period)
        slow = _ema(df["close"], ema_slow_period)

        usable += 1
        if fast.iloc[-1] > slow.iloc[-1]:
            bull_count += 1
            details[tf] = "bullish"
        elif fast.iloc[-1] < slow.iloc[-1]:
            bear_count += 1
            details[tf] = "bearish"
        else:
            details[tf] = "neutral"

    if usable == 0:
        return {"bullish_score": 0, "bearish_score": 0, "details": details}

    bullish_score = round((bull_count / usable) * 100, 2)
    bearish_score = round((bear_count / usable) * 100, 2)

    return {
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "details": details,
    }
