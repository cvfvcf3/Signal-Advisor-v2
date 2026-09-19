"""
Technical indicators layer.

Combines EMA cross, MACD, RSI, volume, and support/resistance proximity
into a single bullish_score / bearish_score pair, each on a TRUE 0-100
scale (the individual sub-signal weights below sum to 100). This keeps
the layer directly comparable to the other layers (multi_timeframe,
orderbook, market_structure, smc), all of which must also output on a
0-100 scale — mixing scales was the bug that broke the original engine.

Only uses CLOSED candles: the caller must pass OHLCV data that excludes
the still-forming candle, so scores don't flicker as the current candle
updates.
"""

import pandas as pd
import numpy as np


# Sub-signal weights — must sum to 100
W_EMA_CROSS = 30
W_MACD = 25
W_RSI = 20
W_VOLUME = 15
W_SUPPORT_RESISTANCE = 10


def _to_dataframe(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    return df


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(series, fast=12, slow=26, signal=9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def analyze(ohlcv, ema_fast_period=9, ema_slow_period=21):
    """
    ohlcv: list of [ts, open, high, low, close, volume], CLOSED candles only,
           most recent candle last. Needs at least ~60 candles for MACD/EMA
           to be meaningful.

    Returns dict:
        bullish_score: 0-100
        bearish_score: 0-100
        details: dict of each sub-signal's raw contribution, for debugging
                 and for the dashboard's "why" breakdown.
    """
    if len(ohlcv) < 60:
        return {"bullish_score": 0, "bearish_score": 0, "details": {"error": "not_enough_candles"}}

    df = _to_dataframe(ohlcv)
    close = df["close"]
    volume = df["volume"]

    bullish = 0.0
    bearish = 0.0
    details = {}

    # 1) EMA cross (fast vs slow) + slope of fast EMA
    ema_fast = _ema(close, ema_fast_period)
    ema_slow = _ema(close, ema_slow_period)
    cross_up = ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]
    cross_down = ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-2] >= ema_slow.iloc[-2]
    above = ema_fast.iloc[-1] > ema_slow.iloc[-1]
    slope_up = ema_fast.iloc[-1] > ema_fast.iloc[-3]

    if cross_up:
        bullish += W_EMA_CROSS
        details["ema_cross"] = "fresh_bull_cross"
    elif above and slope_up:
        bullish += W_EMA_CROSS * 0.6
        details["ema_cross"] = "bull_trend"
    elif cross_down:
        bearish += W_EMA_CROSS
        details["ema_cross"] = "fresh_bear_cross"
    elif not above and not slope_up:
        bearish += W_EMA_CROSS * 0.6
        details["ema_cross"] = "bear_trend"
    else:
        details["ema_cross"] = "neutral"

    # 2) MACD histogram direction + momentum
    macd_line, signal_line, hist = _macd(close)
    hist_now = hist.iloc[-1]
    hist_prev = hist.iloc[-2]
    macd_rising = hist_now > hist_prev

    if hist_now > 0 and macd_rising:
        bullish += W_MACD
        details["macd"] = "bull_momentum"
    elif hist_now > 0:
        bullish += W_MACD * 0.5
        details["macd"] = "bull_fading"
    elif hist_now < 0 and not macd_rising:
        bearish += W_MACD
        details["macd"] = "bear_momentum"
    elif hist_now < 0:
        bearish += W_MACD * 0.5
        details["macd"] = "bear_fading"
    else:
        details["macd"] = "neutral"

    # 3) RSI
    rsi = _rsi(close).iloc[-1]
    if rsi < 30:
        bullish += W_RSI  # oversold -> bullish bias
        details["rsi"] = f"oversold({rsi:.1f})"
    elif rsi > 70:
        bearish += W_RSI  # overbought -> bearish bias
        details["rsi"] = f"overbought({rsi:.1f})"
    elif rsi > 50:
        bullish += W_RSI * 0.4
        details["rsi"] = f"mild_bull({rsi:.1f})"
    elif rsi < 50:
        bearish += W_RSI * 0.4
        details["rsi"] = f"mild_bear({rsi:.1f})"
    else:
        details["rsi"] = f"neutral({rsi:.1f})"

    # 4) Volume confirmation (current candle vs recent average)
    avg_volume = volume.iloc[-21:-1].mean()
    vol_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1.0
    candle_bullish = close.iloc[-1] > df["open"].iloc[-1]

    if vol_ratio > 1.5:
        if candle_bullish:
            bullish += W_VOLUME
            details["volume"] = f"high_vol_bull_candle({vol_ratio:.2f}x)"
        else:
            bearish += W_VOLUME
            details["volume"] = f"high_vol_bear_candle({vol_ratio:.2f}x)"
    else:
        details["volume"] = f"normal({vol_ratio:.2f}x)"

    # 5) Support/Resistance proximity (recent swing high/low over last 50 candles)
    lookback = df.iloc[-50:]
    recent_high = lookback["high"].max()
    recent_low = lookback["low"].min()
    price = close.iloc[-1]
    range_size = recent_high - recent_low if recent_high > recent_low else 1.0
    dist_to_support = (price - recent_low) / range_size
    dist_to_resistance = (recent_high - price) / range_size

    if dist_to_support < 0.1:
        bullish += W_SUPPORT_RESISTANCE
        details["structure"] = "near_support"
    elif dist_to_resistance < 0.1:
        bearish += W_SUPPORT_RESISTANCE
        details["structure"] = "near_resistance"
    else:
        details["structure"] = "mid_range"

    return {
        "bullish_score": round(min(bullish, 100), 2),
        "bearish_score": round(min(bearish, 100), 2),
        "details": details,
    }
