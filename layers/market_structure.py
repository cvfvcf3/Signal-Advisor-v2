"""
Market structure layer (funding rate + open interest).

FIX NOTE (v1 bug): the original version scored each condition as a flat
+10, capping bullish_score/bearish_score at a max of 20 out of an assumed
0-100 scale — so this layer could never contribute more than ~4 weighted
points no matter how strong the signal, which is why composite confidence
never reached the 70 threshold.

This version instead:
  1. Keeps a rolling window (per symbol) of recent funding-rate and
     open-interest readings.
  2. Converts the CURRENT reading into a z-score relative to that rolling
     window, so "extreme" is judged relative to this symbol's own recent
     behavior rather than a fixed cutoff — this avoids the sudden-spike
     problem (e.g. a short squeeze) skewing a fixed-threshold scorer.
  3. Passes the z-score through a sigmoid to get a smooth 0-1 value, then
     scales to a true 0-100 bullish/bearish score.

Funding rate is treated as a CONTRARIAN signal: extremely positive funding
(longs paying shorts heavily) tends to precede downside, and vice versa.
Open interest is combined with price direction: rising OI + rising price
= bullish continuation (new longs entering a bull move); rising OI +
falling price = bearish continuation; falling OI suggests position
unwinding rather than new conviction, so it pulls toward neutral.
"""

import math
from collections import deque

import numpy as np


def _sigmoid(x):
    # clip to avoid overflow on extreme z-scores
    x = max(min(x, 20), -20)
    return 1 / (1 + math.exp(-x))


class MarketStructureTracker:
    """
    Stateful tracker — one instance shared across ticks (held by
    advisor_engine), NOT re-created every tick, so the rolling window
    actually accumulates history over time.
    """

    def __init__(self, window_candles=150):
        self.window_candles = window_candles
        self._funding_history = {}   # symbol -> deque[float]
        self._oi_history = {}        # symbol -> deque[float]

    def _deque_for(self, store, symbol):
        if symbol not in store:
            store[symbol] = deque(maxlen=self.window_candles)
        return store[symbol]

    def _zscore(self, history, current_value):
        if len(history) < 10:
            return 0.0  # not enough history yet to judge "extreme"
        arr = np.array(history)
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return 0.0
        return (current_value - mean) / std

    def score(self, symbol, funding_rate, open_interest, price_change_pct):
        """
        funding_rate: latest funding rate (e.g. 0.0001 = 0.01%)
        open_interest: latest OI value (contracts or notional — just needs
                        to be consistent tick to tick)
        price_change_pct: price change over the same period OI was measured
                           over (e.g. this candle's % change), used to judge
                           OI direction against price direction

        Returns dict: bullish_score, bearish_score (0-100), details
        """
        funding_hist = self._deque_for(self._funding_history, symbol)
        oi_hist = self._deque_for(self._oi_history, symbol)

        funding_z = self._zscore(funding_hist, funding_rate)
        oi_z = self._zscore(oi_hist, open_interest)

        # record AFTER computing z-score so current reading doesn't dilute
        # its own baseline
        funding_hist.append(funding_rate)
        oi_hist.append(open_interest)

        # --- OI + price interaction (0..1, 0.5 = neutral) ---
        direction = 1 if price_change_pct > 0 else (-1 if price_change_pct < 0 else 0)
        oi_price_signal = oi_z * direction
        oi_price_score = _sigmoid(oi_price_signal)

        # --- funding contrarian (0..1, 0.5 = neutral) ---
        # negative funding_z (funding unusually low/negative) -> bullish bias
        # positive funding_z (funding unusually high) -> bearish bias
        funding_score = _sigmoid(-funding_z)

        combined = (0.6 * oi_price_score) + (0.4 * funding_score)  # 0..1

        if combined > 0.5:
            bullish_score = (combined - 0.5) * 2 * 100
            bearish_score = 0.0
        elif combined < 0.5:
            bearish_score = (0.5 - combined) * 2 * 100
            bullish_score = 0.0
        else:
            bullish_score = 0.0
            bearish_score = 0.0

        return {
            "bullish_score": round(min(bullish_score, 100), 2),
            "bearish_score": round(min(bearish_score, 100), 2),
            "details": {
                "funding_rate": funding_rate,
                "funding_z": round(funding_z, 3),
                "open_interest": open_interest,
                "oi_z": round(oi_z, 3),
                "price_change_pct": price_change_pct,
                "history_len": len(oi_hist),
            },
        }
