"""
Scoring: combines per-layer bullish/bearish scores into a single composite
score, applies the BTC macro-filter to altcoin scores, and decides the
final action (BUY / SELL / WAIT).

IMPORTANT: a confidence score here is a measure of how much evidence the
scoring model found (weighted agreement across layers) — it is NOT a
calibrated win probability.

ASYMMETRIC THRESHOLDS: min_score_to_signal can be either a single number
(applied to both BUY and SELL, the original behavior) or a dict like
{"buy": 73, "sell": 65} to require a higher bar for one direction than
the other. This exists because real trade data showed scalp-mode BUY
signals performing meaningfully worse than SELL signals at the same
threshold — raising BUY's bar (while leaving SELL alone) filters out
weaker BUY setups without shutting BUY off entirely, so the system still
keeps producing BUY data to confirm whether the gap persists or closes.
"""


def composite_score(layer_scores, weights):
    bullish = 0.0
    bearish = 0.0

    for layer_name, weight in weights.items():
        reading = layer_scores.get(layer_name)
        if not reading:
            continue
        bullish += reading.get("bullish_score", 0) * (weight / 100)
        bearish += reading.get("bearish_score", 0) * (weight / 100)

    return round(min(bullish, 100), 2), round(min(bearish, 100), 2)


def determine_btc_regime(btc_htf_bullish_score, btc_htf_bearish_score, strong_threshold=70):
    if btc_htf_bullish_score >= strong_threshold:
        return "bullish"
    if btc_htf_bearish_score >= strong_threshold:
        return "bearish"
    return "neutral"


def apply_btc_macro_filter(is_btc_symbol, btc_regime, bullish, bearish, macro_cfg):
    if is_btc_symbol or not macro_cfg.get("enabled", True):
        return bullish, bearish

    if btc_regime == "bullish":
        mult = macro_cfg["bullish"]
        bullish = bullish * mult["buy_multiplier"]
        bearish = bearish * mult["sell_multiplier"]
    elif btc_regime == "bearish":
        mult = macro_cfg["bearish"]
        bullish = bullish * mult["buy_multiplier"]
        bearish = bearish * mult["sell_multiplier"]

    bullish = round(max(0.0, min(bullish, 100.0)), 2)
    bearish = round(max(0.0, min(bearish, 100.0)), 2)
    return bullish, bearish


def _resolve_thresholds(min_score_to_signal):
    """Returns (buy_threshold, sell_threshold) whether min_score_to_signal
    is a single number or a {"buy":.., "sell":..} dict."""
    if isinstance(min_score_to_signal, dict):
        buy_t = min_score_to_signal.get("buy", min_score_to_signal.get("sell", 0))
        sell_t = min_score_to_signal.get("sell", min_score_to_signal.get("buy", 0))
        return buy_t, sell_t
    return min_score_to_signal, min_score_to_signal


def decide_action(bullish, bearish, min_score_to_signal, min_score_gap):
    """
    BUY:  bullish >= buy_threshold AND (bullish - bearish) >= min_score_gap
    SELL: bearish >= sell_threshold AND (bearish - bullish) >= min_score_gap
    Otherwise: WAIT

    Returns dict: {"action": "BUY"|"SELL"|"WAIT", "confidence": float}
    """
    buy_threshold, sell_threshold = _resolve_thresholds(min_score_to_signal)

    if bullish >= buy_threshold and (bullish - bearish) >= min_score_gap:
        return {"action": "BUY", "confidence": bullish}

    if bearish >= sell_threshold and (bearish - bullish) >= min_score_gap:
        return {"action": "SELL", "confidence": bearish}

    return {"action": "WAIT", "confidence": round(max(bullish, bearish), 2)}
