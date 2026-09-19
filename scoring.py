"""
Scoring: combines per-layer bullish/bearish scores into a single composite
score, applies the BTC macro-filter to altcoin scores, and decides the
final action (BUY / SELL / WAIT).

IMPORTANT: a confidence score here is a measure of how much evidence the
scoring model found (weighted agreement across layers) — it is NOT a
calibrated win probability. An 82/100 score does not mean "82% chance of
being correct." Only journal-based accuracy tracking (see evaluator.py)
tells you the real historical win rate for a given (symbol, mode)
combination, and even that is backward-looking, not a guarantee.
"""


def composite_score(layer_scores, weights):
    """
    layer_scores: dict like {"technical": {"bullish_score": .., "bearish_score": ..}, ...}
    weights: dict like {"technical": 20, "multi_timeframe": 15, ...} (per-mode weights from config)

    Returns (bullish_composite, bearish_composite), each 0-100.
    Any layer present in `weights` but missing from `layer_scores` is
    treated as 0/0 (contributes nothing) rather than raising an error,
    so one failed layer fetch doesn't crash the whole tick.
    """
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
    """
    Classifies BTC's higher-timeframe regime for the macro filter.
    "Strong" is defined purely by this threshold on BTC's own HTF
    composite score — kept explicit/configurable rather than hardcoded,
    since it needs to be tunable from journal data later.
    """
    if btc_htf_bullish_score >= strong_threshold:
        return "bullish"
    if btc_htf_bearish_score >= strong_threshold:
        return "bearish"
    return "neutral"


def apply_btc_macro_filter(is_btc_symbol, btc_regime, bullish, bearish, macro_cfg):
    """
    Adjusts an ALTCOIN's composite scores based on BTC's regime. Never
    applied to BTC itself (is_btc_symbol=True passes through unchanged).

    macro_cfg: the `btc_macro_filter` section of config.yaml, e.g.:
        enabled: true
        bullish: {buy_multiplier: 1.15, sell_multiplier: 0.70}
        bearish: {buy_multiplier: 0.50, sell_multiplier: 1.15}

    Result is clamped back to [0, 100] after adjustment.
    """
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
    # regime == "neutral" -> no adjustment

    bullish = round(max(0.0, min(bullish, 100.0)), 2)
    bearish = round(max(0.0, min(bearish, 100.0)), 2)
    return bullish, bearish


def decide_action(bullish, bearish, min_score_to_signal, min_score_gap):
    """
    Final BUY / SELL / WAIT decision.

    BUY:  bullish >= min_score_to_signal AND (bullish - bearish) >= min_score_gap
    SELL: bearish >= min_score_to_signal AND (bearish - bullish) >= min_score_gap
    Otherwise: WAIT

    Returns dict: {"action": "BUY"|"SELL"|"WAIT", "confidence": float}
    confidence is the winning side's score (or the higher of the two if
    WAIT, for dashboard display only — a WAIT confidence is informational,
    not actionable).
    """
    if bullish >= min_score_to_signal and (bullish - bearish) >= min_score_gap:
        return {"action": "BUY", "confidence": bullish}

    if bearish >= min_score_to_signal and (bearish - bullish) >= min_score_gap:
        return {"action": "SELL", "confidence": bearish}

    return {"action": "WAIT", "confidence": round(max(bullish, bearish), 2)}
