"""
Order book layer.

Reads bid/ask depth and computes a volume-weighted imbalance between the
two sides. Imbalance is naturally bounded in [-1, 1], so it is scaled by
*100 to land on the same 0-100 scale as every other layer.

A wide spread (relative to price) reduces confidence in the imbalance
reading, since wide spreads usually mean thin/volatile books where the
imbalance number is less reliable.
"""


def analyze(order_book, depth_levels=20, max_spread_pct=0.001):
    """
    order_book: ccxt-style dict with 'bids' and 'asks', each a list of
                [price, quantity], best price first.

    Returns dict:
        bullish_score: 0-100
        bearish_score: 0-100
        details: imbalance, spread_pct, wall info, for dashboard/debugging
    """
    bids = order_book.get("bids", [])[:depth_levels]
    asks = order_book.get("asks", [])[:depth_levels]

    if not bids or not asks:
        return {"bullish_score": 0, "bearish_score": 0, "details": {"error": "empty_book"}}

    bid_volume = sum(qty for _, qty in bids)
    ask_volume = sum(qty for _, qty in asks)
    total_volume = bid_volume + ask_volume

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2
    spread_pct = (best_ask - best_bid) / mid_price if mid_price > 0 else 1.0

    if total_volume == 0:
        return {"bullish_score": 0, "bearish_score": 0, "details": {"error": "zero_volume"}}

    imbalance = (bid_volume - ask_volume) / total_volume  # -1..1

    # Widen spread => reduce confidence in the imbalance signal.
    # At max_spread_pct the reading is halved; well below it, full strength.
    if spread_pct <= max_spread_pct:
        confidence_mult = 1.0
    elif spread_pct <= max_spread_pct * 3:
        confidence_mult = 0.5
    else:
        confidence_mult = 0.2

    bull_raw = max(imbalance, 0) * 100 * confidence_mult
    bear_raw = max(-imbalance, 0) * 100 * confidence_mult

    # Detect a large single-level "wall" on either side (>3x the average
    # level size) as a secondary signal — walls often act as short-term
    # support/resistance.
    avg_bid_level = bid_volume / len(bids)
    avg_ask_level = ask_volume / len(asks)
    bid_wall = any(qty > avg_bid_level * 3 for _, qty in bids)
    ask_wall = any(qty > avg_ask_level * 3 for _, qty in asks)

    return {
        "bullish_score": round(min(bull_raw, 100), 2),
        "bearish_score": round(min(bear_raw, 100), 2),
        "details": {
            "imbalance": round(imbalance, 4),
            "spread_pct": round(spread_pct, 5),
            "confidence_mult": confidence_mult,
            "bid_wall": bid_wall,
            "ask_wall": ask_wall,
        },
    }
