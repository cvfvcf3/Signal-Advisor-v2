"""
Advisor engine: the central orchestrator. One Advisor instance is created
at startup (by main.py) and its run_tick() is called on a timer
(poll_interval_seconds). Each tick:

  1. Determines BTC's HTF regime (bullish/bearish/neutral) once, since
     every altcoin's score depends on it.
  2. Loops over every (symbol, mode) combination, running all 5 layers,
     combining them into a composite score, applying the BTC macro-filter
     to altcoins, and deciding BUY/SELL/WAIT.
  3. Records new BUY/SELL signals to the journal (with duplicate
     protection — a repeat of the same still-pending action on the same
     symbol+mode is not re-recorded/re-notified).
  4. Evaluates any previously-PENDING signals against newly-closed
     candles and resolves them.

PER-SYMBOL ERROR ISOLATION: a failure processing one (symbol, mode)
combination is caught and logged; it does not stop the rest of the tick.

CLOSED CANDLES ONLY: _get_closed_ohlcv always drops the most recent
(potentially still-forming) candle before handing data to any layer, so
scores are based on confirmed price action only and don't flicker.
"""

import threading
from datetime import datetime, timezone

from exchange_client import ExchangeClient
from cache import TTLCache
from layers import technical, multi_timeframe, orderbook, smc
from layers.market_structure import MarketStructureTracker
from scoring import composite_score, determine_btc_regime, apply_btc_macro_filter, decide_action
import journal
import evaluator


class Advisor:
    def __init__(self, config):
        self.config = config
        self.client = ExchangeClient(config)
        self.cache = TTLCache(ttl_seconds=config["cache"]["ttl_seconds"])
        self.ms_tracker = MarketStructureTracker(
            window_candles=config["market_structure"]["rolling_window_candles"]
        )

        self._lock = threading.Lock()
        self.last_readings = {}          # (symbol, mode) -> reading dict, for dashboard
        self._pending_notifications = [] # list of signal dicts awaiting telegram_notifier

        journal.init_db()

    # ---------- data fetch helpers (cached, closed-candle only) ----------

    def _get_closed_ohlcv(self, symbol, timeframe, limit=200):
        key = f"ohlcv:{symbol}:{timeframe}"
        raw = self.cache.get_or_fetch(
            key, lambda: self.client.fetch_ohlcv(symbol, timeframe, limit=limit + 1)
        )
        if not raw or len(raw) < 2:
            return []
        return raw[:-1]  # drop the still-forming last candle

    def _get_order_book(self, symbol):
        key = f"orderbook:{symbol}"
        return self.cache.get_or_fetch(key, lambda: self.client.fetch_order_book(symbol))

    def _get_funding_and_oi(self, symbol):
        funding = self.cache.get_or_fetch(
            f"funding:{symbol}", lambda: self.client.fetch_funding_rate(symbol)
        )
        oi = self.cache.get_or_fetch(
            f"oi:{symbol}", lambda: self.client.fetch_open_interest(symbol)
        )
        return funding, oi

    # ---------- BTC macro regime ----------

    def _compute_btc_regime(self):
        macro_cfg = self.config["btc_macro_filter"]
        if not macro_cfg.get("enabled", True):
            return "neutral"

        btc_symbol = self.config["btc_symbol"]
        htf_tf = macro_cfg["htf_timeframe"]
        ohlcv = self._get_closed_ohlcv(btc_symbol, htf_tf, limit=120)
        if len(ohlcv) < 60:
            return "neutral"

        reading = technical.analyze(ohlcv)
        return determine_btc_regime(
            reading["bullish_score"], reading["bearish_score"], macro_cfg["strong_threshold"]
        )

    # ---------- per (symbol, mode) processing ----------

    def _process_symbol_mode(self, symbol, mode_name, btc_regime, btc_symbol):
        mode_cfg = self.config["modes"][mode_name]
        primary_tf = mode_cfg["primary_timeframe"]
        confirm_tfs = mode_cfg["confirm_timeframes"]

        primary_ohlcv = self._get_closed_ohlcv(symbol, primary_tf, limit=200)
        if len(primary_ohlcv) < 60:
            return  # not enough closed candles yet to score meaningfully

        confirm_ohlcv = {tf: self._get_closed_ohlcv(symbol, tf, limit=100) for tf in confirm_tfs}
        order_book = self._get_order_book(symbol)
        funding, oi = self._get_funding_and_oi(symbol)

        prev_close = primary_ohlcv[-2][4]
        last_close = primary_ohlcv[-1][4]
        price_change_pct = ((last_close - prev_close) / prev_close) * 100 if prev_close else 0.0

        funding_rate = funding.get("fundingRate", 0.0) if funding else 0.0
        open_interest = 0.0
        if oi:
            open_interest = oi.get("openInterestAmount") or oi.get("openInterestValue") or oi.get("openInterest") or 0.0

        layer_scores = {
            "technical": technical.analyze(primary_ohlcv),
            "multi_timeframe": multi_timeframe.analyze(confirm_ohlcv),
            "orderbook": orderbook.analyze(order_book) if order_book else {"bullish_score": 0, "bearish_score": 0},
            "market_structure": self.ms_tracker.score(symbol, funding_rate, open_interest, price_change_pct),
            "smc": smc.analyze(primary_ohlcv, self.config["smc"]["fractal_left"], self.config["smc"]["fractal_right"]),
        }

        weights = dict(mode_cfg["weights"])

        # Scalp-mode SMC noise control: cap SMC weight unless HTF trend
        # agrees with the SMC-implied direction.
        cap = mode_cfg.get("smc_weight_cap_if_htf_disagrees")
        if cap is not None and weights.get("smc", 0) > cap:
            mtf = layer_scores["multi_timeframe"]
            smc_reading = layer_scores["smc"]
            smc_dir_bull = smc_reading["bullish_score"] > smc_reading["bearish_score"]
            htf_dir_bull = mtf["bullish_score"] > mtf["bearish_score"]
            if smc_dir_bull != htf_dir_bull:
                weights["smc"] = cap

        bullish, bearish = composite_score(layer_scores, weights)

        is_btc = symbol == btc_symbol
        bullish, bearish = apply_btc_macro_filter(
            is_btc, btc_regime, bullish, bearish, self.config["btc_macro_filter"]
        )

        decision = decide_action(bullish, bearish, mode_cfg["min_score_to_signal"], mode_cfg["min_score_gap"])
        entry_price = primary_ohlcv[-1][4]
        entry_ts = primary_ohlcv[-1][0]

        reading = {
            "symbol": symbol,
            "mode": mode_name,
            "bullish_score": bullish,
            "bearish_score": bearish,
            "action": decision["action"],
            "confidence": decision["confidence"],
            "entry_price": entry_price,
            "timestamp": entry_ts,
            "layers": layer_scores,
        }

        with self._lock:
            self.last_readings[(symbol, mode_name)] = reading

        if decision["action"] in ("BUY", "SELL"):
            self._maybe_record_signal(symbol, mode_name, decision, entry_price, mode_cfg, layer_scores)

    def _maybe_record_signal(self, symbol, mode_name, decision, entry_price, mode_cfg, layer_scores):
        """Duplicate protection: don't re-record/re-notify the same action
        while the previous signal for this symbol+mode is still PENDING."""
        last = journal.get_last_signal(symbol, mode_name)
        if last and last["action"] == decision["action"] and last["status"] == "PENDING":
            return

        signal_id = journal.record_signal(
            symbol=symbol,
            mode=mode_name,
            action=decision["action"],
            confidence=decision["confidence"],
            entry_price=entry_price,
            evaluate_after_candles=mode_cfg["evaluation_candles"],
            success_move_pct=mode_cfg["success_move_pct"],
            layers_snapshot=layer_scores,
        )

        with self._lock:
            self._pending_notifications.append({
                "signal_id": signal_id,
                "symbol": symbol,
                "mode": mode_name,
                "action": decision["action"],
                "confidence": decision["confidence"],
                "entry_price": entry_price,
                "layers": layer_scores,
            })

    # ---------- evaluation of pending signals ----------

    def _evaluate_pending(self):
        pending = journal.get_pending_signals()
        for sig in pending:
            try:
                mode_cfg = self.config["modes"][sig["mode"]]
                primary_tf = mode_cfg["primary_timeframe"]
                all_candles = self._get_closed_ohlcv(sig["symbol"], primary_tf, limit=300)

                entry_dt = datetime.fromisoformat(sig["created_at"])
                entry_ts_ms = int(entry_dt.timestamp() * 1000)

                since = evaluator.candles_since(all_candles, entry_ts_ms)
                result = evaluator.evaluate_signal(sig, since)
                if result is not None:
                    journal.resolve_signal(
                        sig["signal_id"], result["status"], result["exit_price"],
                        result["mae_pct"], result["mfe_pct"],
                    )
            except Exception as e:
                print(f"[evaluator error] {sig.get('symbol')}/{sig.get('mode')}: {e}")
                continue

    # ---------- public API ----------

    def run_tick(self):
        btc_symbol = self.config["btc_symbol"]
        btc_regime = self._compute_btc_regime()

        for symbol in self.config["symbols"]:
            for mode_name in self.config["active_modes"]:
                try:
                    self._process_symbol_mode(symbol, mode_name, btc_regime, btc_symbol)
                except Exception as e:
                    # Per-symbol/mode error isolation — one failure must
                    # not stop the rest of the tick.
                    print(f"[tick error] {symbol}/{mode_name}: {e}")
                    continue

        self._evaluate_pending()

    def get_reading(self, symbol, mode):
        with self._lock:
            return self.last_readings.get((symbol, mode))

    def get_all_readings(self):
        with self._lock:
            return dict(self.last_readings)

    def pop_pending_notifications(self):
        """Drains and returns signals awaiting Telegram notification.
        Called by telegram_notifier after each tick."""
        with self._lock:
            items = self._pending_notifications
            self._pending_notifications = []
            return items
