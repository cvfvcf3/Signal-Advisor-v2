"""
Advisor engine: the central orchestrator. One Advisor instance is created
at startup (by main.py) and its run_tick() is called on a timer
(poll_interval_seconds). Each tick:

  1. Determines BTC's HTF regime (bullish/bearish/neutral) once.
  2. Loops over every (symbol, mode) combination, running all 5 layers,
     combining them into a composite score, applying the BTC macro-filter
     to altcoins, and deciding BUY/SELL/WAIT.
  3. On a new BUY/SELL: checks max_concurrent_positions for that mode
     (skips if already at the cap), computes a volatility-adaptive
     take_profit/stop_loss, and records the signal — guarded by
     duplicate protection and a per-mode cooldown.
  4. Evaluates PENDING signals two ways, in order:
       a. Price-action outcome via evaluator.evaluate_signal (path-aware
          TP/SL/trailing/EXPIRED check against newly-closed candles).
       b. If still open, INVALIDATION: compares the pending signal's
          original direction against this tick's freshly-computed live
          reading for that same (symbol, mode). If the live composite
          has flipped hard enough to now itself imply the opposite
          action (gap >= mode's min_score_gap), the position is closed
          early as INVALIDATED at the current price — this is a risk-
          management exit for when the original thesis reverses, rather
          than waiting for a fixed stop-loss or timeout. Guarded by a
          minimum signal age (10 minutes) to avoid first-tick noise.
     Either way, once resolved, computes the paper-trading dollar P&L
     for that mode's simulated balance (see config.yaml paper_trading)
     and persists everything together.

PER-SYMBOL ERROR ISOLATION: a failure processing one (symbol, mode)
combination is caught and logged; it does not stop the rest of the tick.

CLOSED CANDLES ONLY: _get_closed_ohlcv always drops the most recent
(potentially still-forming) candle before handing data to any layer.
"""

import threading
from datetime import datetime, timezone, timedelta

from exchange_client import ExchangeClient
from cache import TTLCache
from layers import technical, multi_timeframe, orderbook, smc
from layers.market_structure import MarketStructureTracker
from scoring import composite_score, determine_btc_regime, apply_btc_macro_filter, decide_action
import journal
import evaluator

MIN_SIGNAL_AGE_MINUTES_FOR_INVALIDATION = 10


class Advisor:
    def __init__(self, config):
        self.config = config
        self.client = ExchangeClient(config)
        self.cache = TTLCache(ttl_seconds=config["cache"]["ttl_seconds"])
        self.ms_tracker = MarketStructureTracker(
            window_candles=config["market_structure"]["rolling_window_candles"]
        )

        self._lock = threading.Lock()
        self.last_readings = {}
        self._pending_notifications = []

        journal.init_db()

    # ---------- data fetch helpers (cached, closed-candle only) ----------

    def _get_closed_ohlcv(self, symbol, timeframe, limit=200):
        key = f"ohlcv:{symbol}:{timeframe}"
        raw = self.cache.get_or_fetch(
            key, lambda: self.client.fetch_ohlcv(symbol, timeframe, limit=limit + 1)
        )
        if not raw or len(raw) < 2:
            return []
        return raw[:-1]

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
            return

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
            self._maybe_record_signal(symbol, mode_name, decision, entry_price, mode_cfg, layer_scores, primary_ohlcv)

    def _compute_tp_sl(self, action, entry_price, mode_cfg, primary_ohlcv):
        atr_mult = self.config.get("adaptive_target", {}).get("atr_multiplier", 1.4)
        atr_pct_value = technical.atr_pct(primary_ohlcv, period=14)

        target_pct = max(mode_cfg["success_move_pct"], atr_mult * atr_pct_value)
        stop_pct = target_pct / mode_cfg["rr"]

        if action == "BUY":
            take_profit = entry_price * (1 + target_pct)
            stop_loss = entry_price * (1 - stop_pct)
        else:
            take_profit = entry_price * (1 - target_pct)
            stop_loss = entry_price * (1 + stop_pct)

        return round(take_profit, 8), round(stop_loss, 8)

    def _cooldown_active(self, symbol, mode_name, cooldown_minutes):
        if not cooldown_minutes:
            return False
        last = journal.get_last_signal(symbol, mode_name)
        if not last:
            return False
        last_created = datetime.fromisoformat(last["created_at"])
        if last_created.tzinfo is None:
            last_created = last_created.replace(tzinfo=timezone.utc)
        elapsed = datetime.now(timezone.utc) - last_created
        return elapsed < timedelta(minutes=cooldown_minutes)

    def _maybe_record_signal(self, symbol, mode_name, decision, entry_price, mode_cfg, layer_scores, primary_ohlcv):
        last = journal.get_last_signal(symbol, mode_name)
        if last and last["action"] == decision["action"] and last["status"] == "PENDING":
            return

        if self._cooldown_active(symbol, mode_name, mode_cfg.get("cooldown_minutes")):
            return

        max_concurrent = mode_cfg.get("max_concurrent_positions")
        if max_concurrent is not None:
            open_count = journal.get_open_positions_count(mode_name)
            if open_count >= max_concurrent:
                return

        take_profit, stop_loss = self._compute_tp_sl(decision["action"], entry_price, mode_cfg, primary_ohlcv)

        signal_id = journal.record_signal(
            symbol=symbol,
            mode=mode_name,
            action=decision["action"],
            confidence=decision["confidence"],
            entry_price=entry_price,
            evaluate_after_candles=mode_cfg["evaluation_candles"],
            success_move_pct=mode_cfg["success_move_pct"],
            take_profit=take_profit,
            stop_loss=stop_loss,
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
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "layers": layer_scores,
            })

    # ---------- invalidation (thesis-reversal early exit) ----------

    def _check_invalidation(self, sig):
        """Returns an evaluator-style result dict if the signal's live
        reading has reversed hard enough to invalidate it, else None."""
        entry_dt = datetime.fromisoformat(sig["created_at"])
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        age_minutes = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60
        if age_minutes < MIN_SIGNAL_AGE_MINUTES_FOR_INVALIDATION:
            return None

        current = self.get_reading(sig["symbol"], sig["mode"])
        if current is None:
            return None

        mode_cfg = self.config["modes"][sig["mode"]]
        gap = mode_cfg["min_score_gap"]
        action = sig["action"]
        reversed_hard = False

        if action == "BUY" and (current["bearish_score"] - current["bullish_score"]) >= gap:
            reversed_hard = True
        elif action == "SELL" and (current["bullish_score"] - current["bearish_score"]) >= gap:
            reversed_hard = True

        if not reversed_hard:
            return None

        exit_price = current["entry_price"]  # latest close used for this tick's reading
        entry = sig["entry_price"]
        return {
            "status": "INVALIDATED",
            "exit_price": round(exit_price, 8),
            "pnl_pct": round(evaluator.pnl_pct(action, entry, exit_price), 4),
            "mae_pct": None,
            "mfe_pct": None,
        }

    # ---------- paper trading ----------

    def _apply_paper_trading(self, sig, result):
        paper_cfg = self.config.get("paper_trading", {})
        if not paper_cfg.get("enabled", True) or result.get("pnl_pct") is None:
            return None, None

        starting_balance = paper_cfg.get("starting_balance", 1000)
        risk_pct = paper_cfg.get("risk_pct_per_trade", 1.0)

        current_balance = journal.get_paper_balance(sig["mode"], starting_balance)

        entry = sig["entry_price"]
        stop_loss = sig["stop_loss"]
        stop_distance_pct = abs(entry - stop_loss) / entry if entry else 0

        if stop_distance_pct <= 0:
            return None, None

        dollar_risk = current_balance * (risk_pct / 100)
        position_size_usd = dollar_risk / stop_distance_pct
        dollar_pnl = position_size_usd * (result["pnl_pct"] / 100)

        journal.apply_paper_pnl(sig["mode"], dollar_pnl, starting_balance)

        return round(dollar_pnl, 4), round(position_size_usd, 2)

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
                result = evaluator.evaluate_signal(sig, since, trailing_cfg=mode_cfg.get("trailing"))

                if result is None:
                    result = self._check_invalidation(sig)

                if result is not None:
                    dollar_pnl, position_size_usd = self._apply_paper_trading(sig, result)
                    journal.resolve_signal(
                        sig["signal_id"], result["status"], result["exit_price"],
                        result["mae_pct"], result["mfe_pct"], result.get("pnl_pct"),
                        dollar_pnl, position_size_usd,
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
        with self._lock:
            items = self._pending_notifications
            self._pending_notifications = []
            return items
