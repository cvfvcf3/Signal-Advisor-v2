import os
import time
import ccxt
import yaml


def _load_config():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def _with_retry(fn, retries=1, delay_seconds=1.5):
    """
    Runs fn() and retries once (by default) on any exception before
    giving up. Meant for transient network hiccups against the exchange
    API — a single retry recovers most of these without masking a
    persistent failure (which will still raise after the retry).
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay_seconds)
    raise last_exc


class ExchangeClient:
    """
    Thin wrapper around ccxt for the exchange configured in config.yaml.
    Handles auth and market_type (futures vs spot).
    READ-ONLY: never calls any order-placing ccxt methods (create_order,
    cancel_order, etc). Only market-data endpoints are used here.
    """

    def __init__(self, config=None):
        self.config = config or _load_config()
        ex_cfg = self.config["exchange"]

        exchange_class = getattr(ccxt, ex_cfg["id"])

        self.exchange = exchange_class({
            "apiKey": os.environ.get("BINANCE_API_KEY", ""),
            "secret": os.environ.get("BINANCE_API_SECRET", ""),
            "enableRateLimit": True,
            "rateLimit": ex_cfg.get("rate_limit_ms", 250),
            "options": {
                "defaultType": "future" if ex_cfg.get("market_type") == "future" else "spot",
                "fetchCurrencies": False,
                "warnOnFetchOpenOrdersWithoutSymbol": False,
            },
        })

    def fetch_ohlcv(self, symbol, timeframe, limit=200):
        """Returns list of [timestamp, open, high, low, close, volume]."""
        return _with_retry(lambda: self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))

    def fetch_order_book(self, symbol, limit=50):
        return _with_retry(lambda: self.exchange.fetch_order_book(symbol, limit=limit))

    def fetch_ticker(self, symbol):
        return _with_retry(lambda: self.exchange.fetch_ticker(symbol))

    def fetch_tickers_bulk(self, symbols=None):
        """Bulk ticker fetch where the exchange supports it — reduces the
        number of per-symbol API calls needed each tick."""
        if not self.exchange.has.get("fetchTickers"):
            return None
        return _with_retry(lambda: self.exchange.fetch_tickers(symbols))

    def fetch_funding_rate(self, symbol):
        """Futures only. Returns latest funding rate info, or None if the
        exchange/market doesn't support it."""
        if not self.exchange.has.get("fetchFundingRate"):
            return None
        return _with_retry(lambda: self.exchange.fetch_funding_rate(symbol))

    def fetch_open_interest(self, symbol):
        """Futures only. Returns current open interest, or None if
        unsupported."""
        if not self.exchange.has.get("fetchOpenInterest"):
            return None
        return _with_retry(lambda: self.exchange.fetch_open_interest(symbol))
