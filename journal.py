"""
Signal journal: persists every generated signal (per symbol + per mode)
to SQLite and tracks its eventual resolution (CORRECT / INCORRECT /
EXPIRED / INVALIDATED), plus a per-mode paper-trading balance ledger.

PERSISTENCE: DB_PATH is read from the DATA_DIR environment variable
(point it at a Railway Volume's mount path, e.g. /data) so history
survives redeploys; falls back to a local ./data folder otherwise.

CONCURRENCY: WAL mode + a module-level write lock + a busy_timeout —
see comments below for why each is needed.

STATUSES:
  PENDING     — still open, not yet resolved
  CORRECT     — hit its target (fixed TP, or a trailing-stop exit)
  INCORRECT   — hit its hard stop-loss
  EXPIRED     — evaluation window ran out with neither triggered
  INVALIDATED — closed early because the live signal reversed against
                the open position (see advisor_engine.py) — a risk-
                management exit, not a price-target outcome. Excluded
                from accuracy_pct like EXPIRED, but included in P&L.

PAPER TRADING: purely simulated — never touches a real exchange
balance. Each MODE has its own independent starting balance (see
paper_balances table). Position size for a resolving trade is computed
by advisor_engine.py using fixed fractional risk (dollar_risk = balance
* risk_pct_per_trade / 100; position_size_usd = dollar_risk /
stop_distance_pct) and passed in as dollar_pnl/position_size_usd — this
module just persists those numbers and updates the running balance.
"""

import os
import sqlite3
import threading
import uuid
import json
from datetime import datetime, timezone

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
DB_PATH = os.path.join(DATA_DIR, "signals.db")

_write_lock = threading.Lock()


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn, table, column, coltype):
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    with _write_lock:
        conn = _connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    evaluate_after_candles INTEGER NOT NULL,
                    success_move_pct REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    resolved_at TEXT,
                    exit_price REAL,
                    mae_pct REAL,
                    mfe_pct REAL,
                    layers_snapshot TEXT,
                    notified INTEGER NOT NULL DEFAULT 0
                )
            """)
            _add_column_if_missing(conn, "signals", "take_profit", "REAL")
            _add_column_if_missing(conn, "signals", "stop_loss", "REAL")
            _add_column_if_missing(conn, "signals", "pnl_pct", "REAL")
            _add_column_if_missing(conn, "signals", "dollar_pnl", "REAL")
            _add_column_if_missing(conn, "signals", "position_size_usd", "REAL")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_balances (
                    mode TEXT PRIMARY KEY,
                    balance REAL NOT NULL,
                    starting_balance REAL NOT NULL,
                    trade_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_mode ON signals(symbol, mode)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON signals(status)")
            conn.commit()
        finally:
            conn.close()


def record_signal(symbol, mode, action, confidence, entry_price,
                   evaluate_after_candles, success_move_pct,
                   take_profit, stop_loss, layers_snapshot=None):
    signal_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    with _write_lock:
        conn = _connect()
        try:
            conn.execute("""
                INSERT INTO signals (
                    signal_id, symbol, mode, action, confidence, entry_price,
                    created_at, evaluate_after_candles, success_move_pct,
                    take_profit, stop_loss, status, layers_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
            """, (
                signal_id, symbol, mode, action, confidence, entry_price,
                created_at, evaluate_after_candles, success_move_pct,
                take_profit, stop_loss, json.dumps(layers_snapshot or {}),
            ))
            conn.commit()
        finally:
            conn.close()

    return signal_id


def resolve_signal(signal_id, status, exit_price, mae_pct=None, mfe_pct=None,
                    pnl_pct=None, dollar_pnl=None, position_size_usd=None):
    """status: 'CORRECT' | 'INCORRECT' | 'EXPIRED' | 'INVALIDATED'"""
    resolved_at = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        conn = _connect()
        try:
            conn.execute("""
                UPDATE signals
                SET status = ?, resolved_at = ?, exit_price = ?, mae_pct = ?, mfe_pct = ?,
                    pnl_pct = ?, dollar_pnl = ?, position_size_usd = ?
                WHERE signal_id = ?
            """, (status, resolved_at, exit_price, mae_pct, mfe_pct,
                  pnl_pct, dollar_pnl, position_size_usd, signal_id))
            conn.commit()
        finally:
            conn.close()


def mark_notified(signal_id):
    with _write_lock:
        conn = _connect()
        try:
            conn.execute("UPDATE signals SET notified = 1 WHERE signal_id = ?", (signal_id,))
            conn.commit()
        finally:
            conn.close()


def get_pending_signals(symbol=None, mode=None):
    conn = _connect()
    try:
        query = "SELECT * FROM signals WHERE status = 'PENDING'"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_open_positions_count(mode):
    """How many PENDING signals currently exist for this mode, across all
    symbols — used to enforce max_concurrent_positions."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM signals WHERE mode = ? AND status = 'PENDING'",
            (mode,),
        ).fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def get_last_signal(symbol, mode):
    conn = _connect()
    try:
        row = conn.execute("""
            SELECT * FROM signals WHERE symbol = ? AND mode = ?
            ORDER BY created_at DESC LIMIT 1
        """, (symbol, mode)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_signal_history(symbol=None, mode=None, limit=50):
    conn = _connect()
    try:
        query = "SELECT * FROM signals WHERE 1=1"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_accuracy(symbol=None, mode=None):
    """
    accuracy_pct = CORRECT / (CORRECT + INCORRECT) — EXPIRED and
    INVALIDATED are tracked but excluded from this hit-rate, since
    neither represents a clean target-hit-or-stopped-out outcome.
    total_pnl_pct / avg_pnl_pct cover ALL resolved statuses with a
    stored pnl_pct (CORRECT, INCORRECT, EXPIRED, INVALIDATED alike).
    """
    conn = _connect()
    try:
        query = "SELECT status, COUNT(*) as cnt FROM signals WHERE 1=1"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        query += " GROUP BY status"
        rows = conn.execute(query, params).fetchall()

        counts = {"PENDING": 0, "CORRECT": 0, "INCORRECT": 0, "EXPIRED": 0, "INVALIDATED": 0}
        for r in rows:
            counts[r["status"]] = r["cnt"]

        resolved = counts["CORRECT"] + counts["INCORRECT"]
        accuracy_pct = round((counts["CORRECT"] / resolved) * 100, 2) if resolved > 0 else None

        pnl_query = "SELECT SUM(pnl_pct) as total, AVG(pnl_pct) as avg, COUNT(pnl_pct) as n FROM signals WHERE pnl_pct IS NOT NULL"
        pnl_params = []
        if symbol:
            pnl_query += " AND symbol = ?"
            pnl_params.append(symbol)
        if mode:
            pnl_query += " AND mode = ?"
            pnl_params.append(mode)
        pnl_row = conn.execute(pnl_query, pnl_params).fetchone()
        total_pnl_pct = round(pnl_row["total"], 4) if pnl_row["total"] is not None else None
        avg_pnl_pct = round(pnl_row["avg"], 4) if pnl_row["avg"] is not None else None

        return {
            "symbol": symbol,
            "mode": mode,
            "correct": counts["CORRECT"],
            "incorrect": counts["INCORRECT"],
            "expired": counts["EXPIRED"],
            "invalidated": counts["INVALIDATED"],
            "pending": counts["PENDING"],
            "accuracy_pct": accuracy_pct,
            "total_pnl_pct": total_pnl_pct,
            "avg_pnl_pct": avg_pnl_pct,
            "pnl_trade_count": pnl_row["n"],
        }
    finally:
        conn.close()


# ---------- paper trading balance ledger (per mode) ----------

def get_paper_balance(mode, starting_balance):
    """Returns the current balance for a mode, creating its row (seeded
    at starting_balance) on first use."""
    with _write_lock:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM paper_balances WHERE mode = ?", (mode,)).fetchone()
            if row is None:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute("""
                    INSERT INTO paper_balances (mode, balance, starting_balance, trade_count, updated_at)
                    VALUES (?, ?, ?, 0, ?)
                """, (mode, starting_balance, starting_balance, now))
                conn.commit()
                return starting_balance
            return row["balance"]
        finally:
            conn.close()


def apply_paper_pnl(mode, dollar_pnl, starting_balance):
    """Adds dollar_pnl to the mode's running balance (creating the row
    seeded at starting_balance first if needed), incrementing trade_count.
    Returns the new balance."""
    with _write_lock:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM paper_balances WHERE mode = ?", (mode,)).fetchone()
            now = datetime.now(timezone.utc).isoformat()
            if row is None:
                new_balance = starting_balance + dollar_pnl
                conn.execute("""
                    INSERT INTO paper_balances (mode, balance, starting_balance, trade_count, updated_at)
                    VALUES (?, ?, ?, 1, ?)
                """, (mode, new_balance, starting_balance, now))
            else:
                new_balance = row["balance"] + dollar_pnl
                conn.execute("""
                    UPDATE paper_balances SET balance = ?, trade_count = trade_count + 1, updated_at = ?
                    WHERE mode = ?
                """, (new_balance, now, mode))
            conn.commit()
            return new_balance
        finally:
            conn.close()


def get_all_paper_balances():
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM paper_balances").fetchall()
        return {r["mode"]: dict(r) for r in rows}
    finally:
        conn.close()
