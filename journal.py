"""
Signal journal: persists every generated signal (per symbol + per mode)
to SQLite and tracks its eventual resolution (correct/incorrect/expired).

PERSISTENCE: Railway (and most container hosts) wipe local container
filesystem on every redeploy — a plain relative path here would silently
lose all history each time new code is pushed. DB_PATH is therefore read
from the DATA_DIR environment variable when set (point it at a Railway
Volume's mount path, e.g. /data) and only falls back to a local ./data
folder for local/dev runs where that isn't set up.

CONCURRENCY: the Flask dashboard (request-driven) and the advisor_engine
tick loop (background thread) both touch this database. To avoid
"database is locked" errors under load:
  - WAL mode is enabled (PRAGMA journal_mode=WAL) so readers don't block
    writers and vice versa.
  - A module-level threading.Lock wraps every write, since SQLite still
    serializes writers even in WAL mode — this avoids write contention
    errors rather than relying on busy_timeout retries alone.
  - A busy_timeout is set on every connection so a momentary lock
    (a write already in progress) waits briefly instead of failing
    immediately.

Every signal record carries both symbol and mode, since accuracy and
history must be tracked per (symbol, mode) combination — never mixed.

Each signal also stores take_profit and stop_loss price levels (computed
by advisor_engine.py from the mode's success_move_pct / stop_loss_pct),
so the journal — and the dashboard — show more than just an entry price.
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
            # Migration for DBs created before take_profit/stop_loss existed.
            _add_column_if_missing(conn, "signals", "take_profit", "REAL")
            _add_column_if_missing(conn, "signals", "stop_loss", "REAL")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_mode ON signals(symbol, mode)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON signals(status)")
            conn.commit()
        finally:
            conn.close()


def record_signal(symbol, mode, action, confidence, entry_price,
                   evaluate_after_candles, success_move_pct,
                   take_profit, stop_loss, layers_snapshot=None):
    """
    Inserts a new PENDING signal. Returns the generated signal_id.
    take_profit / stop_loss: absolute price levels (not percentages),
                              computed by the caller from the mode's
                              success_move_pct / stop_loss_pct.
    layers_snapshot: dict of which layers/values supported this signal
                      (stored as JSON, used for dashboard + Telegram detail).
    """
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


def resolve_signal(signal_id, status, exit_price, mae_pct=None, mfe_pct=None):
    """status: 'CORRECT' | 'INCORRECT' | 'EXPIRED'"""
    resolved_at = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        conn = _connect()
        try:
            conn.execute("""
                UPDATE signals
                SET status = ?, resolved_at = ?, exit_price = ?, mae_pct = ?, mfe_pct = ?
                WHERE signal_id = ?
            """, (status, resolved_at, exit_price, mae_pct, mfe_pct, signal_id))
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


def get_last_signal(symbol, mode):
    """Most recent signal (any status) for duplicate-notification checks."""
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
    Returns per (symbol, mode) accuracy stats. If symbol/mode are None,
    aggregates across all — but callers should almost always pass both,
    since mixing modes/symbols produces a misleading accuracy figure.
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

        counts = {"PENDING": 0, "CORRECT": 0, "INCORRECT": 0, "EXPIRED": 0}
        for r in rows:
            counts[r["status"]] = r["cnt"]

        resolved = counts["CORRECT"] + counts["INCORRECT"]
        accuracy_pct = round((counts["CORRECT"] / resolved) * 100, 2) if resolved > 0 else None

        return {
            "symbol": symbol,
            "mode": mode,
            "correct": counts["CORRECT"],
            "incorrect": counts["INCORRECT"],
            "expired": counts["EXPIRED"],
            "pending": counts["PENDING"],
            "accuracy_pct": accuracy_pct,
        }
    finally:
        conn.close()
