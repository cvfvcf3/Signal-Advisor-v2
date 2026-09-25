"""
Flask dashboard + JSON API. READ-ONLY except one clearly-marked,
token-gated destructive route (clear-history). Nothing here can place,
modify, or cancel a real exchange order. Paper trading balances are
purely simulated numbers this app tracks itself.
"""

import os
import hmac
import csv
import io
import json
from datetime import datetime, timezone
from flask import Flask, jsonify, request, render_template, Response

import journal


ACTIVITY_LOG = []
MAX_LOG_ENTRIES = 200


def log_activity(message):
    ACTIVITY_LOG.append({"time": datetime.now(timezone.utc).isoformat(), "message": message})
    if len(ACTIVITY_LOG) > MAX_LOG_ENTRIES:
        del ACTIVITY_LOG[: len(ACTIVITY_LOG) - MAX_LOG_ENTRIES]


ALLOWED_MODE_FIELDS = {"weights", "min_score_to_signal", "min_score_gap"}
ALLOWED_TELEGRAM_FIELDS = {"enabled", "min_confidence_to_notify"}


def _validate_reload_payload(payload, config):
    if not isinstance(payload, dict):
        return False, "payload must be a JSON object"

    allowed_top = {"modes", "telegram"}
    unknown_top = set(payload.keys()) - allowed_top
    if unknown_top:
        return False, f"fields not allowed: {sorted(unknown_top)}"

    if "modes" in payload:
        if not isinstance(payload["modes"], dict):
            return False, "modes must be an object"
        for mode_name, mode_updates in payload["modes"].items():
            if mode_name not in config["modes"]:
                return False, f"unknown mode: {mode_name}"
            if not isinstance(mode_updates, dict):
                return False, f"modes.{mode_name} must be an object"
            unknown_fields = set(mode_updates.keys()) - ALLOWED_MODE_FIELDS
            if unknown_fields:
                return False, f"modes.{mode_name}: fields not allowed: {sorted(unknown_fields)}"

    if "telegram" in payload:
        if not isinstance(payload["telegram"], dict):
            return False, "telegram must be an object"
        unknown_fields = set(payload["telegram"].keys()) - ALLOWED_TELEGRAM_FIELDS
        if unknown_fields:
            return False, f"telegram: fields not allowed: {sorted(unknown_fields)}"

    return True, None


def _check_admin_token():
    expected = os.environ.get("DASHBOARD_ADMIN_TOKEN", "")
    header_token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    query_token = request.args.get("token", "").strip()
    provided = header_token or query_token
    return bool(expected) and hmac.compare_digest(provided, expected)


CSV_COLUMNS = [
    "signal_id", "created_at", "symbol", "mode", "action", "confidence",
    "entry_price", "take_profit", "stop_loss", "status", "resolved_at",
    "exit_price", "pnl_pct", "dollar_pnl", "position_size_usd",
    "mae_pct", "mfe_pct",
]


def create_app(advisor, config):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            symbols=config["symbols"],
            modes=config["active_modes"],
        )

    @app.route("/api/health")
    def api_health():
        last_tick_time = None
        last_tick_ok = None
        for entry in reversed(ACTIVITY_LOG):
            if "tick completed" in entry["message"] or "tick error" in entry["message"]:
                last_tick_time = entry["time"]
                last_tick_ok = "tick completed" in entry["message"]
                break
        db_reachable = True
        try:
            journal.get_accuracy()
        except Exception:
            db_reachable = False
        return jsonify({
            "status": "ok" if db_reachable else "degraded",
            "db_reachable": db_reachable,
            "last_tick_time": last_tick_time,
            "last_tick_ok": last_tick_ok,
        })

    @app.route("/api/symbols")
    def api_symbols():
        return jsonify({"symbols": config["symbols"], "modes": config["active_modes"]})

    @app.route("/api/current_signal")
    def api_current_signal():
        symbol = request.args.get("symbol")
        mode = request.args.get("mode")
        if symbol and mode:
            reading = advisor.get_reading(symbol, mode)
            return jsonify(reading or {})
        readings = advisor.get_all_readings()
        return jsonify(list(readings.values()))

    @app.route("/api/signals")
    def api_signals():
        symbol = request.args.get("symbol")
        mode = request.args.get("mode")
        limit = int(request.args.get("limit", 50))
        return jsonify(journal.get_signal_history(symbol=symbol, mode=mode, limit=limit))

    @app.route("/api/accuracy")
    def api_accuracy():
        symbol = request.args.get("symbol")
        mode = request.args.get("mode")
        return jsonify(journal.get_accuracy(symbol=symbol, mode=mode))

    @app.route("/api/attribution")
    def api_attribution():
        """
        Level-1 attribution: per layer, accuracy when that layer agreed
        with the trade's direction vs when it disagreed. Read-only,
        computed from resolved CORRECT/INCORRECT signals only.
        """
        mode = request.args.get("mode")
        return jsonify(journal.get_layer_attribution(mode=mode))

    @app.route("/api/paper_balance")
    def api_paper_balance():
        mode = request.args.get("mode")
        starting_balance = config.get("paper_trading", {}).get("starting_balance", 1000)

        if mode:
            balance = journal.get_paper_balance(mode, starting_balance)
            all_rows = journal.get_all_paper_balances()
            row = all_rows.get(mode, {})
            return jsonify({
                "mode": mode,
                "starting_balance": starting_balance,
                "current_balance": round(balance, 2),
                "total_dollar_pnl": round(balance - starting_balance, 2),
                "trade_count": row.get("trade_count", 0),
            })

        all_rows = journal.get_all_paper_balances()
        result = {}
        for m in config.get("active_modes", []):
            bal = all_rows.get(m, {}).get("balance", starting_balance)
            result[m] = {
                "starting_balance": starting_balance,
                "current_balance": round(bal, 2),
                "total_dollar_pnl": round(bal - starting_balance, 2),
                "trade_count": all_rows.get(m, {}).get("trade_count", 0),
            }
        return jsonify(result)

    @app.route("/api/logs")
    def api_logs():
        return jsonify(list(reversed(ACTIVITY_LOG)))

    @app.route("/api/debug/storage")
    def api_debug_storage():
        db_path = journal.DB_PATH
        data_dir = journal.DATA_DIR
        exists = os.path.exists(db_path)
        info = {
            "DATA_DIR_env_var_set": "DATA_DIR" in os.environ,
            "DATA_DIR_value": os.environ.get("DATA_DIR", "(not set)"),
            "resolved_data_dir": data_dir,
            "resolved_db_path": db_path,
            "db_file_exists": exists,
        }
        if exists:
            stat = os.stat(db_path)
            info["db_file_size_bytes"] = stat.st_size
            info["db_last_modified"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        try:
            info["is_on_mounted_volume"] = os.path.ismount(data_dir) or os.path.ismount(os.path.dirname(data_dir))
        except Exception:
            info["is_on_mounted_volume"] = "unknown"
        return jsonify(info)

    @app.route("/api/export")
    def api_export():
        symbol = request.args.get("symbol")
        mode = request.args.get("mode")
        fmt = request.args.get("format", "csv").lower()

        rows = journal.get_signal_history(symbol=symbol, mode=mode, limit=100000)
        rows.sort(key=lambda r: r.get("created_at") or "")

        if fmt == "json":
            for r in rows:
                if isinstance(r.get("layers_snapshot"), str):
                    try:
                        r["layers_snapshot"] = json.loads(r["layers_snapshot"])
                    except Exception:
                        pass
            body = json.dumps(rows, indent=2)
            return Response(
                body,
                mimetype="application/json",
                headers={"Content-Disposition": "attachment; filename=signal_history.json"},
            )

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=signal_history.csv"},
        )

    @app.route("/api/admin/clear-history", methods=["GET", "POST"])
    def admin_clear_history():
        if not _check_admin_token():
            return jsonify({"error": "unauthorized"}), 401
        if request.args.get("confirm") != "yes":
            return jsonify({
                "error": "confirmation required",
                "hint": "add &confirm=yes to actually clear history",
            }), 400
        journal.clear_all_history()
        log_activity("history cleared via admin endpoint")
        return jsonify({"status": "ok", "message": "all signal history and paper balances cleared"})

    @app.route("/api/admin/reload-config", methods=["POST"])
    def admin_reload_config():
        if not _check_admin_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = request.get_json(silent=True) or {}
        valid, error = _validate_reload_payload(payload, config)
        if not valid:
            return jsonify({"error": error}), 400

        if "modes" in payload:
            for mode_name, updates in payload["modes"].items():
                config["modes"][mode_name].update(updates)
        if "telegram" in payload:
            config["telegram"].update(payload["telegram"])

        log_activity(f"config reloaded: {list(payload.keys())}")
        return jsonify({"status": "ok", "applied": payload})

    return app
