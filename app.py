"""
Flask dashboard + JSON API. READ-ONLY: every route here only reads data
(advisor's in-memory readings, or the journal). Nothing here can place,
modify, or cancel an exchange order — there is no such endpoint, and the
one write-capable route (admin config reload) only ever touches scoring
weights/thresholds/notification settings, never exchange credentials or
trading behavior. The export routes are also read-only — they generate a
CSV/JSON snapshot from the journal, never touch or clear it.

create_app(advisor, config) is a factory so main.py can construct the
Advisor once and hand it to Flask, rather than each module creating its
own instance.
"""

import os
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


# Only these top-level sections, and only these fields within them, can be
# changed via the admin reload endpoint. Anything else (exchange config,
# symbols list, credentials, market_type, etc.) is rejected outright.
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


CSV_COLUMNS = [
    "signal_id", "created_at", "symbol", "mode", "action", "confidence",
    "entry_price", "take_profit", "stop_loss", "status", "resolved_at",
    "exit_price", "mae_pct", "mfe_pct",
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

    @app.route("/api/logs")
    def api_logs():
        return jsonify(list(reversed(ACTIVITY_LOG)))

    @app.route("/api/export")
    def api_export():
        """
        Read-only export of the full (or filtered) signal history.
        ?format=csv (default) or ?format=json
        ?symbol=... and ?mode=... optionally filter, same as /api/signals.
        This only reads the journal — it never deletes or modifies it.
        """
        symbol = request.args.get("symbol")
        mode = request.args.get("mode")
        fmt = request.args.get("format", "csv").lower()

        rows = journal.get_signal_history(symbol=symbol, mode=mode, limit=100000)
        rows.sort(key=lambda r: r.get("created_at") or "")

        if fmt == "json":
            # Full detail including layers_snapshot for each signal.
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

        # Default: CSV (layers_snapshot omitted — use format=json for that detail)
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

    @app.route("/api/admin/reload-config", methods=["POST"])
    def admin_reload_config():
        token_env = config["dashboard"]["admin_token_env"]
        expected = os.environ.get(token_env, "")
        provided = request.headers.get("Authorization", "").replace("Bearer ", "").strip()

        if not expected or provided != expected:
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
