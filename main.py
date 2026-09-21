"""
Entrypoint. Exposes a module-level `app` so it works both:
  - under a production WSGI server: `gunicorn main:app` (Railway's
    Railpack builder auto-detects Flask and does this by default)
  - run directly for local/dev: `python main.py`

IMPORTANT: only ONE gunicorn worker must be used (see Procfile:
--workers 1). The background tick-loop thread is started once at module
import time below — if gunicorn ran multiple worker PROCESSES, each
would start its own independent tick loop, multiplying exchange API
calls, duplicating journal writes, and duplicating Telegram
notifications. Threads within the single worker are fine (Flask request
handling + the tick-loop thread share one process safely, guarded by
Advisor's internal lock and journal.py's SQLite write lock).
"""

import os
import time
import threading

import yaml
from dotenv import load_dotenv

load_dotenv()

with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)

from advisor_engine import Advisor
import telegram_notifier
import app as app_module

advisor = Advisor(CONFIG)

# Module-level `app` — this is what `gunicorn main:app` looks for.
app = app_module.create_app(advisor, CONFIG)


def tick_loop():
    poll_interval = CONFIG.get("poll_interval_seconds", 90)
    while True:
        try:
            advisor.run_tick()
            telegram_notifier.send_pending_notifications(advisor, CONFIG)
            app_module.log_activity("tick completed")
        except Exception as e:
            print(f"[main tick_loop error] {e}")
            app_module.log_activity(f"tick error: {e}")
        time.sleep(poll_interval)


_tick_thread_started = False


def _start_tick_thread_once():
    global _tick_thread_started
    if not _tick_thread_started:
        thread = threading.Thread(target=tick_loop, daemon=True)
        thread.start()
        _tick_thread_started = True


# Runs once at import — whether imported by gunicorn or run as __main__.
_start_tick_thread_once()


if __name__ == "__main__":
    # Local/dev fallback only. In production, gunicorn (see Procfile)
    # imports this module and uses `app` directly without hitting this.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
