"""
Entrypoint.

Production (Railway): served by gunicorn, e.g.
    gunicorn main:flask_app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
gunicorn imports this module and looks for the `flask_app` WSGI object —
it never executes the `if __name__ == "__main__":` block. --workers 1 is
required: the background tick loop below starts once at import time, and
running more than one worker process would start multiple tick loops,
each independently hitting the exchange and duplicating signals.

Local/dev: `python main.py` runs this module as __main__, which starts
the same background thread (module-level code always runs once on
import/execution) and then serves via Flask's own dev server.

Responsibilities:
  1. Load .env and config.yaml.
  2. Construct one Advisor instance (holds exchange client, cache,
     market-structure tracker, journal access, in-memory readings).
  3. Start a background thread that calls advisor.run_tick() then
     telegram_notifier.send_pending_notifications() every
     poll_interval_seconds — started at MODULE LEVEL so it runs
     regardless of how this file is loaded (gunicorn or `python main.py`).
  4. Expose flask_app (the dashboard) as a module-level WSGI object.
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
flask_app = app_module.create_app(advisor, CONFIG)


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


_tick_thread = threading.Thread(target=tick_loop, daemon=True)
_tick_thread.start()


if __name__ == "__main__":
    # Local/dev only — Railway uses gunicorn (see module docstring).
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host="0.0.0.0", port=port)
