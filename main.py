"""
Entrypoint. Run with: python main.py (or via Railway's start command).

Responsibilities:
  1. Load .env and config.yaml.
  2. Construct one Advisor instance (holds exchange client, cache,
     market-structure tracker, journal access, in-memory readings).
  3. Start a background thread that calls advisor.run_tick() then
     telegram_notifier.send_pending_notifications() every
     poll_interval_seconds.
  4. Start the Flask dashboard (app.create_app) in the foreground.

Both the background tick loop and Flask requests share the same Advisor
instance and the same SQLite journal — safe because journal.py already
handles WAL mode + a write lock, and Advisor's shared state is guarded by
its own lock.
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


if __name__ == "__main__":
    thread = threading.Thread(target=tick_loop, daemon=True)
    thread.start()

    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host="0.0.0.0", port=port)
