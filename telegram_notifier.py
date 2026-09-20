"""
Telegram notifications for new BUY/SELL signals.

Reads bot_token/chat_id from environment (.env), never from config.yaml.
Called once per tick, after advisor.run_tick(): pulls whatever new
signals the engine queued up (advisor.pop_pending_notifications()),
applies the min-confidence filter, and sends one message per signal.

DUPLICATE PROTECTION is primarily handled upstream in advisor_engine.py
(a repeat of the same pending action isn't queued at all). This module
additionally marks each sent signal as notified=1 in the journal so a
restart of the process can't accidentally resend it.
"""

import os
import requests

TELEGRAM_API_BASE = "https://api.telegram.org"


def _format_message(signal):
    layers = signal.get("layers", {})
    layer_lines = []
    for name, reading in layers.items():
        b = reading.get("bullish_score", 0)
        s = reading.get("bearish_score", 0)
        layer_lines.append(f"  {name}: bull={b} bear={s}")

    layers_text = "\n".join(layer_lines) if layer_lines else "  (no layer detail)"

    return (
        f"{signal['action']} signal — {signal['symbol']} ({signal['mode']})\n"
        f"Confidence: {signal['confidence']}\n"
        f"Entry: {signal['entry_price']}\n"
        f"Take-Profit: {signal.get('take_profit', '--')}\n"
        f"Stop-Loss: {signal.get('stop_loss', '--')}\n\n"
        f"Layers:\n{layers_text}\n\n"
        f"Read-only advisor — no trade was placed."
    )


def _send_message(text):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        print("[telegram_notifier] missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID, skipping send")
        return False

    url = f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        if resp.status_code != 200:
            print(f"[telegram_notifier] send failed: {resp.status_code} {resp.text}")
            return False
        return True
    except Exception as e:
        print(f"[telegram_notifier] send error: {e}")
        return False


def send_pending_notifications(advisor, config):
    """
    Call once per tick, after advisor.run_tick(). Drains queued signals,
    filters by min_confidence_to_notify, sends each via Telegram, and
    marks it notified in the journal.
    """
    telegram_cfg = config.get("telegram", {})
    if not telegram_cfg.get("enabled", False):
        advisor.pop_pending_notifications()  # drain without sending
        return

    min_confidence = telegram_cfg.get("min_confidence_to_notify", 0)
    signals = advisor.pop_pending_notifications()

    for signal in signals:
        if signal["confidence"] < min_confidence:
            continue

        text = _format_message(signal)
        sent = _send_message(text)
        if sent:
            import journal
            journal.mark_notified(signal["signal_id"])
