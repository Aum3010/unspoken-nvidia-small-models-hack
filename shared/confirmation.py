"""Deterministic speech authorization (plan.md §4 policy gate).

This module is the only path to a TTS authorization token. It lives outside
the coordinator model: the model can *request* speech for a confirmation
event id, but only code here can turn a confirmed message into a token that
the TTS service accepts.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path

from shared.schemas import ALLOWED_CONFIRMATION_METHODS, DEMO_AUTO_METHOD, AuditEvent, Message

def _load_secret() -> str:
    """Shared between gateway and TTS: env var, else a local file both processes read."""
    if os.environ.get("UNSPOKEN_TTS_SECRET"):
        return os.environ["UNSPOKEN_TTS_SECRET"]
    path = Path(__file__).resolve().parents[1] / ".run" / "tts_secret"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_text().strip()


_SECRET = _load_secret()


class PolicyViolation(Exception):
    """Raised when something tries to speak text that was not explicitly confirmed."""


def secret() -> str:
    return _SECRET


def sign(message_id: str, text: str, key: str | None = None) -> str:
    payload = f"{message_id}\x1f{text}".encode()
    return hmac.new((key or _SECRET).encode(), payload, hashlib.sha256).hexdigest()


def phrase_preload_token(preload_id: str, text: str) -> str:
    """A TTS token to pre-render one of the fixed phrases (message id `preload-…`). Rendering is not speaking:
    the audio is played only when the gateway marks a message spoken, after authorize_speech passed for
    exactly that text."""
    return sign(f"preload-{preload_id}", text)


def verify_token(message_id: str, text: str, token: str, key: str | None = None) -> bool:
    return hmac.compare_digest(sign(message_id, text, key), token)


def authorize_speech(message: Message, audit_log: list[AuditEvent], confirmation_event_id: str, demo_auto: bool = False) -> str:
    """Return a TTS token for exactly `message.confirmed_text`, or raise PolicyViolation.

    demo_auto: the gateway's demo auto-speak mode. Only the gateway passes it, the audit records
    the message as spoken automatically (not by the person), and every other check still applies."""
    if message.status != "confirmed":
        raise PolicyViolation(f"message status is {message.status!r}, not 'confirmed'")
    if demo_auto:
        auto = [e for e in audit_log if e.id == confirmation_event_id and e.kind == "auto_confirmed" and e.message_id == message.id]
        if message.confirmation_method != DEMO_AUTO_METHOD or not auto or auto[0].data.get("text") != message.confirmed_text:
            raise PolicyViolation("auto-speak needs its own audited auto_confirmed event for this exact text")
        return sign(message.id, message.confirmed_text)
    if message.confirmation_method not in ALLOWED_CONFIRMATION_METHODS:
        raise PolicyViolation(f"confirmation method {message.confirmation_method!r} is not an explicit user action")
    if not message.confirmed_text:
        raise PolicyViolation("no confirmed text")
    if confirmation_event_id != message.confirmation_event_id:
        raise PolicyViolation("confirmation event does not belong to this message")

    confirm_events = [
        e for e in audit_log
        if e.id == confirmation_event_id and e.kind == "user_confirmed" and e.message_id == message.id
    ]
    if not confirm_events:
        raise PolicyViolation("confirmation event is not in the audit log")
    if confirm_events[0].actor not in ("user", "operator"):
        raise PolicyViolation("confirmation was not made by a person")
    if confirm_events[0].data.get("text") != message.confirmed_text:
        raise PolicyViolation("confirmed text changed after confirmation")

    # Recorded EEG is a benchmark replay, not new live consent: an operator
    # confirms that the labelled replay result is read out.
    if message.source_mode == "prerecorded" and not message.confirmed_text.startswith("Recorded EEG replay"):
        raise PolicyViolation("recorded data must be spoken with its replay label")

    return sign(message.id, message.confirmed_text)
