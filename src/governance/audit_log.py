"""
Privacy / audit logging.

Records what happened during a pipeline run — not sensitive data itself.
Each entry captures: timestamp, event, and non-sensitive context (record
counts, file paths, pass/fail status). This creates a trace of how data
moved through the pipeline, supporting accountability and reproducibility,
and gives us real evidence to show rather than a decorative log.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "data" / "processed" / "_audit_log.jsonl"


def log_event(event: str, **context) -> dict:
    """Append one structured event to the audit log (JSON Lines format).
    context kwargs must not include raw sensitive data — only counts,
    paths, statuses, and similar non-sensitive metadata."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **context,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_log() -> list:
    """Read all events currently in the audit log."""
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clear_log() -> None:
    """Remove the audit log (used to reset state between full pipeline runs
    for demonstration purposes; not part of normal operation)."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()
