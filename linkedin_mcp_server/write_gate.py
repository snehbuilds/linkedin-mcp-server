"""Global write gate, account safety, rate limiting, and audit logging.

Protects the account and user against unintended mutations or prompt-injection-driven
writes. All write tools are gated behind ``MCP_LINKEDIN_WRITE_ENABLED=true``, enforce
daily action quotas, apply human-like jitter delays, and write an immutable audit log.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import random
from typing import Any

from linkedin_mcp_server.config.loaders import _env
from linkedin_mcp_server.core.exceptions import (
    RateLimitError,
    WritesDisabledError,
)

logger = logging.getLogger(__name__)

WRITE_ENABLED_ENV = "MCP_LINKEDIN_WRITE_ENABLED"
AUDIT_LOG_ENV = "LINKEDIN_WRITE_AUDIT_LOG_FILE"
DEFAULT_AUDIT_DIR = "~/.linkedin-mcp"
DEFAULT_AUDIT_FILE = "write-audit.log"
DEFAULT_COUNTS_FILE = "write-counts.json"

# Per-action daily limits to protect against automated restrictions.
# AGENTS.md rules: invitations max 30/day, 10s apart; general tool calls max 10/min, 100/day.
DEFAULT_DAILY_LIMITS: dict[str, int] = {
    "connect_with_person": 30,
    "remove_connections": 10,
    "publish_post": 20,
    "comment_on_post": 50,
    "reply_to_comment": 50,
    "react_to_post": 100,
    "send_message": 100,
    "reply_to_conversation": 100,
    "archive_conversation": 100,
    "save_job": 100,
    "dismiss_job": 100,
    "apply_profile_changes": 20,
}

WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "connect_with_person",
        "remove_connections",
        "publish_post",
        "comment_on_post",
        "reply_to_comment",
        "react_to_post",
        "send_message",
        "reply_to_conversation",
        "archive_conversation",
        "save_job",
        "dismiss_job",
        "apply_profile_changes",
    }
)


def is_write_enabled() -> bool:
    """Return True if MCP_LINKEDIN_WRITE_ENABLED is explicitly enabled."""
    val = _env(WRITE_ENABLED_ENV) or os.getenv(WRITE_ENABLED_ENV) or ""
    return val.strip().lower() in {"1", "true", "yes", "on"}


def get_audit_dir() -> Path:
    """Return the base directory where audit logs and counters reside."""
    override = os.getenv("LINKEDIN_PROFILE_EDITS_DIR") or os.getenv(
        "LINKEDIN_AUDIT_DIR"
    )
    if override:
        path = Path(override).expanduser().resolve()
    else:
        path = Path(DEFAULT_AUDIT_DIR).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_audit_log_path() -> Path:
    """Return the audit log file path."""
    override = os.getenv(AUDIT_LOG_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return get_audit_dir() / DEFAULT_AUDIT_FILE


def get_counts_path() -> Path:
    """Return the daily write counters file path."""
    return get_audit_dir() / DEFAULT_COUNTS_FILE


def _load_daily_counts() -> tuple[str, dict[str, int]]:
    """Load today's write counts from disk, rolling over on a new UTC day."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    counts_file = get_counts_path()
    if not counts_file.exists():
        return today, {}

    try:
        data = json.loads(counts_file.read_text(encoding="utf-8"))
        if data.get("date") == today and isinstance(data.get("counts"), dict):
            return today, data["counts"]
    except Exception as exc:
        logger.debug("Could not read write counts file: %s", exc)
    return today, {}


def _save_daily_counts(today: str, counts: dict[str, int]) -> None:
    """Save daily counts to disk."""
    counts_file = get_counts_path()
    try:
        counts_file.write_text(
            json.dumps({"date": today, "counts": counts}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("Could not save write counts file: %s", exc)


def check_daily_limit(action_name: str, increment: int = 1) -> None:
    """Verify that action_name has not exceeded its daily quota."""
    limit = DEFAULT_DAILY_LIMITS.get(action_name)
    if limit is None:
        return

    today, counts = _load_daily_counts()
    current = counts.get(action_name, 0)
    if current + increment > limit:
        raise RateLimitError(
            f"Daily safety limit for '{action_name}' reached ({current}/{limit}). "
            "Action blocked to safeguard your LinkedIn account against restrictions."
        )


def record_write_audit(
    tool_name: str,
    target: str | Sequence[str] | None = None,
    status: str = "SUCCESS",
    details: dict[str, Any] | None = None,
    count_increment: int = 1,
) -> None:
    """Record an entry in the persistent audit log and increment today's count."""
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    record = {
        "timestamp": now_iso,
        "pid": os.getpid(),
        "tool": tool_name,
        "target": target,
        "status": status,
        "details": details or {},
    }

    try:
        log_path = get_audit_log_path()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Failed to append to write audit log: %s", exc)

    if status == "SUCCESS" and count_increment > 0:
        try:
            today, counts = _load_daily_counts()
            counts[tool_name] = counts.get(tool_name, 0) + count_increment
            _save_daily_counts(today, counts)
        except Exception as exc:
            logger.debug("Failed to increment daily write count: %s", exc)


async def apply_write_jitter(
    min_seconds: float = 2.0, max_seconds: float = 4.0
) -> float:
    """Introduce human-cadence delay before executing write actions.

    Skipped under unit tests unless explicitly enabled with LINKEDIN_ENFORCE_TEST_JITTER.
    """
    if os.getenv("PYTEST_CURRENT_TEST") and not os.getenv(
        "LINKEDIN_ENFORCE_TEST_JITTER"
    ):
        return 0.0

    delay = random.uniform(min_seconds, max_seconds)
    logger.info("Applying safety jitter: pausing %.2fs before write action", delay)
    await asyncio.sleep(delay)
    return delay


async def enforce_write_gate(
    tool_name: str,
    target: str | Sequence[str] | None = None,
    details: dict[str, Any] | None = None,
    min_jitter: float = 2.0,
    max_jitter: float = 4.0,
    count_increment: int = 1,
) -> None:
    """Enforce the global write gate, daily quota, and pacing jitter for a write tool."""
    if not is_write_enabled():
        record_write_audit(
            tool_name,
            target=target,
            status="BLOCKED_WRITES_DISABLED",
            details=details,
            count_increment=0,
        )
        raise WritesDisabledError(tool_name)

    check_daily_limit(tool_name, increment=count_increment)
    await apply_write_jitter(min_jitter, max_jitter)
