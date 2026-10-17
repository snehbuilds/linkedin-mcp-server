"""Unit tests for the write gate, rate limits, audit logging, and tool allowlist filtering."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from fastmcp import FastMCP
from linkedin_mcp_server.core.exceptions import WritesDisabledError, RateLimitError
from linkedin_mcp_server.tool_filter import (
    filter_server_tools,
    resolve_enabled_tools,
)
from linkedin_mcp_server.tools.connection_removal import (
    register_connection_removal_tools,
)
from linkedin_mcp_server.write_gate import (
    DEFAULT_DAILY_LIMITS,
    apply_write_jitter,
    check_daily_limit,
    enforce_write_gate,
    is_write_enabled,
    record_write_audit,
)


def test_write_gate_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MCP_LINKEDIN_WRITE_ENABLED", raising=False)
    assert not is_write_enabled()


def test_write_gate_truthy_values(monkeypatch):
    for val in ("true", "1", "yes", "on", "True", "TRUE"):
        monkeypatch.setenv("MCP_LINKEDIN_WRITE_ENABLED", val)
        assert is_write_enabled()


def test_write_gate_falsy_values(monkeypatch):
    for val in ("false", "0", "no", "off", "invalid", ""):
        monkeypatch.setenv("MCP_LINKEDIN_WRITE_ENABLED", val)
        assert not is_write_enabled()


@pytest.mark.anyio
async def test_enforce_write_gate_raises_when_disabled(monkeypatch):
    monkeypatch.delenv("MCP_LINKEDIN_WRITE_ENABLED", raising=False)
    with pytest.raises(WritesDisabledError) as exc_info:
        await enforce_write_gate("send_message", target="testuser")
    assert "send_message" in str(exc_info.value)
    assert "MCP_LINKEDIN_WRITE_ENABLED=true" in str(exc_info.value)


@pytest.mark.anyio
async def test_enforce_write_gate_passes_when_enabled(monkeypatch):
    monkeypatch.setenv("MCP_LINKEDIN_WRITE_ENABLED", "true")
    # Should not raise
    await enforce_write_gate("send_message", target="testuser")


def test_daily_quotas_enforced(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_LINKEDIN_WRITE_ENABLED", "true")
    counts_file = tmp_path / "counts.json"
    monkeypatch.setattr(
        "linkedin_mcp_server.write_gate.get_counts_path",
        lambda: counts_file,
    )

    action = "connect_with_person"
    limit = DEFAULT_DAILY_LIMITS[action]

    # Under the limit
    for i in range(limit):
        check_daily_limit(action)
        record_write_audit(action, target=f"user_{i}")

    # Exceeding the limit must raise RateLimitError
    with pytest.raises(RateLimitError) as exc_info:
        check_daily_limit(action)
    assert f"safety limit for '{action}' reached" in str(exc_info.value)


def test_audit_log_persists_jsonl(monkeypatch, tmp_path):
    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setattr(
        "linkedin_mcp_server.write_gate.get_audit_log_path",
        lambda: audit_file,
    )

    record_write_audit("save_job", target="12345", details={"status": "saved"})
    record_write_audit(
        "remove_connections", target=["user1", "user2"], details={"count": 2}
    )

    lines = audit_file.read_text().strip().split("\n")
    assert len(lines) == 2

    entry1 = json.loads(lines[0])
    assert entry1["tool"] == "save_job"
    assert entry1["target"] == "12345"
    assert entry1["details"]["status"] == "saved"
    assert "timestamp" in entry1

    entry2 = json.loads(lines[1])
    assert entry2["tool"] == "remove_connections"
    assert entry2["target"] == ["user1", "user2"]


@pytest.mark.anyio
async def test_jitter_skips_in_pytest_by_default():
    # In pytest, apply_write_jitter returns immediately without sleeping
    import anyio

    t0 = anyio.current_time()
    await apply_write_jitter(min_seconds=5.0, max_seconds=10.0)
    elapsed = anyio.current_time() - t0
    assert elapsed < 1.0


def test_tool_filter_groups_and_aliases():
    # Test read group
    read_tools = resolve_enabled_tools("read")
    assert read_tools is not None
    assert "get_person_profile" in read_tools
    assert "send_message" not in read_tools
    assert "close_session" in read_tools

    # Test alias readonly -> read
    assert resolve_enabled_tools("readonly") == read_tools
    assert resolve_enabled_tools("read_only") == read_tools

    # Test messaging group
    messaging_tools = resolve_enabled_tools("messaging")
    assert messaging_tools is not None
    assert "send_message" in messaging_tools
    assert "get_inbox" in messaging_tools
    assert "get_person_profile" not in messaging_tools

    # Test all / empty / None
    assert resolve_enabled_tools(None) is None
    assert resolve_enabled_tools("") is None
    assert resolve_enabled_tools("all") is None

    # Test combined groups and individual tools
    combined = resolve_enabled_tools("jobs,react_to_post")
    assert combined is not None
    assert "search_jobs" in combined
    assert "react_to_post" in combined
    assert "send_message" not in combined


@pytest.mark.anyio
async def test_filter_server_tools_filters_unwanted_tools():
    mcp = FastMCP("test_filter")

    @mcp.tool()
    def get_feed() -> str:
        return "feed"

    @mcp.tool()
    def send_message() -> str:
        return "sent"

    @mcp.tool()
    def close_session() -> str:
        return "closed"

    enabled = frozenset({"get_feed", "close_session"})
    filter_server_tools(mcp, enabled)

    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert tool_names == {"get_feed", "close_session"}


@pytest.mark.anyio
async def test_remove_connections_hard_cap_3(monkeypatch):
    monkeypatch.setenv("MCP_LINKEDIN_WRITE_ENABLED", "true")
    mcp = FastMCP("test_removal")
    register_connection_removal_tools(mcp)

    tool = await mcp.get_tool("remove_connections")
    assert tool is not None

    mock_ctx = MagicMock()
    mock_ctx.report_progress = AsyncMock()

    # Calling with 4 profiles should refuse before acquiring browser
    tool_fn = getattr(tool, "fn")
    result = await tool_fn(
        usernames=["u1", "u2", "u3", "u4"],
        ctx=mock_ctx,
        confirm=True,
    )
    assert result["status"] == "refused"
    assert "at most 3" in result["message"]
