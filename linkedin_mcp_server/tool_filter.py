"""Tool allowlist and grouping for MCP token budget and context optimization.

Allows running the server with only a desired subset of tools (e.g. read-only, messaging,
jobs, posts, or profile) to avoid context bloat and improve LLM tool selection.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

TOOL_FILTER_ENV = "LINKEDIN_ENABLED_TOOLS"

TOOL_GROUPS: dict[str, frozenset[str]] = {
    "read": frozenset(
        {
            "get_person_profile",
            "get_company_profile",
            "get_company_employees",
            "get_company_posts",
            "get_feed",
            "get_inbox",
            "get_conversation",
            "search_conversations",
            "get_job_details",
            "get_job_apply_url",
            "get_easy_apply_form",
            "get_saved_jobs",
            "search_jobs",
            "search_people",
            "search_companies",
            "search_posts",
            "get_sidebar_profiles",
            "get_post_comments",
            "list_my_connections",
            "get_my_profile",
            "get_my_editable_profile",
            "get_my_experience",
            "get_my_skills",
            "describe_profile_form",
            "prepare_post",
            "propose_profile_changes",
            "preview_profile_changes",
            "rehearse_profile_changes",
            "discard_profile_changes",
        }
    ),
    "messaging": frozenset(
        {
            "get_inbox",
            "get_conversation",
            "search_conversations",
            "send_message",
            "reply_to_conversation",
            "archive_conversation",
        }
    ),
    "jobs": frozenset(
        {
            "search_jobs",
            "get_job_details",
            "get_job_apply_url",
            "get_easy_apply_form",
            "get_saved_jobs",
            "save_job",
            "dismiss_job",
        }
    ),
    "posts": frozenset(
        {
            "get_feed",
            "get_company_posts",
            "search_posts",
            "get_post_comments",
            "prepare_post",
            "publish_post",
            "react_to_post",
            "comment_on_post",
            "reply_to_comment",
        }
    ),
    "profile": frozenset(
        {
            "get_my_profile",
            "get_person_profile",
            "get_sidebar_profiles",
            "get_my_editable_profile",
            "get_my_experience",
            "get_my_skills",
            "propose_profile_changes",
            "preview_profile_changes",
            "rehearse_profile_changes",
            "describe_profile_form",
            "apply_profile_changes",
            "discard_profile_changes",
            "list_my_connections",
            "remove_connections",
            "connect_with_person",
        }
    ),
    "company": frozenset(
        {
            "get_company_profile",
            "get_company_employees",
            "get_company_posts",
            "search_companies",
        }
    ),
    "write": frozenset(
        {
            "send_message",
            "reply_to_conversation",
            "archive_conversation",
            "save_job",
            "dismiss_job",
            "connect_with_person",
            "remove_connections",
            "publish_post",
            "react_to_post",
            "comment_on_post",
            "reply_to_comment",
            "apply_profile_changes",
        }
    ),
}

# Aliases for convenience
GROUP_ALIASES: dict[str, str] = {
    "read_only": "read",
    "readonly": "read",
    "reading": "read",
    "writes": "write",
    "message": "messaging",
    "messages": "messaging",
    "job": "jobs",
    "post": "posts",
    "profiles": "profile",
    "companies": "company",
}

ALWAYS_ENABLED_TOOLS: frozenset[str] = frozenset({"close_session"})


def resolve_enabled_tools(
    spec: str | Sequence[str] | None = None,
) -> frozenset[str] | None:
    """Resolve an allowlist spec into an explicit set of tool names.

    Returns None if all tools should be enabled (unrestricted mode).
    Spec may be:
    - None or empty string: unrestricted (all tools)
    - "all": unrestricted
    - Comma-separated groups or tool names (e.g., "read,messaging" or "search_jobs,get_job_details")
    """
    if spec is None:
        raw_env = os.getenv(TOOL_FILTER_ENV, "").strip()
        if not raw_env:
            return None
        spec = raw_env

    if isinstance(spec, str):
        tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
    else:
        tokens = [str(t).strip().lower() for t in spec if str(t).strip()]

    if not tokens or "all" in tokens:
        return None

    resolved: set[str] = set(ALWAYS_ENABLED_TOOLS)
    for token in tokens:
        canonical_group = GROUP_ALIASES.get(token, token)
        if canonical_group in TOOL_GROUPS:
            resolved.update(TOOL_GROUPS[canonical_group])
        else:
            resolved.add(token)

    return frozenset(resolved)


def is_tool_enabled(tool_name: str, enabled_set: frozenset[str] | None = None) -> bool:
    """Return True if tool_name should be registered on the MCP server."""
    if tool_name in ALWAYS_ENABLED_TOOLS:
        return True
    if enabled_set is None:
        return True
    return tool_name in enabled_set


def filter_server_tools(mcp: Any, enabled_set: frozenset[str] | None = None) -> None:
    """Remove any registered tools that are not in the enabled set."""
    if enabled_set is None:
        return
    for provider in getattr(mcp, "providers", []):
        components = getattr(provider, "_components", None)
        if isinstance(components, dict):
            tool_names = [
                comp.name
                for comp in list(components.values())
                if hasattr(comp, "name") and comp.name not in enabled_set
            ]
            for name in tool_names:
                try:
                    provider.remove_tool(name)
                    logger.info("Tool '%s' excluded by tool allowlist filter", name)
                except Exception as exc:
                    logger.debug("Could not remove tool '%s': %s", name, exc)
