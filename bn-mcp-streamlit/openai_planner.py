"""LLM → Beyond News MCP tool + arguments (OpenAI or Anthropic Claude).

Uses structured tool/function calls when possible (closer to Cursor / Claude Desktop MCP),
with JSON-in-text fallback for models that do not support tools well.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from openai import OpenAI

ALLOWED_TOOLS = frozenset(
    {
        "check_health",
        "explore",
        "list_fields",
        "links_aggregations",
        "search_links",
        "search_links_post",
        "search_documents",
        "search_documents_post",
    }
)

TOOL_ID = "beyond_news_mcp_plan"

_SYSTEM_CORE = """You choose exactly one Beyond News MCP tool and its arguments (same job as MCP in Cursor or Claude Desktop).

Rules:
- Call the tool `{tool_id}` once per user request. Put the real MCP tool name in field `tool` and all API parameters in `arguments`.
- Never put api_key in arguments (MCP uses HTTP x-api-key).
- Prefer **search_links** for web pages / news; **search_documents** for PDFs, agendas, minutes, attachments.
- **search_links** requires **q** (string). Use rich keywords / boolean style from the explore reference. Omit **search_in** for broad "find all" unless the user restricts title vs body.
- **search_documents** / **search_documents_post** match the MCP tools: required **q** (same NewsCatcher-style string as links: AND/OR/NOT, parentheses, phrases). Use **page** and **page_size** (not `size`). Filters: **sources**, **search_in**, **dynamic_search**, **theme**, **document_type** (not `doc_type`), **from_**, **to_**, **url_contains**, **parent_url**, **link_id**, **fields**, **sort_by**, **sort_order**. There is no **keywords** or **keyword_operator** — put terms in **q**.
- **content_type** only if it appears in the content_type list in context (exact string).
- **dynamic_search** on search_links (GET) must be a **string** of compact JSON, not a nested object.
- Omit **fields** on search_links unless the user wants a slimmer response; without **fields**, the API returns full index rows (best for raw JSON). Use **fields** only to limit payload size.
- Follow the live **explore** payload in the user message for names, limits, and examples.

The `{tool_id}` schema enforces a valid tool enum; arguments must match that tool.""".format(
    tool_id=TOOL_ID
)


def _openai_json_only_model(model: str) -> bool:
    """Some reasoning models handle tools or JSON mode poorly — use plain JSON text."""
    m = (model or "").strip().lower()
    if re.match(r"^o\d", m):
        return True
    if m.startswith("gpt-5"):
        return True
    return False


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t)
    if m:
        return m.group(1).strip()
    return t


def _normalize_documents_args(out: dict[str, Any]) -> dict[str, Any]:
    """MCP search_documents* uses q + page_size (+ document_type, from_, to_, sources), not keywords/size."""
    if "q" not in out and "keywords" in out:
        kw = out.pop("keywords")
        out["q"] = kw if isinstance(kw, str) else str(kw)
    if "page_size" not in out and "size" in out:
        out["page_size"] = out.pop("size")
    if "sources" not in out and "domains" in out:
        out["sources"] = out.pop("domains")
    if "document_type" not in out and "doc_type" in out:
        out["document_type"] = out.pop("doc_type")
    if "from_" not in out and "date_from" in out:
        out["from_"] = out.pop("date_from")
    if "to_" not in out and "date_to" in out:
        out["to_"] = out.pop("date_to")
    for stale in ("keyword_operator", "exclude_keywords"):
        out.pop(stale, None)
    if "dynamic_search" in out:
        ds = out["dynamic_search"]
        if isinstance(ds, dict):
            out["dynamic_search"] = json.dumps(ds, separators=(",", ":"), ensure_ascii=False)
        elif ds == "":
            del out["dynamic_search"]
    return out


def normalize_mcp_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in dict(arguments).items() if k != "api_key" and v is not None}
    if tool == "search_links" and "dynamic_search" in out:
        ds = out["dynamic_search"]
        if isinstance(ds, dict):
            out["dynamic_search"] = json.dumps(ds, separators=(",", ":"), ensure_ascii=False)
        elif ds == "":
            del out["dynamic_search"]
    if tool in ("search_documents", "search_documents_post"):
        out = _normalize_documents_args(out)
    return out


def _finalize_plan(tool: Any, arguments: Any, raw_for_log: str) -> tuple[str, dict[str, Any], str]:
    if not isinstance(tool, str) or tool not in ALLOWED_TOOLS:
        raise ValueError(f"Invalid or disallowed tool: {tool!r}")
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a JSON object")
    arguments = normalize_mcp_arguments(tool, arguments)
    return tool, arguments, raw_for_log


def _parse_plan_response(raw_text: str) -> tuple[str, dict[str, Any], str]:
    raw = _strip_json_fence(raw_text.strip() or "{}")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Model returned non-object JSON")
    tool = data.get("tool")
    arguments = data.get("arguments")
    return _finalize_plan(tool, arguments, raw)


def _anthropic_tool_def() -> dict[str, Any]:
    return {
        "name": TOOL_ID,
        "description": "Select one Beyond News MCP tool and pass its arguments object exactly as the API expects.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "enum": sorted(ALLOWED_TOOLS),
                    "description": "Name of the MCP tool to run",
                },
                "arguments": {
                    "type": "object",
                    "description": "Parameters for that tool only (see explore in user message).",
                },
            },
            "required": ["tool", "arguments"],
        },
    }


def _openai_tool_def() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_ID,
            "description": "Select one Beyond News MCP tool and its arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": sorted(ALLOWED_TOOLS),
                        "description": "MCP tool name",
                    },
                    "arguments": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Tool-specific parameters",
                    },
                },
                "required": ["tool", "arguments"],
            },
        },
    }


def plan_mcp_call_openai(
    *,
    api_key: str,
    model: str,
    user_message: str,
    mcp_context: str,
) -> tuple[str, dict[str, Any], str]:
    client = OpenAI(api_key=api_key)
    user_content = (
        f"{mcp_context}\n\n---\nUser request:\n{user_message}\n\n"
        f"Call function `{TOOL_ID}` with the MCP tool name and arguments."
    )

    if _openai_json_only_model(model):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_CORE},
                {"role": "user", "content": user_content + '\nOutput JSON only: {"tool":"...","arguments":{...}}'},
            ],
            response_format={"type": "json_object"},
        )
        raw_text = resp.choices[0].message.content or "{}"
        return _parse_plan_response(raw_text)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_CORE},
                {"role": "user", "content": user_content},
            ],
            tools=[_openai_tool_def()],
            tool_choice={"type": "function", "function": {"name": TOOL_ID}},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_CORE},
                {
                    "role": "user",
                    "content": user_content
                    + '\nOutput JSON only: {"tool":"<mcp_tool_name>","arguments":{...}}',
                },
            ],
            response_format={"type": "json_object"},
        )
        raw_text = resp.choices[0].message.content or "{}"
        return _parse_plan_response(raw_text)
    msg = resp.choices[0].message
    tcalls = getattr(msg, "tool_calls", None) or []
    if tcalls:
        raw_args = tcalls[0].function.arguments or "{}"
        data = json.loads(raw_args)
        if isinstance(data, dict):
            return _finalize_plan(
                data.get("tool"),
                data.get("arguments"),
                json.dumps(data, ensure_ascii=False),
            )
    # Fallback if API returns text only
    if msg.content:
        return _parse_plan_response(msg.content)
    return _parse_plan_response("{}")


def plan_mcp_call_anthropic(
    *,
    api_key: str,
    model: str,
    user_message: str,
    mcp_context: str,
) -> tuple[str, dict[str, Any], str]:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    user_content = (
        f"{mcp_context}\n\n---\nUser request:\n{user_message}\n\n"
        f"Use the `{TOOL_ID}` tool (required) with the correct MCP tool and arguments."
    )
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=8192,
            system=_SYSTEM_CORE,
            tools=[_anthropic_tool_def()],
            tool_choice={"type": "tool", "name": TOOL_ID},
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception:
        msg = client.messages.create(
            model=model,
            max_tokens=8192,
            system=_SYSTEM_CORE
            + "\n\nIf tools are unavailable, reply with JSON only: {\"tool\":\"...\",\"arguments\":{...}}",
            messages=[
                {
                    "role": "user",
                    "content": user_content
                    + '\n\nOutput JSON only: {"tool":"<mcp_tool_name>","arguments":{...}}',
                }
            ],
        )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == TOOL_ID:
            inp = getattr(block, "input", None)
            if isinstance(inp, dict):
                raw = json.dumps(inp, ensure_ascii=False)
                return _finalize_plan(inp.get("tool"), inp.get("arguments"), raw)
    # Fallback: text JSON (older models / refusals)
    parts: list[str] = []
    for block in msg.content:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            parts.append(block.text)
    return _parse_plan_response("".join(parts) if parts else "{}")


Provider = Literal["openai", "anthropic"]


def plan_mcp_call(
    *,
    provider: Provider,
    api_key: str,
    model: str,
    user_message: str,
    mcp_context: str,
) -> tuple[str, dict[str, Any], str]:
    """
    Returns (tool_name, arguments_dict, raw_json_string).
    """
    p = str(provider or "").strip().lower()
    if p not in ("openai", "anthropic"):
        raise ValueError(f"Unknown LLM provider: {provider!r} (use openai or anthropic)")
    if p == "anthropic":
        return plan_mcp_call_anthropic(
            api_key=api_key,
            model=model,
            user_message=user_message,
            mcp_context=mcp_context,
        )
    return plan_mcp_call_openai(
        api_key=api_key,
        model=model,
        user_message=user_message,
        mcp_context=mcp_context,
    )
