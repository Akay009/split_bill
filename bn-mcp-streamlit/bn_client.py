"""Beyond News MCP over Streamable HTTP (`x-api-key` header)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastmcp.client import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StreamableHttpTransport


def _drop_empty(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None and v != ""}


def _tool_result_to_obj(res: CallToolResult) -> Any:
    if res.is_error:
        msg = ""
        for c in res.content or []:
            if hasattr(c, "text") and c.text:
                msg = c.text
                break
        raise ValueError(msg or "MCP tool returned an error")

    if res.structured_content and isinstance(res.structured_content, dict):
        inner = res.structured_content.get("result")
        if inner is not None:
            if isinstance(inner, str):
                if inner.strip().startswith("Error:"):
                    raise ValueError(inner.strip())
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    return inner
            return inner

    if res.data is not None:
        if isinstance(res.data, str):
            if res.data.strip().startswith("Error:"):
                raise ValueError(res.data.strip())
            try:
                return json.loads(res.data)
            except json.JSONDecodeError:
                return res.data
        return res.data

    for c in res.content or []:
        if hasattr(c, "text") and c.text:
            t = c.text.strip()
            if t.startswith("Error:"):
                raise ValueError(t)
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                return {"text": t}
    return {}


class BNClient:
    def __init__(self, mcp_url: str, api_key: str) -> None:
        self.mcp_url = mcp_url.rstrip("/")
        key = (api_key or "").strip()
        headers: dict[str, str] = {}
        if key:
            headers["x-api-key"] = key
        transport = StreamableHttpTransport(url=self.mcp_url, headers=headers)
        self._client = Client(transport)

    def close(self) -> None:
        return

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        args = _drop_empty(dict(arguments))
        tr = await self._client.call_tool(name, args)
        return _tool_result_to_obj(tr)

    async def _one_shot(self, name: str, arguments: dict[str, Any]) -> Any:
        async with self._client:
            return await self._call_tool(name, arguments)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke any MCP tool; `arguments` are the tool parameters (same names as in Cursor / server)."""
        return asyncio.run(self._one_shot(name, dict(arguments or {})))
