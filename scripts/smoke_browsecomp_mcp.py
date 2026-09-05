#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json

from fastmcp import Client

from checker_agent.browsecomp_plus.mcp_bridge import build_mcp


def tool_data(result):
    if hasattr(result, "data"):
        return result.data
    if isinstance(result, list) and result and hasattr(result[0], "text"):
        return json.loads(result[0].text)
    raise TypeError(f"Unsupported FastMCP tool result: {type(result).__name__}")


async def main() -> None:
    server = build_mcp()
    async with Client(server) as client:
        started = await client.call_tool("start_trace", {"task_uid": "bcp::smoke", "query": "smoke query"})
        start_data = tool_data(started)
        trace_uid = start_data["trace_uid"]
        observation = json.dumps([{"docid": "smoke-doc", "score": 1.0, "snippet": "A smoke-test document."}])
        audited = await client.call_tool(
            "audit_search",
            {"trace_uid": trace_uid, "thought": "Inspect the smoke document.", "query": "smoke query", "observation_json": observation},
        )
        finished = await client.call_tool("audit_answer", {"trace_uid": trace_uid, "response": "Smoke answer"})
        exported = await client.call_tool("get_audit_trace", {"trace_uid": trace_uid})
    print(json.dumps({
        "tools_called": ["start_trace", "audit_search", "audit_answer", "get_audit_trace"],
        "semantic_status": tool_data(audited)["semantic"]["status"],
        "finish_status": tool_data(finished)["status"],
        "stored_steps": len(tool_data(exported)["steps"]),
    }))


if __name__ == "__main__":
    asyncio.run(main())
