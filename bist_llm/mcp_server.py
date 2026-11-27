"""MCP server exposing the deterministic tools (brief §22) over stdio.

The tools, argument validation, 2026-09-10 look-ahead block and compact outputs are exactly those of tools/registry.py,
which the harness also calls in-process. build_evidence_bundle returns the evidence bundle of a logged analysis.
Run from bist_llm/:  python mcp_server.py        (needs: pip install "mcp>=2.3,<3")
MCP client config:   {"command": "python", "args": ["mcp_server.py"], "cwd": "<repo>/bist_llm"}"""
import asyncio, json, sys
from pathlib import Path

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.registry import TOOL_SPECS, call_tool   # noqa: E402

EVIDENCE_TOOL = {"name": "build_evidence_bundle",
                 "description": "Full observed context and committed evidence set of a logged analysis (decision log id).",
                 "parameters": {"type": "object", "properties": {"analysis_id": {"type": "string"}}, "required": ["analysis_id"]}}
SPECS = TOOL_SPECS + [EVIDENCE_TOOL]


def evidence_bundle(analysis_id):
    logs = sorted((ROOT / "logs").glob(f"*-{analysis_id}.json")) if str(analysis_id).isalnum() else []
    if not logs:
        return {"tool": "build_evidence_bundle", "error": "UNKNOWN_ENTITY", "message": f"no decision log with id {analysis_id}"}
    t = json.loads(logs[-1].read_text())
    return {"tool": "build_evidence_bundle", "analysis_id": analysis_id, "question": t.get("question"),
            "states": t.get("states"), "evidence_bundle": t.get("evidence_bundle")}


async def list_tools(ctx, params):
    return types.ListToolsResult(tools=[types.Tool(name=s["name"], description=s["description"], inputSchema=s["parameters"])
                                        for s in SPECS])


async def call(ctx, params):
    args = params.arguments or {}
    res = evidence_bundle(args.get("analysis_id")) if params.name == "build_evidence_bundle" else call_tool(params.name, args)
    return types.CallToolResult(content=[types.TextContent(text=json.dumps(res, ensure_ascii=False))],
                                structuredContent=res, isError="error" in res)


server = Server("bist-analyst-tools", version="1.0", on_list_tools=list_tools, on_call_tool=call,
                instructions="Deterministic BIST research tools. Data ends 2026-09-10; later dates are refused (LOOKAHEAD_BLOCKED).")


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
