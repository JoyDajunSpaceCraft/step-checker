# MCP integration

Use the official BrowseComp-Plus MCP as the retrieval server and this package
as a separate audit server. The agent runner, not the language model, should
invoke the audit server after every retrieval result:

1. `start_trace(task_uid, query)` before generation;
2. call the official `search` or `get_document` tool;
3. call `audit_search` with the reasoning text, query, and exact tool output;
4. persist the returned typed decision for evaluation/reward;
5. call `audit_answer` before accepting the final response;
6. call `get_audit_trace` to save a replayable record.

This observer arrangement preserves the official retrieval-tool schema and
prevents the policy from skipping checker calls. An optional active-feedback
condition may insert the audit result into the next model observation; the
passive reward-only condition must not do so.

The default bridge deliberately implements only deterministic session logging
and exact-query redundancy. Semantic output is `not_evaluated` until a real
claim-decomposition and local/joint-evidence backend is injected. It must not
be reported as a complete StepGap checker.

Run over stdio:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. \
  /ocean/projects/med250011p/yji3/venvs/browsecomp-plus-mcp/bin/python \
  -m checker_agent.browsecomp_plus.mcp_bridge
```

Or expose a local HTTP MCP endpoint:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. \
  /ocean/projects/med250011p/yji3/venvs/browsecomp-plus-mcp/bin/python \
  -m checker_agent.browsecomp_plus.mcp_bridge \
  --transport streamable-http --port 8011
```

The MCP environment is deliberately separate from the RL environment because
the official server pins FastMCP 2.9.2 / MCP 1.9.4 / Pydantic 2.9.2, while the
existing RL stack requires newer MCP and Pydantic packages.
