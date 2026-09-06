# StepGap-Agent

This folder extends StepGap from search-interleaved QA traces to tool-use agent
traces. It is intentionally isolated from `scripts/run_pipeline_v17.py`: none of
the original QA checker, reward code, prompts, or paper-reproduction settings is
modified.

## What problem does it solve?

The QA taxonomy checks whether retrieved evidence supports a reasoning claim.
An agent can fail even when every claim is factually supported: it may use the
wrong tool, violate a policy precondition, repeat a tool call, or announce
success before changing the environment. StepGap-Agent therefore evaluates two
independent axes:

| Axis | Labels | Relation checked |
|---|---|---|
| Cognitive | `contradicted_claim`, `irrelevant_evidence`, `missing_bridge` | observation -> thought |
| Normative | `policy_violation`, `wrong_tool`, `redundant_action`, `premature_commit` | policy/state -> action |

The axes are not mutually exclusive. A thought can contradict an observation
and its action can violate policy in the same step. The checker preserves both
decisions and derives `primary_label` only for consumers that require one scalar
reward category.

## Pipeline

1. Normalize tool names, arguments, record identifiers, and state.
2. Verbalize a structured observation into a minimal proposition.
3. Run cognitive NLI against the thought; prior observations can rescue a local
   missing-bridge decision.
4. Check tool existence, allowed record status, and object targeting.
5. Check structured policy preconditions, including explicit confirmation.
6. Detect repeated read actions deterministically.
7. On final-response steps, compare actual and requested final state.
8. Return both axes, the selected primary label, repair actions, and a readable
   `pipeline_path`.

Policy checking is deliberately not implemented as one raw NLI call. NLI
`NEUTRAL` can mean an irrelevant clause, missing state, or model uncertainty; it
does not prove that an action is permitted. Policy clauses are first represented
as structured preconditions, then the trace state is checked against them.

## Input

Each JSONL line is one step:

```json
{
  "uid": "trace12-step4",
  "thought": "Order W123 is delivered and eligible for exchange.",
  "action": {
    "tool": "exchange_delivered_order_items",
    "arguments": {"order_id": "W123"},
    "is_write": true
  },
  "observation": {"order_id": "W123", "status": "delivered"},
  "user_turns": ["I want to exchange W123."],
  "known_state": {"order_id": "W123", "status": "delivered"},
  "expected_final_state": {},
  "prior_steps": []
}
```

Keep raw environment fields in a separate source artifact if an adapter drops
them. `prior_steps` must contain the original observations and actions needed for
cross-step and redundancy checks.

## Run the dependency-free smoke test

From the repository root:

```bash
python3 -m checker_agent.cli \
  --input checker_agent/examples/demo_trace.jsonl \
  --output /tmp/stepgap_agent_demo.jsonl \
  --tool-specs checker_agent/examples/tool_specs.json \
  --policy-rules checker_agent/examples/policy_rules.json

python3 -m checker_agent.evaluate --input /tmp/stepgap_agent_demo.jsonl
python3 -m unittest tests/test_agent_checker.py
```

The default backend exists for schema, routing, and integration tests only. Its
`HeuristicNLI` scores must **not** be reported as model results.

## Run with local Qwen and a real NLI model

The repository already has a local Qwen model at the following path in the
current environment:

```text
/ocean/projects/med250011p/yji3/models/Qwen3-8B
```

Qwen is used only to verbalize JSON observations; it is not allowed to decide
the final cognitive or policy label. Use a three-way NLI checkpoint for the
deterministic entailment stage:

```bash
python3 -m checker_agent.cli \
  --input YOUR_AGENT_TRACE.jsonl \
  --output results/qwen_nli_predictions.jsonl \
  --tool-specs checker_agent/examples/tool_specs.json \
  --policy-rules checker_agent/examples/policy_rules.json \
  --qwen-model /ocean/projects/med250011p/yji3/models/Qwen3-8B \
  --nli-model cross-encoder/nli-deberta-v3-large
```

For a reproducible experiment, download the NLI checkpoint once, record its
exact Hugging Face revision, and then pass the local snapshot path. Do not mix
model revisions across ablations. Qwen generation is greedy (`do_sample=False`).

### Why use Qwen only for verbalization?

Cross-encoder NLI models are trained on sentence pairs, not arbitrary JSON.
Qwen converts a relevant field into a short sentence such as `Order W123 has
status delivered.` NLI then checks that proposition against the thought. This
preserves the StepGap division of labor: an LLM handles open-ended reading while
NLI anchors the categorical judgment.

Before paper-scale use, add a faithfulness guard that verifies every value in the
Qwen sentence occurs in the source observation. The current deterministic
verbalizer is safer and should remain a required ablation.

## What the output means

```json
{
  "cognitive": {"label": "contradicted_claim", "has_gap": true},
  "normative": {"label": "policy_violation", "has_gap": true},
  "primary_label": "policy_violation",
  "pipeline_path": [
    "normalize:success",
    "cognitive:contradicted_claim",
    "wrong_tool:pass",
    "policy:policy_violation"
  ],
  "metadata": {"multi_axis": true}
}
```

Use the axis labels for evaluation and error analysis. Use `primary_label` only
when integrating a single typed process reward. The current priority ordering is
policy violation > wrong tool > premature commit > CC > IE > MB > redundancy.
It is an engineering default, not an empirically validated reward hierarchy.

## Minimum scientific evaluation before RL

Do not start GRPO first. Freeze 100-150 steps from one pinned agent benchmark
version and annotate:

1. whether the step contains a failure;
2. whether CC/IE/MB alone covers it;
3. cognitive label, if applicable;
4. normative label, if applicable;
5. the exact policy clause and state fields supporting the decision.

Report cognitive-taxonomy coverage, residual normative distribution, axis
co-occurrence, per-label precision/recall/F1, binary balanced accuracy, latency,
and automatic-vs-model decision rate. Split random and targeted samples when
oversampling rare labels. Preserve raw annotator labels rather than consensus
alone.

The first useful result is a boundary measurement, not a high score. For
example, finding that CC/IE/MB covers 60% of agent failures while 40% requires
action-level types would directly motivate the second axis.

## Current status and limitations

- Implemented: schemas, two-axis output, structured policy rules, wrong-tool,
  redundancy, termination checks, optional Qwen verbalization, optional NLI,
  CLI, summary script, and 20 targeted cases.
- Not implemented: benchmark-specific adapter, automatic free-text policy-rule
  extraction, annotation UI, calibrated reward weights, or agent GRPO.
- The deterministic checks assume environment state is available and trustworthy.
- Confirmation matching is currently pattern-based and must be audited per domain.
- The optional Qwen verbalizer needs a source-faithfulness validator before its
  outputs can support paper claims.

## Recommended next implementation

1. Choose and pin one benchmark/domain and implement `adapters/<benchmark>.py`.
2. Convert 20 real trajectories and manually audit all normalized fields.
3. Run deterministic verbalization and Qwen verbalization side by side.
4. Add the NLI checkpoint revision and measure latency/error types.
5. Annotate 100-150 steps and report taxonomy coverage.
6. Only after checker quality is established, adapt
   `verl/verl/tools/checker_tool.py` to consume both axes for RL.
