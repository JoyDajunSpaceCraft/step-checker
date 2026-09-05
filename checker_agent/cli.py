from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checker import AgentStepGapChecker
from .models import QwenObservationVerbalizer, TransformersNLI
from .schema import AgentStep, PolicyRule, ToolSpec


def load_json(path: str | None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StepGap-Agent on JSONL traces")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tool-specs")
    parser.add_argument("--policy-rules")
    parser.add_argument("--qwen-model", help="Optional local Qwen observation verbalizer")
    parser.add_argument("--nli-model", help="Optional local/Hugging Face three-way NLI model")
    args = parser.parse_args()

    specs = [ToolSpec.from_dict(item) for item in load_json(args.tool_specs)]
    rules = [PolicyRule.from_dict(item) for item in load_json(args.policy_rules)]
    verbalizer = QwenObservationVerbalizer(args.qwen_model) if args.qwen_model else None
    nli = TransformersNLI(args.nli_model) if args.nli_model else None
    checker = AgentStepGapChecker(specs, rules, verbalizer, nli)

    with Path(args.input).open(encoding="utf-8") as source, Path(args.output).open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if line.strip():
                result = checker.check(AgentStep.from_dict(json.loads(line)))
                target.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
