#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from checker_agent.infoseek.semantic import GPT54Analyzer, evaluate_analysis
from checker_agent.models import TransformersNLI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit-trajectories", type=int, default=5)
    parser.add_argument("--start-trajectory", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--nli-model", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    analyzer = GPT54Analyzer(args.model)
    nli = TransformersNLI(args.nli_model, args.device)
    counts: Counter[str] = Counter()
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink:
        for trace_index, line in enumerate(source):
            if trace_index < args.start_trajectory:
                continue
            if trace_index >= args.limit_trajectories:
                break
            trajectory = json.loads(line)
            accumulated: list[str] = []
            question = trajectory["steps"][0]["user_turns"][0] if trajectory["steps"] else ""
            for step in trajectory["steps"]:
                if step.get("observation"):
                    accumulated.append(str(step["observation"]))
                evidence = str(step.get("observation") or "") if step["step_type"] == "search" else "\n\n".join(accumulated[-3:])
                analysis, usage = analyzer.analyze(question=question, step=step, evidence=evidence)
                decision = evaluate_analysis(analysis, nli)
                record = {
                    "trace_uid": trajectory["trace_uid"], "step_uid": step["uid"],
                    "step_index": step["step_index"], "step_type": step["step_type"],
                    "model": args.model, "nli_model": args.nli_model,
                    **decision, "usage": usage,
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                counts["steps"] += 1
                counts[f"label::{decision['label']}"] += 1
                counts["input_tokens"] += usage["input_tokens"]
                counts["output_tokens"] += usage["output_tokens"]
            counts["trajectories"] += 1
    args.summary.write_text(json.dumps(dict(counts), indent=2), encoding="utf-8")
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
