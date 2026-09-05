#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from checker_agent.infoseek.adapter import convert_rft_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.trajectories, args.steps, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    with args.input.open(encoding="utf-8") as source, \
         args.trajectories.open("w", encoding="utf-8") as trajectories, \
         args.steps.open("w", encoding="utf-8") as steps:
        for fallback_index, line in enumerate(source):
            raw = json.loads(line)
            source_index = int(raw.get("rft_index", fallback_index))
            trajectory = convert_rft_record(raw, source_index)
            trajectories.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            stats["trajectories"] += 1
            stats["complete"] += int(trajectory["parse"]["complete"])
            for step in trajectory["steps"]:
                steps.write(json.dumps({"trace_uid": trajectory["trace_uid"], **step}, ensure_ascii=False) + "\n")
                stats["steps"] += 1
                stats[f"steps_{step['step_type']}"] += 1
    summary = dict(stats)
    summary["all_complete"] = stats["complete"] == stats["trajectories"]
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
