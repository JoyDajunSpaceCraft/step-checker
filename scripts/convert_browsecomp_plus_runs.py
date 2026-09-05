#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from checker_agent.browsecomp_plus.adapter import convert_official_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official BrowseComp-Plus runs to StepGap-Agent v1")
    parser.add_argument("--input", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--benchmark-version", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    trajectory_path = Path(args.trajectories)
    observation_path = Path(args.observations)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    observation_path.parent.mkdir(parents=True, exist_ok=True)

    store = {}
    count = 0
    with open(args.input, encoding="utf-8") as source, trajectory_path.open("w", encoding="utf-8") as out:
        for line in source:
            if not line.strip():
                continue
            trajectory, _ = convert_official_record(
                json.loads(line), source_run=args.source_run, model=args.model,
                retriever=args.retriever, benchmark_version=args.benchmark_version,
                seed=args.seed, observation_store=store,
            )
            json.dump(trajectory.to_dict(), out, ensure_ascii=False)
            out.write("\n")
            count += 1

    with observation_path.open("w", encoding="utf-8") as out:
        for uid in sorted(store):
            json.dump(store[uid].to_dict(), out, ensure_ascii=False)
            out.write("\n")

    print(json.dumps({"trajectories": count, "unique_observations": len(store)}))


if __name__ == "__main__":
    main()
