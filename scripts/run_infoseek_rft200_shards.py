#!/usr/bin/env python3
"""Resumable two-GPU launcher for the InfoSeek RFT-200 semantic audit."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def valid(summary: Path, expected: int) -> bool:
    try:
        return json.loads(summary.read_text())["trajectories"] == expected
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def run_shard(args, start: int, end: int) -> tuple[int, int, str]:
    stem = args.out / "shards" / f"traj_{start:03d}_{end:03d}"
    pred, summary, log = stem.with_suffix(".predictions.jsonl"), stem.with_suffix(".summary.json"), stem.with_suffix(".log")
    if valid(summary, end - start):
        return start, end, "cached"
    env = os.environ.copy()
    env.update({"PYTHONPATH": ".", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    command = [
        args.python, "-u", "scripts/evaluate_infoseek_rft.py",
        "--input", str(args.input), "--output", str(pred), "--summary", str(summary),
        "--start-trajectory", str(start), "--limit-trajectories", str(end),
        "--model", args.model, "--nli-model", args.nli_model,
        "--device", str((start // args.shard_size) % args.workers),
    ]
    for attempt in range(1, args.retries + 1):
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\nATTEMPT {attempt}\n")
            result = subprocess.run(command, cwd=args.workspace, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode == 0 and valid(summary, end - start):
            return start, end, f"completed_attempt_{attempt}"
        time.sleep(min(30, 5 * attempt))
    return start, end, "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--nli-model", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--shard-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    args.input = args.input.resolve()
    args.out = args.out.resolve()
    (args.out / "shards").mkdir(parents=True, exist_ok=True)

    jobs = [(start, min(start + args.shard_size, args.count)) for start in range(0, args.count, args.shard_size)]
    statuses = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_shard, args, start, end): (start, end) for start, end in jobs}
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            print(status, flush=True)

    failures = [status for status in statuses if status[2] == "failed"]
    manifest = {"model": args.model, "nli_model": args.nli_model, "count": args.count,
                "shard_size": args.shard_size, "workers": args.workers, "statuses": sorted(statuses)}
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"Failed shards: {failures}")

    records = []
    for start, end in jobs:
        path = args.out / "shards" / f"traj_{start:03d}_{end:03d}.predictions.jsonl"
        records.extend(json.loads(line) for line in path.open(encoding="utf-8"))
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    labels = Counter(record["label"] for record in records)
    usage = Counter()
    for record in records:
        usage.update(record["usage"])
    final = {"trajectories": args.count, "steps": len(records), "labels": dict(labels), "usage": dict(usage)}
    (args.out / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
