#!/usr/bin/env python3
"""Reconnoiter InfoSeek trees and RFT trajectories without exposing gold data.

Detailed/sample outputs belong in a git-ignored private directory.
This script uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TAGS = re.compile(r"<(\/)?(think|search|information|answer)>", re.I)


def rows(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            if line.strip():
                yield json.loads(line)


def walk(node: dict[str, Any], depth: int = 0) -> Iterable[tuple[int, dict[str, Any]]]:
    yield depth, node
    for child in node.get("children") or []:
        if isinstance(child, dict):
            yield from walk(child, depth + 1)


def title_from_href(href: str | None) -> str | None:
    if not href:
        return None
    match = re.search(r"/wiki/([^#?]+)", href)
    encoded = match.group(1) if match else href.split("#", 1)[0].split("?", 1)[0]
    return urllib.parse.unquote(encoded).replace("_", " ").strip() or None


def schema(value: Any, depth: int = 0) -> Any:
    """Return key/type structure, not benchmark text."""
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {key: schema(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return {"type": "list", "item": schema(value[0], depth + 1) if value else None}
    return type(value).__name__


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def scan_wiki_titles(wiki_jsonl: Path, needed: set[str]) -> set[str]:
    found: set[str] = set()
    with wiki_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = record.get("title") or record.get("name")
            if not title and record.get("contents"):
                title = record["contents"].splitlines()[0].strip().strip('"')
            if title in needed:
                found.add(title)
                if len(found) == len(needed):
                    break
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wiki-jsonl", type=Path)
    parser.add_argument("--pilot-max-vertices", type=int, default=4)
    parser.add_argument("--pilot-size", type=int, default=5000)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--rft-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = {
        "tree": args.data_dir / "InfoSeek.jsonl",
        "qa": args.data_dir / "InfoSeekQA.jsonl",
        "hard": args.data_dir / "InfoSeek-Hard-18K.jsonl",
        "rft": args.data_dir / "Trajectory-RFT-17K.jsonl",
        "eval": args.data_dir / "infoseek_eval.jsonl",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing InfoSeek files: {missing}")

    schemas = {name: schema(next(rows(path, 1))) for name, path in files.items()}
    (args.out / "schema_summary.json").write_text(
        json.dumps(schemas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    vertices_hist: Counter[int] = Counter()
    tree_depth_hist: Counter[int] = Counter()
    constraint_hist: Counter[int] = Counter()
    titles: set[str] = set()
    constraint_rows: list[dict[str, Any]] = []
    for record in rows(files["tree"]):
        root = record["root"]
        nodes = list(walk(root))
        constraints = [
            {"depth": depth, "entity": node.get("entity"), "claim": claim}
            for depth, node in nodes for claim in (node.get("claims") or [])
        ]
        vertices = int(record.get("vertices", len(nodes)))
        vertices_hist[vertices] += 1
        tree_depth_hist[max((depth for depth, _ in nodes), default=0)] += 1
        constraint_hist[len(constraints)] += 1
        for _, node in nodes:
            title = title_from_href(node.get("href"))
            if title:
                titles.add(title)
        constraint_rows.append({
            "question_id": root.get("id"), "question": root.get("question"),
            "vertices": vertices, "n_constraints": len(constraints), "constraints": constraints,
        })

    write_jsonl(args.out / "constraints_per_q.jsonl", constraint_rows)
    (args.out / "titles_needed.txt").write_text("\n".join(sorted(titles)), encoding="utf-8")
    with (args.out / "distribution_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "value", "count"])
        writer.writerows(("vertices", key, value) for key, value in sorted(vertices_hist.items()))
        writer.writerows(("constraints", key, value) for key, value in sorted(constraint_hist.items()))

    rng = random.Random(args.seed)
    shallow = [record for record in rows(files["hard"])
               if int(record.get("vertices", 10**9)) <= args.pilot_max_vertices]
    rng.shuffle(shallow)
    write_jsonl(args.out / "pilot_dev.jsonl", shallow[:args.dev_size])
    write_jsonl(args.out / "pilot_train.jsonl",
                shallow[args.dev_size:args.dev_size + args.pilot_size])

    tag_hist: Counter[str] = Counter()
    turn_hist: Counter[int] = Counter()
    rft_sample = []
    for index, record in enumerate(rows(files["rft"], args.rft_limit)):
        conversation = record.get("conversation") or []
        turn_hist[len(conversation)] += 1
        assistant = "\n".join(turn.get("content", "") for turn in conversation
                              if turn.get("role") == "assistant")
        for closing, tag in TAGS.findall(assistant):
            tag_hist[("/" if closing else "") + tag.lower()] += 1
        rft_sample.append({"rft_index": index, "conversation": conversation})
    write_jsonl(args.out / f"rft_sample_{len(rft_sample)}.jsonl", rft_sample)

    summary: dict[str, Any] = {
        "source_files": {name: str(path) for name, path in files.items()},
        "n_tree_questions": len(constraint_rows),
        "distinct_wikipedia_titles": len(titles),
        "vertices_histogram": dict(sorted(vertices_hist.items())),
        "tree_max_depth_histogram": dict(sorted(tree_depth_hist.items())),
        "constraints_histogram": dict(sorted(constraint_hist.items())),
        "shallow_candidates": len(shallow),
        "pilot_train": min(args.pilot_size, max(0, len(shallow) - args.dev_size)),
        "pilot_dev": min(args.dev_size, len(shallow)),
        "rft_records_scanned": len(rft_sample),
        "rft_turn_histogram": dict(sorted(turn_hist.items())),
        "rft_tag_histogram": dict(sorted(tag_hist.items())),
    }
    if args.wiki_jsonl:
        found = scan_wiki_titles(args.wiki_jsonl, titles)
        summary["wiki_title_hits"] = len(found)
        summary["wiki_title_coverage"] = len(found) / len(titles) if titles else 1.0
        (args.out / "titles_missing.txt").write_text("\n".join(sorted(titles - found)), encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
