#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt_string(value: str, password: str) -> str:
    encrypted = base64.b64decode(value)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8")


def decrypt_tree(value: Any, password: str) -> Any:
    if isinstance(value, str):
        return decrypt_string(value, password)
    if isinstance(value, list):
        return [decrypt_tree(item, password) for item in value]
    if isinstance(value, dict):
        return {key: decrypt_tree(item, password) for key, item in value.items()}
    return value


def write_tasks(query_dir: Path, agent_output: Path, gold_output: Path, canary: str) -> dict:
    files = sorted((query_dir / "data").glob("test-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No test parquet shards under {query_dir / 'data'}")
    agent_output.parent.mkdir(parents=True, exist_ok=True)
    gold_output.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = set()
    count = 0
    with agent_output.open("w", encoding="utf-8") as agent_file, gold_output.open("w", encoding="utf-8") as gold_file:
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    query_id = str(row["query_id"])
                    if query_id in seen_ids:
                        raise ValueError(f"Duplicate query_id: {query_id}")
                    seen_ids.add(query_id)
                    query = decrypt_string(row["query"], canary)
                    answer = decrypt_string(row["answer"], canary)

                    def docids(field: str):
                        return [decrypt_string(str(doc["docid"]), canary) for doc in (row.get(field) or [])]

                    json.dump({"task_uid": f"bcp::{query_id}", "query_id": query_id, "query": query}, agent_file, ensure_ascii=False)
                    agent_file.write("\n")
                    json.dump(
                        {
                            "task_uid": f"bcp::{query_id}",
                            "query_id": query_id,
                            "answer": answer,
                            "gold_docids": docids("gold_docs"),
                            "evidence_docids": docids("evidence_docs"),
                            "negative_docids": docids("negative_docs"),
                        },
                        gold_file,
                        ensure_ascii=False,
                    )
                    gold_file.write("\n")
                    count += 1
    return {"queries": count, "agent_fields": ["task_uid", "query_id", "query"], "gold_is_separate": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create label-separated BrowseComp-Plus task files")
    parser.add_argument("--query-dir", required=True, type=Path)
    parser.add_argument("--agent-output", required=True, type=Path)
    parser.add_argument("--gold-output", required=True, type=Path)
    parser.add_argument("--canary", required=True)
    args = parser.parse_args()
    print(json.dumps(write_tasks(args.query_dir, args.agent_output, args.gold_output, args.canary)))


if __name__ == "__main__":
    main()
