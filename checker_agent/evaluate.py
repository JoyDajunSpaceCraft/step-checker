from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize StepGap-Agent predictions")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    multi = sum(bool(row.get("metadata", {}).get("multi_axis")) for row in rows)
    report = {
        "n_steps": len(rows),
        "cognitive_distribution": Counter(row["cognitive"]["label"] for row in rows),
        "normative_distribution": Counter(row["normative"]["label"] for row in rows),
        "primary_distribution": Counter(row["primary_label"] for row in rows),
        "multi_axis_count": multi,
        "multi_axis_rate": multi / len(rows) if rows else 0.0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
