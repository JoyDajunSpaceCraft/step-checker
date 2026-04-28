#!/usr/bin/env python3
"""
build_sft_data.py
=================
Converts v17 pipeline output (JSONL) into SFT training data for Qwen3-8B.

Strategy: Chain-of-Thought Distillation
  - Teacher: GPT-4.1-mini + NLI (v17)
  - Student: Qwen3-8B (to be fine-tuned)
  - Format: instruction-following with full reasoning chain as output

Input:  v17 pipeline output JSONL (has_gap, gap_type, tree_path, tree_steps)
Output: SFT JSONL with (instruction, response) pairs

Usage:
python build_sft_data/build_sft_data.py \
    --input  data/annotated/v17_n1000_combined.jsonl \
    --output data/sft/checker_train_v3.jsonl \
    --val_ratio 0.1 --seed 42
"""

import json
import random
import argparse
import os
from typing import List, Dict, Optional

# ===================== SYSTEM PROMPT =====================
SYSTEM_PROMPT = """\
You are a step-level evidence analyzer for multi-hop QA.

Your job is ONLY to:
1. Check if the retrieved evidence is about the correct entity/topic (entity alignment)
2. Find a verbatim quote from evidence that supports the step's claim
3. Detect if the step is an abstention (N/A / cannot determine)
4. Detect if the step targets the wrong entity/relation (alignment drift)

You do NOT decide if there is a gap — a separate NLI model handles that.

Output EXACTLY these tags in order, nothing else:

<is_off_target>true or false</is_off_target>
<drift_type>none OR entity_drift OR relation_drift OR scope_drift</drift_type>
<alignment_reasoning>brief explanation</alignment_reasoning>
<is_abstention>true or false</is_abstention>
<abstention_accurate>true or false (only meaningful if is_abstention=true)</abstention_accurate>
<entity_match>true or false</entity_match>
<entity_reasoning>brief explanation of whether evidence is about the right entity</entity_reasoning>
<found_quote>exact verbatim 5-20 word span from evidence, or empty if none</found_quote>
<quote_reasoning>brief explanation of quote search result</quote_reasoning>

Rules:
- is_off_target=true only if the step TEXT itself names wrong entity/relation
- Retrieval failure is NOT off-target; step query correctness is what matters
- entity_match=false if evidence document is about a different entity than the step targets
- found_quote must be exact text from evidence, not a paraphrase
- If step says N/A/unknown, set is_abstention=true"""


# ===================== INPUT BUILDER =====================
def build_input(question: str, step: Dict, prev_steps: List[Dict]) -> str:
    """Build the user-side instruction from a step record."""
    prev_text = ""
    for i, s in enumerate(prev_steps[-3:]):
        prev_text += f"  Step {s.get('step_id','?')} [{s.get('step_type','?')}]: {s.get('step_text','')[:100]}\n"
    if not prev_text:
        prev_text = "  (first step)\n"

    # Top-2 evidence snippets
    evidence_pool = step.get("evidence_pool", [])
    ev_lines = ""
    for i, e in enumerate(evidence_pool[:2]):
        title   = e.get("title", "")
        snippet = (e.get("snippet", "") or "")[:300]
        ev_lines += f"  [{i+1}] {title}: {snippet}\n"
    if not ev_lines:
        ev_lines = "  (no evidence retrieved)\n"

    step_type = step.get("step_type", "inference")
    step_text = step.get("step_text", "")

    return (
        f"Question: {question}\n\n"
        f"Previous steps:\n{prev_text}\n"
        f"Current step (Step {step.get('step_id','?')}) [{step_type}]:\n"
        f"  {step_text}\n\n"
        f"Evidence pool:\n{ev_lines}"
    )


# ===================== OUTPUT BUILDER =====================
def build_output(step: Dict) -> Optional[str]:
    """
    Build teacher output with ONLY LLM-responsible fields.
    NLI decisions (has_gap, gap_type, tree_path) are NOT included —
    those will be computed by DeBERTa NLI at inference time.

    LLM fields:
      Step -1: is_off_target, drift_type, alignment_reasoning
      Node  0: is_abstention, abstention_accurate
      Node  1: entity_match, entity_reasoning, found_quote, quote_reasoning
    """
    tree_path = step.get("tree_path", "")
    has_gap   = step.get("has_gap")
    ts        = step.get("tree_steps", {})

    # Skip errored or skipped steps
    if has_gap is None or tree_path in ("error", "skipped", ""):
        return None

    # ── Step -1: alignment ────────────────────────────────────────────────
    sm1 = ts.get("step_m1") or {}
    is_off_target = sm1.get("is_off_target", False)
    drift_type    = sm1.get("drift_type", "none") or "none"
    align_reason  = (sm1.get("alignment_reasoning") or "")[:120]
    if not align_reason:
        align_reason = ("Step targets wrong entity/relation." if is_off_target
                        else "Step targets the correct entity and relation.")

    # ── Node 0: abstention ────────────────────────────────────────────────
    s0 = ts.get("step_0") or {}
    is_abstention      = s0.get("is_abstention_step", False)
    abstention_accurate = s0.get("abstention_is_accurate", False)

    # ── Node 1: entity match + quote search ───────────────────────────────
    s1 = ts.get("step_1") or {}
    entity_match     = s1.get("entity_match", True)
    entity_reasoning = (s1.get("entity_match_reasoning") or "")[:120]
    found_quote      = (s1.get("evidence_quote") or "")[:150]
    quote_reasoning  = (s1.get("quote_search_reasoning") or "")[:120]

    if not entity_reasoning:
        entity_reasoning = ("Entity in evidence matches step target." if entity_match
                            else "Evidence is about a different entity than the step targets.")
    if not quote_reasoning:
        quote_reasoning = ("Found exact supporting span in evidence." if found_quote
                           else "No verbatim supporting span found in evidence.")

    output_lines = [
        f"<is_off_target>{'true' if is_off_target else 'false'}</is_off_target>",
        f"<drift_type>{drift_type}</drift_type>",
        f"<alignment_reasoning>{align_reason}</alignment_reasoning>",
        f"<is_abstention>{'true' if is_abstention else 'false'}</is_abstention>",
        f"<abstention_accurate>{'true' if abstention_accurate else 'false'}</abstention_accurate>",
        f"<entity_match>{'true' if entity_match else 'false'}</entity_match>",
        f"<entity_reasoning>{entity_reasoning}</entity_reasoning>",
        f"<found_quote>{found_quote}</found_quote>",
        f"<quote_reasoning>{quote_reasoning}</quote_reasoning>",
    ]
    return "\n".join(output_lines)


# ===================== DIFFICULTY CLASSIFIER =====================
def classify_difficulty(step: Dict, has_gap: bool) -> str:
    """
    Classify sample difficulty for weighted training.
    easy:   clear-cut decisions (entity_mismatch, entailment, inaccurate abstention)
    medium: moderately complex (NLI neutral, cross-verify, abstention)
    hard:   edge cases that the model tends to get wrong (FP/FN patterns)
    """
    path = step.get("tree_path", "")
    if ("entity_mismatch" in path or
        "NLI:entailment" in path or
        "inaccurate" in path or
        "off_target" in path):
        return "easy"
    elif ("NLI:neutral" in path or
          "cross_verified" in path or
          "accurate" in path or
          "NLI:contradiction" in path):
        return "medium"
    else:
        # not_found inference, not_cross_verified conclusion
        return "hard"


# ===================== MAIN =====================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       type=str, required=True,
                   help="v17 pipeline output JSONL")
    p.add_argument("--output",      type=str, required=True,
                   help="SFT training JSONL output path")
    p.add_argument("--val_ratio",   type=float, default=0.1,
                   help="Fraction of data for validation (default: 0.1)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--min_steps",   type=int, default=1,
                   help="Minimum steps per question to include")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    val_path  = args.output.replace(".jsonl", "_val.jsonl")
    stat_path = args.output.replace(".jsonl", "_stats.json")

    # Load records
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"[INFO] Loaded {len(records)} records from {args.input}")

    # Build SFT samples
    samples = []
    skipped = 0
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    gap_dist = {"gap": 0, "no_gap": 0}

    for rec in records:
        question = rec.get("question", "")
        steps    = rec.get("steps", [])

        if len(steps) < args.min_steps:
            continue

        for i, step in enumerate(steps):
            prev = steps[:i]
            instruction = build_input(question, step, prev)
            response    = build_output(step)

            if response is None:
                skipped += 1
                continue

            has_gap = step.get("has_gap", False)
            diff    = classify_difficulty(step, has_gap)
            difficulty_counts[diff] += 1
            gap_dist["gap" if has_gap else "no_gap"] += 1

            samples.append({
                "system":     SYSTEM_PROMPT,
                "instruction": instruction,
                "response":   response,
                "metadata": {
                    "question_id": rec.get("question_id", ""),
                    "dataset":     rec.get("dataset", ""),
                    "step_id":     step.get("step_id"),
                    "step_type":   step.get("step_type"),
                    "has_gap":     has_gap,
                    "gap_type":    step.get("gap_type"),
                    "tree_path":   step.get("tree_path"),
                    "difficulty":  diff,
                }
            })

    print(f"[INFO] Built {len(samples)} samples ({skipped} skipped)")
    print(f"  Gap distribution: {gap_dist}")
    print(f"  Difficulty: {difficulty_counts}")

    # Train/val split
    random.seed(args.seed)
    random.shuffle(samples)
    n_val   = int(len(samples) * args.val_ratio)
    val_set = samples[:n_val]
    trn_set = samples[n_val:]

    # Save train
    with open(args.output, "w", encoding="utf-8") as f:
        for s in trn_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[INFO] Train: {len(trn_set)} samples → {args.output}")

    # Save val
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[INFO] Val:   {len(val_set)} samples → {val_path}")

    # Save stats
    stats = {
        "total": len(samples),
        "train": len(trn_set),
        "val":   len(val_set),
        "skipped": skipped,
        "gap_distribution": gap_dist,
        "difficulty": difficulty_counts,
        "records_processed": len(records),
    }
    with open(stat_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Stats → {stat_path}")
    print(f"\n{'='*50}")
    print(f"Ready for SFT: {len(trn_set)} train / {len(val_set)} val samples")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()