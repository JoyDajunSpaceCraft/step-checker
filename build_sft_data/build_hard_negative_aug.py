#!/usr/bin/env python3
"""


python build_sft_data/build_hard_negative_aug.py \
    --gt      ablation/benchmark_step_level_gt.json \
    --q25     data/annotated/qwen2_5_v4_output.jsonl \
    --teacher data/annotated/teacher_v17_output.jsonl \
    --val     data/sft/checker_train_v3_val.jsonl \
    --output  data/sft/checker_train_v3_hard_neg.jsonl \
    --difficulty hard


    
build_hard_negative_aug.py
==========================
从 Qwen2.5-7B v4 的 FN 案例中构建 hard negative 训练数据。

Qwen2.5-7B v4 的主要漏检模式（47 FN cases）：

  23 cases: 1(found)->inference->no_gap
    → student 找到了 quote 就直接判 no_gap
    → 但 quote 其实是关于 wrong entity 的
    → 需要教会 student: found_quote ≠ correct entity ≠ no gap

   9 cases: 1(found)->2A(NLI:entailment)->no_gap  
    → NLI 算出 entailment，但人类说有 gap
    → 属于 NLI 本身的局限（semantic leap）
    → 训练数据里明确标注这类有 gap 的例子

   8 cases: 0(abstention)->accurate->no_gap
    → student 接受了 N/A 作为 grounded abstention
    → 但 evidence 其实有信息、model 没有利用
    → 需要更多"看起来是 grounded abstention 但实际是错的"的例子

   7 cases: 1(not_found)->2B(cross_verified)->no_gap
    → cross-verify 误触，prior evidence 不够可靠
    → 需要更多 cross-verify 失败的例子

用法:
    python build_hard_negative_aug.py \
        --gt     benchmark_step_level_gt.json \
        --q25    qwen2_5n82__v17__checker-gpt-4_1-mini__nli-cross-4.jsonl \
        --teacher n82__v15__checker-gpt-4_1-mini__nli-cross.jsonl \
        --val    checker_train_v3_val.jsonl \
        --output checker_train_v3_hard_neg.jsonl \
        --include_teacher_fn   包含 teacher 也漏检的
        --difficulty hard       全部标为 hard
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


# ── System prompt（和原始 SFT 数据保持一致）────────────────────────────────
SYSTEM_PROMPT = """You are a step-level evidence analyzer for multi-hop QA.

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
"""


# ── Build instruction ──────────────────────────────────────────────────────

def format_evidence_pool(evidence_pool: list, max_items: int = 4) -> str:
    lines = []
    for i, ev in enumerate(evidence_pool[:max_items]):
        title = ev.get("title", "").strip()
        snippet = (ev.get("snippet") or "").strip()[:400]
        if title:
            lines.append(f"  [{i+1}] \"{title}\"\n{snippet}")
        elif snippet:
            lines.append(f"  [{i+1}] {snippet}")
    return "\n".join(lines) if lines else "  (no evidence retrieved)"


def build_instruction(question: str, step: dict, prev_steps: list) -> str:
    # Previous steps context (last 3)
    if prev_steps:
        prev_text = "\n".join(
            f"  Step {s.get('step_id',i)} [{s.get('step_type','inference')}]: "
            f"{s.get('step_text','')[:100]}"
            for i, s in enumerate(prev_steps[-3:])
        )
    else:
        prev_text = "  (first step)"

    step_type = step.get("step_type", "inference")
    step_id   = step.get("step_id", 0)
    step_text = step.get("step_text", "").strip()
    ev_pool   = step.get("evidence_pool", [])

    return (
        f"Question: {question}\n\n"
        f"Previous steps:\n{prev_text}\n\n"
        f"Current step (Step {step_id}) [{step_type}]:\n"
        f"  {step_text}\n\n"
        f"Evidence pool:\n{format_evidence_pool(ev_pool)}"
    )


# ── Build response ─────────────────────────────────────────────────────────

def build_response(
    is_off_target: bool,
    drift_type: str,
    alignment_reasoning: str,
    is_abstention: bool,
    abstention_accurate: bool,
    entity_match: bool,
    entity_reasoning: str,
    found_quote: str,
    quote_reasoning: str,
) -> str:
    return (
        f"<is_off_target>{'true' if is_off_target else 'false'}</is_off_target>\n"
        f"<drift_type>{drift_type}</drift_type>\n"
        f"<alignment_reasoning>{alignment_reasoning}</alignment_reasoning>\n"
        f"<is_abstention>{'true' if is_abstention else 'false'}</is_abstention>\n"
        f"<abstention_accurate>{'true' if abstention_accurate else 'false'}</abstention_accurate>\n"
        f"<entity_match>{'true' if entity_match else 'false'}</entity_match>\n"
        f"<entity_reasoning>{entity_reasoning}</entity_reasoning>\n"
        f"<found_quote>{found_quote}</found_quote>\n"
        f"<quote_reasoning>{quote_reasoning}</quote_reasoning>"
    )


# ── Infer correct response from tree_steps and human label ──────────────────

def infer_response_from_step(
    step: dict,
    human_gap_type: Optional[str],
    teacher_step: Optional[dict],
) -> dict:
    """
    Given what we know (student tree_steps, teacher tree_steps, human label),
    construct the correct response the student SHOULD have produced.
    
    Strategy per FN pattern:
    
    Pattern A: 1(found)->inference->no_gap  (student found quote but entity was wrong)
      Correct response: entity_match=False (or found_quote=empty to prevent no_gap shortcut)
    
    Pattern B: 1(found)->2A(NLI:entailment)->no_gap (NLI entailment but human says gap)
      Correct response: entity_match=True, found_quote=empty
      (force student to NOT find supporting quote → then gap path kicks in)
    
    Pattern C: 0(abstention)->accurate->no_gap (accepted N/A but evidence had info)
      Correct response: is_abstention=True, abstention_accurate=False
    
    Pattern D: 1(not_found)->2B(cross_verified)->no_gap (cross-verify was wrong)
      Correct response: entity_match=True, found_quote=empty
      (cross-verify being wrong is handled by NLI, not the LLM response)
    """
    path = step.get("tree_path", "")
    ts   = step.get("tree_steps", {})
    s1   = ts.get("step_1", {})
    s0   = ts.get("step_0", {})
    sm1  = ts.get("step_m1", {})
    
    # Default values from student's actual tree_steps
    is_off_target = sm1.get("is_off_target", False)
    drift_type    = sm1.get("drift_type", "none") or "none"
    align_reason  = sm1.get("alignment_reasoning", "No alignment issue detected.") or "No alignment issue detected."
    
    is_abstention    = s0.get("is_abstention_step", False)
    abstention_acc   = s0.get("abstention_is_accurate", False)
    entity_match     = s1.get("entity_match", True)
    entity_reason    = s1.get("entity_match_reasoning", "") or ""
    found_quote_val  = s1.get("evidence_quote", "") or ""
    quote_reason     = s1.get("quote_search_reasoning", "") or ""
    
    # ── Pattern A: found quote but it's about wrong entity or supports wrong claim ──
    if "found->inference->no_gap" in path:
        # Student found a quote and auto-assumed no_gap for inference step
        # The fix: the quote is either from wrong entity, or doesn't actually support
        # We teach: entity_match=False OR found_quote="" (so it doesn't short-circuit)
        
        # Check if teacher said entity_mismatch
        teacher_path = (teacher_step or {}).get("tree_path", "")
        if "entity_mismatch" in teacher_path or "off_target" in teacher_path:
            # Teacher caught the entity issue - teach student to do same
            entity_match  = False
            entity_reason = (
                f"Evidence is about a different entity than the step targets. "
                f"The step's query targets a specific entity but retrieved document "
                f"is about a different entity. Quote from wrong document does not count."
            )
            found_quote_val = ""
            quote_reason    = "No valid quote found because evidence is about wrong entity."
        else:
            # NLI issue - quote exists but doesn't truly support, teach found_quote=""
            # so node 2B cross-verify runs instead of auto-no_gap
            found_quote_val = ""
            quote_reason    = (
                "Although a passage was found, it does not directly verify "
                "the specific claim in this step. No reliable verbatim quote available."
            )
    
    # ── Pattern C: accepted N/A as accurate but it wasn't ──
    elif "abstention->accurate->no_gap" in path:
        is_abstention  = True
        abstention_acc = False  # KEY: flip to False - abstention was NOT accurate
        align_reason   = align_reason or "Step targets the correct entity and relation."
    
    # ── Pattern B/D: NLI entailment or cross-verified but human says gap ──
    elif "NLI:entailment" in path or "cross_verified" in path:
        # Force found_quote to empty so NLI doesn't auto-pass
        found_quote_val = ""
        quote_reason    = (
            "No clear verbatim quote directly supports the specific claim in this step. "
            "The retrieved passage is related but does not constitute direct evidence."
        )
    
    return {
        "is_off_target":        is_off_target,
        "drift_type":           drift_type,
        "alignment_reasoning":  align_reason,
        "is_abstention":        is_abstention,
        "abstention_accurate":  abstention_acc,
        "entity_match":         entity_match,
        "entity_reasoning":     entity_reason,
        "found_quote":          found_quote_val,
        "quote_reasoning":      quote_reason,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Build hard negative augmentation data")
    p.add_argument("--gt",      required=True, help="benchmark_step_level_gt.json")
    p.add_argument("--q25",     required=True, help="Qwen2.5 student output JSONL")
    p.add_argument("--teacher", required=True, help="Teacher output JSONL")
    p.add_argument("--val",     required=True, help="Original SFT val JSONL (for format reference)")
    p.add_argument("--output",  required=True, help="Output JSONL path")
    p.add_argument("--include_teacher_fn", action="store_true",
                   help="Include cases where teacher ALSO misses (harder negatives)")
    p.add_argument("--difficulty", default="hard", choices=["hard","medium","auto"])
    args = p.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    gt_data = json.load(open(args.gt))
    gt_by_qid = {r["question_id"]: r for r in gt_data}

    q25_by_qid = {}; q25_step = {}
    for r in [json.loads(l) for l in open(args.q25)]:
        q25_by_qid[r["question_id"]] = r
        for s in r["steps"]: q25_step[(r["question_id"], s["step_id"])] = s

    teacher_step = {}
    for r in [json.loads(l) for l in open(args.teacher)]:
        for s in r["steps"]: teacher_step[(r["question_id"], s["step_id"])] = s

    human_dict = {}
    for r in gt_data:
        for s in r["steps"]: human_dict[(r["question_id"], s["step_id"])] = s

    # ── Collect FN cases ───────────────────────────────────────────────────
    fn_examples = []
    skipped = 0

    for key in set(human_dict) & set(q25_step):
        h_gap = human_dict[key]["human_has_gap"]
        q_gap = q25_step[key]["has_gap"]

        if not (h_gap and not q_gap):   # only FN cases
            continue

        qid, sid = key
        if qid not in gt_by_qid or qid not in q25_by_qid:
            skipped += 1
            continue

        r = gt_by_qid[qid]
        h_s = human_dict[key]
        q_s = q25_step[key]
        t_s = teacher_step.get(key, {})

        teacher_also_fn = not t_s.get("has_gap", False)
        if teacher_also_fn and not args.include_teacher_fn:
            # Skip cases where teacher also fails - less useful for student
            # (teacher-derived labels are wrong here anyway)
            pass  # still include, just note it

        # Get all steps for this question to build context
        all_steps_q25 = q25_by_qid[qid]["steps"]
        prev_steps    = [s for s in all_steps_q25 if s["step_id"] < sid]

        # The current step needs evidence_pool from q25 output
        current_step = q_s.copy()

        # Build correct response
        resp_fields = infer_response_from_step(
            step=current_step,
            human_gap_type=h_s.get("human_gap_type"),
            teacher_step=t_s if not teacher_also_fn else None,
        )

        instruction = build_instruction(r["question"], current_step, prev_steps)
        response    = build_response(**resp_fields)

        difficulty = args.difficulty
        if args.difficulty == "auto":
            # Harder if teacher also missed
            difficulty = "hard" if teacher_also_fn else "medium"

        example = {
            "system":      SYSTEM_PROMPT,
            "instruction": instruction,
            "response":    response,
            "metadata": {
                "question_id":     qid,
                "dataset":         r.get("dataset", "unknown"),
                "step_id":         sid,
                "step_type":       h_s.get("step_type", "inference"),
                "has_gap":         True,   # ground truth
                "gap_type":        h_s.get("human_gap_type"),
                "tree_path":       q_s.get("tree_path", ""),
                "difficulty":      difficulty,
                "aug_type":        "hard_negative_fn",
                "teacher_also_fn": teacher_also_fn,
                "q25_path":        q_s.get("tree_path", ""),
            }
        }
        fn_examples.append(example)

    # ── Save ───────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in fn_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ── Stats ──────────────────────────────────────────────────────────────
    from collections import Counter
    path_dist  = Counter(ex["metadata"]["q25_path"] for ex in fn_examples)
    type_dist  = Counter(ex["metadata"]["gap_type"] for ex in fn_examples)
    stype_dist = Counter(ex["metadata"]["step_type"] for ex in fn_examples)
    teacher_fn = sum(1 for ex in fn_examples if ex["metadata"]["teacher_also_fn"])

    print(f"\n{'='*55}")
    print(f"Hard Negative Augmentation — Summary")
    print(f"{'='*55}")
    print(f"Total examples generated : {len(fn_examples)}")
    print(f"Skipped (missing data)   : {skipped}")
    print(f"Teacher also missed      : {teacher_fn}/{len(fn_examples)}")
    print(f"\nStep type breakdown:")
    for k, v in stype_dist.most_common(): print(f"  {k}: {v}")
    print(f"\nHuman gap type breakdown:")
    for k, v in type_dist.most_common():  print(f"  {k}: {v}")
    print(f"\nStudent FN path breakdown:")
    for k, v in path_dist.most_common():  print(f"  {v:2d}  {k}")
    print(f"\nOutput: {args.output}")
    print(f"\nNext step:")
    print(f"  cat checker_train_v3.jsonl {args.output} > checker_train_v4.jsonl")
    print(f"  # Then retrain Qwen2.5-7B on checker_train_v4.jsonl")


if __name__ == "__main__":
    main()
