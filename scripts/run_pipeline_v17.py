#!/usr/bin/env python3
"""
run_pipeline_v15.py
===================
Step-Gap Pipeline v15 — NLI-based checker redesign.

What changed from v14:
  Node 2A (quote supports claim):
    Old: GPT-4.1-mini judges "does quote support claim?" (over-triggers on partial support)
    New: NLI model — entailment/neutral/contradiction maps to no_gap/missing_bridge/unsupported_claim

  Node 2B (common knowledge → necessity + cross-verify):
    Old: GPT-4.1-mini decides "is this common knowledge?" (model-dependent, subjective)
    New: Two NLI tests:
         Test 1 — Necessity: does step_claim entail final_answer? neutral → not necessary → no_gap
         Test 2 — Cross-verify: does any prior step's evidence entail step_claim? yes → no_gap

  Node 3A + Node 4 (semantic leap):
    Old: GPT-4.1-mini flags "does conclusion require an unstated premise?" (over-triggers)
    New: Removed — handled implicitly by NLI entailment in Node 2A
         entailment=no_gap / neutral=missing_bridge / contradiction=unsupported_claim

LLM still handles (reasoning tasks requiring language understanding):
  Step -1: entity/relation alignment
  Node 0:  abstention detection
  Node 1:  entity consistency check + verbatim quote search
  Node 3B: conflict check (contradiction confirmation, feeds NLI label)

NLI model handles (deterministic logical inference):
  Node 2A: evidence ⊢ step_claim?
  Node 2B: step_claim ⊢ final_answer? (necessity)  /  prior_evidence ⊢ step_claim? (cross-verify)

Usage:

  # Full pipeline (Qwen + NLI checker)
  CUDA_VISIBLE_DEVICES=1 python scripts/run_pipeline_v17.py \
    --freeze_qwen ablation/v10_correct_for_ablation.jsonl \
    --benchmark ablation/benchmark_step_level_gt.json \
    --output data/annotated/v17_n82 \
    --nli_model cross-encoder/nli-deberta-v3-large \
    --nli_device 0 \
    --checker_model gpt-4.1-mini


    
  CUDA_VISIBLE_DEVICES=1 python run_pipeline_v15.py \
      --input data/processed/train.json \
      --output data/annotated/v15 \
      --qwen_model /path/to/Qwen2.5-7B-Instruct \
      --nli_model cross-encoder/nli-deberta-v3-large \
      --sample_total 100 --seed 42 \
      --use_dynamic_retrieval \
      --save_qwen_steps data/annotated/v15/qwen_steps_frozen.jsonl

  # Checker-only (frozen Qwen, no GPU for Qwen)
  CUDA_VISIBLE_DEVICES=1 python run_pipeline_v15.py \
      --freeze_qwen data/annotated/ablation/v10_correct_for_ablation.jsonl \
      --benchmark benchmark_step_level_gt.json \
      --output data/annotated/v15 \
      --nli_model cross-encoder/nli-deberta-v3-large

NLI model options (in order of preference):
  cross-encoder/nli-deberta-v3-large   Best accuracy, ~180MB, requires sentence-transformers
  cross-encoder/nli-deberta-v3-base    Faster, slightly lower accuracy
  typeform/distilbert-base-uncased-mnli  Fastest, lowest accuracy
"""
#!/usr/bin/env python3
"""
run_pipeline_v15.py
===================
Step-Gap Pipeline v15 — NLI-based checker redesign.

What changed from v14:
  Node 2A (quote supports claim):
    Old: GPT-4.1-mini judges "does quote support claim?" (over-triggers on partial support)
    New: NLI model — entailment/neutral/contradiction maps to no_gap/missing_bridge/unsupported_claim

  Node 2B (common knowledge → necessity + cross-verify):
    Old: GPT-4.1-mini decides "is this common knowledge?" (model-dependent, subjective)
    New: Two NLI tests:
         Test 1 — Necessity: does step_claim entail final_answer? neutral → not necessary → no_gap
         Test 2 — Cross-verify: does any prior step's evidence entail step_claim? yes → no_gap

  Node 3A + Node 4 (semantic leap):
    Old: GPT-4.1-mini flags "does conclusion require an unstated premise?" (over-triggers)
    New: Removed — handled implicitly by NLI entailment in Node 2A
         entailment=no_gap / neutral=missing_bridge / contradiction=unsupported_claim

LLM still handles (reasoning tasks requiring language understanding):
  Step -1: entity/relation alignment
  Node 0:  abstention detection
  Node 1:  entity consistency check + verbatim quote search
  Node 3B: conflict check (contradiction confirmation, feeds NLI label)

NLI model handles (deterministic logical inference):
  Node 2A: evidence ⊢ step_claim?
  Node 2B: step_claim ⊢ final_answer? (necessity)  /  prior_evidence ⊢ step_claim? (cross-verify)

Usage:
  # Full pipeline (Qwen + NLI checker)
  CUDA_VISIBLE_DEVICES=1 python run_pipeline_v15.py \
      --input data/processed/train.json \
      --output data/annotated/v15 \
      --qwen_model /path/to/Qwen2.5-7B-Instruct \
      --nli_model cross-encoder/nli-deberta-v3-large \
      --sample_total 100 --seed 42 \
      --use_dynamic_retrieval \
      --save_qwen_steps data/annotated/v15/qwen_steps_frozen.jsonl

  # Checker-only (frozen Qwen, no GPU for Qwen)
  CUDA_VISIBLE_DEVICES=1 python run_pipeline_v15.py \
      --freeze_qwen data/annotated/ablation/v10_correct_for_ablation.jsonl \
      --benchmark benchmark_step_level_gt.json \
      --output data/annotated/v15 \
      --nli_model cross-encoder/nli-deberta-v3-large

NLI model options (in order of preference):
  cross-encoder/nli-deberta-v3-large   Best accuracy, ~180MB, requires sentence-transformers
  cross-encoder/nli-deberta-v3-base    Faster, slightly lower accuracy
  typeform/distilbert-base-uncased-mnli  Fastest, lowest accuracy
"""

import os
import re
import json
import argparse
import random
import requests
import pandas as pd

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

DEBUG = True


def debug_print(msg: str, data: Any = None):
    if DEBUG:
        print(f"\n{'='*60}")
        print(f"[DEBUG] {msg}")
        if data is not None:
            content = data if isinstance(data, str) else json.dumps(data, indent=2, ensure_ascii=False)
            print(content[:2000])
        print("=" * 60)


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.replace("/", "-").replace(":", "-"))
    return re.sub(r"-+", "-", text).strip("-") or "unknown"


# ===================== NLI MODULE =====================
class NLIChecker:
    """
    NLI-based inference checker.
    Replaces both Node 2B (common knowledge) and Node 3A/4 (semantic leap)
    with deterministic NLI inference.

    Labels returned: 'entailment' / 'neutral' / 'contradiction'

    Mapping to gap decisions:
      Node 2A (quote vs claim):
        entailment   → no_gap
        neutral      → missing_bridge   (claim needs unstated bridging premise)
        contradiction→ unsupported_claim

      Node 2B Test 1 (claim vs final_answer — necessity):
        neutral      → claim not necessary for answer → no_gap (replaces CK)
        entailment   → claim IS necessary, proceed to cross-verify

      Node 2B Test 2 (prior_evidence vs claim — cross-verify):
        entailment   → claim supported by prior evidence → no_gap
        neutral/contr→ claim genuinely unsupported → insufficient_context
    """

    # NLI label thresholds (score from cross-encoder is a continuous value)
    ENTAILMENT_THRESHOLD = 0.5    # above this → entailment
    CONTRADICTION_THRESHOLD = 0.5  # above this → contradiction (for cross-encoder)

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large", device: int = 0):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.model_name,
                num_labels=3,
                device=f"cuda:{self.device}" if self.device >= 0 else "cpu",
            )
            # CrossEncoder NLI models output [contradiction, entailment, neutral]
            # Label order varies by model — we calibrate below
            self._label_order = self._detect_label_order()
            print(f"[INFO] NLI model loaded: {self.model_name} (device=cuda:{self.device})")
            print(f"[INFO]   Label order: {self._label_order}")
        except ImportError:
            print("[WARNING] sentence-transformers not installed. Falling back to transformers pipeline.")
            self._init_pipeline_fallback()
        except Exception as e:
            print(f"[ERROR] NLI model init failed: {e}")
            self.model = None

    def _detect_label_order(self) -> List[str]:
        """
        Detect label order by running a known entailment pair.
        Returns list like ['contradiction', 'entailment', 'neutral'].
        """
        try:
            test_scores = self.model.predict([("A cat is an animal.", "A cat is an animal.")])
            idx = int(test_scores[0].argmax())
            # Known entailment should score highest on entailment label
            # Most cross-encoder NLI models: [contradiction=0, entailment=1, neutral=2]
            # But some use: [entailment=0, neutral=1, contradiction=2]
            # We detect by checking which index is highest for a clear entailment
            label_candidates = [
                ['contradiction', 'entailment', 'neutral'],  # most common
                ['entailment', 'neutral', 'contradiction'],
                ['entailment', 'contradiction', 'neutral'],
            ]
            for order in label_candidates:
                if order[idx] == 'entailment':
                    return order
            return label_candidates[0]  # default
        except Exception:
            return ['contradiction', 'entailment', 'neutral']

    def _init_pipeline_fallback(self):
        """Fallback using transformers pipeline."""
        try:
            from transformers import pipeline
            self.model = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device if self.device >= 0 else -1,
            )
            self._label_order = None  # pipeline returns labeled scores
            self._use_pipeline = True
            print(f"[INFO] NLI pipeline loaded: {self.model_name}")
        except Exception as e:
            print(f"[ERROR] NLI pipeline fallback failed: {e}")
            self.model = None

    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Returns dict with scores for each label.
        {'entailment': 0.92, 'neutral': 0.06, 'contradiction': 0.02}
        """
        if self.model is None:
            # Fallback: conservative — always return neutral
            return {'entailment': 0.33, 'neutral': 0.34, 'contradiction': 0.33}

        try:
            if getattr(self, '_use_pipeline', False):
                result = self.model(
                    premise,
                    candidate_labels=['entailment', 'neutral', 'contradiction'],
                    hypothesis_template="{}",
                )
                return {l: s for l, s in zip(result['labels'], result['scores'])}

            # CrossEncoder returns raw logits or softmax scores
            import numpy as np
            scores = self.model.predict([(premise, hypothesis)])
            score_arr = scores[0]

            # Softmax if not already probabilities
            if score_arr.max() > 1.0 or score_arr.min() < 0.0:
                e = np.exp(score_arr - score_arr.max())
                score_arr = e / e.sum()

            return {
                label: float(score_arr[i])
                for i, label in enumerate(self._label_order)
            }
        except Exception as e:
            print(f"[NLI ERROR] {e}")
            return {'entailment': 0.33, 'neutral': 0.34, 'contradiction': 0.33}

    def classify(self, premise: str, hypothesis: str) -> str:
        """Returns the top label: 'entailment' / 'neutral' / 'contradiction'."""
        scores = self.predict(premise, hypothesis)
        return max(scores, key=scores.get)

    def batch_classify(self, pairs: List[Tuple[str, str]]) -> List[str]:
        """Batch classify multiple premise-hypothesis pairs."""
        if not pairs:
            return []
        if self.model is None or getattr(self, '_use_pipeline', False):
            return [self.classify(p, h) for p, h in pairs]
        try:
            import numpy as np
            scores_arr = self.model.predict(pairs)
            results = []
            for scores in scores_arr:
                if scores.max() > 1.0 or scores.min() < 0.0:
                    e = np.exp(scores - scores.max())
                    scores = e / e.sum()
                idx = int(scores.argmax())
                results.append(self._label_order[idx])
            return results
        except Exception as e:
            print(f"[NLI BATCH ERROR] {e}")
            return [self.classify(p, h) for p, h in pairs]


# ===================== RETRIEVAL CLIENT =====================
class RetrievalClient:
    def __init__(self, url: str = "http://localhost:8003/retrieve", topk: int = 5, timeout: int = 30):
        self.url = url
        self.topk = topk
        self.timeout = timeout
        self._check_connection()

    def _check_connection(self):
        try:
            resp = requests.get(self.url.replace("/retrieve", "/health"), timeout=5)
            if resp.status_code == 200:
                print(f"[INFO] Retrieval server connected: {self.url}")
        except Exception as e:
            print(f"[WARNING] Cannot connect to retrieval server: {e}")

    def retrieve(self, query: str, topk: Optional[int] = None) -> List[Dict]:
        return self.batch_retrieve([query], topk)[0]

    def batch_retrieve(self, queries: List[str], topk: Optional[int] = None) -> List[List[Dict]]:
        topk = topk or self.topk
        try:
            resp = requests.post(
                self.url,
                json={"queries": queries, "topk": topk, "return_scores": True},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            formatted = []
            for qr in resp.json().get("result", []):
                docs = []
                for item in qr:
                    doc = item.get("document", {})
                    docs.append({
                        "title":   doc.get("title", ""),
                        "snippet": doc.get("text", doc.get("contents", doc.get("snippet", ""))),
                        "score":   item.get("score", 0.0),
                    })
                formatted.append(docs)
            return formatted
        except Exception as e:
            print(f"[ERROR] Retrieval failed: {e}")
            return [[] for _ in queries]


# ===================== CHECKER v15 =====================
class GapCheckerV15:
    """
    Step-Gap Checker v15 — hybrid LLM + NLI design.

    Decision tree:
      Step -1  [LLM]  : entity/relation alignment check
      Node  0  [LLM]  : abstention detection
      Node  1  [LLM]  : entity consistency check + verbatim quote search
        ├─ found_quote=True:
        │    Node 2A [NLI]  : NLI(quote, step_claim)
        │      entailment   → no_gap
        │      neutral      → missing_bridge
        │      contradiction→ check 3B then unsupported_claim
        └─ found_quote=False:
             Node 2B [NLI]  : necessity test + cross-verify
               NLI(step_claim, final_answer) == neutral → not necessary → no_gap
               NLI(any_prior_evidence, step_claim) == entailment → cross-verified → no_gap
               else → insufficient_context gap

    Global Prior Evidence: for conclusion steps, prior inference evidence is
    injected and used in both Node 0 and Node 2B cross-verify.
    """

    # Simplified schema — LLM only handles alignment, abstention, and quote search
    RESPONSE_SCHEMA = {
        "name": "gap_check_v15",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "step_minus1_alignment": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "is_off_target":       {"type": "boolean"},
                        "drift_type":          {"type": "string",
                                                "enum": ["none","entity_drift","relation_drift","scope_drift"]},
                        "alignment_reasoning": {"type": "string"},
                    },
                    "required": ["is_off_target","drift_type","alignment_reasoning"],
                },
                "step_0_abstention_check": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "is_abstention_step":     {"type": "boolean"},
                        "abstention_is_accurate": {"type": "boolean"},
                        "abstention_reasoning":   {"type": "string"},
                    },
                    "required": ["is_abstention_step","abstention_is_accurate","abstention_reasoning"],
                },
                "step_1_quote_search": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "entity_match":           {"type": "boolean"},
                        "entity_match_reasoning": {"type": "string"},
                        "found_quote":            {"type": "boolean"},
                        "evidence_quote":         {"type": "string"},
                        "quote_search_reasoning": {"type": "string"},
                    },
                    "required": ["entity_match","entity_match_reasoning",
                                 "found_quote","evidence_quote","quote_search_reasoning"],
                },
            },
            "required": [
                "step_minus1_alignment",
                "step_0_abstention_check",
                "step_1_quote_search",
            ],
        },
    }

    def __init__(
        self,
        llm_model: str = "gpt-4.1-mini",
        llm_max_tokens: int = 800,
        nli_checker: Optional[NLIChecker] = None,
        nli_entailment_threshold: float = 0.5,
        ablate_no_alignment: bool = False,
        ablate_no_global_prior: bool = False,
        ablate_no_crossverify: bool = False,
        checker_base_url: Optional[str] = None,
    ):
        self.llm_model = llm_model
        self.llm_max_tokens = llm_max_tokens
        self.nli = nli_checker
        self.nli_threshold = nli_entailment_threshold
        self.ablate_no_alignment      = ablate_no_alignment
        self.ablate_no_global_prior   = ablate_no_global_prior
        self.ablate_no_crossverify    = ablate_no_crossverify
        self.checker_base_url         = checker_base_url
        self.client = None
        self.call_count = 0

        try:
            from openai import OpenAI
            if checker_base_url:
                # Local student model via vLLM OpenAI-compatible API
                self.client = OpenAI(base_url=checker_base_url, api_key="dummy")
                print(f"[INFO] Checker v17 (LOCAL): model={llm_model} | base_url={checker_base_url}")
            else:
                self.client = OpenAI()
                print(f"[INFO] Checker v17: LLM={llm_model} | NLI={'loaded' if nli_checker else 'MISSING'}")
        except Exception as e:
            print(f"[ERROR] Failed to init OpenAI: {e}")

    # ── LLM prompt (simplified: alignment + abstention + EC + quote) ─────────
    def _build_llm_prompt(
        self,
        question: str,
        step: Dict,
        prev_steps: List[Dict],
        prior_evidence_snippets: List[str],
    ) -> str:

        prev_text = "\n".join(
            f"Step {s['step_id']}: [{s.get('step_type','?')}] {s.get('step_text','')[:120]}"
            for s in prev_steps[-3:]
        ) or "(first step)"

        evidence_txt = json.dumps(step.get("evidence_pool", []), ensure_ascii=False)[:2000]
        is_conclusion = step.get("step_type") == "conclusion"

        # Global prior evidence section (conclusion steps only)
        global_section = ""
        if is_conclusion and prior_evidence_snippets and not self.ablate_no_global_prior:
            global_section = (
                "\nGLOBAL EVIDENCE (from all prior inference steps):\n"
                + json.dumps(prior_evidence_snippets[:8], ensure_ascii=False)
                + "\nNode 0: if any snippet answers the main question, N/A is inaccurate.\n"
                "Node 1: search BOTH evidence_pool AND this global evidence.\n"
            )

        # Step -1 section
        if self.ablate_no_alignment:
            step_m1_section = """
STEP -1: ALIGNMENT [ABLATED]
Set is_off_target=false, drift_type="none", alignment_reasoning="ablated".
"""
        elif not is_conclusion:
            step_m1_section = """
STEP -1: SUBQUESTION ALIGNMENT
Does this step target the CORRECT entity and relation?
CRITICAL: retrieval failure != off-target. Only flag if step_text ITSELF names wrong entity.

EC quick test — is_off_target=FALSE if:
  - Entity in step_text appears verbatim in the question
  - Entity is the correct intermediate result
  - step_text names correct film/person but retrieval returned wrong doc

TRUE drift: Q asks "AUTHOR of book" → step searches "DIRECTOR of film adaptation"
NOT drift: retrieval failed but query is correct

If is_off_target=TRUE → has_gap=TRUE, gap_type="unsupported_claim", STOP.
"""
        else:
            step_m1_section = """
STEP -1: CONCLUSION ALIGNMENT (MANDATORY)
RULE 0: answer contains N/A/unknown → is_off_target=FALSE, bypass to Node 0.

Otherwise C-1→C-2→C-3:
  C-1: target attribute type (PLACE / BIRTH_YEAR / SPOUSE / AGENT / COUNTRY / YES_NO)
  C-2: actual type of answer value
  C-3: TYPE mismatch = relation_drift. Wrong value but same type = NOT drift.

[DRIFT]    Q:"Where born?" → "Naresh Kumar" (person≠place) → TRUE
[NO DRIFT] Q:"Who did X?" → "Viktor Belash" (person=person) → FALSE
"""

        prompt = f"""You are a gap detector for multi-hop QA reasoning steps.

{step_m1_section}
STEP 0: ABSTENTION CHECK (skip if Step -1 triggered)
Is this step saying N/A / cannot determine?
  Evidence lacks it → GROUNDED ABSTENTION → is_abstention_step=True, abstention_is_accurate=True
  Evidence has it   → WRONG ABSTENTION   → is_abstention_step=True, abstention_is_accurate=False
{global_section}

STEP 1: ENTITY CONSISTENCY + QUOTE SEARCH (skip if Step -1 or 0 triggered)

Part A — Entity Consistency Check:
  EC-1: What entity/topic does this step's query target?
  EC-2: What entity/topic is the retrieved evidence actually about?
        (Check the document title and first sentence)
  EC-3: Same entity? → entity_match=TRUE, proceed to quote search
        Different entity? → entity_match=FALSE, found_quote=FALSE (stop here)

  ╔═══════════════════════════════════════════════════════════╗
  ║  Quote from the WRONG document does NOT count.           ║
  ║  Step: "Sruthilayalu composer"                          ║
  ║  Evidence: "M.G. Sreekumar" (different person) → FALSE  ║
  ╚═══════════════════════════════════════════════════════════╝

Part B — Verbatim Quote Search (only if entity_match=TRUE):
  Find 5-20 word EXACT span from evidence.
  Must be: (1) exact text, (2) same entity, (3) relevant to step claim.

NOTE: Node 2A (does quote support claim?) and Node 2B (is it verifiable?)
will be handled by a separate NLI model. Your job here is ONLY:
  1. Check entity consistency
  2. Find or not find a verbatim quote

Main Question: {question}

Previous Steps:
{prev_text}

CURRENT STEP (Step {step['step_id']}) [{step.get('step_type','inference')}]:
{step['step_text']}

Evidence Pool:
{evidence_txt}

Fill ALL fields carefully.
"""
        return prompt

    # ── Node 2A: NLI entailment check ────────────────────────────────────────
    def _node_2a_nli(self, evidence_quote: str, step_claim: str,
                     is_conclusion: bool = False) -> Dict:
        """
        NLI(premise=evidence_quote, hypothesis=step_claim)
        entailment   → no_gap
        neutral      → missing_bridge  (conclusion steps only)
                     → insufficient_context (inference steps — search query ≠ logical claim)
        contradiction→ unsupported_claim

        Fix: neutral on inference steps was mis-triggering because search query text
        ("Brer Rabbit 1946 Disney film") is not a logical claim — NLI correctly sees
        no entailment, but that's expected, not a missing_bridge gap.
        Only conclusion steps make logical claims that can have missing_bridge.
        """
        if not self.nli or not evidence_quote or not step_claim:
            return {"label": "entailment", "has_gap": False, "gap_type": None,
                    "gap_cause": "none", "nli_scores": {}}

        scores = self.nli.predict(evidence_quote, step_claim)
        label = max(scores, key=scores.get)

        if label == "entailment":
            gap_type, gap_cause = None, "none"
        elif label == "neutral":
            if is_conclusion:
                # Conclusion step: evidence exists but answer needs an unstated bridge
                gap_type, gap_cause = "missing_bridge", "semantic_leap"
            else:
                # Inference step: NLI neutral just means the retrieved doc doesn't
                # perfectly entail the search query — but that's okay for an
                # intermediate search step. The step found relevant evidence.
                # Fix v15.1→v15.2: treat as no_gap (not insufficient_context).
                gap_type, gap_cause = None, "none"
        else:  # contradiction
            if is_conclusion:
                gap_type, gap_cause = "unsupported_claim", "reasoning"
            else:
                # Inference step: contradiction means the found quote explicitly
                # contradicts the search direction — this is a real gap.
                gap_type, gap_cause = "unsupported_claim", "reasoning"

        return {
            "label":     label,
            "has_gap":   label != "entailment",
            "gap_type":  gap_type,
            "gap_cause": gap_cause,
            "nli_scores": {k: round(v, 3) for k, v in scores.items()},
        }

    # ── Node 2B: cross-verify only (no necessity test) ───────────────────────
    def _node_2b_nli(
        self,
        step_claim: str,
        final_answer: str,
        prior_evidence_snippets: List[str],
        prior_evidence_entity_matched: List[bool],
    ) -> Dict:
        """
        Fix: removed neutral_not_necessary test entirely.
        Reason: in multi-hop QA, intermediate step claims are naturally
        neutral w.r.t. the final answer (different granularity) — the test
        was incorrectly releasing ~4 FN cases per 20 questions.

        Fix: cross-verify only uses entity-matched evidence snippets.
        Reason: NLI was finding semantic matches in irrelevant documents
        (e.g. "Devon Martinus nationality" cross-verified by a basketball
        team article), causing FN cases.

        New logic:
          1. Filter prior snippets to only those from entity-matched steps
          2. NLI(filtered_snippet, step_claim) == entailment → cross-verified → no_gap
          3. No entity-matched snippets or none entail → insufficient_context gap
        """
        if not self.nli or not step_claim:
            return {"has_gap": True, "gap_type": "insufficient_context",
                    "gap_cause": "retrieval", "test1": "skipped", "test2": "skipped",
                    "nli_scores_test1": {}, "cross_verified_by": None}

        # Only use snippets from steps where entity matched (relevant documents)
        trusted_snippets = [
            snip for snip, matched in
            zip(prior_evidence_snippets, prior_evidence_entity_matched)
            if matched and snip.strip()
        ]

        if trusted_snippets:
            pairs = [(snip, step_claim) for snip in trusted_snippets]
            labels2 = self.nli.batch_classify(pairs)
            for snip, lbl in zip(trusted_snippets, labels2):
                if lbl == "entailment":
                    return {
                        "has_gap":   False,
                        "gap_type":  None,
                        "gap_cause": "none",
                        "test1": "cross_verify_only",
                        "test2": "cross_verified",
                        "nli_scores_test1": {},
                        "cross_verified_by": snip[:80],
                    }

        return {
            "has_gap":   True,
            "gap_type":  "insufficient_context",
            "gap_cause": "retrieval",
            "test1": "cross_verify_only",
            "test2": "not_cross_verified",
            "nli_scores_test1": {},
            "cross_verified_by": None,
        }

    # ── Main check_step ───────────────────────────────────────────────────────
    def check_step(
        self,
        question: str,
        step: Dict,
        prev_steps: List[Dict],
        all_prior_evidence: Optional[List[Dict]] = None,
        final_answer: str = "",
    ) -> Dict:
        self.call_count += 1
        if not self.client:
            return self._error_result("No OpenAI client")

        is_conclusion = step.get("step_type") == "conclusion"

        # Build prior evidence snippets
        prior_snippets: List[str] = []
        if all_prior_evidence:
            seen: set = set()
            for e in all_prior_evidence:
                snip = (e.get("snippet") or "")[:300].strip()
                if snip and snip not in seen:
                    seen.add(snip)
                    prior_snippets.append(snip)

        prompt = self._build_llm_prompt(question, step, prev_steps, prior_snippets)
        raw_response = ""

        try:
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.llm_max_tokens,
                temperature=0,
                response_format={"type": "json_schema", "json_schema": self.RESPONSE_SCHEMA},
            )
            raw_response = resp.choices[0].message.content or ""
            parsed = json.loads(raw_response)
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return self._error_result(str(e), raw_response)

        # ── Step -1: alignment ────────────────────────────────────────────────
        if not self.ablate_no_alignment:
            m1 = parsed.get("step_minus1_alignment", {})
            if m1.get("is_off_target", False):
                drift = m1.get("drift_type", "entity_drift")
                return self._result(
                    tree_path=f"-1(off_target:{drift})",
                    has_gap=True, gap_type="unsupported_claim", gap_cause="alignment",
                    parsed=parsed, prior_snippets=prior_snippets,
                )

        # ── Node 0: abstention ────────────────────────────────────────────────
        s0 = parsed.get("step_0_abstention_check", {})
        if s0.get("is_abstention_step", False):
            if s0.get("abstention_is_accurate", False):
                return self._result(
                    tree_path="0(abstention)->accurate->no_gap",
                    has_gap=False, gap_type=None, gap_cause="none",
                    parsed=parsed, prior_snippets=prior_snippets,
                )
            return self._result(
                tree_path="0(abstention)->inaccurate->gap",
                has_gap=True, gap_type="unsupported_claim", gap_cause="reasoning",
                parsed=parsed, prior_snippets=prior_snippets,
            )

        # ── Node 1: entity check + quote search ───────────────────────────────
        s1 = parsed.get("step_1_quote_search", {})
        entity_match = s1.get("entity_match", True)
        found_quote  = s1.get("found_quote", False)
        quote        = s1.get("evidence_quote", "")

        # Entity mismatch → retrieval failure, skip to gap
        if not entity_match:
            return self._result(
                tree_path="1(entity_mismatch)->insufficient_context",
                has_gap=True, gap_type="insufficient_context", gap_cause="retrieval",
                evidence_quote="", parsed=parsed, prior_snippets=prior_snippets,
            )

        # ── Node 2A + 2B: data-driven rules by step_type x entity_match x found_quote ──
        #
        # Based on empirical analysis of 181 steps (n=82 benchmark):
        #
        #   inference + entity_match=True  + found_quote=True  → 12% gap rate → no_gap
        #     Rationale: found relevant evidence for the right entity. NLI neutral
        #     on search queries is unreliable (query text ≠ logical claim).
        #     Rule: no_gap regardless of NLI label.
        #
        #   inference + entity_match=True  + found_quote=False → 61% gap rate → gap
        #     Rationale: right entity retrieved but nothing found in evidence → retrieval fail.
        #     Rule: insufficient_context gap.
        #
        #   inference + entity_match=False → 78% gap rate → gap
        #     Rationale: wrong document retrieved entirely.
        #     Rule: already handled above as entity_mismatch → gap.
        #
        #   conclusion + found_quote=True  → 59% gap rate → use NLI
        #     Rationale: NLI is reliable here — quote vs answer is a proper
        #     entailment question. entailment=no_gap, neutral=missing_bridge,
        #     contradiction=unsupported_claim.
        #
        #   conclusion + found_quote=False → 83-89% gap rate → gap (use cross-verify first)
        #     Rationale: conclusion without direct evidence almost always has a gap.
        #     Only override if cross-verify finds support in prior steps.

        if not is_conclusion:
            # ── Inference step ────────────────────────────────────────────────
            if found_quote and quote and len(quote) > 3:
                # entity_match=True + found_quote=True → no_gap (12% gap rate)
                nli_2a = self._node_2a_nli(quote, step.get("step_text", ""),
                                            is_conclusion=False)
                return self._result(
                    tree_path="1(found)->inference->no_gap",
                    has_gap=False, gap_type=None, gap_cause="none",
                    evidence_quote=quote,
                    nli_2a=nli_2a, parsed=parsed, prior_snippets=prior_snippets,
                )
            else:
                # entity_match=True + found_quote=False (61% gap rate)
                # v15.4: Before judging gap, run two NLI rescue tests:
                #
                # Test 1 — Necessity:
                #   NLI(step_claim → final_answer)
                #   neutral → this claim is NOT necessary for the final answer
                #           → the missing evidence doesn't affect the conclusion → no_gap
                #
                # Test 2 — Cross-verify:
                #   NLI(prior_entity_matched_evidence → step_claim)
                #   entailment → another step's evidence already covers this claim → no_gap
                #
                # Only if both tests fail → insufficient_context gap

                # v17: Necessity Test removed entirely.
                # Reason: in multi-hop QA, intermediate inference steps are
                # structurally neutral w.r.t. the final answer — the test
                # always fired and incorrectly released 10 true gaps (FN).
                #
                # Only Cross-Verify remains:
                #   NLI(entity-matched prior evidence → step_claim) == entailment
                #   → another step already covers this claim → no_gap
                #   → else → insufficient_context gap

                step_claim = step.get("step_text", "")
                nli_v17    = {}

                if self.nli and step_claim and not self.ablate_no_crossverify:
                    trusted = []
                    for ps in prev_steps:
                        ps_match = (ps.get("tree_steps") or {}).get(
                            "step_1", {}).get("entity_match", True)
                        if ps_match:
                            for e in ps.get("evidence_pool", []):
                                snip = (e.get("snippet") or "")[:300].strip()
                                if snip:
                                    trusted.append(snip)

                    if trusted:
                        pairs  = [(snip, step_claim) for snip in trusted]
                        labels = self.nli.batch_classify(pairs)
                        for snip, lbl in zip(trusted, labels):
                            if lbl == "entailment":
                                nli_v17["test2_label"]       = "cross_verified"
                                nli_v17["cross_verified_by"] = snip[:80]
                                return self._result(
                                    tree_path="1(not_found)->inf->cross_verified->no_gap",
                                    has_gap=False, gap_type=None, gap_cause="none",
                                    nli_2b=nli_v17, parsed=parsed, prior_snippets=prior_snippets,
                                )
                        nli_v17["test2_label"] = "not_cross_verified"

                # Cross-verify failed or ablated → gap
                return self._result(
                    tree_path="1(not_found)->inference->insufficient_context",
                    has_gap=True, gap_type="insufficient_context", gap_cause="retrieval",
                    nli_2b=nli_v17, parsed=parsed, prior_snippets=prior_snippets,
                )

        # ── Conclusion step: use NLI (reliable here) ─────────────────────────
        if found_quote and quote and len(quote) > 3:
            nli_2a = self._node_2a_nli(quote, step.get("step_text", ""),
                                        is_conclusion=True)
            label = nli_2a["label"]
            if label == "entailment":
                return self._result(
                    tree_path="1(found)->2A(NLI:entailment)->no_gap",
                    has_gap=False, gap_type=None, gap_cause="none",
                    evidence_quote=quote,
                    nli_2a=nli_2a, parsed=parsed, prior_snippets=prior_snippets,
                )
            elif label == "neutral":
                return self._result(
                    tree_path="1(found)->2A(NLI:neutral)->missing_bridge",
                    has_gap=True, gap_type="missing_bridge", gap_cause="semantic_leap",
                    evidence_quote=quote,
                    nli_2a=nli_2a, parsed=parsed, prior_snippets=prior_snippets,
                )
            else:  # contradiction
                return self._result(
                    tree_path="1(found)->2A(NLI:contradiction)->unsupported_claim",
                    has_gap=True, gap_type="unsupported_claim", gap_cause="reasoning",
                    evidence_quote=quote,
                    nli_2a=nli_2a, parsed=parsed, prior_snippets=prior_snippets,
                )

        # Conclusion + no quote: try cross-verify with entity-matched prior evidence
        prior_entity_matched: List[bool] = []
        for s in prev_steps:
            s_match = (s.get("tree_steps") or {}).get("step_1", {}).get("entity_match", True)
            for e in s.get("evidence_pool", []):
                snip = (e.get("snippet") or "")[:300].strip()
                if snip:
                    prior_entity_matched.append(bool(s_match))

        nli_2b = self._node_2b_nli(
            step_claim=step.get("step_text", ""),
            final_answer=final_answer,
            prior_evidence_snippets=prior_snippets if not self.ablate_no_global_prior else [],
            prior_evidence_entity_matched=(
                prior_entity_matched if not self.ablate_no_global_prior else []
            ),
        )

        if not nli_2b["has_gap"]:
            return self._result(
                tree_path="1(not_found)->2B(cross_verified)->no_gap",
                has_gap=False, gap_type=None, gap_cause="none",
                nli_2b=nli_2b, parsed=parsed, prior_snippets=prior_snippets,
            )

        return self._result(
            tree_path="1(not_found)->2B(not_cross_verified)->insufficient_context",
            has_gap=True, gap_type="insufficient_context", gap_cause="retrieval",
            nli_2b=nli_2b, parsed=parsed, prior_snippets=prior_snippets,
        )

    def _result(
        self,
        tree_path: str,
        has_gap: bool,
        gap_type: Optional[str],
        gap_cause: str,
        evidence_quote: str = "",
        nli_2a: Optional[Dict] = None,
        nli_2b: Optional[Dict] = None,
        parsed: Optional[Dict] = None,
        prior_snippets: Optional[List[str]] = None,
    ) -> Dict:
        parsed = parsed or {}
        s1 = (parsed.get("step_1_quote_search") or {})
        return {
            "has_gap":           has_gap,
            "gap_type":          gap_type,
            "gap_cause":         gap_cause,
            "has_semantic_leap": gap_cause == "semantic_leap",
            "semantic_leap_description": (
                f"NLI: {nli_2a.get('label','?')} (evidence does not entail claim)"
                if nli_2a and nli_2a.get("label") == "neutral" else ""
            ),
            "is_common_knowledge": False,
            "evidence_quote":    evidence_quote or s1.get("evidence_quote", ""),
            "tree_path":         tree_path,
            "confidence":        0.9 if not has_gap else 0.7,
            "explanation":       f"NLI-based: {tree_path}",
            "evidence_sufficient": "yes" if not has_gap else "no",
            "tree_steps": {
                "step_m1":  parsed.get("step_minus1_alignment", {}),
                "step_0":   parsed.get("step_0_abstention_check", {}),
                "step_1":   s1,
                "nli_2a":   nli_2a or {},
                "nli_2b":   nli_2b or {},
            },
        }

    def _error_result(self, error_msg: str, raw_response: str = "") -> Dict:
        return {
            "has_gap": None, "gap_type": None, "gap_cause": "error",
            "has_semantic_leap": False, "semantic_leap_description": "",
            "is_common_knowledge": False, "evidence_quote": "",
            "tree_path": "error", "confidence": 0.0,
            "explanation": f"Error: {error_msg}",
            "evidence_sufficient": "unclear",
            "tree_steps": {},
        }


# ===================== GENERATOR =====================
class HFGenerator:
    def __init__(self, model_path: str, device: str = "auto", max_new_tokens: int = 1024):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._init_model()

    def _init_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.torch = torch
        print(f"[INFO] Loading Qwen: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else None,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()
        print(f"[INFO] Qwen loaded on {self.device}")

    def _build_prompt(self, question: str, context: List[Dict]) -> str:
        initial_results = "".join(
            f"{c.get('title','')}: {(c.get('snippet','') or '')[:400]}\n"
            for c in context[:5] if (c.get('snippet','') or '')
        ).strip()
        return f"""Answer the given question. Follow this exact format:
<think>[Your reasoning plan]</think>
<search>[first search query]</search>
<search>[second search query if needed]</search>
<answer>[your final short answer]</answer>
RULES: 1. Start with <think>. 2. Each <search> has ONE query. 3. End with <answer>.
---
Initial retrieved evidence:
<r>{initial_results}</r>
Question: {question}
"""

    def _generate(self, prompt: str) -> str:
        try:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    @staticmethod
    def _extract_answer(text: str) -> str:
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else "unknown"

    @staticmethod
    def _extract_think(text: str) -> str:
        blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        return "\n".join(b.strip() for b in blocks) if blocks else ""

    def generate(self, question: str, context: List[Dict]) -> Tuple[str, str, str]:
        prompt = self._build_prompt(question, context)
        raw = self._generate(prompt)
        return raw, self._extract_think(raw), self._extract_answer(raw)


# ===================== STEP SEGMENTATION =====================
def parse_structured_output(raw_output: str) -> Tuple[str, List[Tuple[str, str]], str]:
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL | re.IGNORECASE)
    think_content = think_match.group(1).strip() if think_match else ""

    search_spans = [
        (m.start(), m.end(), m.group(1).strip())
        for m in re.finditer(r"<search>(.*?)</search>", raw_output, re.DOTALL | re.IGNORECASE)
    ]

    def _extract_r(text, after):
        tag   = re.compile(r"<(?:r|result)>",  re.IGNORECASE)
        close = re.compile(r"</(?:r|result)>", re.IGNORECASE)
        nxt   = re.compile(r"<search>|<answer>", re.IGNORECASE)
        mo = tag.search(text, after)
        if not mo: return -1, -1, ""
        cs = mo.end()
        mc = close.search(text, cs)
        mn = nxt.search(text, cs)
        if mc and (mn is None or mc.start() < mn.start()):
            raw, ep = text[cs:mc.start()], mc.end()
        elif mn:
            raw, ep = text[cs:mn.start()], mn.start()
        else:
            raw, ep = text[cs:], len(text)
        return mo.start(), ep, re.sub(r'^\s*:\s*"?\s*', "", raw).strip().strip('"').strip()

    pairs = []
    for _, s_end, query in search_spans:
        _, _, result_text = _extract_r(raw_output, s_end)
        pairs.append((query, result_text))

    answer_match = re.search(r"<answer>(.*?)</answer>", raw_output, re.DOTALL | re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else "unknown"
    return think_content, pairs, answer


def segment_think_into_steps(raw_output: str, context: List[Dict]) -> List[Dict]:
    _, pairs, answer = parse_structured_output(raw_output)
    if pairs:
        steps = []
        for i, (query, result_text) in enumerate(pairs):
            if not query: continue
            evidence_pool = (
                [{"title": "", "snippet": result_text[:1000], "score": None}]
                if result_text else context[:3]
            )
            steps.append({"step_id": i, "step_type": "inference",
                           "step_text": query[:500], "evidence_pool": evidence_pool})
        if answer and answer != "unknown":
            steps.append({"step_id": len(steps), "step_type": "conclusion",
                           "step_text": f"Answer is: {answer}", "evidence_pool": context[:2]})
        return steps[:8]

    think_match = re.search(r"<think>(.*?)(?:</think>|<answer>|$)", raw_output, re.DOTALL | re.IGNORECASE)
    think_text = think_match.group(1).strip() if think_match else raw_output
    sentences = re.split(r"(?<=[.!?])\s+", think_text)
    steps = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if sent and len(sent) > 15:
            step_type = ("evidence" if i == 0 else
                         "conclusion" if i == len(sentences)-1 else "inference")
            steps.append({"step_id": i, "step_type": step_type,
                           "step_text": sent[:500], "evidence_pool": context[:5]})
    return steps[:6]


def _token_set(text: str) -> set:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", (text or "").lower())
    return set(t for t in text.split() if len(t) > 2)


def is_evidence_like_step(step_text: str, evidence_pool: List[Dict]) -> bool:
    step_tokens = _token_set(step_text)
    if len(step_tokens) < 6: return False
    for e in evidence_pool or []:
        snippet = e.get("snippet", "")
        if snippet:
            overlap = len(step_tokens & _token_set(snippet)) / max(1, len(step_tokens))
            if overlap >= 0.35: return True
    return False


class AnswerEvaluator:
    def is_correct(self, pred: str, gold: str) -> bool:
        if not pred or not gold: return False
        p = re.sub(r"[^\w\s]", "", pred.lower()).strip()
        g = re.sub(r"[^\w\s]", "", gold.lower()).strip()
        return p == g or g in p or p in g


# ===================== FROZEN QWEN SUPPORT =====================
def load_frozen_qwen(path: str) -> Dict[str, Dict]:
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[r["question_id"]] = r
    print(f"[INFO] Loaded {len(records)} frozen Qwen records from: {path}")
    return records


def save_qwen_steps(records: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            entry = {
                "question_id":       r["question_id"],
                "dataset":           r.get("dataset"),
                "question":          r["question"],
                "gold_answer":       r["gold_answer"],
                "predicted_answer":  r.get("predicted_answer", ""),
                "is_correct":        r.get("is_correct", False),
                "context":           r.get("context", []),
                "steps": [
                    {"step_id": s["step_id"], "step_type": s["step_type"],
                     "step_text": s["step_text"], "evidence_pool": s.get("evidence_pool", []),
                     "tree_path": s.get("tree_path", "")}
                    for s in r.get("steps", [])
                ],
                "raw_think_content": r.get("raw_think_content", ""),
                "raw_output":        r.get("raw_output", ""),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved frozen steps: {path} ({len(records)} records)")


# ===================== CSV GENERATION =====================
def generate_human_annotation_csv(records: List[Dict], output_path: str) -> str:
    rows = []
    for record in records:
        qid = record.get("question_id", "")
        for step in record.get("steps", []):
            ts  = step.get("tree_steps", {})
            n2a = ts.get("nli_2a", {})
            n2b = ts.get("nli_2b", {})
            rows.append({
                "question_id":       qid,
                "dataset":           record.get("dataset", ""),
                "question":          record.get("question", ""),
                "gold_answer":       record.get("gold_answer", ""),
                "predicted_answer":  record.get("predicted_answer", ""),
                "answer_correct":    record.get("is_correct", False),
                "step_id":           step.get("step_id", ""),
                "step_type":         step.get("step_type", ""),
                "step_text":         step.get("step_text", ""),
                "nm1_is_off_target":       ts.get("step_m1", {}).get("is_off_target", ""),
                "nm1_drift_type":          ts.get("step_m1", {}).get("drift_type", ""),
                "n0_is_abstention":        ts.get("step_0", {}).get("is_abstention_step", ""),
                "n0_abstention_accurate":  ts.get("step_0", {}).get("abstention_is_accurate", ""),
                "n1_entity_match":         ts.get("step_1", {}).get("entity_match", ""),
                "n1_found_quote":          ts.get("step_1", {}).get("found_quote", ""),
                "n1_evidence_quote":       ts.get("step_1", {}).get("evidence_quote", ""),
                "n2a_nli_label":           n2a.get("label", ""),
                "n2a_nli_entail":          n2a.get("nli_scores", {}).get("entailment", ""),
                "n2a_nli_neutral":         n2a.get("nli_scores", {}).get("neutral", ""),
                "n2a_nli_contra":          n2a.get("nli_scores", {}).get("contradiction", ""),
                "n2b_test1":               n2b.get("test1", ""),
                "n2b_test2":               n2b.get("test2", ""),
                "n2b_cross_verified_by":   n2b.get("cross_verified_by", ""),
                "ai_tree_path":            step.get("tree_path", ""),
                "ai_has_gap":              step.get("has_gap", ""),
                "ai_gap_type":             step.get("gap_type", ""),
                "ai_gap_cause":            step.get("gap_cause", ""),
                "human_has_gap": "", "human_gap_type": "", "human_notes": "",
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] CSV saved: {output_path} ({len(rows)} rows)")
    return output_path


# ===================== PIPELINE =====================
def load_inputs(paths: List[str]) -> List[Dict]:
    data = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            if p.endswith(".jsonl"):
                for line in f:
                    if line.strip(): data.append(json.loads(line))
            else:
                obj = json.load(f)
                data.extend(obj if isinstance(obj, list) else [obj])
    return data


def run_checker_on_steps(qid: str, src: Dict, checker: GapCheckerV15) -> List[Dict]:
    question    = src["question"]
    final_answer = src.get("predicted_answer", "")
    steps        = src["steps"]
    out_steps    = []

    for i, step in enumerate(steps):
        if step.get("tree_path") == "skipped":
            out_steps.append({**step})
            continue

        prior_evidence: List[Dict] = []
        if step.get("step_type") == "conclusion":
            for s in steps[:i]:
                prior_evidence.extend(s.get("evidence_pool", []))

        r = checker.check_step(
            question, step, steps[:i],
            all_prior_evidence=prior_evidence or None,
            final_answer=final_answer,
        )
        out_step = {**step}
        out_step.update({
            "has_gap":                   r.get("has_gap"),
            "gap_type":                  r.get("gap_type"),
            "gap_cause":                 r.get("gap_cause"),
            "evidence_sufficient":       r.get("evidence_sufficient"),
            "evidence_quote":            r.get("evidence_quote"),
            "has_semantic_leap":         r.get("has_semantic_leap"),
            "semantic_leap_description": r.get("semantic_leap_description"),
            "is_common_knowledge":       r.get("is_common_knowledge"),
            "tree_path":                 r.get("tree_path"),
            "tree_steps":                r.get("tree_steps", {}),
        })
        out_steps.append(out_step)

    return out_steps


def run_pipeline(args) -> str:
    global DEBUG
    DEBUG = args.debug
    _safe_mkdir(args.output)

    # ── Init NLI ──────────────────────────────────────────────────────────────
    nli_checker = None
    if args.nli_model:
        nli_device = args.nli_device if args.nli_device >= 0 else 0
        nli_checker = NLIChecker(model_name=args.nli_model, device=nli_device)
    else:
        print("[WARNING] No NLI model specified. Node 2A will accept all found quotes, "
              "Node 2B will always flag as insufficient_context.")

    # ── Init LLM checker ──────────────────────────────────────────────────────
    checker = GapCheckerV15(
        llm_model=args.checker_model,
        llm_max_tokens=args.checker_max_tokens,
        nli_checker=nli_checker,
        ablate_no_alignment=getattr(args, 'ablate_no_alignment', False),
        ablate_no_global_prior=getattr(args, 'ablate_no_global_prior', False),
        ablate_no_crossverify=getattr(args, 'ablate_no_crossverify', False),
        checker_base_url=getattr(args, 'checker_base_url', None),
    )

    checker_tag = sanitize(args.checker_model)
    nli_tag     = sanitize(args.nli_model or "no-nli").split("-")[0]
    base_tag    = f"v17__checker-{checker_tag}__nli-{nli_tag}"

    # ── MODE A: Frozen Qwen ───────────────────────────────────────────────────
    if args.freeze_qwen:
        print(f"[INFO] FROZEN QWEN MODE: {args.freeze_qwen}")
        frozen = load_frozen_qwen(args.freeze_qwen)

        if args.benchmark:
            with open(args.benchmark) as f:
                bm_ids = {r["question_id"] for r in json.load(f)}
            frozen = {k: v for k, v in frozen.items() if k in bm_ids}
            print(f"[INFO] Filtered to {len(frozen)} benchmark questions")

        out_records = []
        for idx, (qid, src) in enumerate(frozen.items()):
            q = src["question"]
            print(f"[{idx+1}/{len(frozen)}] {qid}: {q[:50]}...")
            out_steps = run_checker_on_steps(qid, src, checker)
            gap_count = sum(1 for s in out_steps if s.get("has_gap") is True)
            print(f"  Steps: {len(out_steps)} | Gaps: {gap_count}")
            out_records.append({
                "question_id":      qid,
                "dataset":          src.get("dataset"),
                "question":         q,
                "gold_answer":      src.get("gold_answer", ""),
                "predicted_answer": src.get("predicted_answer", ""),
                "is_correct":       src.get("is_correct", False),
                "context":          src.get("context", []),
                "steps":            out_steps,
            })

        n        = len(out_records)
        out_file = os.path.join(args.output, f"n{n}__{base_tag}.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total_steps = sum(len(r["steps"]) for r in out_records)
        gaps = sum(1 for r in out_records for s in r["steps"] if s.get("has_gap") is True)
        print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
        print(f"Questions: {n} | Steps: {total_steps} | Gaps: {gaps}")
        print(f"Output: {out_file}")
        return out_file

    # ── MODE B: Full pipeline ─────────────────────────────────────────────────
    if not args.input:
        raise ValueError("Provide --input or --freeze_qwen")

    data = load_inputs(args.input)
    if args.sample_total and len(data) > args.sample_total:
        data = random.Random(args.seed).sample(data, args.sample_total)

    generator = HFGenerator(args.qwen_model, args.device, args.max_new_tokens)

    retrieval_client = None
    if args.use_dynamic_retrieval:
        retrieval_client = RetrievalClient(
            url=args.retrieval_url, topk=args.retrieval_topk, timeout=args.retrieval_timeout,
        )

    evaluator   = AnswerEvaluator()
    out_records = []
    stats       = defaultdict(int)

    for idx, item in enumerate(data):
        qid      = item.get("question_id", item.get("id", str(idx)))
        question = item.get("question", "")
        gold     = item.get("gold_answer", item.get("answer", ""))

        if args.use_dynamic_retrieval and retrieval_client:
            context = retrieval_client.retrieve(question, args.retrieval_topk)
        else:
            context = [
                {"title": c.get("title",""), "snippet": c.get("snippet", c.get("text","")), "score": c.get("score")}
                for c in item.get("context", [])
            ]
        if not context:
            context = [{"title": "", "snippet": f"Question: {question}", "score": None}]

        raw_output, think_content, predicted = generator.generate(question, context)
        steps = segment_think_into_steps(raw_output, context)

        if steps and steps[-1]["step_type"] != "conclusion":
            steps.append({"step_id": len(steps), "step_type": "conclusion",
                           "step_text": f"Answer is: {predicted}", "evidence_pool": context[:2]})
        if not steps:
            steps = [{"step_id": 0, "step_type": "conclusion",
                      "step_text": f"Answer is: {predicted}", "evidence_pool": context[:2]}]

        is_correct = evaluator.is_correct(predicted, gold)
        if is_correct: stats["correct"] += 1

        for i, step in enumerate(steps):
            if step.get("step_type") == "evidence" and is_evidence_like_step(
                step.get("step_text",""), step.get("evidence_pool",[])
            ):
                step.update({"has_gap": False, "gap_type": None, "gap_cause": "none",
                              "evidence_sufficient": "yes", "tree_path": "skipped",
                              "has_semantic_leap": False, "semantic_leap_description": "",
                              "is_common_knowledge": False, "tree_steps": {}})
                continue

            prior_evidence: List[Dict] = []
            if step.get("step_type") == "conclusion":
                for s in steps[:i]:
                    prior_evidence.extend(s.get("evidence_pool", []))

            r = checker.check_step(question, step, steps[:i],
                                   all_prior_evidence=prior_evidence or None,
                                   final_answer=predicted)
            step.update({
                "has_gap":                   r.get("has_gap"),
                "gap_type":                  r.get("gap_type"),
                "gap_cause":                 r.get("gap_cause"),
                "evidence_sufficient":       r.get("evidence_sufficient"),
                "evidence_quote":            r.get("evidence_quote"),
                "has_semantic_leap":         r.get("has_semantic_leap"),
                "semantic_leap_description": r.get("semantic_leap_description"),
                "is_common_knowledge":       r.get("is_common_knowledge"),
                "tree_path":                 r.get("tree_path"),
                "tree_steps":                r.get("tree_steps", {}),
            })

        gap_count = sum(1 for s in steps if s.get("has_gap") is True)
        print(f"[{idx+1}/{len(data)}] {qid}: {'✓' if is_correct else '✗'} | Steps:{len(steps)} Gaps:{gap_count}")

        out_records.append({
            "question_id": qid, "dataset": item.get("dataset"),
            "question": question, "gold_answer": gold,
            "predicted_answer": predicted, "is_correct": is_correct,
            "context": context, "steps": steps,
            "raw_think_content": think_content, "raw_output": raw_output,
        })

    n        = len(out_records)
    out_file = os.path.join(args.output, f"n{n}__{base_tag}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.save_qwen_steps:
        save_qwen_steps(out_records, args.save_qwen_steps)

    csv_file = out_file.replace(".jsonl", ".csv")
    generate_human_annotation_csv(out_records, csv_file)

    total_steps = sum(len(r["steps"]) for r in out_records)
    gaps = sum(1 for r in out_records for s in r["steps"] if s.get("has_gap") is True)
    print(f"\n{'='*55}\nSUMMARY v15\n{'='*55}")
    print(f"Samples: {n} | Correct: {stats['correct']} ({100*stats['correct']/n:.0f}%)")
    print(f"Steps: {total_steps} | Gaps: {gaps} ({100*gaps/total_steps:.0f}%)")
    return out_file


def main():
    p = argparse.ArgumentParser(description="Step-Gap Pipeline v15 — NLI-based checker")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--input",       type=str, nargs="+")
    g.add_argument("--freeze_qwen", type=str)

    p.add_argument("--output",          type=str, required=True)
    p.add_argument("--benchmark",       type=str, default=None)
    p.add_argument("--save_qwen_steps", type=str, default=None)
    p.add_argument("--sample_total",    type=int, default=100)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--debug",           action="store_true", default=True)

    # Qwen generator
    p.add_argument("--qwen_model",     type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device",         type=str, default="auto")
    p.add_argument("--max_new_tokens", type=int, default=1024)

    # LLM checker
    p.add_argument("--checker_model",      type=str, default="gpt-4.1-mini")
    p.add_argument("--checker_max_tokens", type=int, default=800)
    p.add_argument("--checker_base_url",   type=str, default=None,
                   help="OpenAI-compatible base URL for local student checker "
                        "(e.g. http://localhost:8100/v1). If not set, uses OpenAI API.")

    # NLI model
    p.add_argument("--nli_model",  type=str,
                   default="cross-encoder/nli-deberta-v3-large",
                   help="HuggingFace NLI model name")
    p.add_argument("--nli_device", type=int, default=0,
                   help="GPU device for NLI model (-1 for CPU)")

    # Retrieval
    p.add_argument("--use_dynamic_retrieval", action="store_true")
    p.add_argument("--retrieval_url",         type=str, default="http://localhost:8003/retrieve")
    p.add_argument("--retrieval_topk",        type=int, default=5)
    p.add_argument("--retrieval_timeout",     type=int, default=30)

    # Ablation flags (3 components)
    p.add_argument("--ablate_no_alignment",    action="store_true",
                   help="Disable Step -1 entity/relation alignment check")
    p.add_argument("--ablate_no_global_prior", action="store_true",
                   help="Disable global prior evidence injection for conclusion steps")
    p.add_argument("--ablate_no_crossverify",  action="store_true",
                   help="Disable NLI cross-verify for inference+not_found steps")

    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()