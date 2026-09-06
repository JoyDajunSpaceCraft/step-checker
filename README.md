# Step-Gap Guided Search: Evidence Gap Detection for Multi-Hop QA

A hybrid NLI + LLM pipeline for detecting evidence gaps at the step level in multi-hop question answering. The checker identifies when a reasoning step lacks sufficient evidence support, enabling step-level reward signals for generator training (GRPO/RLHF).

---

## Overview

> **Artifact status:** The surviving 100-question / 223-step benchmark is the primary reconstruction input. The legacy 82-question count-matched artifact is diagnostic only; it is not the recovered original split and must not be used for headline metrics. Benchmark gold is excluded from this migration branch.

Multi-hop QA systems (e.g., Qwen 7B with chain-of-thought reasoning) often produce intermediate reasoning steps that are not grounded in retrieved evidence. This project builds a **step-level evidence gap checker** that:

1. Accepts a reasoning step + its retrieved evidence pool
2. Classifies each step as `has_gap=True/False`
3. Assigns a gap type: `insufficient_context`, `missing_bridge`, `unsupported_claim`, or `null`
4. Produces a decision path (tree path) explaining the classification

The final system (v17) achieves **F1 = 72.0%** on the legacy diagnostic subset (n=82, κ=0.704 on 7B-generated steps), outperforming a pure-LLM baseline (v14 F1 = 70.0%).

---

## Key Results

### Step-Level Checker Performance (diagnostic n=82 subset)

| Version | Precision | Recall | F1 | TP | FP | TN | FN |
|---------|-----------|--------|-----|----|----|----|----|
| v14 (LLM only) | 72.9% | 67.3% | 70.0% | 70 | 26 | 51 | 34 |
| v15.3 (NLI hybrid) | 69.2% | 71.2% | 70.1% | 74 | 33 | 44 | 30 |
| v16 w/o necessity | 71.0% | 73.1% | 72.0% | 76 | 31 | 46 | 28 |
| **v17 (final)** | **71.0%** | **73.1%** | **72.0%** | 76 | 31 | 46 | 28 |

### Ablation Study (v17, n=82)

| Component | F1 | ΔF1 |
|-----------|-----|-----|
| Full system | 72.0% | — |
| w/o Step -1 alignment | 63.5% | -8.5% |
| w/o Global Prior | 67.6% | -4.4% |
| w/o Cross-Verify | 63.8% | -8.2% |

### Inter-Annotator Agreement

| Generator | n steps | Cohen's κ | Agreement |
|-----------|---------|-----------|-----------|
| Qwen 7B | 223 | **0.704** | 85.2% |
| Qwen 32B | 225 | 0.283 | 63.1% |

---

## System Architecture (v17)

```
Input: question + reasoning step + evidence pool + prior steps

Step -1 [LLM]   Entity/relation alignment check
    ↓ off_target → unsupported_claim (gap)

Node 0  [LLM]   Abstention detection (N/A steps)
    ↓ accurate abstention → no_gap
    ↓ inaccurate abstention → gap

Node 1  [LLM]   Entity consistency check + verbatim quote search
    ↓ entity_mismatch → insufficient_context (gap)

    ┌── found_quote=True + inference step ──────→ no_gap (rule-based)
    │
    ├── found_quote=True + conclusion step ──────→ NLI(quote, answer)
    │     entailment → no_gap
    │     neutral    → missing_bridge (gap)
    │     contradiction → unsupported_claim (gap)
    │
    └── found_quote=False + inference step
          NLI cross-verify: prior entity-matched evidence → step_claim
          entailment → no_gap
          else → insufficient_context (gap)

          found_quote=False + conclusion step
          NLI cross-verify with global prior → no_gap or gap
```

**Key design insight**: NLI is only applied where it is reliable — conclusion steps with found quotes (direct entailment check) and cross-verification against entity-matched prior evidence. Inference steps with found quotes are handled by rule (12% gap rate empirically), avoiding NLI over-triggering on search query text.

---

## Gap Taxonomy

| Gap Type | Definition | Example |
|----------|-----------|---------|
| `insufficient_context` | Evidence does not contain the needed information | Search for director returns wrong document |
| `missing_bridge` | Evidence found but answer requires an unstated premise | Quote found but entailment fails (NLI neutral) |
| `unsupported_claim` | Evidence directly contradicts or mismatches the step | NLI contradiction, or wrong entity retrieved |

---

## Repository Structure

```
gap_checker/
├── scripts/
│   ├── run_pipeline_v17.py      # Main pipeline (final version)
│   ├── prepare_datasets.py      # Load & combine 5 QA datasets
│   ├── build_sft_data.py        # Convert v17 output → SFT training format
│   └── run_full_pipeline.sh     # End-to-end shell script
│
├── train_checker/
│   └── train_checker.py         # LoRA SFT training (Qwen/Llama/DeepSeek)
│
├── ablation/
│   ├── v10_correct_for_ablation.jsonl   # Frozen 7B Qwen steps (82 questions)
│   └── benchmark_step_level_gt.json     # Human GT annotations (181 steps)
│
├── data/
│   ├── processed/               # Combined 1000-question datasets
│   ├── annotated/               # v17 pipeline output (checker-labeled steps)
│   └── sft/                     # SFT training data (train/val JSONL)
│
└── checkpoints/                 # Trained student checker models
    ├── checker-qwen25-7b/
    ├── checker-qwen3-8b/
    ├── checker-llama31-8b/
    └── checker-deepseek-r1-qwen7b/
```

---

## Setup

```bash
conda create -n gap_checker python=3.10
conda activate gap_checker
pip install transformers>=4.51.0 peft accelerate bitsandbytes \
            sentence-transformers openai datasets trl
```

**Required models** (download to `/path/to/models/`):
- Generator: `Qwen/Qwen2.5-7B-Instruct`
- NLI: `cross-encoder/nli-deberta-v3-large` (auto-downloaded)
- Checker LLM: OpenAI `gpt-4.1-mini` (requires API key)

---

## Running the Pipeline

### Step 1 — Prepare datasets

```bash
python scripts/prepare_datasets.py \
    --output data/processed/combined_1000.json \
    --per_dataset 200 \
    --datasets 2wiki hotpot musique triviaqa iirc \
    --wiki_path    data/processed/train_combined.json \
    --hotpot_path  data/processed/train_combined.json \
    --musique_path data/processed/train_combined.json \
    --triviaqa_path   data/raw/triviaqa-unfiltered/unfiltered-web-train.json \
    --iirc_path       data/raw/iirc_train_dev/train.json
```

### Step 2 — Run v17 pipeline (Qwen generator + hybrid checker)

```bash

CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline_v17.py \
    --input data/processed/combined_1000.json \
    --output data/annotated/v17_n1000 \
    --qwen_model /path/to/Qwen2.5-7B-Instruct \
    --nli_model cross-encoder/nli-deberta-v3-large \
    --nli_device 0 \
    --checker_model gpt-4.1-mini \
    --sample_total 1000 --seed 42 \
    --use_dynamic_retrieval \
    --save_qwen_steps data/annotated/v17_n1000/frozen_n1000.jsonl
```

### Step 3 — Diagnostic evaluation (n=82 subset)

```bash
# Checker-only mode using frozen Qwen steps
python scripts/run_pipeline_v17.py \
    --freeze_qwen ablation/v10_correct_for_ablation.jsonl \
    --benchmark   ablation/benchmark_step_level_gt.json \
    --output      data/annotated/v17_eval \
    --nli_model   cross-encoder/nli-deberta-v3-large \
    --nli_device  0 \
    --checker_model gpt-4.1-mini
```

### Step 4 — Ablation study

```bash
for FLAG in "" "--ablate_no_alignment" "--ablate_no_global_prior" "--ablate_no_crossverify"; do
    python scripts/run_pipeline_v17.py \
        --freeze_qwen ablation/v10_correct_for_ablation.jsonl \
        --benchmark   ablation/benchmark_step_level_gt.json \
        --output      data/annotated/ablation \
        --nli_model   cross-encoder/nli-deberta-v3-large \
        --nli_device  0 \
        --checker_model gpt-4.1-mini $FLAG
done
```

---

## Student Checker (SFT Distillation)

Distill v17 (GPT-4.1-mini + NLI teacher) into a local Qwen/Llama student model.

### Build training data

```bash
# Merge two 500-question runs first
cat data/annotated/v17_n1000/n500__v17__*.jsonl \
    > data/annotated/v17_n1000_combined.jsonl

# Convert to SFT format
python scripts/build_sft_data.py \
    --input  data/annotated/v17_n1000_combined.jsonl \
    --output data/sft/checker_train.jsonl \
    --val_ratio 0.1 --seed 42
```

### Train student models

```bash
BASE=/path/to/models
TRAIN=data/sft/checker_train.jsonl
VAL=data/sft/checker_train_val.jsonl

# Qwen2.5-7B (fp16)
CUDA_VISIBLE_DEVICES=0 python train_checker/train_checker.py \
    --model_path $BASE/Qwen2.5-7B-Instruct \
    --train_data $TRAIN --val_data $VAL \
    --output_dir checkpoints/checker-qwen25-7b \
    --epochs 3 --batch_size 4 --grad_accum 8 --lora_r 32 --fp16

# Qwen3-8B (bf16, /no_think mode)
CUDA_VISIBLE_DEVICES=0 python train_checker/train_checker.py \
    --model_path $BASE/Qwen3-8B \
    --train_data $TRAIN --val_data $VAL \
    --output_dir checkpoints/checker-qwen3-8b \
    --epochs 3 --batch_size 4 --grad_accum 8 --lora_r 32 --bf16

# Llama-3.1-8B (fp16)
CUDA_VISIBLE_DEVICES=0 python train_checker/train_checker.py \
    --model_path $BASE/Meta-Llama-3.1-8B-Instruct \
    --train_data $TRAIN --val_data $VAL \
    --output_dir checkpoints/checker-llama31-8b \
    --epochs 3 --batch_size 4 --grad_accum 8 --lora_r 32 --fp16

# DeepSeek-R1-Distill-Qwen-7B (bf16)
CUDA_VISIBLE_DEVICES=0 python train_checker/train_checker.py \
    --model_path $BASE/DeepSeek-R1-Distill-Qwen-7B \
    --train_data $TRAIN --val_data $VAL \
    --output_dir checkpoints/checker-deepseek-r1-qwen7b \
    --epochs 3 --batch_size 4 --grad_accum 8 --lora_r 32 --bf16
```

**Hardware**: Each model trains in ~1.5 hours on a single H100 80GB GPU.

### Evaluate student checker

```bash
# 1. Serve the student model
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model checkpoints/checker-qwen3-8b \
    --port 8100 \
    --served-model-name local-checker

# 2. Run benchmark evaluation with student checker
python scripts/run_pipeline_v17.py \
    --freeze_qwen ablation/v10_correct_for_ablation.jsonl \
    --benchmark   ablation/benchmark_step_level_gt.json \
    --output      data/annotated/student_eval \
    --nli_model   cross-encoder/nli-deberta-v3-large \
    --nli_device  0 \
    --checker_model local-checker \
    --checker_base_url http://localhost:8100/v1
```

**Distillation targets**:

| Metric | Target | Excellent |
|--------|--------|-----------|
| Step-level F1 vs human GT | ≥ 65% | ≥ 70% |
| Cohen's κ vs teacher (v17) | ≥ 0.60 | ≥ 0.70 |
| Tree path KL divergence | < 0.3 | < 0.15 |

---

## Version History

| Version | Key Change | F1 |
|---------|-----------|-----|
| v14 | LLM-only baseline (GPT-4.1-mini full pipeline) | 70.0% |
| v15.1 | Add NLI (DeBERTa) for all nodes | 67.2% |
| v15.2 | Rule-based inference steps | 60.1% |
| v15.3 | Data-driven rules (step_type × entity_match × found_quote) | 70.1% |
| v16 | Add necessity test + cross-verify for inference+not_found | 66.7% |
| **v17** | Remove necessity test (logically invalid in multi-hop), keep cross-verify | **72.0%** |

The key finding across versions: **NLI is only reliable on conclusion steps with direct evidence quotes**. Applying NLI to inference steps (search queries) causes over-triggering because search query text is not a logical proposition — it has no entailment relationship with the retrieved document.

---

## Diagnostic Human-Annotation Subset

The legacy diagnostic artifact (`ablation/benchmark_step_level_gt.json`) contains:
- **82 questions** from 2WikiMultihopQA
- **181 steps** generated by Qwen 7B
- Human annotations for each step: `has_gap`, `gap_type`, `human_note`
- Cohen's κ = 0.704 between two annotators

Gap type distribution in human annotations:

| Gap Type | Count | % |
|----------|-------|---|
| `unsupported_claim` | 49 | 46% |
| `insufficient_context` | 26 | 24% |
| `missing_bridge` | 19 | 18% |
| `relation_drift` | 7 | 7% |
| Other | 4 | 4% |

---

## Citation

```bibtex
@article{gap_checker_2025,
  title   = {Step-Gap Guided Search: Evidence Gap Detection for Multi-Hop QA},
  author  = {},
  year    = {2025},
  note    = {Submitted to EMNLP 2026 ARR}
}
```

---

## Hardware Requirements

| Task | GPU | Memory | Time |
|------|-----|--------|------|
| v17 pipeline (1000 questions) | 1× H100 80GB | ~18GB | ~2.5h |
| LoRA SFT (7-8B model) | 1× H100 80GB | ~22GB | ~1.5h |
| vLLM inference (student checker) | 1× H100 80GB | ~16GB | — |
| NLI model (DeBERTa-large) | CPU or GPU | ~1.5GB | — |

---

## Server migration and external resources

This branch contains source code, tests, and lightweight configuration only.
Private experiment data, trajectory and observation bodies, benchmark gold,
model weights, caches, Wikipedia corpora, FAISS indexes, checkpoints, and API
logs are excluded. Keep downloads and generated artifacts outside the Git
checkout on persistent storage.

### Official downloads

- InfoSeek annotations, mappings, images, and Wikipedia metadata:
  [open-vision-language/infoseek](https://github.com/open-vision-language/infoseek).
- BrowseComp-Plus code and documentation:
  [texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus).
- BrowseComp-Plus corpus:
  [Tevatron/browsecomp-plus-corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus),
  pinned here to revision `b27b02bc3e45511b8b82a13e6f90ce761df726f6`.
- FRAMES: [google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark).
- Generator models: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  and [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).
- NLI model: [nli-deberta-v3-large](https://huggingface.co/cross-encoder/nli-deberta-v3-large).
- Wikipedia-2018 E5 corpus/index:
  [Search-R1 retriever documentation](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md).

Example downloads (set `RESOURCE_ROOT` to persistent storage outside this
checkout):

```bash
git clone https://github.com/open-vision-language/infoseek.git \
  "$RESOURCE_ROOT/infoseek"
git clone https://github.com/texttron/BrowseComp-Plus.git \
  "$RESOURCE_ROOT/BrowseComp-Plus"
huggingface-cli download Tevatron/browsecomp-plus-corpus \
  --repo-type dataset \
  --revision b27b02bc3e45511b8b82a13e6f90ce761df726f6 \
  --local-dir "$RESOURCE_ROOT/browsecomp-plus-corpus"
huggingface-cli download google/frames-benchmark \
  --repo-type dataset --local-dir "$RESOURCE_ROOT/frames"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir "$RESOURCE_ROOT/models/Qwen2.5-7B-Instruct"
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir "$RESOURCE_ROOT/models/Qwen3-8B"
huggingface-cli download cross-encoder/nli-deberta-v3-large \
  --local-dir "$RESOURCE_ROOT/models/nli-deberta-v3-large"
```

For the flat Wikipedia-2018 E5 retriever:

```bash
git clone https://github.com/PeterGriffinJin/Search-R1.git \
  "$RESOURCE_ROOT/Search-R1"
python "$RESOURCE_ROOT/Search-R1/scripts/download.py" \
  --save_path "$RESOURCE_ROOT/wiki18"
cat "$RESOURCE_ROOT/wiki18"/part_* > "$RESOURCE_ROOT/wiki18/e5_Flat.index"
gzip -dk "$RESOURCE_ROOT/wiki18/wiki-18.jsonl.gz"
```

### Regenerating derived artifacts

Set `RUN_ROOT` to an external persistent directory. Outputs below intentionally
use a `private` subdirectory and must not be committed.

```bash
python scripts/infoseek_recon.py \
  --data-dir "$INFOSEEK_DATA" --wiki-jsonl "$INFOSEEK_WIKI" \
  --out "$RUN_ROOT/infoseek-recon"

python scripts/convert_infoseek_rft.py \
  --input "$RFT_INPUT" \
  --trajectories "$RUN_ROOT/private/infoseek-trajectories.jsonl" \
  --steps "$RUN_ROOT/private/infoseek-steps.jsonl" \
  --summary "$RUN_ROOT/private/infoseek-conversion-summary.json"

python scripts/evaluate_infoseek_rft.py \
  --input "$RUN_ROOT/private/infoseek-trajectories.jsonl" \
  --output "$RUN_ROOT/private/infoseek-evaluation.jsonl" \
  --summary "$RUN_ROOT/private/infoseek-evaluation-summary.json" \
  --nli-model "$NLI_MODEL"

python scripts/prepare_browsecomp_plus_tasks.py \
  --query-dir "$BROWSECOMP_QUERY_DIR" \
  --agent-output "$RUN_ROOT/private/browsecomp-agent.jsonl" \
  --gold-output "$RUN_ROOT/private/browsecomp-gold.jsonl" \
  --canary "$BROWSECOMP_CANARY"

python scripts/convert_browsecomp_plus_runs.py \
  --input "$BROWSECOMP_RUN" \
  --trajectories "$RUN_ROOT/private/browsecomp-trajectories.jsonl" \
  --observations "$RUN_ROOT/private/browsecomp-observations.jsonl" \
  --source-run official --model "$MODEL" --retriever "$RETRIEVER" \
  --benchmark-version "$BROWSECOMP_VERSION"
```

Use `python scripts/run_infoseek_rft200_shards.py --help` for the sharded
200-example evaluator. Never place credentials in command lines, committed
files, logs, or generated artifacts.
