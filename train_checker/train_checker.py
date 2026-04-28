#!/usr/bin/env python3
"""
train_checker.py
================
LoRA SFT training for Qwen3-8B gap checker (student model).

Trains Qwen3-8B to reproduce v17 checker behavior via chain-of-thought distillation.

Requirements:
  pip install transformers peft accelerate bitsandbytes datasets trl

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/train_checker.py \
      --model_path /ocean/projects/med230010p/yji3/models/Qwen3-8B \
      --train_data data/sft/checker_train.jsonl \
      --val_data   data/sft/checker_train_val.jsonl \
      --output_dir checkpoints/checker-qwen3-8b \
      --epochs 3 \
      --batch_size 4 \
      --grad_accum 8

For Qwen2.5-7B (already on your server):
  CUDA_VISIBLE_DEVICES=1 python scripts/train_checker.py \
      --model_path /ocean/projects/med230010p/yji3/models/Qwen2.5-7B-Instruct \
      --train_data data/sft/checker_train.jsonl \
      --val_data   data/sft/checker_train_val.jsonl \
      --output_dir checkpoints/checker-qwen25-7b \
      --epochs 3
"""

import os
import json
import argparse
import torch
from typing import Optional

# ── Silence tokenizer warnings ───────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_jsonl(path: str):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_chat_sample(sample: dict, tokenizer) -> str:
    """
    Format one SFT sample into the model's chat template.
    Uses /no_think for Qwen3 to disable chain-of-thought mode.
    """
    system  = sample.get("system", "You are a gap detector.")
    user    = sample.get("instruction", "")
    assist  = sample.get("response", "")

    # Qwen3 non-thinking mode: append /no_think to system prompt
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assist},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            return text
        except Exception:
            pass

    # Fallback format
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assist}"


class CheckerDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, max_length: int = 2048):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = []

        for s in samples:
            text = format_chat_sample(s, tokenizer)
            enc  = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            # Only compute loss on assistant response
            input_ids = enc["input_ids"]
            labels    = self._mask_input_labels(input_ids, tokenizer)

            # Weight hard samples 2x
            weight = 2.0 if s.get("metadata", {}).get("difficulty") == "hard" else 1.0

            self.data.append({
                "input_ids":      input_ids,
                "attention_mask": enc["attention_mask"],
                "labels":         labels,
                "weight":         weight,
            })

    def _mask_input_labels(self, input_ids, tokenizer):
        """
        Set labels=-100 for everything except the assistant's response.
        This ensures loss is only computed on what the model should generate.
        """
        labels = list(input_ids)

        # Find assistant token to mask everything before it
        assist_token_strs = ["<|assistant|>", "[/INST]", "<|im_start|>assistant"]
        text = tokenizer.decode(input_ids)

        # Find the last occurrence of assistant marker
        last_assist_pos = -1
        for marker in assist_token_strs:
            if marker in text:
                # Encode up to the marker to find token position
                prefix = text[:text.rfind(marker) + len(marker)]
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                last_assist_pos = max(last_assist_pos, len(prefix_ids))

        if last_assist_pos > 0:
            for i in range(last_assist_pos):
                labels[i] = -100

        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids      = []
    attention_masks = []
    labels_list    = []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [0] * pad_len)
        attention_masks.append(x["attention_mask"] + [0] * pad_len)
        labels_list.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids":      torch.tensor(input_ids,       dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels":         torch.tensor(labels_list,     dtype=torch.long),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  type=str, required=True)
    p.add_argument("--train_data",  type=str, required=True)
    p.add_argument("--val_data",    type=str, default=None)
    p.add_argument("--output_dir",  type=str, required=True)

    # Training hyperparams
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--grad_accum",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--max_length",  type=int,   default=2048)
    p.add_argument("--warmup_ratio",type=float, default=0.05)

    # LoRA params
    p.add_argument("--lora_r",      type=int,   default=32)
    p.add_argument("--lora_alpha",  type=int,   default=64)
    p.add_argument("--lora_dropout",type=float, default=0.05)

    # System
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--fp16",        action="store_true", default=True)
    p.add_argument("--bf16",        action="store_true", default=False)
    p.add_argument("--no_think",    action="store_true", default=False,
                   help="Append /no_think to system prompt (Qwen3) or "
                        "'Respond only in JSON' (DeepSeek-R1) to suppress thinking output")
    args = p.parse_args()

    # Auto-detect model family for system prompt override
    model_lower = args.model_path.lower()
    if "qwen3" in model_lower or args.no_think:
        SYSTEM_PROMPT_OVERRIDE = "You are a strict step-level evidence gap detector. /no_think"
        print("[INFO] Qwen3 detected — appending /no_think to system prompt")
    elif "deepseek-r1" in model_lower or "r1-distill" in model_lower:
        SYSTEM_PROMPT_OVERRIDE = (
            "You are a strict step-level evidence gap detector. "
            "Respond ONLY with a valid JSON object. Do NOT produce any thinking, "
            "reasoning, or <think> blocks. Output JSON immediately."
        )
        print("[INFO] DeepSeek-R1 detected — suppressing thinking in system prompt")
    else:
        SYSTEM_PROMPT_OVERRIDE = None  # use default from data

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model & tokenizer ────────────────────────────────────────────────
    print(f"[INFO] Loading model: {args.model_path}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA config ───────────────────────────────────────────────────────────
    # Target all attention and MLP projection layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"[INFO] Loading train data: {args.train_data}")
    train_samples = load_jsonl(args.train_data)
    # Apply system prompt override if needed
    if SYSTEM_PROMPT_OVERRIDE:
        for s in train_samples:
            s["system"] = SYSTEM_PROMPT_OVERRIDE
    train_dataset = CheckerDataset(train_samples, tokenizer, args.max_length)
    print(f"[INFO] Train samples: {len(train_dataset)}")

    val_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        val_samples = load_jsonl(args.val_data)
        if SYSTEM_PROMPT_OVERRIDE:
            for s in val_samples:
                s["system"] = SYSTEM_PROMPT_OVERRIDE
        val_dataset = CheckerDataset(val_samples, tokenizer, args.max_length)
        print(f"[INFO] Val samples: {len(val_dataset)}")

    # ── Training ──────────────────────────────────────────────────────────────
    from transformers import TrainingArguments, Trainer

    total_steps = (len(train_dataset) // (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=val_dataset is not None,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print(f"[INFO] Starting training...")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size} × {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    print(f"  LR:            {args.lr}")
    print(f"  LoRA r:        {args.lora_r}  alpha: {args.lora_alpha}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Warmup steps:  {warmup_steps}")

    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[INFO] Model saved to: {args.output_dir}")
    print(f"[INFO] To run inference:")
    print(f"  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \\")
    print(f"      --model {args.output_dir} \\")
    print(f"      --port 8100 \\")
    print(f"      --served-model-name local-checker")


if __name__ == "__main__":
    main()