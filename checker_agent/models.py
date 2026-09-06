from __future__ import annotations

import json
from typing import Any, Protocol, Tuple


class Verbalizer(Protocol):
    def verbalize(self, observation: Any, thought: str) -> str: ...


class NLIModel(Protocol):
    def classify(self, premise: str, hypothesis: str) -> Tuple[str, float]: ...


class DeterministicVerbalizer:
    """Dependency-free, faithful fallback; it adds no inferred facts."""

    def verbalize(self, observation: Any, thought: str = "") -> str:
        if observation is None:
            return ""
        if isinstance(observation, str):
            return observation.strip()
        if isinstance(observation, dict):
            identity = next((f"{k} {v}" for k, v in observation.items() if k.endswith("_id")), "The record")
            facts = [
                f"{k.replace('_', ' ')} is {json.dumps(v, ensure_ascii=False)}"
                for k, v in observation.items() if not k.endswith("_id")
            ]
            return f"{identity}: " + "; ".join(facts) + "." if facts else f"{identity}."
        return str(observation)


class HeuristicNLI:
    """Conservative smoke-test backend. Never use its scores as paper results."""

    def classify(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        p, h = premise.lower().strip(), hypothesis.lower().strip()
        if not p or not h:
            return "neutral", 0.5
        if h in p or p in h:
            return "entailment", 0.75
        opposing = ((" is not ", " is "), (" cannot ", " can "), ("false", "true"))
        if any((a in p and b in h) or (a in h and b in p) for a, b in opposing):
            return "contradiction", 0.7
        return "neutral", 0.55


class TransformersNLI:
    """Lazy Hugging Face three-way NLI adapter."""

    def __init__(self, model_path: str, device: int = 0):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model_path, tokenizer=model_path, device=device)

    def classify(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        result = self.pipe({"text": premise, "text_pair": hypothesis}, top_k=None)
        raw = {str(x["label"]).lower(): float(x["score"]) for x in result}
        aliases = {
            "entailment": ("entailment", "label_2"),
            "neutral": ("neutral", "label_1"),
            "contradiction": ("contradiction", "label_0"),
        }
        scores = {name: max(raw.get(k, 0.0) for k in keys) for name, keys in aliases.items()}
        label = max(scores, key=scores.get)
        return label, scores[label]


class QwenObservationVerbalizer:
    """Optional local Qwen backend used only to verbalize structured observations."""

    def __init__(self, model_path: str, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map=device_map, trust_remote_code=True
        )

    def verbalize(self, observation: Any, thought: str) -> str:
        payload = json.dumps(observation, ensure_ascii=False, sort_keys=True)
        prompt = (
            "Convert this JSON observation into one faithful English sentence of 5-25 words. "
            "Use only literal field values and add no inference. Return only that sentence.\n"
            f"Observation: {payload}\nClaim: {thought}"
        )
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
