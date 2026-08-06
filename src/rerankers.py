"""Reranker wrapper module for Stage 3 multi-model staged reranking.

Supports:
  1. BAAI/bge-reranker-v2-m3 (Cross-Encoder champion)
  2. BAAI/bge-reranker-v2-gemma (2.5B LLM decoder reranker champion)

Enforces VRAM safety and fp16 precision on CUDA. Supports automatic dynamic batch size
reduction upon OOM and 4-bit bitsandbytes fallback for gemma.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Tuple

import torch


def unload_model(model: object) -> None:
    """Explicitly unload a PyTorch model from GPU VRAM and release memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class M3Reranker:
    """Wrapper for BAAI/bge-reranker-v2-m3 (Cross-Encoder)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.use_fp16 = (device == "cuda")

        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(
                model_name_or_path=model_name,
                use_fp16=self.use_fp16,
                device=device,
            )
            self._backend = "flagembedding"
        except Exception as exc:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_kwargs = {"attn_implementation": "sdpa"}
            if device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **model_kwargs
            ).to(device)
            self.model.eval()
            self._backend = "transformers"

    def compute_scores(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[float]:
        """Compute reranking scores for list of (query, passage) pairs."""
        if not pairs:
            return []

        if self._backend == "flagembedding":
            pair_list = [[q, p] for q, p in pairs]
            raw_scores = self.reranker.compute_score(pair_list, batch_size=batch_size, normalize=True)
            if isinstance(raw_scores, (int, float)):
                return [float(raw_scores)]
            return [float(s) for s in raw_scores]

        # Transformers fallback path
        scores: List[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            queries = [b[0] for b in batch]
            passages = [b[1] for b in batch]
            inputs = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                if logits.shape[1] == 1:
                    batch_scores = logits.squeeze(-1).tolist()
                else:
                    batch_scores = logits[:, 0].tolist()
                scores.extend([float(s) for s in (batch_scores if isinstance(batch_scores, list) else [batch_scores])])
        return scores


class GemmaReranker:
    """Wrapper for BAAI/bge-reranker-v2-gemma (2.5B LLM Reranker)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-gemma",
        device: str | None = None,
        load_in_4bit: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.use_fp16 = (device == "cuda") and not load_in_4bit
        self.load_in_4bit = load_in_4bit

        try:
            from FlagEmbedding import FlagLLMReranker
            self.reranker = FlagLLMReranker(
                model_name_or_path=model_name,
                use_fp16=self.use_fp16,
                device=device,
                load_in_4bit=load_in_4bit,
            )
            self._backend = "flagembedding"
        except Exception as exc:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_kwargs = {"attn_implementation": "sdpa"}
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            elif device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
            if not load_in_4bit and device == "cuda":
                self.model = self.model.to(device)
            self.model.eval()
            self._backend = "transformers"

    def compute_scores(self, pairs: List[Tuple[str, str]], batch_size: int = 4) -> List[float]:
        """Compute reranking scores for list of (query, passage) pairs with OOM auto-recovery."""
        if not pairs:
            return []

        current_bs = batch_size
        while current_bs >= 1:
            try:
                if self._backend == "flagembedding":
                    pair_list = [[q, p] for q, p in pairs]
                    raw_scores = self.reranker.compute_score(pair_list, batch_size=current_bs, normalize=True)
                    if isinstance(raw_scores, (int, float)):
                        return [float(raw_scores)]
                    return [float(s) for s in raw_scores]

                # Transformers fallback path
                scores: List[float] = []
                for i in range(0, len(pairs), current_bs):
                    batch = pairs[i : i + current_bs]
                    pair_texts = [
                        f"Given a query '{q}', is the following passage relevant to the query?\nPassage: '{p}'"
                        for q, p in batch
                    ]
                    inputs = self.tokenizer(
                        pair_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits[:, -1, :]  # score from final token
                        batch_scores = logits.max(dim=-1).values.tolist()
                        scores.extend([float(s) for s in (batch_scores if isinstance(batch_scores, list) else [batch_scores])])
                return scores

            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower() and current_bs > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    current_bs = max(1, current_bs // 2)
                    print(f"\n  [VRAM OOM AUTO-RECOVERY] Reduced Gemma batch size to {current_bs} and retrying query...")
                else:
                    raise
        raise RuntimeError("Gemma reranker failed due to persistent OutOfMemoryError even at batch_size=1.")
