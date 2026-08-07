"""Local Answer Generator module using Qwen/Qwen2.5-1.5B-Instruct.

Provides citation-aware RAG prompt construction, greedy deterministic decoding (fp16, SDPA),
and CUDA VRAM cleanup helpers.
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


def format_citation_prompt(query_text: str, context_docs: List[Tuple[str, str]]) -> str:
    """Construct a citation-aware RAG prompt for scientific claim answering."""
    doc_blocks = []
    for doc_id, text in context_docs:
        doc_blocks.append(f"[Document {doc_id}]: {text}")
    context_str = "\n\n".join(doc_blocks)

    system_prompt = (
        "You are an expert scientific assistant. Your task is to answer the user's scientific claim query "
        "based ONLY on the provided context documents. Cite source documents inline as [Document doc_id]. "
        "If the context does not contain enough information to verify or answer the claim, state that clearly."
    )

    user_prompt = f"Context Documents:\n{context_str}\n\nQuery Claim: {query_text}\n\nAnswer:"
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


class LocalGenerator:
    """Wrapper for Qwen/Qwen2.5-1.5B-Instruct local causal generator."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"attn_implementation": "sdpa"}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        ).to(device)
        self.model.eval()

        # Suppress transformers sample-based flags warning during greedy decoding
        self.model.generation_config.do_sample = False
        self.model.generation_config.top_k = None
        self.model.generation_config.top_p = None
        self.model.generation_config.temperature = None

    def generate_answer(
        self,
        query_text: str,
        context_docs: List[Tuple[str, str]],
        max_new_tokens: int = 384,
    ) -> Tuple[str, bool]:
        """Generate answer for query and context documents using greedy decoding.

        Returns:
            Tuple of (generated_answer_text, hit_max_tokens_flag)
        """
        prompt = format_citation_prompt(query_text, context_docs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0][input_len:]
        generated_tokens = len(gen_ids)
        hit_max = (generated_tokens >= max_new_tokens)

        answer_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return answer_text, hit_max
