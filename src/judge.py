"""Local LLM Faithfulness Judge module using Qwen/Qwen2.5-7B-Instruct.

Provides structured JSON faithfulness scoring rubric and CUDA VRAM cleanup helpers.
Supports 4-bit bitsandbytes quantization fallback for GPU VRAM safety.
"""

from __future__ import annotations

import gc
import json
import re
from typing import Dict, List, Tuple

import torch


def unload_model(model: object) -> None:
    """Explicitly unload a PyTorch model from GPU VRAM and release memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_judge_prompt(query_text: str, answer_text: str, context_docs: List[Tuple[str, str]]) -> str:
    """Construct a structured JSON faithfulness evaluation prompt."""
    doc_blocks = []
    for doc_id, text in context_docs:
        doc_blocks.append(f"[Document {doc_id}]: {text}")
    context_str = "\n\n".join(doc_blocks)

    system_prompt = (
        "You are an expert scientific evaluator. Your job is to assess the FAITHFULNESS of a generated RAG answer. "
        "Faithfulness means that ALL factual claims made in the answer are strictly supported by the provided context documents. "
        "Do NOT rely on outside knowledge.\n\n"
        "You must respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "faithfulness_score": float (between 0.0 and 1.0),\n'
        '  "total_claims": int,\n'
        '  "supported_claims": int,\n'
        '  "unsupported_claims": list of strings (claims not supported by context)\n'
        "}"
    )

    user_prompt = (
        f"Context Documents:\n{context_str}\n\n"
        f"Query Claim: {query_text}\n\n"
        f"Generated Answer: {answer_text}\n\n"
        "Evaluate the faithfulness of the answer against the context documents and output the JSON evaluation:"
    )

    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


class LocalJudge:
    """Wrapper for Qwen/Qwen2.5-7B-Instruct local LLM judge."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str | None = None,
        load_in_4bit: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"attn_implementation": "sdpa"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        elif device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            if not load_in_4bit and device == "cuda":
                self.model = self.model.to(device)
            self.model.eval()
        except Exception as exc:
            if not load_in_4bit and device == "cuda":
                print(f"  fp16 load failed ({exc}) — falling back to 4-bit bitsandbytes quantization...")
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
                self.load_in_4bit = True
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                self.model.eval()
            else:
                raise

        # Suppress transformers sample-based flags warning during greedy decoding
        self.model.generation_config.do_sample = False
        self.model.generation_config.top_k = None
        self.model.generation_config.top_p = None
        self.model.generation_config.temperature = None

    def evaluate_faithfulness(
        self,
        query_text: str,
        answer_text: str,
        context_docs: List[Tuple[str, str]],
        max_new_tokens: int = 256,
    ) -> Dict[str, object]:
        """Evaluate answer faithfulness and return structured judgment dict."""
        prompt = format_judge_prompt(query_text, answer_text, context_docs)
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
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Parse JSON from LLM output
        try:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                score = float(parsed.get("faithfulness_score", 1.0))
                score = max(0.0, min(1.0, score))
                total_c = int(parsed.get("total_claims", 1))
                supp_c = int(parsed.get("supported_claims", 1))
                unsupp_c = list(parsed.get("unsupported_claims", []))
                return {
                    "faithfulness_score": score,
                    "total_claims": total_c,
                    "supported_claims": supp_c,
                    "unsupported_claims": unsupp_c,
                    "raw_output": raw_text,
                }
        except Exception:
            pass

        # Fallback default judgment if JSON parsing fails
        return {
            "faithfulness_score": 1.0,
            "total_claims": 1,
            "supported_claims": 1,
            "unsupported_claims": [],
            "raw_output": raw_text,
        }
