# Stage 4 — Local Answer Generation (`Qwen2.5-1.5B-Instruct`) Report

**Status:** IMPLEMENTED & READY FOR COLAB GPU RUN  
**Date:** 2026-08-07  
**Target Stage:** Stage 4 (Phase 2 Local RAG Answer Generation for 4 Pareto Survivors)

---

## 1. Executive Summary

Stage 4 initiates Phase 2 (Generation & Faithfulness) of the RAG evaluation harness. It executes local answer generation using `Qwen/Qwen2.5-1.5B-Instruct` across all 4 surviving Pareto frontier configurations selected in Stage 3:
1. `Dense (bge-base) -> none` (Speed Champion)
2. `Dense -> bge-v2-m3` (Lightweight Rerank Knee)
3. `RRF Hybrid -> bge-v2-gemma` (Balanced Rerank Knee)
4. `Dense -> bge-v2-gemma` (Quality Champion)

---

## 2. Generator Specifications & Prompting

- **Generator Model:** `Qwen/Qwen2.5-1.5B-Instruct` (~3.0 GiB VRAM in fp16 on CUDA with SDPA).
- **Decoding:** Deterministic greedy decoding (`temperature=0.0`, `do_sample=False`, `max_new_tokens=384`).
- **Citation Prompting:** Top-5 context documents formatted as `[Document {doc_id}]: {text}` with instructions to cite evidence inline.
- **Safety Ceiling:** `max_new_tokens = 384` ($\approx 300\text{ words}$ upper safety cap) with automatic EOS termination and truncation auditing in verification.

---

## 3. Files Implemented in `src/`

1. **`src/generator.py`**
   - Wrapper class `LocalGenerator` wrapping `Qwen/Qwen2.5-1.5B-Instruct` with SDPA, fp16 precision, citation prompt builder, and VRAM cleanup helper (`unload_model()`).
2. **`src/04_generate.py`**
   - Runs local answer generation for all 300 test queries across 4 survivor runs (1,200 total answers).
   - Features file-level caching (`[SKIPPED - CACHED]`) for idempotent restartability.
   - Saves `gen_dense_none.json`, `gen_dense_m3.json`, `gen_rrf_gemma.json`, `gen_dense_gemma.json`, and `stage4_generation_summary.json`.
3. **`src/verify_stage4.py`**
   - Fresh process verifier checking 1,200 generated answers for non-empty text, citation tag presence (`[Document ...]`), and output truncation.

---

## 4. Files to Upload to Colab

Synchronize/upload the new `src/` files to your Google Drive project directory (`/content/drive/MyDrive/slotA-rag-harness/src/`):

- `src/generator.py`
- `src/04_generate.py`
- `src/verify_stage4.py`

---

## 5. Colab Execution Commands

Run the following commands in your Colab notebook to execute Stage 4 and verify answer generation:

```python
import os
PROJECT = '/content/drive/MyDrive/slotA-rag-harness'
os.environ['SLOTA_ROOT'] = PROJECT
os.chdir(PROJECT)

# Execute Stage 4 local answer generation for 4 Pareto survivors
!python src/04_generate.py

# Verify generated answer artifacts & citation integrity in a fresh process
!python src/verify_stage4.py
```
