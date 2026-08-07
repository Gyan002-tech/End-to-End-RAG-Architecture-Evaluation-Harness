# Stage 5 — Local LLM-as-a-Judge Faithfulness Scoring (`Qwen2.5-7B-Instruct`) Report

**Status:** IMPLEMENTED & READY FOR COLAB GPU RUN  
**Date:** 2026-08-07  
**Target Stage:** Stage 5 (Phase 2 Final Stage — Local LLM Faithfulness Judging for 4 Pareto Survivors)

---

## 1. Executive Summary

Stage 5 completes Phase 2 (Generation & Faithfulness) of the RAG evaluation harness. It executes local LLM-as-a-Judge evaluation using `Qwen/Qwen2.5-7B-Instruct` across all 1,200 generated RAG answers (300 test queries x 4 Pareto survivor runs):
1. `Dense (bge-base) -> none` (Speed Champion)
2. `Dense -> bge-v2-m3` (Lightweight Rerank Knee)
3. `RRF Hybrid -> bge-v2-gemma` (Balanced Rerank Knee)
4. `Dense -> bge-v2-gemma` (Quality Champion)

---

## 2. Judge Specifications & Per-Query Auto-Save Architecture

- **Judge Model:** `Qwen/Qwen2.5-7B-Instruct` (~14.0 GiB VRAM in fp16 on CUDA, with automatic 4-bit `bitsandbytes` fallback for T4 16GB VRAM safety).
- **Per-Query Auto-Saving & Partial Resume:**
  Saves results to disk after **EVERY query**. If a Colab session disconnects or times out mid-run, re-running `!python src/05_judge.py` automatically detects completed queries in `judge_{config_slug}.json` and resumes from where it left off!
- **Faithfulness Rubric:**
  Deconstructs generated answers into factual claims, verifies context support, and outputs structured JSON:
  $$\text{Faithfulness} = \frac{\text{Supported Claims}}{\text{Total Claims}}$$

---

## 3. Files Implemented in `src/`

1. **`src/judge.py`**
   - Wrapper class `LocalJudge` wrapping `Qwen/Qwen2.5-7B-Instruct` with automatic 4-bit quantization fallback, structured JSON output parser, and VRAM cleanup helper (`unload_model()`).
2. **`src/05_judge.py`**
   - Runs local LLM faithfulness judging across 4 survivor runs with per-query auto-saving and partial-resume support.
   - Saves `judge_dense_none.json`, `judge_dense_m3.json`, `judge_rrf_gemma.json`, `judge_dense_gemma.json`, and `stage5_faithfulness_summary.json`.
3. **`src/verify_stage5.py`**
   - Fresh process verifier checking all 1,200 faithfulness judgments and rendering the **Final Phase 2 Pareto Survivor Matrix**.

---

## 4. Files to Upload to Colab

Synchronize/upload the new `src/` files to your Google Drive project directory (`/content/drive/MyDrive/slotA-rag-harness/src/`):

- `src/04_generate.py` (updated with per-query auto-save & partial resume)
- `src/judge.py`
- `src/05_judge.py`
- `src/verify_stage5.py`

---

## 5. Colab Execution Commands

Run the following commands in your Colab notebook to execute Stage 5 and verify final Phase 2 results:

```python
import os
PROJECT = '/content/drive/MyDrive/slotA-rag-harness'
os.environ['SLOTA_ROOT'] = PROJECT
os.chdir(PROJECT)

# Execute Stage 5 local LLM faithfulness judging for 4 Pareto survivors (with per-query auto-save & resume)
!python src/05_judge.py

# Verify final Phase 2 results & display summary table in a fresh process
!python src/verify_stage5.py
```
