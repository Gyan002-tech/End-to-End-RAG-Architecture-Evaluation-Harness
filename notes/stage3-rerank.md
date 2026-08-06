# Stage 3 — Multi-Model Staged Reranking & 3x3 Factorial Grid Implementation Report

**Status:** IMPLEMENTED & READY FOR COLAB GPU RUN  
**Date:** 2026-08-06  
**Target Stage:** Stage 3 (Phase 1 Reranking & Pareto Frontier Selection)

---

## 1. Executive Summary

Stage 3 completes Phase 1 of the experimental design specified in `slotA-methodology.md`. It scores all 3 candidate sets generated in Stage 2 (BM25, Dense, RRF Hybrid) through two reranker model champions:
1. `BAAI/bge-reranker-v2-m3` (Cross-Encoder)
2. `BAAI/bge-reranker-v2-gemma` (2.5B parameter causal LLM reranker)

It strictly enforces staged memory residency on the Tesla T4 GPU (loading one model at a time, scoring candidates, and explicitly unloading before loading the next) to remain within the 14.5 GiB VRAM ceiling.

---

## 2. Stage 3 Code Components Implemented

The following files were created in `src/`:

1. **`src/rerankers.py`**
   - Implements `M3Reranker` class wrapping `BAAI/bge-reranker-v2-m3` in fp16 with SDPA attention.
   - Implements `GemmaReranker` class wrapping `BAAI/bge-reranker-v2-gemma` with automatic 4-bit `bitsandbytes` fallback if fp16 exceeds VRAM bounds.
   - Includes `unload_model()` utility for explicit CUDA VRAM cleanup (`del model`, `torch.cuda.empty_cache()`, `gc.collect()`).

2. **`src/03_rerank.py`**
   - **Idempotent & Restartable:** Checks disk for existing output JSON files (`rerank_m3_*.json`, `rerank_gemma_*.json`). If an interrupted run is re-triggered, completed combinations are loaded instantly from disk and skipped without re-running model inference. Pass `--force` to rebuild all.
   - Loads Stage 2 candidate runs (`retrieval_bm25.json`, `retrieval_dense.json`, `retrieval_rrf.json`).
   - Reranks candidate sets with `bge-v2-m3` $\rightarrow$ saves `rerank_m3_*.json` $\rightarrow$ unloads `m3`.
   - Reranks candidate sets with `bge-v2-gemma` $\rightarrow$ saves `rerank_gemma_*.json` $\rightarrow$ unloads `gemma`.
   - Assembles the complete **3x3 Factorial Grid Matrix (9 cells)**.
   - Calculates the Quality-vs-Latency Pareto Frontier and selects the **top 2–3 survivor configurations** for Phase 2.

3. **`src/verify_stage3.py`**
   - Cold reload verifier running in a fresh process.
   - Validates candidate run structures across all 9 cells (300 test queries, 50 candidates/query, docid resolution).
   - Cross-validates metrics against `pytrec_eval` across all 9 cells.
   - Displays the 3x3 Factorial Grid and lists the top Pareto survivors.

---

## 3. Files to Upload to Colab

Synchronize the `src/` directory to your Google Drive project root (`/content/drive/MyDrive/slotA-rag-harness/src/`):

- `src/rerankers.py`
- `src/03_rerank.py`
- `src/verify_stage3.py`

---

## 4. Colab Execution Commands

In your Colab notebook, run the following code blocks to execute Stage 3 and verify the complete 3x3 Factorial Grid:

```python
import os
PROJECT = '/content/drive/MyDrive/slotA-rag-harness'
os.environ['SLOTA_ROOT'] = PROJECT
os.chdir(PROJECT)

# Execute staged multi-model reranking across all 3 candidate sets
!python src/03_rerank.py

# Verify all 9 grid cells, pytrec_eval parity & Pareto frontier in fresh process
!python src/verify_stage3.py
```
