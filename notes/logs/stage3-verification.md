# Stage 3 Factorial Grid & Pareto Frontier Verification Report

**Status:** VERIFIED & PHASE 1 SIGNED OFF  
**Date:** 2026-08-06  
**Target Stage:** Stage 3 (Phase 1 Multi-Model Reranking & Pareto Frontier Assembly)

---

## 1. Complete 3x3 Factorial Grid Matrix (9 Cells Measured)

| Retriever ↓ / Reranker → | None | bge-v2-m3 (Cross-Encoder) | bge-v2-gemma (2.5B LLM) |
|---|---|---|---|
| **BM25** | 0.5597 (20.28 ms) | 0.6693 (1801.40 ms) | 0.6956 (6227.42 ms) |
| **Dense (bge-base)** | **0.7407** (3.19 ms) | **0.7420** (1831.80 ms) | **0.7844** (6376.81 ms) |
| **RRF Hybrid** | 0.6758 (23.54 ms) | 0.7347 (1839.97 ms) | **0.7796** (6256.78 ms) |

*(Format: nDCG@10 score with total query latency in parentheses)*

---

## 2. Quality-vs-Latency Pareto Frontier Analysis

| Rank | Configuration | Recall@10 | MRR | nDCG@10 | P@1 | Total Latency (ms) | Frontier Role |
|---|---|---|---|---|---|---|---|
| 1 | **Dense -> bge-v2-gemma** | **0.9092** | **0.7508** | **0.7844** | **0.6667** | 6376.81 ms | **★ Quality Champion** |
| 2 | **RRF Hybrid -> bge-v2-gemma** | 0.9059 | 0.7453 | 0.7796 | 0.6633 | 6256.78 ms | **★ Balanced Rerank Knee** |
| 3 | **Dense -> bge-v2-m3** | 0.8559 | 0.7166 | 0.7420 | 0.6367 | 1831.80 ms | **★ Lightweight Rerank Knee** |
| 4 | **Dense (bge-base) -> none** | 0.8709 | 0.7085 | 0.7407 | 0.6200 | **3.19 ms** | **★ Speed Champion** |
| 5 | RRF Hybrid -> bge-v2-m3 | 0.8512 | 0.7076 | 0.7347 | 0.6267 | 1839.97 ms | Dominated |
| 6 | BM25 -> bge-v2-gemma | 0.7668 | 0.6796 | 0.6956 | 0.6200 | 6227.42 ms | Dominated |
| 7 | RRF Hybrid -> none | 0.8267 | 0.6395 | 0.6758 | 0.5367 | 23.54 ms | Dominated |
| 8 | BM25 -> bge-v2-m3 | 0.7462 | 0.6518 | 0.6693 | 0.5900 | 1801.40 ms | Dominated |
| 9 | BM25 (sparse) -> none | 0.6862 | 0.5288 | 0.5597 | 0.4367 | 20.28 ms | Dominated |

---

## 3. Key Findings & Interview Talking Points

1. **Frontier Reranker Uplift (+4.37 nDCG@10 points):**  
   `bge-v2-gemma` on Dense candidates elevated nDCG@10 from **0.7407 to 0.7844** (+4.37 points), Recall@10 from **0.8709 to 0.9092**, and P@1 from **0.6200 to 0.6667**. This quality gain requires a **2,000x latency multiplier** (3.19 ms -> 6376.81 ms).

2. **Reranker Rescues BM25 Candidates (+13.59 points):**  
   Reranking BM25 candidates with `bge-v2-gemma` boosted nDCG@10 from 0.5597 to 0.6956 (+13.59 points), proving that strong cross-encoders/LLMs can salvage keyword-only initial retrieval.

3. **RRF Hybrid Recovery under LLM Reranking:**  
   While `RRF Hybrid -> none` (0.6758) was dominated by `Dense -> none` (0.7407) in Stage 2, adding `bge-v2-gemma` brought RRF Hybrid to **0.7796 nDCG@10**, earning it a spot on the Pareto Frontier.

---

## 4. Phase 1 Verification & Sign-Off

- **Artifact Cold Reload:** All 9 run JSONs + `stage3_rerank_metrics.json` loaded cleanly.
- **Candidate Integrity:** 300 test queries, 50 candidates/query, 0 unresolved docids.
- **Library Parity (`pytrec_eval`):** 100% numerical match across all 9 cells.

Phase 1 (Retrieval & Reranking Factorial Grid) is **100% COMPLETE**. Proceeding to Phase 2 (Generation & LLM-as-Judge Faithfulness Scoring).
