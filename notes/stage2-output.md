# Output of !python src/retrive.py

==========================================================================
[1] Load Stage 1 artifacts & SciFact test dataset
==========================================================================
100% 5183/5183 [00:00<00:00, 99026.45it/s]
  test queries        : 300
  corpus docs         : 5183
  docmap units        : 5661 (dedup_needed=True)
  FAISS ntotal        : 5661
2026-08-06 07:18:24.088518: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

==========================================================================
[2] BM25 Candidate Retrieval (Sparse Arm)
==========================================================================
  BM25 retrieval total: 6.10s (20.28 ms/query avg)

==========================================================================
[3] Dense Candidate Retrieval (BAAI/bge-base-en-v1.5 + Max-Collapse)
==========================================================================
  Dense embed time    : 0.62s (2.08 ms/query)
  Dense FAISS search  : 0.33s (1.11 ms/query avg)
  Total Dense latency : 3.19 ms/query avg

==========================================================================
[4] RRF Hybrid Candidate Retrieval (BM25 + Dense Fusion, k=60)
==========================================================================
  RRF fusion total    : 18.91 ms (0.0630 ms/query avg)

==========================================================================
[5] Compute Hand-Written Retrieval Metrics & Latency Summary
==========================================================================

Phase 1 Retrieval Performance Baseline (3 Arms):
Arm                  | Recall@10  | MRR      | nDCG@10  | P@1    | Latency (ms)
---------------------------------------------------------------------------
BM25 (sparse)        | 0.6862     | 0.5288   | 0.5597   | 0.4367 | 20.28       
Dense (bge-base)     | 0.8709     | 0.7085   | 0.7407   | 0.6200 | 3.19        
RRF Hybrid           | 0.8267     | 0.6395   | 0.6758   | 0.5367 | 23.54       

==========================================================================
[6] Persist Run Artifacts to artifacts/runs/
==========================================================================
  Wrote: retrieval_bm25.json
  Wrote: retrieval_dense.json
  Wrote: retrieval_rrf.json
  Wrote: stage2_retrieval_metrics.json

==========================================================================
Stage 2 complete
==========================================================================
  Candidate runs & evaluation summary persisted cleanly.
  Next: python src/verify_stage2.py


# Output of !python src/verify_stage2.py

==========================================================================
[1] Canary Math Verification
==========================================================================
  Canary metric calculations : PASS

==========================================================================
[2] Cold Reload Candidate Runs from disk
==========================================================================
  Loaded : retrieval_bm25.json (1499.8 KiB)
  Loaded : retrieval_dense.json (1505.0 KiB)
  Loaded : retrieval_rrf.json (1523.3 KiB)
  Loaded : stage2_retrieval_metrics.json (2.0 KiB)
100% 5183/5183 [00:00<00:00, 56813.25it/s]

==========================================================================
[3] Validate Candidate Structure & Docid Resolution
==========================================================================
  BM25         : queries=300  candidates/query=50 (OK)  unresolved_docs=0
  Dense        : queries=300  candidates/query=50 (OK)  unresolved_docs=0
  RRF Hybrid   : queries=300  candidates/query=50 (OK)  unresolved_docs=0

==========================================================================
[4] Metric Library Parity Cross-Validation
==========================================================================

  Checking BM25 arm:
  pytrec_eval parity  : OK (Rec@10: 0.6862, MRR: 0.5288, nDCG@10: 0.5597)
  ranx not installed — skipping library parity check

  Checking Dense arm:
  pytrec_eval parity  : OK (Rec@10: 0.8709, MRR: 0.7085, nDCG@10: 0.7407)
  ranx not installed — skipping library parity check

  Checking RRF Hybrid arm:
  pytrec_eval parity  : OK (Rec@10: 0.8267, MRR: 0.6395, nDCG@10: 0.6758)
  ranx not installed — skipping library parity check

==========================================================================
[5] Phase 1 Baseline Retrieval Performance Table
==========================================================================
Arm                  | Recall@10  | MRR      | nDCG@10  | P@1    | Latency (ms)
---------------------------------------------------------------------------
BM25 (sparse)        | 0.6862     | 0.5288   | 0.5597   | 0.4367 | 20.28       
Dense (bge-base)     | 0.8709     | 0.7085   | 0.7407   | 0.6200 | 3.19        
RRF Hybrid           | 0.8267     | 0.6395   | 0.6758   | 0.5367 | 23.54       

==========================================================================
VERDICT
==========================================================================
  All Stage 2 checks passed. Candidate runs & metrics verified.
  Stage 3 (Reranking) is unblocked.
