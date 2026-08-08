# Comparing BAAI/bge-base-en-v1.5 Vs intfloat/e5-large-v2

==========================================================================
[1] Cold Reload e5 Artifacts from disk
==========================================================================
  Loaded : faiss_e5.index           (20732.0 KiB)
  Loaded : docmap_e5.json           ( 7851.2 KiB)
  Loaded : index_meta_e5.json       (    0.5 KiB)
  Loaded : retrieval_e5_dense.json  ( 1504.9 KiB)
  Loaded : e5_vs_bge_comparison.json (    0.9 KiB)
100% 5183/5183 [00:00<00:00, 78855.06it/s]

==========================================================================
[2] FAISS & Docmap Assertions (e5-large-v2)
==========================================================================
  index.ntotal        : 5183 (expect 5183)
  index.d             : 1024 (expect 1024 for e5-large)

==========================================================================
[3] Candidate Run Structure & Docid Resolution
==========================================================================
  queries in run      : 300
  candidates/query=50 : OK
  unresolved docids   : 0

==========================================================================
[4] Dense Embedding Model Comparison Baseline
==========================================================================
Model                        | Recall@10  | MRR      | nDCG@10  | P@1   
------------------------------------------------------------------------
BAAI/bge-base-en-v1.5        | 0.8709     | 0.7085   | 0.7407   | 0.6200
intfloat/e5-large-v2         | 0.8438     | 0.6936   | 0.7230   | 0.6067

==========================================================================
VERDICT
==========================================================================
  All e5 dense embedding checks passed cleanly.


# Comparing BAAI/bge-base-en-v1.5 Vs BAAI/bge-small-en-v1.5


==========================================================================
[1] Cold Reload bge-small Artifacts from disk
==========================================================================
  Loaded : faiss_bge_small.index          ( 7774.5 KiB)
  Loaded : docmap_bge_small.json          ( 7851.2 KiB)
  Loaded : index_meta_bge_small.json      (    0.5 KiB)
  Loaded : retrieval_bge_small_dense.json ( 1505.0 KiB)
  Loaded : dense_model_size_sweetspot.json (    0.9 KiB)
100% 5183/5183 [00:00<00:00, 92285.23it/s]

==========================================================================
[2] FAISS & Docmap Assertions (bge-small-en-v1.5)
==========================================================================
  index.ntotal        : 5183 (expect 5183)
  index.d             : 384 (expect 384 for bge-small)

==========================================================================
[3] Candidate Run Structure & Docid Resolution
==========================================================================
  queries in run      : 300
  candidates/query=50 : OK
  unresolved docids   : 0

==========================================================================
[4] Render 3-Tier Dense Model Size Sweet-Spot Benchmark
==========================================================================
Tier                   | Model                    | Dim   | Recall@10  | MRR      | nDCG@10  | P@1   
-----------------------------------------------------------------------------------------------
Small (33M params)     | BAAI/bge-small-en-v1.5   | 384   | 0.8362     | 0.6864   | 0.7127   | 0.6067
Base (109M params) ★   | BAAI/bge-base-en-v1.5    | 768   | 0.8709     | 0.7085   | 0.7407   | 0.6200
Large (335M params)    | intfloat/e5-large-v2     | 1024  | 0.8438     | 0.6936   | 0.7230   | 0.6067

==========================================================================
VERDICT
==========================================================================
  All bge-small dense embedding checks passed cleanly.
