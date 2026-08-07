# Output of !python src/05_judge.py

==========================================================================
[1] Load SciFact Document Corpus & Stage 3 Rerank Metrics
==========================================================================

==========================================================================
[2] Load Local LLM Judge onto CUDA
==========================================================================
2026-08-07 16:21:25.773976: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:bitsandbytes.cextension:Could not find the bitsandbytes CUDA binary at PosixPath('/usr/local/lib/python3.12/dist-packages/bitsandbytes/libbitsandbytes_cuda128.so')
WARNING:bitsandbytes.cextension:The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  4-bit bitsandbytes load failed (Failed to import transformers.integrations.bitsandbytes because of the following error (look up to see its traceback):
No module named 'triton.ops') — falling back to Qwen2.5-3B-Instruct in fp16...
  Loading judge model: Qwen/Qwen2.5-3B-Instruct (fp16 on CUDA)...
tokenizer_config.json: 7.30kB [00:00, 4.98MB/s]
vocab.json: 2.78MB [00:00, 44.0MB/s]
merges.txt: 1.67MB [00:00, 94.6MB/s]
tokenizer.json: 7.03MB [00:00, 149MB/s]
config.json: 100% 661/661 [00:00<00:00, 3.69MB/s]
model.safetensors.index.json: 35.6kB [00:00, 115MB/s]
Downloading shards:   0% 0/2 [00:00<?, ?it/s]
Downloading shards: 100% 2/2 [01:28<00:00, 44.19s/it]
Loading checkpoint shards: 100% 2/2 [00:26<00:00, 13.11s/it]
generation_config.json: 100% 242/242 [00:00<00:00, 2.32MB/s]

==========================================================================
[3] Execute LLM Faithfulness Judging across Survivor Runs (Per-Query Auto-Save)
==========================================================================

  Judging answers for: Dense -> none...
  Judging [Dense -> none]: 100% 300/300 [15:34<00:00,  3.11s/query, mean_faithfulness=0.4825]
    Wrote: judge_dense_none.json    (faithfulness: 0.4825, judge latency: 3103.72 ms/query)

  Judging answers for: Dense -> bge-v2-m3...
  Judging [Dense -> bge-v2-m3]: 100% 300/300 [15:22<00:00,  3.08s/query, mean_faithfulness=0.5002]
    Wrote: judge_dense_m3.json      (faithfulness: 0.5002, judge latency: 3064.93 ms/query)

  Judging answers for: RRF Hybrid -> bge-v2-gemma...
  Judging [RRF Hybrid -> bge-v2-gemma]: 100% 300/300 [15:12<00:00,  3.04s/query, mean_faithfulness=0.5183]
    Wrote: judge_rrf_gemma.json     (faithfulness: 0.5183, judge latency: 3031.63 ms/query)

  Judging answers for: Dense -> bge-v2-gemma...
  Judging [Dense -> bge-v2-gemma]: 100% 300/300 [15:06<00:00,  3.02s/query, mean_faithfulness=0.5425]
    Wrote: judge_dense_gemma.json   (faithfulness: 0.5425, judge latency: 3011.82 ms/query)

  Unloading judge model from GPU VRAM...

==========================================================================
[4] Persist Stage 5 Faithfulness Summary & Final Phase 2 Matrix
==========================================================================

Final Phase 2 Pareto Survivor Matrix (Quality vs Latency vs Faithfulness):
Config                       | Retrieval nDCG@10  | Total Latency (ms)   | Faithfulness
-------------------------------------------------------------------------------------
Dense -> none                | 0.0000             | 0.00                 | 0.4825
Dense -> bge-v2-m3           | 0.7420             | 1831.80              | 0.5002
RRF Hybrid -> bge-v2-gemma   | 0.7796             | 6256.78              | 0.5183
Dense -> bge-v2-gemma        | 0.7844             | 6376.81              | 0.5425

==========================================================================
Stage 5 complete — Phase 2 Fully Finished!
==========================================================================
  Persisted final summary: stage5_faithfulness_summary.json
  Next: python src/verify_stage5.py

# Output of !python src/verify_stage5.py

==========================================================================
[1] Cold Reload All 4 Survivor Judgment Artifacts from disk
==========================================================================
  Loaded : judge_dense_none.json          (  330.0 KiB)
  Loaded : judge_dense_m3.json            (  327.8 KiB)
  Loaded : judge_rrf_gemma.json           (  332.1 KiB)
  Loaded : judge_dense_gemma.json         (  328.3 KiB)
  Loaded : stage5_faithfulness_summary.json (    1.2 KiB)
100% 5183/5183 [00:00<00:00, 95327.18it/s]

==========================================================================
[2] Validate Judgment Completeness & Score Bounds across Survivor Runs
==========================================================================
  Dense -> none                : judgments=300 invalid=0 (OK) mean_faithfulness=0.4825
  Dense -> bge-v2-m3           : judgments=300 invalid=0 (OK) mean_faithfulness=0.5002
  RRF Hybrid -> bge-v2-gemma   : judgments=300 invalid=0 (OK) mean_faithfulness=0.5183
  Dense -> bge-v2-gemma        : judgments=300 invalid=0 (OK) mean_faithfulness=0.5425

==========================================================================
[3] Render Final Phase 2 Pareto Survivor Matrix
==========================================================================
Config                       | Retrieval nDCG@10  | Total Latency (ms)   | Faithfulness
-------------------------------------------------------------------------------------
Dense -> none                | 0.0000             | 0.00                 | 0.4825
Dense -> bge-v2-m3           | 0.7420             | 1831.80              | 0.5002
RRF Hybrid -> bge-v2-gemma   | 0.7796             | 6256.78              | 0.5183
Dense -> bge-v2-gemma        | 0.7844             | 6376.81              | 0.5425

==========================================================================
FINAL PROJECT VERDICT
==========================================================================
  All Stage 5 faithfulness judgment checks passed cleanly.
  CONGRATULATIONS! Phase 1 & Phase 2 End-to-End RAG Harness is 100% COMPLETE & VERIFIED!
