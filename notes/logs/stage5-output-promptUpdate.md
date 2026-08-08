# Output of !python src/05_judge.py --force

==========================================================================
[1] Load SciFact Document Corpus & Stage 3 Rerank Metrics
==========================================================================

==========================================================================
[2] Load Local LLM Judge onto CUDA
==========================================================================
tokenizer_config.json: 7.30kB [00:00, 33.0MB/s]
vocab.json: 2.78MB [00:00, 65.6MB/s]
merges.txt: 1.67MB [00:00, 144MB/s]
tokenizer.json: 7.03MB [00:00, 170MB/s]
config.json: 100% 663/663 [00:00<00:00, 6.03MB/s]
2026-08-08 09:58:47.168471: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:bitsandbytes.cextension:Could not find the bitsandbytes CUDA binary at PosixPath('/usr/local/lib/python3.12/dist-packages/bitsandbytes/libbitsandbytes_cuda128.so')
WARNING:bitsandbytes.cextension:The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  4-bit bitsandbytes load failed (Failed to import transformers.integrations.bitsandbytes because of the following error (look up to see its traceback):
No module named 'triton.ops') — falling back to Qwen2.5-3B-Instruct in fp16...
  Loading judge model: Qwen/Qwen2.5-3B-Instruct (fp16 on CUDA)...
tokenizer_config.json: 7.30kB [00:00, 3.94MB/s]
vocab.json: 2.78MB [00:00, 132MB/s]
merges.txt: 1.67MB [00:00, 135MB/s]
tokenizer.json: 7.03MB [00:00, 182MB/s]
config.json: 100% 661/661 [00:00<00:00, 4.72MB/s]
model.safetensors.index.json: 35.6kB [00:00, 121MB/s]
Downloading shards:   0% 0/2 [00:00<?, ?it/s]
model-00001-of-00002.safetensors:   0% 0.00/3.97G [00:00<?, ?B/s]
model-00001-of-00002.safetensors: 100% 3.97G/3.97G [00:49<00:00, 80.8MB/s]
Downloading shards:  50% 1/2 [00:49<00:49, 49.60s/it]
model-00002-of-00002.safetensors:   0% 0.00/2.20G [00:00<?, ?B/s]
model-00002-of-00002.safetensors:   1% 21.0M/2.20G [00:00<00:17, 125MB/s]
model-00002-of-00002.safetensors: 100% 2.20G/2.20G [00:28<00:00, 77.0MB/s]
Downloading shards: 100% 2/2 [01:18<00:00, 39.32s/it]
Loading checkpoint shards: 100% 2/2 [00:26<00:00, 13.27s/it]
generation_config.json: 100% 242/242 [00:00<00:00, 2.56MB/s]

==========================================================================
[3] Execute LLM Faithfulness Judging across Survivor Runs (Per-Query Auto-Save)
==========================================================================

  Judging answers for: Dense -> none...
  Judging [Dense -> none]: 100% 300/300 [15:14<00:00,  3.05s/query, mean_faithfulness=0.5388]
    Wrote: judge_dense_none.json    (faithfulness: 0.5388, judge latency: 3037.07 ms/query)

  Judging answers for: Dense -> bge-v2-m3...
  Judging [Dense -> bge-v2-m3]: 100% 300/300 [15:32<00:00,  3.11s/query, mean_faithfulness=0.5642]
    Wrote: judge_dense_m3.json      (faithfulness: 0.5642, judge latency: 3096.03 ms/query)

  Judging answers for: RRF Hybrid -> bge-v2-gemma...
  Judging [RRF Hybrid -> bge-v2-gemma]: 100% 300/300 [15:12<00:00,  3.04s/query, mean_faithfulness=0.5713]
    Wrote: judge_rrf_gemma.json     (faithfulness: 0.5713, judge latency: 3028.33 ms/query)

  Judging answers for: Dense -> bge-v2-gemma...
  Judging [Dense -> bge-v2-gemma]: 100% 300/300 [15:19<00:00,  3.06s/query, mean_faithfulness=0.5497]
    Wrote: judge_dense_gemma.json   (faithfulness: 0.5497, judge latency: 3051.01 ms/query)

  Unloading judge model from GPU VRAM...

==========================================================================
[4] Persist Stage 5 Faithfulness Summary & Final Phase 2 Matrix
==========================================================================

Final Phase 2 Pareto Survivor Matrix (Quality vs Latency vs Faithfulness):
Config                       | Retrieval nDCG@10  | Total Latency (ms)   | Faithfulness
-------------------------------------------------------------------------------------
Dense -> none                | 0.0000             | 0.00                 | 0.5388
Dense -> bge-v2-m3           | 0.7420             | 1831.80              | 0.5642
RRF Hybrid -> bge-v2-gemma   | 0.7796             | 6256.78              | 0.5713
Dense -> bge-v2-gemma        | 0.7844             | 6376.81              | 0.5497

==========================================================================
Stage 5 complete — Phase 2 Fully Finished!
==========================================================================
  Persisted final summary: stage5_faithfulness_summary.json
  Next: python src/verify_stage5.py

# Output of !python src/verify_stage5.py


==========================================================================
[1] Cold Reload All 4 Survivor Judgment Artifacts from disk
==========================================================================
  Loaded : judge_dense_none.json          (  332.0 KiB)
  Loaded : judge_dense_m3.json            (  328.7 KiB)
  Loaded : judge_rrf_gemma.json           (  332.3 KiB)
  Loaded : judge_dense_gemma.json         (  329.4 KiB)
  Loaded : stage5_faithfulness_summary.json (    1.2 KiB)
100% 5183/5183 [00:00<00:00, 86907.64it/s]

==========================================================================
[2] Validate Judgment Completeness & Score Bounds across Survivor Runs
==========================================================================
  Dense -> none                : judgments=300 invalid=0 (OK) mean_faithfulness=0.5388
  Dense -> bge-v2-m3           : judgments=300 invalid=0 (OK) mean_faithfulness=0.5642
  RRF Hybrid -> bge-v2-gemma   : judgments=300 invalid=0 (OK) mean_faithfulness=0.5713
  Dense -> bge-v2-gemma        : judgments=300 invalid=0 (OK) mean_faithfulness=0.5497

==========================================================================
[3] Render Final Phase 2 Pareto Survivor Matrix
==========================================================================
Config                       | Retrieval nDCG@10  | Total Latency (ms)   | Faithfulness
-------------------------------------------------------------------------------------
Dense -> none                | 0.0000             | 0.00                 | 0.5388
Dense -> bge-v2-m3           | 0.7420             | 1831.80              | 0.5642
RRF Hybrid -> bge-v2-gemma   | 0.7796             | 6256.78              | 0.5713
Dense -> bge-v2-gemma        | 0.7844             | 6376.81              | 0.5497

==========================================================================
FINAL PROJECT VERDICT
==========================================================================
  All Stage 5 faithfulness judgment checks passed cleanly.
  CONGRATULATIONS! Phase 1 & Phase 2 End-to-End RAG Harness is 100% COMPLETE & VERIFIED!
