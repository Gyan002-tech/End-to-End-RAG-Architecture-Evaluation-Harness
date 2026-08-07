#!/usr/bin/env python3
"""Stage 4 — Local Answer Generation for Pareto Frontier Survivors.

Runs local RAG answer generation using `Qwen/Qwen2.5-1.5B-Instruct` across all 4 surviving
Pareto configurations identified in Stage 3:
  1. Dense (bge-base) -> none
  2. Dense -> bge-v2-m3
  3. RRF Hybrid -> bge-v2-gemma
  4. Dense -> bge-v2-gemma

Includes tqdm progress tracking for batch generation.

Outputs:
  - artifacts/runs/gen_dense_none.json
  - artifacts/runs/gen_dense_m3.json
  - artifacts/runs/gen_rrf_gemma.json
  - artifacts/runs/gen_dense_gemma.json
  - artifacts/runs/stage4_generation_summary.json

Usage:
    python src/04_generate.py
    python src/04_generate.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.common import (  # noqa: E402
    RUNS_DIR,
    SCIFACT_SPLIT,
    load_docmap,
    load_scifact,
)
from src.generator import LocalGenerator, unload_model  # noqa: E402

BANNER = "=" * 74


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def run_generation_for_survivor(
    generator: LocalGenerator,
    survivor_name: str,
    runs: Dict[str, List[dict]],
    queries: Dict[str, str],
    doc_texts: Dict[str, str],
    top_k_context: int = 5,
    max_new_tokens: int = 384,
) -> Tuple[Dict[str, dict], List[float], int]:
    """Generate RAG answers for all 300 test queries in a survivor run with tqdm progress tracking."""
    gen_results: Dict[str, dict] = {}
    latencies: List[float] = []
    truncations_count = 0

    qids = sorted(runs.keys(), key=lambda q: int(q) if q.isdigit() else q)

    pbar = tqdm(qids, desc=f"  Generating [{survivor_name}]", unit="query", leave=True)
    for qid in pbar:
        qtext = queries[qid]
        hits = runs[qid][:top_k_context]

        context_docs: List[Tuple[str, str]] = []
        context_docids: List[str] = []
        for h in hits:
            did = h["doc_id"]
            dtext = doc_texts.get(did, "")
            context_docs.append((did, dtext))
            context_docids.append(did)

        st = time.perf_counter()
        answer, hit_max = generator.generate_answer(
            query_text=qtext,
            context_docs=context_docs,
            max_new_tokens=max_new_tokens,
        )
        dt = (time.perf_counter() - st) * 1000.0  # ms
        latencies.append(dt)

        if hit_max:
            truncations_count += 1

        gen_results[qid] = {
            "query": qtext,
            "answer": answer,
            "context_docids": context_docids,
            "latency_ms": round(dt, 2),
            "hit_max_tokens": hit_max,
            "word_count": len(answer.split()),
        }

        # Update tqdm status with current mean latency
        pbar.set_postfix({"mean_lat": f"{np.mean(latencies):.0f}ms"})

    return gen_results, latencies, truncations_count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k-context", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="force re-running generation")
    args = ap.parse_args()

    # Define the 4 Pareto survivor input run files from Stage 2 & Stage 3
    survivors_config = [
        ("Dense -> none", RUNS_DIR / "retrieval_dense.json", RUNS_DIR / "gen_dense_none.json"),
        ("Dense -> bge-v2-m3", RUNS_DIR / "rerank_m3_dense.json", RUNS_DIR / "gen_dense_m3.json"),
        ("RRF Hybrid -> bge-v2-gemma", RUNS_DIR / "rerank_gemma_rrf.json", RUNS_DIR / "gen_rrf_gemma.json"),
        ("Dense -> bge-v2-gemma", RUNS_DIR / "rerank_gemma_dense.json", RUNS_DIR / "gen_dense_gemma.json"),
    ]

    for name, in_p, out_p in survivors_config:
        if not in_p.exists():
            print(f"Missing required input survivor run file: {in_p.name}")
            print("Run Stage 2 / Stage 3 first.")
            return 1

    hdr("[1] Load SciFact Dataset & Document Corpus")
    docmap = load_docmap()
    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)
    doc_texts = docmap.docid_to_text

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = None
    runs_to_process = []
    summary_rows = []

    # Check for existing cached output JSONs
    for name, in_p, out_p in survivors_config:
        if out_p.exists() and not args.force:
            with open(out_p) as f:
                cached = json.load(f)
            mean_lat = cached.get("mean_gen_latency_ms", 0.0)
            truncs = cached.get("truncations_count", 0)
            print(f"  [SKIPPED - CACHED] {out_p.name:<24} (gen: {mean_lat:.2f} ms/query, truncs: {truncs})")
            summary_rows.append({
                "config": name,
                "output_file": out_p.name,
                "mean_gen_latency_ms": round(mean_lat, 2),
                "truncations_count": truncs,
                "total_queries": len(cached["generations"]),
            })
        else:
            with open(in_p) as f:
                in_data = json.load(f)
            runs_to_process.append((name, in_data["runs"], out_p))

    if runs_to_process:
        hdr("[2] Load Qwen/Qwen2.5-1.5B-Instruct Generator onto CUDA")
        print("  Loading Qwen/Qwen2.5-1.5B-Instruct...")
        generator = LocalGenerator(device=device)

        hdr("[3] Execute Local Answer Generation across Survivor Runs")
        for name, runs, out_p in runs_to_process:
            print(f"\n  Generating answers for: {name}...")
            t0 = time.perf_counter()
            gen_results, latencies, truncs = run_generation_for_survivor(
                generator=generator,
                survivor_name=name,
                runs=runs,
                queries=queries,
                doc_texts=doc_texts,
                top_k_context=args.top_k_context,
                max_new_tokens=args.max_new_tokens,
            )
            total_time = time.perf_counter() - t0
            mean_gen_lat = float(np.mean(latencies))

            out_data = {
                "config": name,
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "top_k_context": args.top_k_context,
                "max_new_tokens": args.max_new_tokens,
                "mean_gen_latency_ms": mean_gen_lat,
                "truncations_count": truncs,
                "generations": gen_results,
            }
            with open(out_p, "w") as f:
                json.dump(out_data, f, indent=2)

            print(f"    Wrote: {out_p.name:<24} (gen latency: {mean_gen_lat:.2f} ms/query, truncations: {truncs})")

            summary_rows.append({
                "config": name,
                "output_file": out_p.name,
                "mean_gen_latency_ms": round(mean_gen_lat, 2),
                "truncations_count": truncs,
                "total_queries": len(gen_results),
            })

        print("\n  Unloading Qwen2.5-1.5B-Instruct from GPU VRAM...")
        unload_model(generator)

    hdr("[4] Persist Stage 4 Generation Summary")
    summary_path = RUNS_DIR / "stage4_generation_summary.json"
    summary_data = {
        "schema_version": 1,
        "stage": "04_generate",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "survivor_generations": summary_rows,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nStage 4 Generation Summary:")
    print(f"{'Config':<28} | {'Output File':<24} | {'Gen Latency (ms)':<18} | {'Truncations'}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['config']:<28} | {r['output_file']:<24} | {r['mean_gen_latency_ms']:<18.2f} | {r['truncations_count']}")

    hdr("Stage 4 complete")
    print(f"  Generated RAG answers for 4 Pareto survivors. Persisted: {summary_path.name}")
    print("  Next: python src/verify_stage4.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
