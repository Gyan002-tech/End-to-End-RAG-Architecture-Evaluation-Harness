#!/usr/bin/env python3
"""Stage 5 — Local LLM-as-a-Judge Faithfulness Scoring for Pareto Survivors.

Runs local faithfulness judging using `Qwen/Qwen2.5-7B-Instruct` (with automatic `Qwen2.5-3B-Instruct` fp16 fallback)
across all 4 surviving Pareto configurations identified in Stage 3:
  1. Dense (bge-base) -> none
  2. Dense -> bge-v2-m3
  3. RRF Hybrid -> bge-v2-gemma
  4. Dense -> bge-v2-gemma

PER-QUERY AUTO-SAVING & RESUME:
  Saves results to disk after EVERY query. If interrupted mid-run, re-running automatically
  detects completed queries and resumes from where it left off. Pass `--force` to restart.

Outputs:
  - artifacts/runs/judge_dense_none.json
  - artifacts/runs/judge_dense_m3.json
  - artifacts/runs/judge_rrf_gemma.json
  - artifacts/runs/judge_dense_gemma.json
  - artifacts/runs/stage5_faithfulness_summary.json

Usage:
    python src/05_judge.py
    python src/05_judge.py --force
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.engine.common import (  # noqa: E402
    RUNS_DIR,
    SCIFACT_SPLIT,
    load_docmap,
    load_scifact,
)
from src.engine.judge import LocalJudge, unload_model  # noqa: E402

BANNER = "=" * 74


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def run_judgement_for_survivor(
    judge: LocalJudge,
    survivor_name: str,
    gen_data: dict,
    doc_texts: Dict[str, str],
    out_path: Path,
    force: bool = False,
) -> Tuple[Dict[str, dict], List[float], List[float]]:
    """Judge RAG answer faithfulness with per-query auto-saving and partial-resume capability."""
    judge_results: Dict[str, dict] = {}
    latencies: List[float] = []
    faithfulness_scores: List[float] = []

    # Load existing partial judgments if out_path exists and not force
    if out_path.exists() and not force:
        try:
            with open(out_path) as f:
                existing_data = json.load(f)
            judge_results = existing_data.get("judgments", {})
            for entry in judge_results.values():
                if "judge_latency_ms" in entry:
                    latencies.append(entry["judge_latency_ms"])
                if "faithfulness_score" in entry:
                    faithfulness_scores.append(entry["faithfulness_score"])
            if judge_results:
                print(f"  [PARTIAL RESUME] Loaded {len(judge_results)} completed judgments from {out_path.name}")
        except Exception:
            judge_results = {}
            latencies = []
            faithfulness_scores = []

    generations = gen_data["generations"]
    qids = sorted(generations.keys(), key=lambda q: int(q) if q.isdigit() else q)

    pbar = tqdm(qids, desc=f"  Judging [{survivor_name}]", unit="query", leave=True)
    for qid in pbar:
        if qid in judge_results and not force:
            continue  # Skip already completed query judgment

        gen_entry = generations[qid]
        qtext = gen_entry["query"]
        answer = gen_entry["answer"]
        context_docids = gen_entry["context_docids"]

        context_docs: List[Tuple[str, str]] = [
            (did, doc_texts.get(did, "")) for did in context_docids
        ]

        st = time.perf_counter()
        eval_dict = judge.evaluate_faithfulness(
            query_text=qtext,
            answer_text=answer,
            context_docs=context_docs,
        )
        dt = (time.perf_counter() - st) * 1000.0  # ms
        latencies.append(dt)

        f_score = float(eval_dict["faithfulness_score"])
        faithfulness_scores.append(f_score)

        judge_results[qid] = {
            "query": qtext,
            "answer": answer,
            "context_docids": context_docids,
            "faithfulness_score": f_score,
            "total_claims": eval_dict["total_claims"],
            "supported_claims": eval_dict["supported_claims"],
            "unsupported_claims": eval_dict["unsupported_claims"],
            "judge_latency_ms": round(dt, 2),
        }

        # Per-query incremental auto-save to disk
        out_data = {
            "config": survivor_name,
            "judge_model": judge.model_name,
            "quantized_4bit": judge.load_in_4bit,
            "mean_faithfulness": float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            "mean_judge_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "judgments": judge_results,
        }
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)

        # Update tqdm status with current mean faithfulness score
        pbar.set_postfix({"mean_faithfulness": f"{np.mean(faithfulness_scores):.4f}"})

    return judge_results, latencies, faithfulness_scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="force re-running judging")
    args = ap.parse_args()

    # Define the 4 Stage 4 survivor generation input files
    survivors_config = [
        ("Dense -> none", RUNS_DIR / "gen_dense_none.json", RUNS_DIR / "judge_dense_none.json"),
        ("Dense -> bge-v2-m3", RUNS_DIR / "gen_dense_m3.json", RUNS_DIR / "judge_dense_m3.json"),
        ("RRF Hybrid -> bge-v2-gemma", RUNS_DIR / "gen_rrf_gemma.json", RUNS_DIR / "judge_rrf_gemma.json"),
        ("Dense -> bge-v2-gemma", RUNS_DIR / "gen_dense_gemma.json", RUNS_DIR / "judge_dense_gemma.json"),
    ]

    for name, gen_p, out_p in survivors_config:
        if not gen_p.exists():
            print(f"Missing required Stage 4 generation file: {gen_p.name}")
            print("Run Stage 4 (python src/04_generate.py) first.")
            return 1

    hdr("[1] Load SciFact Document Corpus & Stage 3 Rerank Metrics")
    docmap = load_docmap()
    doc_texts = docmap.docid_to_text

    stage3_metrics_path = RUNS_DIR / "stage3_rerank_metrics.json"
    stage3_table_map = {}
    if stage3_metrics_path.exists():
        with open(stage3_metrics_path) as f:
            s3 = json.load(f)
        for row in s3.get("grid_cells", []):
            cfg = row["config"]
            stage3_table_map[cfg] = row
            if "Dense (bge-base) -> none" in cfg or "Dense -> none" in cfg:
                stage3_table_map["Dense -> none"] = row

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    judge_model = None
    runs_to_process = []
    summary_rows = []

    # Check for fully completed output JSONs
    for name, gen_p, out_p in survivors_config:
        if out_p.exists() and not args.force:
            with open(out_p) as f:
                cached = json.load(f)
            judgs = cached.get("judgments", {})
            if len(judgs) == 300:
                mean_f = cached.get("mean_faithfulness", 0.0)
                mean_lat = cached.get("mean_judge_latency_ms", 0.0)
                s3_row = stage3_table_map.get(name, {})
                ret_ndcg = s3_row.get("ndcg@10", 0.0)
                tot_lat = s3_row.get("latency_ms_mean", 0.0)
                print(f"  [SKIPPED - FULLY COMPLETED] {out_p.name:<24} (faithfulness: {mean_f:.4f}, judge lat: {mean_lat:.2f} ms/query)")
                summary_rows.append({
                    "config": name,
                    "output_file": out_p.name,
                    "ndcg@10": ret_ndcg,
                    "total_ret_lat_ms": tot_lat,
                    "faithfulness_score": round(mean_f, 4),
                    "judge_latency_ms": round(mean_lat, 2),
                    "total_queries": len(judgs),
                })
                continue

        with open(gen_p) as f:
            gen_data = json.load(f)
        runs_to_process.append((name, gen_data, out_p))

    if runs_to_process:
        hdr("[2] Load Local LLM Judge onto CUDA")
        judge_model = LocalJudge(device=device, load_in_4bit=True)

        hdr("[3] Execute LLM Faithfulness Judging across Survivor Runs (Per-Query Auto-Save)")
        for name, gen_data, out_p in runs_to_process:
            print(f"\n  Judging answers for: {name}...")
            t0 = time.perf_counter()
            judge_results, latencies, f_scores = run_judgement_for_survivor(
                judge=judge_model,
                survivor_name=name,
                gen_data=gen_data,
                doc_texts=doc_texts,
                out_path=out_p,
                force=args.force,
            )
            total_time = time.perf_counter() - t0
            mean_f = float(np.mean(f_scores)) if f_scores else 0.0
            mean_judge_lat = float(np.mean(latencies)) if latencies else 0.0

            s3_row = stage3_table_map.get(name, {})
            ret_ndcg = s3_row.get("ndcg@10", 0.0)
            tot_lat = s3_row.get("latency_ms_mean", 0.0)

            summary_rows.append({
                "config": name,
                "output_file": out_p.name,
                "ndcg@10": ret_ndcg,
                "total_ret_lat_ms": tot_lat,
                "faithfulness_score": round(mean_f, 4),
                "judge_latency_ms": round(mean_judge_lat, 2),
                "total_queries": len(judge_results),
            })

            print(f"    Wrote: {out_p.name:<24} (faithfulness: {mean_f:.4f}, judge latency: {mean_judge_lat:.2f} ms/query)")

        print("\n  Unloading judge model from GPU VRAM...")
        unload_model(judge_model)

    hdr("[4] Persist Stage 5 Faithfulness Summary & Final Phase 2 Matrix")
    summary_path = RUNS_DIR / "stage5_faithfulness_summary.json"
    summary_data = {
        "schema_version": 1,
        "stage": "05_judge",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge_model": judge_model.model_name if judge_model else "Qwen/Qwen2.5-3B-Instruct",
        "phase2_pareto_survivors": summary_rows,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nFinal Phase 2 Pareto Survivor Matrix (Quality vs Latency vs Faithfulness):")
    print(f"{'Config':<28} | {'Retrieval nDCG@10':<18} | {'Total Latency (ms)':<20} | {'Faithfulness'}")
    print("-" * 85)
    for r in summary_rows:
        print(f"{r['config']:<28} | {r['ndcg@10']:<18.4f} | {r['total_ret_lat_ms']:<20.2f} | {r['faithfulness_score']:.4f}")

    hdr("Stage 5 complete — Phase 2 Fully Finished!")
    print(f"  Persisted final summary: {summary_path.name}")
    print("  Next: python src/verify_stage5.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
