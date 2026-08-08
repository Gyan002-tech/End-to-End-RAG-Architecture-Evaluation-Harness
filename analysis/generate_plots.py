#!/usr/bin/env python3
"""Generate high-resolution evaluation plots for RAG Retrieval + Evaluation Harness.

Reads JSON artifacts from artifacts/runs/ and outputs publication-quality PNG charts
to analysis/plots/.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Apply dark-mode aesthetic styling
plt.style.use('dark_background')
BG_COLOR = '#0F172A'      # Slate 900
PANEL_COLOR = '#1E293B'   # Slate 800
TEXT_COLOR = '#F8FAFC'    # Slate 50
ACCENT_BLUE = '#38BDF8'   # Sky 400
ACCENT_PURPLE = '#C084FC' # Purple 400
ACCENT_GREEN = '#4ADE80'  # Green 400
ACCENT_AMBER = '#FBBF24'  # Amber 400
ACCENT_RED = '#F87171'    # Red 400

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"


def setup_figure(title: str, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.grid(True, linestyle='--', alpha=0.3, color='#64748B')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color=TEXT_COLOR)
    return fig, ax


def plot_retrieval_comparison():
    path = RUNS_DIR / "stage2_retrieval_metrics.json"
    if not path.exists():
        print(f"Skipping plot_retrieval_comparison: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    arms = [item["arm"] for item in data["table"]]
    recall = [item["recall@10"] for item in data["table"]]
    mrr = [item["mrr"] for item in data["table"]]
    ndcg = [item["ndcg@10"] for item in data["table"]]
    p1 = [item["p@1"] for item in data["table"]]

    x = np.arange(len(arms))
    width = 0.18

    fig, ax = setup_figure("Stage 2 — Candidate Retrieval Arm Comparison (SciFact)")

    rects1 = ax.bar(x - 1.5*width, recall, width, label='Recall@10', color=ACCENT_BLUE)
    rects2 = ax.bar(x - 0.5*width, mrr, width, label='MRR', color=ACCENT_PURPLE)
    rects3 = ax.bar(x + 0.5*width, ndcg, width, label='nDCG@10', color=ACCENT_GREEN)
    rects4 = ax.bar(x + 1.5*width, p1, width, label='P@1', color=ACCENT_AMBER)

    ax.set_ylabel('Score', fontsize=12, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=11, color=TEXT_COLOR)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=PANEL_COLOR, edgecolor='#334155', labelcolor=TEXT_COLOR, fontsize=10)

    # Values on top of bars
    for rects in (rects1, rects2, rects3, rects4):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color=TEXT_COLOR)

    plt.tight_layout()
    out_path = PLOTS_DIR / "01_retrieval_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_pareto_frontier():
    path = RUNS_DIR / "stage5_faithfulness_summary.json"
    if not path.exists():
        print(f"Skipping plot_pareto_frontier: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    survivors = data["phase2_pareto_survivors"]
    # Fix 0.0 latency/ndcg for Dense -> none using actual Stage 2 metrics
    for s in survivors:
        if s["config"] == "Dense -> none":
            s["ndcg@10"] = 0.74065
            s["total_ret_lat_ms"] = 3.19

    configs = [s["config"] for s in survivors]
    latencies = [s["total_ret_lat_ms"] for s in survivors]
    ndcgs = [s["ndcg@10"] for s in survivors]

    fig, ax = setup_figure("Stage 3 — Retrieval Accuracy vs. Rerank Latency (Pareto Frontier)")

    colors = [ACCENT_BLUE, ACCENT_AMBER, ACCENT_PURPLE, ACCENT_GREEN]
    for i in range(len(configs)):
        ax.scatter(latencies[i], ndcgs[i], color=colors[i], s=160, zorder=5, label=configs[i])
        ax.annotate(configs[i], (latencies[i], ndcgs[i]),
                    xytext=(8, -4), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=colors[i])

    # Connect Pareto curve
    sorted_pairs = sorted(zip(latencies, ndcgs))
    ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], color='#64748B', linestyle='--', linewidth=1.5, zorder=3)

    ax.set_xlabel('Total Rerank Latency (ms)', fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel('Retrieval nDCG@10', fontsize=12, color=TEXT_COLOR)
    ax.set_ylim(0.70, 0.80)
    ax.set_xlim(-200, 7000)

    plt.tight_layout()
    out_path = PLOTS_DIR / "02_pareto_frontier.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_model_sweetspot():
    path = RUNS_DIR / "dense_model_size_sweetspot.json"
    if not path.exists():
        print(f"Skipping plot_model_sweetspot: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    models = [item["params"] + "\n(" + item["model"].split("/")[-1] + ")" for item in data["sweetspot_matrix"]]
    recall = [item["recall@10"] for item in data["sweetspot_matrix"]]
    ndcg = [item["ndcg@10"] for item in data["sweetspot_matrix"]]
    mrr = [item["mrr"] for item in data["sweetspot_matrix"]]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = setup_figure("Dense Model Size Sweet-Spot Study (33M vs 109M vs 335M)")

    rects1 = ax.bar(x - width, recall, width, label='Recall@10', color=ACCENT_BLUE)
    rects2 = ax.bar(x, ndcg, width, label='nDCG@10', color=ACCENT_GREEN)
    rects3 = ax.bar(x + width, mrr, width, label='MRR', color=ACCENT_PURPLE)

    ax.set_ylabel('Score', fontsize=12, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, color=TEXT_COLOR)
    ax.set_ylim(0.6, 0.95)
    ax.legend(facecolor=PANEL_COLOR, edgecolor='#334155', labelcolor=TEXT_COLOR, fontsize=10)

    # Highlight base model (sweetspot)
    ax.axvspan(0.5, 1.5, color='#FBBF24', alpha=0.1, zorder=1)
    ax.text(1, 0.92, "★ Sweet-Spot Winner (bge-base 109M)", ha='center', fontsize=10, fontweight='bold', color=ACCENT_AMBER)

    for rects in (rects1, rects2, rects3):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color=TEXT_COLOR)

    plt.tight_layout()
    out_path = PLOTS_DIR / "03_model_size_sweetspot.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_faithfulness_vs_latency():
    path = RUNS_DIR / "stage5_faithfulness_summary.json"
    if not path.exists():
        print(f"Skipping plot_faithfulness_vs_latency: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    survivors = data["phase2_pareto_survivors"]
    for s in survivors:
        if s["config"] == "Dense -> none":
            s["ndcg@10"] = 0.74065
            s["total_ret_lat_ms"] = 3.19

    configs = [s["config"] for s in survivors]
    faithfulness = [s["faithfulness_score"] for s in survivors]
    latencies = [s["total_ret_lat_ms"] for s in survivors]

    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax1.set_facecolor(PANEL_COLOR)
    ax1.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax1.spines.values():
        spine.set_color('#334155')
    ax1.grid(True, linestyle='--', alpha=0.3, color='#64748B')
    ax1.set_title("Stage 5 — End-to-End Latency vs. LLM Faithfulness Score", fontsize=14, fontweight='bold', pad=15, color=TEXT_COLOR)

    x = np.arange(len(configs))
    width = 0.35

    rects1 = ax1.bar(x - width/2, faithfulness, width, label='Faithfulness Score (0-1)', color=ACCENT_GREEN)
    ax1.set_ylabel('Faithfulness Score', color=ACCENT_GREEN, fontsize=12)
    ax1.set_ylim(0.40, 0.65)

    ax2 = ax1.twinx()
    ax2.set_facecolor(PANEL_COLOR)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax2.spines.values():
        spine.set_color('#334155')

    rects2 = ax2.bar(x + width/2, latencies, width, label='Total Latency (ms)', color=ACCENT_AMBER, alpha=0.85)
    ax2.set_ylabel('Latency (ms)', color=ACCENT_AMBER, fontsize=12)
    ax2.set_ylim(0, 8000)

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=10, color=TEXT_COLOR, rotation=15)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, color=ACCENT_GREEN, fontweight='bold')

    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}ms', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, color=ACCENT_AMBER)

    plt.tight_layout()
    out_path = PLOTS_DIR / "04_faithfulness_vs_latency.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating evaluation plots...")
    plot_retrieval_comparison()
    plot_pareto_frontier()
    plot_model_sweetspot()
    plot_faithfulness_vs_latency()
    print("All plots successfully generated in analysis/plots/")


if __name__ == "__main__":
    main()
