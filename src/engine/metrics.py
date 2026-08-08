"""Pure-Python retrieval evaluation metrics math & verification module.

Provides exact, hand-written implementations of standard IR metrics:
- Recall@k
- MRR (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain, binary relevance)
- Precision@k / P@1

Includes canary unit test verifying math against reference repo B's
(rag-evals-demo) pinned test vector (Recall@5 = 0.750, MRR = 0.625, nDCG@5 = 0.627).
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Sequence


def recall_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> float:
    """Recall@k = |relevant ∩ top-k| / |relevant|"""
    if not gold:
        return 0.0
    cutoff = retrieved[:k]
    hits = sum(1 for doc_id in cutoff if doc_id in gold)
    return hits / len(gold)


def reciprocal_rank(retrieved: Sequence[str], gold: Set[str]) -> float:
    """Reciprocal rank = 1 / rank of first relevant document (1-indexed), or 0.0."""
    if not gold:
        return 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> float:
    """DCG@k = Σ_{i=1}^k (2^{rel_i} - 1) / log2(i + 1). For binary rel: Σ rel_i / log2(i + 1)."""
    if not gold:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def idcg_at_k(gold_count: int, k: int) -> float:
    """Ideal DCG@k assuming all relevant docs are ranked first."""
    if gold_count == 0:
        return 0.0
    ideal_hits = min(gold_count, k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))


def ndcg_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k"""
    if not gold:
        return 0.0
    idcg = idcg_at_k(len(gold), k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, gold, k) / idcg


def precision_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> float:
    """Precision@k = |relevant ∩ top-k| / k"""
    if not gold or k <= 0:
        return 0.0
    cutoff = retrieved[:k]
    hits = sum(1 for doc_id in cutoff if doc_id in gold)
    return hits / k


def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k_list: Sequence[int] = (1, 5, 10, 20, 50),
) -> Dict[str, float]:
    """Evaluate a query -> docid_list run against qrels over standard k cutoffs.

    qrels: {qid: {docid: rel_score}}
    Returns dictionary of averaged metric scores across all queries.
    """
    valid_qids = [qid for qid in run if qid in qrels and any(rel > 0 for rel in qrels[qid].values())]
    if not valid_qids:
        return {}

    metrics: Dict[str, float] = {}

    for k in k_list:
        rec_sum = 0.0
        ndcg_sum = 0.0
        prec_sum = 0.0
        for qid in valid_qids:
            gold = {d for d, r in qrels[qid].items() if r > 0}
            retrieved = run[qid]
            rec_sum += recall_at_k(retrieved, gold, k)
            ndcg_sum += ndcg_at_k(retrieved, gold, k)
            prec_sum += precision_at_k(retrieved, gold, k)

        n = len(valid_qids)
        metrics[f"recall@{k}"] = round(rec_sum / n, 5)
        metrics[f"ndcg@{k}"] = round(ndcg_sum / n, 5)
        metrics[f"p@{k}"] = round(prec_sum / n, 5)

    mrr_sum = sum(
        reciprocal_rank(run[qid], {d for d, r in qrels[qid].items() if r > 0})
        for qid in valid_qids
    )
    metrics["mrr"] = round(mrr_sum / len(valid_qids), 5)
    metrics["n_queries"] = len(valid_qids)
    return metrics


# ---------------------------------------------------------------------------
# Unit Test / Verification Vector (Repo B Canary Data)
# ---------------------------------------------------------------------------
def test_canary_metrics() -> bool:
    """Verify hand-written metrics against reference repo B's pinned canary vector."""
    gold = {
        "q1": {"d3"},
        "q2": {"d7", "d2"},
        "q3": {"d11"},
        "q4": {"d5"},
    }
    runs = {
        "q1": ["d8", "d3", "d1", "d4", "d2", "d9", "d6", "d10", "d12", "d13"],
        "q2": ["d2", "d6", "d4", "d7", "d1", "d3", "d8", "d11", "d5", "d9"],
        "q3": ["d11", "d2", "d3", "d4", "d1", "d6", "d7", "d8", "d10", "d12"],
        "q4": ["d1", "d2", "d3", "d6", "d8", "d9", "d10", "d12", "d13", "d14"],
    }

    rec5 = sum(recall_at_k(runs[q], gold[q], 5) for q in gold) / len(gold)
    mrr = sum(reciprocal_rank(runs[q], gold[q]) for q in gold) / len(gold)
    ndcg5 = sum(ndcg_at_k(runs[q], gold[q], 5) for q in gold) / len(gold)

    pass_rec = abs(rec5 - 0.750) < 1e-3
    pass_mrr = abs(mrr - 0.625) < 1e-3
    pass_ndcg = abs(ndcg5 - 0.627) < 1e-3

    return pass_rec and pass_mrr and pass_ndcg


if __name__ == "__main__":
    ok = test_canary_metrics()
    print(f"Canary metrics unit test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)
