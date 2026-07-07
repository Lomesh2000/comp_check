#!/usr/bin/env python3
"""
eval_retrieval.py — Compute Recall@K, MRR, nDCG@K for dense/sparse/hybrid retrieval.
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eval_retrieval")

sys.path.insert(0, str(Path(__file__).parent.parent / "retriever_rag"))
from retriever import UnifiedRetriever


def recall_at_k(relevant_ids, retrieved_ids, k):
    """Recall@K: |relevant ∩ retrieved| / |relevant|"""
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(relevant & retrieved_k) / len(relevant)


def mrr(relevant_ids, retrieved_ids):
    """MRR: 1 / rank of first relevant item"""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevant_ids, retrieved_ids, k):
    """DCG@K"""
    score = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            score += 1.0 / np.log2(i + 2)  # log2(i+2) because i starts at 0
    return score


def ndcg_at_k(relevant_ids, retrieved_ids, k):
    """nDCG@K = DCG@K / IDCG@K"""
    dcg = dcg_at_k(relevant_ids, retrieved_ids, k)
    ideal = dcg_at_k(relevant_ids, relevant_ids, k)  # perfect ranking
    if ideal == 0:
        return 0.0
    return dcg / ideal


def evaluate(retriever, gold_data, k_values=[5, 10]):
    """
    gold_data: list of {"query": str, "relevant_chunk_ids": [int, ...]}
    """
    metrics = {k: {"dense": [], "sparse": [], "hybrid": []} for k in k_values}
    metrics["mrr"] = {"dense": [], "sparse": [], "hybrid": []}

    for item in gold_data:
        query = item["query"]
        relevant = item["relevant_chunk_ids"]

        # Dense
        dense_results = retriever.dense_search(query, k=max(k_values))
        dense_ids = [r["id"] for r in dense_results]

        # Sparse
        sparse_results = retriever.sparse_search(query, k=max(k_values))
        sparse_ids = [r["id"] for r in sparse_results]

        # Hybrid
        hybrid_results = retriever.hybrid_search(query, k=max(k_values), alpha=0.5)
        hybrid_ids = [r["id"] for r in hybrid_results]

        for k in k_values:
            metrics[k]["dense"].append(recall_at_k(relevant, dense_ids, k))
            metrics[k]["sparse"].append(recall_at_k(relevant, sparse_ids, k))
            metrics[k]["hybrid"].append(recall_at_k(relevant, hybrid_ids, k))

        metrics["mrr"]["dense"].append(mrr(relevant, dense_ids))
        metrics["mrr"]["sparse"].append(mrr(relevant, sparse_ids))
        metrics["mrr"]["hybrid"].append(mrr(relevant, hybrid_ids))

    # Print results
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*60)
    print(f"Queries evaluated: {len(gold_data)}")
    print()

    for k in k_values:
        print(f"--- Recall@{k} ---")
        for method in ["dense", "sparse", "hybrid"]:
            mean = np.mean(metrics[k][method])
            print(f"  {method:10s}: {mean:.4f}")
        print()

    print("--- MRR ---")
    for method in ["dense", "sparse", "hybrid"]:
        mean = np.mean(metrics["mrr"][method])
        print(f"  {method:10s}: {mean:.4f}")

    print("="*60)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, default="data/gold_retrieval.json",
                    help="Gold standard: list of {query, relevant_chunk_ids}")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    args = ap.parse_args()

    retriever = UnifiedRetriever()

    if not Path(args.gold).exists():
        logger.error("Gold file not found: %s", args.gold)
        logger.error("Create it manually or use build_gold.py")
        return

    with open(args.gold, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    evaluate(retriever, gold_data, k_values=args.k)


if __name__ == "__main__":
    main()