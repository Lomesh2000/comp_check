#!/usr/bin/env python3
"""
eval_retrieval_scaled.py — FIXED for mixed gold formats.

Handles both:
  - LIST format (hand-labeled): [{"query": "...", "relevant_chunk_ids": [...]}, ...]
  - DICT format (build_gold_from_labels): {"cat_0": {"query_text": "...", "relevant_chunk_ids": [...]}, ...}

GPU requirement: NONE.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "retriever_rag"))
from retriever_fixed import UnifiedRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def recall_at_k(relevant_ids, retrieved_ids, k):
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(relevant & retrieved_k) / len(relevant)


def mrr(relevant_ids, retrieved_ids):
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevant_ids, retrieved_ids, k):
    score = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            score += 1.0 / np.log2(i + 2)
    return score


def ndcg_at_k(relevant_ids, retrieved_ids, k):
    dcg = dcg_at_k(relevant_ids, retrieved_ids, k)
    ideal = dcg_at_k(relevant_ids, relevant_ids, k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def normalize_gold(gold_data):
    """Convert gold to standard list format regardless of input structure."""
    if isinstance(gold_data, list):
        # Already a list — check if items have "query" or "query_text"
        normalized = []
        for item in gold_data:
            if "query" in item:
                normalized.append({
                    "query": item["query"],
                    "relevant_chunk_ids": item["relevant_chunk_ids"],
                    "source_corpus": item.get("source_corpus", "overall")
                })
            elif "query_text" in item:
                normalized.append({
                    "query": item["query_text"],
                    "relevant_chunk_ids": item["relevant_chunk_ids"],
                    "source_corpus": item.get("source_corpus", "overall")
                })
        return normalized

    elif isinstance(gold_data, dict):
        # Dict format from build_gold_from_labels (keyed by category)
        normalized = []
        for key, item in gold_data.items():
            query_text = item.get("query_text", item.get("query", ""))
            normalized.append({
                "query": query_text,
                "relevant_chunk_ids": item["relevant_chunk_ids"],
                "source_corpus": item.get("source_corpus", key)
            })
        return normalized

    else:
        raise ValueError(f"Unknown gold format: {type(gold_data)}")


def evaluate(retriever, gold_data, k_values, chunks):
    gold_list = normalize_gold(gold_data)

    metrics = {
        "overall": {k: {"dense": [], "sparse": [], "hybrid": []} for k in k_values},
        "latency_ms": {"dense": [], "sparse": [], "hybrid": []},
    }
    metrics["overall"]["mrr"] = {"dense": [], "sparse": [], "hybrid": []}
    metrics["overall"]["ndcg"] = {k: {"dense": [], "sparse": [], "hybrid": []} for k in k_values}

    for item in gold_list:
        query = item["query"]
        relevant = item["relevant_chunk_ids"]

        for method_name, method_fn in [
            ("dense", retriever.dense_search),
            ("sparse", retriever.sparse_search),
            ("hybrid", retriever.hybrid_search),
        ]:
            t0 = time.time()
            if method_name == "hybrid":
                results = method_fn(query, k=max(k_values), alpha=0.5)
            else:
                results = method_fn(query, k=max(k_values))
            latency_ms = (time.time() - t0) * 1000
            metrics["latency_ms"][method_name].append(latency_ms)

            retrieved_ids = [r["id"] for r in results]

            for k in k_values:
                metrics["overall"][k][method_name].append(
                    recall_at_k(relevant, retrieved_ids, k)
                )
            metrics["overall"]["mrr"][method_name].append(
                mrr(relevant, retrieved_ids)
            )
            for k in k_values:
                metrics["overall"]["ndcg"][k][method_name].append(
                    ndcg_at_k(relevant, retrieved_ids, k)
                )

    return metrics


def print_report(metrics, k_values, n_queries):
    print("\n" + "=" * 70)
    print(" SCALED RETRIEVAL EVALUATION REPORT")
    print("=" * 70)
    print(f"Queries evaluated: {n_queries}")
    print()

    for k in k_values:
        print(f"--- Recall@{k} ---")
        for method in ["dense", "sparse", "hybrid"]:
            mean = np.mean(metrics["overall"][k][method])
            print(f"  {method:10s}: {mean:.4f}")
        print()

    print("--- MRR ---")
    for method in ["dense", "sparse", "hybrid"]:
        mean = np.mean(metrics["overall"]["mrr"][method])
        print(f"  {method:10s}: {mean:.4f}")
    print()

    for k in k_values:
        print(f"--- nDCG@{k} ---")
        for method in ["dense", "sparse", "hybrid"]:
            mean = np.mean(metrics["overall"]["ndcg"][k][method])
            print(f"  {method:10s}: {mean:.4f}")
        print()

    print("--- Latency ---")
    for method in ["dense", "sparse", "hybrid"]:
        latencies = metrics["latency_ms"][method]
        print(f"  {method:10s}: p50={np.percentile(latencies, 50):.2f}ms, "
              f"p95={np.percentile(latencies, 95):.2f}ms, "
              f"max={max(latencies):.2f}ms")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Scaled retrieval evaluation with per-corpus breakdown.")
    parser.add_argument("--gold", default="data/gold_retrieval.json")
    parser.add_argument("--chunks", default="data/chunks.json")
    parser.add_argument("--index", default="data/faiss.index")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    args = parser.parse_args()

    if not Path(args.gold).exists():
        logger.error("Gold file not found: %s", args.gold)
        return

    with open(args.gold, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    gold_list = normalize_gold(gold_data)
    logger.info("Loaded %d gold queries, %d chunks", len(gold_list), len(chunks))

    retriever = UnifiedRetriever(chunks_path=args.chunks, faiss_path=args.index)
    retriever.verify_indexes()

    metrics = evaluate(retriever, gold_data, args.k, chunks)
    print_report(metrics, args.k, len(gold_list))


if __name__ == "__main__":
    main()