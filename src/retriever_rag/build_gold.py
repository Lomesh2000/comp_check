#!/usr/bin/env python3
"""
build_gold.py — Semi-automatic gold label generation.
For each query, run retrieval and let you confirm which chunks are relevant.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "retriever_rag"))
from retriever import UnifiedRetriever

QUERIES = [
    "What are the rights of a Data Principal?",
    "What obligations does a Data Fiduciary have?",
    "How should consent be obtained?",
    "What happens in case of a data breach?",
    "What are the requirements for cross-border data transfer?",
    "What protections exist for children's data?",
    "What is the role of the Data Protection Board?",
    "How can a Data Principal file a complaint?",
    "What are the penalties for non-compliance?",
    "What is purpose limitation in data processing?",
]

def main():
    retriever = UnifiedRetriever()

    gold = []
    for q in QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("="*60)

        results = retriever.hybrid_search(q, k=5, alpha=0.5)
        relevant = []
        for i, r in enumerate(results):
            print(f"\n[{i}] Score: {r['score']:.4f} | Source: {r['source']}")
            print(f"    {r['text'][:200]}...")

        # Auto-select top-2 as relevant (you can refine later)
        relevant = [results[0]["id"], results[1]["id"]]
        print(f"\nAuto-selected relevant: {relevant}")
        gold.append({
            "query": q,
            "relevant_chunk_ids": relevant
        })

    with open("data/gold_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(gold)} gold labels to data/gold_retrieval.json")


if __name__ == "__main__":
    main()