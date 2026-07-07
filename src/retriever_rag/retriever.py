#!/usr/bin/env python3
"""
retriever.py — Unified dense (FAISS) + sparse (BM25) + hybrid retrieval.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("retriever")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class UnifiedRetriever:
    def __init__(self, chunks_path="data/chunks.json", faiss_path="data/faiss.index"):
        self.chunks_path = Path(chunks_path).resolve()
        self.faiss_path = Path(faiss_path).resolve()
        self.chunks = []
        self.chunk_map = {}
        self.index = None
        self.bm25 = None
        self.tokenized_corpus = []
        self.model = None
        self._load_chunks()
        self._load_faiss()
        self._build_bm25()
        self._load_embedder()

    def _load_chunks(self):
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        for c in self.chunks:
            self.chunk_map[c["id"]] = c
        logger.info("Loaded %d chunks from %s", len(self.chunks), self.chunks_path)
        docs = set(c["filename"] for c in self.chunks)
        logger.info("Documents: %s", docs)

    def _load_faiss(self):
        if not self.faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_path}")
        self.index = faiss.read_index(str(self.faiss_path))
        logger.info("Loaded FAISS index: %d vectors", self.index.ntotal)
        assert self.index.ntotal == len(self.chunks), \
            f"FAISS desync: {self.index.ntotal} vs {len(self.chunks)} chunks"

    def _build_bm25(self):
        if not BM25_AVAILABLE:
            logger.warning("rank-bm25 not installed — sparse search disabled")
            return
        self.tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("Built BM25 index over %d chunks", len(self.tokenized_corpus))

    def _load_embedder(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        logger.info("Loaded embedder: %s", EMBED_MODEL)

    def dense_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        qvec = self.model.encode([query], convert_to_numpy=True)[0]
        qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
        scores, ids = self.index.search(qvec.reshape(1, -1), k)
        scores = scores[0]
        ids = ids[0]
        results = []
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunk_map[idx]
            results.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "score": float(score),
                "source": chunk["filename"],
                "method": "dense"
            })
        logger.info("Dense '%s...' -> %d results (top: %.4f)",
                    query[:40], len(results), results[0]["score"] if results else 0)
        return results

    def sparse_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            logger.warning("BM25 not available")
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_k:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "score": float(scores[idx]),
                "source": chunk["filename"],
                "method": "sparse"
            })
        logger.info("Sparse '%s...' -> %d results (top: %.4f)",
                    query[:40], len(results), results[0]["score"] if results else 0)
        return results

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        dense_results = self.dense_search(query, k=k*2)
        sparse_results = self.sparse_search(query, k=k*2)
        score_map = {}
        for r in dense_results:
            score_map[r["id"]] = {"dense": r["score"], "sparse": 0.0, "chunk": r}
        if sparse_results:
            max_s = max(r["score"] for r in sparse_results)
            min_s = min(r["score"] for r in sparse_results)
            rng = max_s - min_s if max_s > min_s else 1.0
            for r in sparse_results:
                norm = (r["score"] - min_s) / rng
                if r["id"] in score_map:
                    score_map[r["id"]]["sparse"] = norm
                else:
                    score_map[r["id"]] = {"dense": 0.0, "sparse": norm, "chunk": r}
        fused = []
        for cid, data in score_map.items():
            fscore = alpha * data["dense"] + (1 - alpha) * data["sparse"]
            chunk = data["chunk"].copy()
            chunk["score"] = fscore
            chunk["dense_score"] = data["dense"]
            chunk["sparse_score"] = data["sparse"]
            chunk["method"] = "hybrid"
            fused.append(chunk)
        fused.sort(key=lambda x: x["score"], reverse=True)
        top_k = fused[:k]
        logger.info("Hybrid '%s...' -> %d results (a=%.2f, top: %.4f)",
                    query[:40], len(top_k), alpha, top_k[0]["score"] if top_k else 0)
        return top_k

    def verify_indexes(self):
        ok = self.index is not None and self.index.ntotal == len(self.chunks)
        ok = ok and (self.bm25 is not None or not BM25_AVAILABLE)
        logger.info("Index verification: %s", "PASS" if ok else "FAIL")
        return ok


if __name__ == "__main__":
    r = UnifiedRetriever()
    r.verify_indexes()
    q = "What are the rights of a Data Principal?"
    print("\n=== DENSE ===")
    for res in r.dense_search(q, k=3):
        print(f"[{res['score']:.4f}] {res['text'][:100]}...")
    print("\n=== SPARSE ===")
    for res in r.sparse_search(q, k=3):
        print(f"[{res['score']:.4f}] {res['text'][:100]}...")
    print("\n=== HYBRID ===")
    for res in r.hybrid_search(q, k=3, alpha=0.5):
        print(f"[{res['score']:.4f} d={res['dense_score']:.4f} s={res['sparse_score']:.4f}] {res['text'][:100]}...")