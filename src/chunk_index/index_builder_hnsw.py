#!/usr/bin/env python3
"""
index_builder_hnsw.py
----------------------
NEW FILE — Phase 6B extension (index upgrade for 10K+ scale).

What it does
------------
Builds a FAISS HNSW (Hierarchical Navigable Small World) index from the
unified chunks.json. This replaces your current IndexFlatIP with an
approximate-nearest-neighbor graph index that scales to millions of vectors.

Why HNSW and not Flat:
  - IndexFlatIP: O(N) exact search. At 10K vectors ~50-100ms/query on CPU.
  - IndexHNSWFlat: O(log N) approximate search. At 10K vectors ~2-5ms/query.
  - At 100K vectors: Flat becomes unusable (500ms+), HNSW stays <10ms.
  - Recall@10 vs exact: ~0.99+ with M=32, efSearch=128.

GPU requirement: NONE. HNSW runs on CPU and is actually faster there than
on GPU for typical dimensions (384-d MiniLM). GPU only helps for IVF/PQ
indexes with batch queries.

Parameters (tunable):
  M=32              — links per node (4-64). Higher = more accurate, more RAM.
  efConstruction=200 — build-time quality (higher = better graph, slower build).
  efSearch=128      — search-time quality (higher = better recall, slower).

Output:
  data/faiss_hnsw.index  — the HNSW index file
  Also writes a sidecar JSON with build metadata for reproducibility.

Usage:
  python src/chunk_index/index_builder_hnsw.py       --chunks data/chunks.json       --out_index data/faiss_hnsw.index       --m 32 --ef_construction 200
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # MiniLM-L6-v2 output dimension


def build_hnsw_index(chunks_path: str, out_index: str, m: int = 32, ef_construction: int = 200):
    chunks_path = Path(chunks_path).resolve()
    out_index = Path(out_index).resolve()
    out_index.parent.mkdir(parents=True, exist_ok=True)

    # Load chunks
    logger.info("Loading chunks from %s ...", chunks_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks", len(chunks))

    # Verify IDs
    ids = [c["id"] for c in chunks]
    assert ids == list(range(len(chunks))), "Chunk IDs must be contiguous 0..N-1"

    # Encode all chunks with batching
    texts = [c["text"] for c in chunks]
    logger.info("Encoding %d chunks with %s ...", len(texts), EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    # Use larger batch size for speed — adjust based on your RAM
    batch_size = 64 if len(chunks) > 1000 else 32
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )
    faiss.normalize_L2(embeddings)

    # Build HNSW index
    logger.info("Building HNSW index (M=%d, efConstruction=%d) ...", m, ef_construction)
    index = faiss.IndexHNSWFlat(EMBED_DIM, m)
    index.hnsw.efConstruction = ef_construction
    index.verbose = True  # Show progress during add

    t0 = time.time()
    index.add(embeddings)
    build_time = time.time() - t0
    logger.info("Index built in %.2f seconds", build_time)

    # Save index
    faiss.write_index(index, str(out_index))
    logger.info("HNSW index saved: %s (%d vectors)", out_index, index.ntotal)

    # Save metadata sidecar
    meta_path = out_index.with_suffix(".meta.json")
    meta = {
        "index_type": "IndexHNSWFlat",
        "num_vectors": index.ntotal,
        "dimension": EMBED_DIM,
        "m": m,
        "ef_construction": ef_construction,
        "ef_search_default": index.hnsw.efSearch,
        "build_time_seconds": round(build_time, 2),
        "embed_model": EMBED_MODEL,
        "source_chunks": str(chunks_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved: %s", meta_path)

    # Verification
    assert index.ntotal == len(chunks), "FAISS index and chunks.json out of sync!"
    logger.info("VERIFIED: index.ntotal (%d) == len(chunks) (%d)", index.ntotal, len(chunks))

    # Quick latency benchmark
    logger.info("Running latency benchmark ...")
    test_query = "What are the rights of a Data Principal?"
    qvec = model.encode([test_query], convert_to_numpy=True)
    faiss.normalize_L2(qvec)

    # Warm-up
    index.search(qvec, 5)

    # Timed runs
    n_runs = 100
    t0 = time.time()
    for _ in range(n_runs):
        index.search(qvec, 10)
    avg_latency_ms = (time.time() - t0) / n_runs * 1000
    logger.info("Average query latency: %.3f ms (over %d runs)", avg_latency_ms, n_runs)

    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS HNSW index for scaled retrieval.")
    parser.add_argument("--chunks", default="data/chunks.json", help="Path to chunks.json")
    parser.add_argument("--out_index", default="data/faiss_hnsw.index", help="Output index path")
    parser.add_argument("--m", type=int, default=32, help="HNSW links per node (4-64)")
    parser.add_argument("--ef_construction", type=int, default=200, help="Build-time quality")
    args = parser.parse_args()

    build_hnsw_index(args.chunks, args.out_index, args.m, args.ef_construction)
