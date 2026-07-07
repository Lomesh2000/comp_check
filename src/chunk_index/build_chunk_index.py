#!/usr/bin/env python3
"""
build_chunk_index.py — Unified chunking + FAISS index builder.
Reads ALL documents from data/ recursively, skips artifacts,
assigns clean IDs 0..N-1, builds synced FAISS index.
"""

import os
import re
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chunk_index")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_WORDS = 200
OVERLAP_WORDS = 30

SKIP_NAMES = {
    "chunks.json", "faiss.index", "eval.json", "gold.json", "preds.json",
    "test_chunks.json", "test_chunks.txt", "queries_io_cl_privacy.txt"
}


def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        texts = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n".join(texts)
    except Exception as e:
        logger.warning("PDF parse failed for %s: %s", path, e)
        return ""


def read_text_file(path):
    for enc in ("utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    return open(path, "r", encoding="utf-8", errors="ignore").read()


def split_long_segment(seg, max_words):
    words = seg.split()
    if len(words) <= max_words:
        return [seg]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - OVERLAP_WORDS
        if start < 0:
            start = 0
    return chunks


def chunk_document(text, max_words=MAX_WORDS):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    for para in paragraphs:
        word_count = len(para.split())
        if word_count <= max_words:
            chunks.append(para)
            continue
        sentences = re.split(r'(?<=[.!?;])\s+', para)
        buffer = []
        buffer_words = 0
        for sent in sentences:
            sent_words = len(sent.split())
            if sent_words > max_words:
                if buffer:
                    chunks.append(" ".join(buffer))
                    buffer = []
                    buffer_words = 0
                chunks.extend(split_long_segment(sent, max_words))
                continue
            if buffer_words + sent_words > max_words:
                chunks.append(" ".join(buffer))
                buffer = [sent]
                buffer_words = sent_words
            else:
                buffer.append(sent)
                buffer_words += sent_words
        if buffer:
            chunks.append(" ".join(buffer))
    return chunks


def build_index(data_dir, out_index="data/faiss.index", metadata_out="data/chunks.json"):
    data_dir = Path(data_dir).resolve()
    out_index = Path(out_index).resolve()
    metadata_out = Path(metadata_out).resolve()
    out_index.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_dir():
            continue
        lower = path.name.lower()
        if lower in SKIP_NAMES or "test" in lower:
            continue
        if lower.endswith('.pdf'):
            text = extract_text_from_pdf(str(path))
        elif lower.endswith(('.txt', '.md', '.text')):
            text = read_text_file(str(path))
        else:
            continue
        if not text.strip():
            continue
        docs.append({"filename": path.name, "text": text})
        logger.info("Loaded %s (%d chars)", path.name, len(text))

    if not docs:
        raise RuntimeError("No documents found in %s" % data_dir)

    all_chunks = []
    for doc_id, doc in enumerate(docs):
        raw_chunks = chunk_document(doc["text"], max_words=MAX_WORDS)
        for chunk_text in raw_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text or len(chunk_text) < 20:
                continue
            all_chunks.append({
                "id": len(all_chunks),
                "doc_id": doc_id,
                "filename": doc["filename"],
                "text": chunk_text,
                "word_count": len(chunk_text.split())
            })

    if not all_chunks:
        raise RuntimeError("No chunks generated.")

    ids = [c["id"] for c in all_chunks]
    assert ids == list(range(len(all_chunks))), "Chunk IDs must be 0..N-1"

    texts = [c["text"] for c in all_chunks]
    logger.info("Encoding %d chunks with %s ...", len(texts), EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, str(out_index))

    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    assert index.ntotal == len(all_chunks), "SYNC ERROR"
    logger.info("VERIFIED: %d chunks, %d vectors", len(all_chunks), index.ntotal)
    return all_chunks, index


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data"
    build_index(str(data_dir))