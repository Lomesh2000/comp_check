"""
hf_to_chunks.py
----------------
NEW FILE — Phase 6 extension (big-data ingestion, step 2).

What it does
------------
Reads data/external/opp115_raw.jsonl (produced by fetch_hf_dataset.py) and
converts every row into your project's canonical chunk schema:

    {
        "id":            int,   # contiguous 0..N-1 across the WHOLE corpus
        "text":          str,
        "source_doc":    str,   # e.g. "opp115:record_00042"
        "source_corpus": str,   # NEW field: "iocl" | "hdfc" | "dpdp_act" | "opp115"
        "page_num":      int,   # not meaningful for OPP-115 -> set to -1
        "chunk_index":   int,   # position within its own source_doc
        "meta": {
            "label_ids":   [int, ...],   # kept for eval / gold generation
            "hf_split":    "train" | "validation" | "test"
        }
    }

Why "source_corpus" is a new required field: once you mix DPDP Act +
IOCL + HDFC + OPP-115 in one chunks.json, your eval script needs to be
able to report metrics per corpus (Phase 5/6 in your debug plan says
"log if Recall drops" — you can't tell WHERE it dropped without this
field).

Critical checks enforced (same spirit as your Phase 1 assertions):
  1. Every new chunk gets a unique, contiguous id that continues on from
     the max id already present in the existing chunks.json (no restart
     at 0, no collision).
  2. No duplicate text is added twice (hash-based de-dup) — cheap
     insurance against re-running this script and doubling the corpus.
  3. max_words is enforced by splitting any OPP-115 segment longer than
     the limit — mirrors the bug you already fixed in build_chunk_index.py
     ("max_words not enforced in fallback path").
  4. Final assembled file is re-validated with the same
     unique/contiguous-id check from your Phase 1 verification script.

GPU requirement: NONE. This is pure text/JSON processing (string
splitting, hashing). No model inference happens here.

Usage:
  python src/chunk_index/hf_to_chunks.py \
      --raw_jsonl data/external/opp115_raw.jsonl \
      --existing_chunks data/chunks.json \
      --out_chunks data/chunks.json \
      --max_words 200
"""

import argparse
import hashlib
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _split_long_text(text: str, max_words: int):
    """Splits text into <= max_words word chunks. Mirrors the fix applied
    to build_chunk_index.py's segment_text() fallback path."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def load_existing_chunks(path: str):
    if not path or not os.path.exists(path):
        logger.info("No existing chunks.json found at %s — starting fresh.", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d existing chunks from %s", len(chunks), path)
    return chunks


def convert(raw_jsonl: str, existing_chunks_path: str, out_path: str, max_words: int):
    existing_chunks = load_existing_chunks(existing_chunks_path)

    existing_ids = [c["id"] for c in existing_chunks]
    assert len(existing_ids) == len(set(existing_ids)), "Existing chunks.json already has duplicate IDs — fix Phase 1 first."
    next_id = (max(existing_ids) + 1) if existing_ids else 0

    seen_hashes = {
        hashlib.sha256(c["text"].encode("utf-8")).hexdigest() for c in existing_chunks
    }

    new_chunks = []
    n_skipped_dupe = 0
    n_split = 0

    with open(raw_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            row = json.loads(line)
            pieces = _split_long_text(row["text"], max_words)
            if len(pieces) > 1:
                n_split += 1

            for chunk_pos, piece in enumerate(pieces):
                h = hashlib.sha256(piece.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    n_skipped_dupe += 1
                    continue
                seen_hashes.add(h)

                new_chunks.append({
                    "id": next_id,
                    "text": piece,
                    "source_doc": f"opp115:record_{line_idx:05d}",
                    "source_corpus": "opp115",
                    "page_num": -1,
                    "chunk_index": chunk_pos,
                    "meta": {
                        "label_ids": row["label_ids"],
                        "hf_split": row["split"],
                    },
                })
                next_id += 1

    merged = existing_chunks + new_chunks

    # Re-run the same Phase 1 verification checks against the merged file.
    ids = [c["id"] for c in merged]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found after merge"
    assert set(ids) == set(range(len(ids))), "IDs are not contiguous 0..N-1 after merge"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info(
        "PASS: %d existing + %d new OPP-115 chunks = %d total (%d split for max_words, %d exact-dupe skipped) -> %s",
        len(existing_chunks), len(new_chunks), len(merged), n_split, n_skipped_dupe, out_path,
    )
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw OPP-115 JSONL into canonical chunk schema and merge.")
    parser.add_argument("--raw_jsonl", default="data/external/opp115_raw.jsonl")
    parser.add_argument("--existing_chunks", default="data/chunks.json")
    parser.add_argument("--out_chunks", default="data/chunks.json")
    parser.add_argument("--max_words", type=int, default=200)
    args = parser.parse_args()
    convert(args.raw_jsonl, args.existing_chunks, args.out_chunks, args.max_words)
