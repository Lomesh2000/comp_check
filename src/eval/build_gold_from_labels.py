"""
build_gold_from_labels.py
--------------------------
NEW FILE — Phase 5/6 extension (evaluation, do not skip).

Why this file exists
---------------------
Your existing build_gold.py needs a human to hand-label 1-3 relevant
chunk ids per query. That's the right approach for your 30 DPDP-specific
queries, but it does not scale to the 100s/1000s of OPP-115 chunks you
are about to add — nobody is hand-labeling that.

Instead, this script builds a SECOND, complementary gold set using
information OPP-115 already ships with: every segment is annotated with
one or more data-practice category labels (label_ids, carried through by
hf_to_chunks.py into each chunk's "meta" field).

Method: for each category, take one representative chunk's text as a
pseudo-query, and mark every OTHER chunk that shares at least one label
as "relevant". This is a category-level relevance proxy, not a
human-verified answer key.

IMPORTANT LIMITATION (state this explicitly in your eval report/resume,
do not hide it):
  - This measures "can hybrid search find other segments about the same
    broad privacy-practice category", NOT "can it answer a specific
    natural-language question". It is weaker evidence than your
    hand-labeled DPDP gold set, and should be reported SEPARATELY, never
    averaged together with it.
  - Category-label relevance is a common, published technique for
    building large pseudo-relevance eval sets when a labeled corpus is
    available (used because it's cheap and derived from real human
    annotation, not from the model being evaluated) — but the resulting
    Recall/MRR numbers should be labeled "OPP-115 category-proxy" in any
    report, not just "Recall@5", so they're never confused with your
    hand-verified DPDP query numbers.

GPU requirement: NONE. Pure list/dict processing over existing chunk
metadata already produced by hf_to_chunks.py.

Output: data/gold_retrieval_opp115_proxy.json
  {
    "cat_0": {
        "query_chunk_id": 42,
        "query_text": "...",
        "category": "First Party Collection/Use",
        "relevant_chunk_ids": [42, 108, 233, ...]
    },
    ...
  }

Usage:
  python src/eval/build_gold_from_labels.py \
      --chunks data/chunks.json \
      --label_names data/external/opp115_label_names.json \
      --out data/gold_retrieval_opp115_proxy.json \
      --max_relevant_per_category 20
"""

import argparse
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build(chunks_path: str, label_names_path: str, out_path: str, max_relevant_per_category: int):
    chunks = json.load(open(chunks_path, "r", encoding="utf-8"))
    label_names = json.load(open(label_names_path, "r", encoding="utf-8")) if label_names_path else {}

    opp115_chunks = [c for c in chunks if c.get("source_corpus") == "opp115" and c.get("meta", {}).get("label_ids")]
    if not opp115_chunks:
        raise ValueError(
            "No OPP-115 chunks with label metadata found. Run hf_to_chunks.py before this script."
        )

    by_category = defaultdict(list)
    for c in opp115_chunks:
        for label_id in c["meta"]["label_ids"]:
            by_category[label_id].append(c["id"])

    gold = {}
    for label_id, chunk_ids in by_category.items():
        if len(chunk_ids) < 2:
            # Need at least a query chunk + 1 other relevant chunk to be useful.
            continue
        query_chunk_id = chunk_ids[0]
        query_text = next(c["text"] for c in opp115_chunks if c["id"] == query_chunk_id)
        relevant = chunk_ids[:max_relevant_per_category]

        gold[f"cat_{label_id}"] = {
            "query_chunk_id": query_chunk_id,
            "query_text": query_text,
            "category": label_names.get(str(label_id), f"label_{label_id}"),
            "relevant_chunk_ids": relevant,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)

    logger.info(
        "PASS: built %d category-proxy queries from %d OPP-115 chunks -> %s",
        len(gold), len(opp115_chunks), out_path,
    )
    return gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a category-proxy gold retrieval set from OPP-115 labels.")
    parser.add_argument("--chunks", default="data/chunks.json")
    parser.add_argument("--label_names", default="data/external/opp115_label_names.json")
    parser.add_argument("--out", default="data/gold_retrieval_opp115_proxy.json")
    parser.add_argument("--max_relevant_per_category", type=int, default=20)
    args = parser.parse_args()
    build(args.chunks, args.label_names, args.out, args.max_relevant_per_category)
