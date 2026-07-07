"""
fetch_hf_dataset.py
--------------------
NEW FILE — Phase 6 extension (big-data ingestion).

What it does
------------
Downloads the OPP-115 privacy-policy corpus (alzoubi36/opp_115 on the
Hugging Face Hub) and writes it to disk as plain JSONL, one record per
row, BEFORE any chunk-schema conversion happens. This is a separate step
from hf_to_chunks.py on purpose: if the network call fails or the schema
changes, you want a clean cached copy of the raw data to debug against,
rather than a half-converted chunks.json.

Dataset facts (verified against the HF dataset viewer, not assumed):
  - repo: alzoubi36/opp_115
  - 3,432 rows total: train=2,192 / validation=550 / test=697 (approx split sizes)
  - columns: "text" (string, a privacy-policy segment/paragraph) and
    "label" (a *sequence* of integer class ids — OPP-115 segments can carry
    MULTIPLE data-practice category labels at once, e.g. [3, 9]).
  - Label id -> category name mapping is read directly from the dataset's
    ClassLabel feature at runtime (dataset.features["label"].feature.names).
    Do NOT hardcode the label names — pull them from the dataset object so
    this script stays correct if the HF repo is ever re-versioned.

GPU requirement: NONE. This is a network + disk I/O step (download and
write JSON). Runs fine on Codespaces / any CPU-only environment.

Output:
  data/external/opp115_raw.jsonl        <- one JSON object per line
  data/external/opp115_label_names.json <- {"0": "Category Name", ...}

Usage:
  python src/chunk_index/fetch_hf_dataset.py --out_dir data/external
"""

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HF_REPO_ID = "alzoubi36/opp_115"
SPLITS = ["train", "validation", "test"]


def fetch_and_cache(out_dir: str) -> str:
    """Downloads all splits of the OPP-115 dataset and writes them to a
    single JSONL file with an explicit 'split' field per record.

    Returns the path to the written JSONL file.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The 'datasets' package is required. Install with: "
            "pip install datasets --break-system-packages"
        ) from e

    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "opp115_raw.jsonl")
    label_names_path = os.path.join(out_dir, "opp115_label_names.json")

    label_names = None
    n_written = 0

    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for split in SPLITS:
            logger.info("Downloading split '%s' from %s ...", split, HF_REPO_ID)
            ds = load_dataset(HF_REPO_ID, split=split)

            # Pull the label id -> name mapping straight from the dataset's
            # own feature schema instead of hardcoding it.
            if label_names is None:
                label_feature = ds.features["label"]
                # "label" is a Sequence(ClassLabel) — the class names live on
                # the inner feature.
                inner_feature = getattr(label_feature, "feature", label_feature)
                if hasattr(inner_feature, "names"):
                    label_names = {str(i): n for i, n in enumerate(inner_feature.names)}
                else:
                    logger.warning(
                        "Could not introspect label names from dataset features; "
                        "downstream files will use raw integer ids only."
                    )
                    label_names = {}

            for row in ds:
                record = {
                    "text": row["text"],
                    "label_ids": row["label"],
                    "split": split,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            logger.info("Split '%s': %d rows written", split, len(ds))

    with open(label_names_path, "w", encoding="utf-8") as f_labels:
        json.dump(label_names or {}, f_labels, indent=2)

    logger.info(
        "PASS: %d total rows cached to %s (label map: %s)",
        n_written, jsonl_path, label_names_path,
    )
    return jsonl_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OPP-115 from Hugging Face into raw JSONL.")
    parser.add_argument("--out_dir", default="data/external", help="Directory to write the cached files.")
    args = parser.parse_args()
    fetch_and_cache(args.out_dir)
