# make_gold_template.py
#
# REPLACES THE OLD make_gold_from_preds.py.
#
# The old script copied model predictions directly into gold.json:
#     gold.append({"chunk_id": p["chunk_id"], "label": p["verdict"]})
# Running that against any preds.json produces a gold file that is
# guaranteed to match the model's own predictions, which makes
# eval_mcc.py report a perfect score (MCC=1.0, accuracy=1.0) regardless
# of whether the model is actually correct. That is not an evaluation —
# it is the model grading its own homework. The function is intentionally
# NOT preserved here, even as a flag-gated option, to make it impossible
# to regenerate gold.json this way by accident.
#
# This script instead produces a CSV/JSON annotation template with the
# "label" field left blank, for a human to fill in by reading each chunk
# against the source regulation. That is the only valid way to create
# evaluation ground truth for this project.

import json
import csv
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_annotation_template(chunks_path: str, out_json: str, out_csv: str):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    rows = []
    for idx, c in enumerate(chunks):
        rows.append({
            "chunk_id": idx,
            "filename": c.get("filename", ""),
            "text_preview": (c.get("text", "") or "")[:300],
            "label": None,            # human fills in: "pass" or "fail"
            "regulation_ref": None,   # e.g. "DPDPA Section 6(1)"
            "violation_type": None,   # e.g. "consent", "retention", "notice"
            "annotator": None,
            "confidence": None,       # 1 = low, 2 = medium, 3 = high
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Annotation template written for {len(rows)} chunks:")
    print(f"  JSON: {out_json}")
    print(f"  CSV:  {out_csv}  (open in Excel/Sheets to label faster)")
    print("Fill in the 'label' field for every row, then save the JSON "
          "as data/gold.json before running src/eval/eval_mcc.py.")


def main():
    ap = argparse.ArgumentParser(
        description="Generate a human-annotation template for compliance gold labels."
    )
    ap.add_argument("--chunks", default=str(_PROJECT_ROOT / "data" / "chunks.json"))
    ap.add_argument("--out-json", default=str(_PROJECT_ROOT / "data" / "gold_template.json"))
    ap.add_argument("--out-csv", default=str(_PROJECT_ROOT / "data" / "gold_template.csv"))
    args = ap.parse_args()
    create_annotation_template(args.chunks, args.out_json, args.out_csv)


if __name__ == "__main__":
    main()
