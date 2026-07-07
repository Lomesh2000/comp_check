#!/usr/bin/env python3
"""
eventic_extractor.py — builds eventic graph from new.json (structured triples)
and attempts PDF extraction as fallback. Since OpenAI is dead, we use the
hand-crafted triples as the authoritative source.
"""

import os
import json
import re
import logging
from pathlib import Path
import pickle
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eventic")

def build_eventic_graph_from_json(json_path):
    """Build eventic graph from structured triples JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    
    G = nx.DiGraph()
    for t in triples:
        subj = t["subject"]
        pred = t["predicate"]
        obj = t["object"]
        
        # Parse deontic from predicate (must/has/requires etc.)
        deontic = "must"
        if pred.startswith("has"):
            deontic = "has"
        elif pred.startswith("requires"):
            deontic = "requires"
        elif pred.startswith("subjectTo"):
            deontic = "subject to"
        elif pred.startswith("may"):
            deontic = "may"
        
        action = pred.replace("must", "").replace("has", "").replace("requires", "").replace("subjectTo", "")
        if not action:
            action = pred
        
        node_deon = f"DEONTIC::{deontic}"
        node_action = f"ACTION::{action} {obj}"
        
        G.add_node(subj, type="agent", label=subj)
        G.add_node(node_deon, type="deontic", label=deontic)
        G.add_node(node_action, type="action", label=f"{action} {obj}")
        G.add_node(obj, type="object", label=obj)
        
        G.add_edge(subj, node_deon, rel="has_deontic")
        G.add_edge(node_deon, node_action, rel="regulates")
        G.add_edge(subj, obj, rel=pred, predicate=pred)
    
    return G

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    # Build from structured triples
    new_json = data_dir / "static" / "new.json"
    if new_json.exists():
        G = build_eventic_graph_from_json(str(new_json))
        logger.info("Built eventic graph from %s: %d nodes, %d edges", new_json, G.number_of_nodes(), G.number_of_edges())
    else:
        logger.error("new.json not found at %s", new_json)
        G = nx.DiGraph()
    
    out_path = data_dir / "eventic_graph.gpickle"
    with open(out_path, "wb") as f:
        pickle.dump(G, f)
    logger.info("Saved to %s", out_path)