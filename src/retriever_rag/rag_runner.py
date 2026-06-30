#!/usr/bin/env python3
"""
rag_runner.py

Runs RAG-style fusion + LLM verdict for each chunk.

Usage:
  python src/retriever_rag/rag_runner.py \
        --static-graph data/static_graph.gpickle \
        --eventic data/eventic_graph.gpickle \
        --chunks data/chunks.json \
        --faiss data/faiss.index \
        --out results.json
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any
import numpy as np
import networkx as nx
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_runner")

# Try import faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# SentenceTransformer for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# OpenAI / local LLM fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

_client = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
_client = None

if OPENAI_API_KEY:
    try:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client is available for LLM calls.")

    except Exception:
        logger.exception("Failed to initialize OpenAI client")
        _client = None

else:
    logger.info("No OPENAI_API_KEY found — will try local model fallback.")

if _client is None:
    logger.warning(
        "STARTUP WARNING: No usable OpenAI client (OPENAI_API_KEY unset or "
        "client init failed). All compliance checks will depend on the local "
        "Flan-T5 fallback loading successfully on first use. If that also "
        "fails, every verdict will be 'unknown'."
    )

if _client is None:
    logger.warning(
        "STARTUP WARNING: No usable OpenAI client (OPENAI_API_KEY unset or "
        "client init failed). All compliance checks will depend on the local "
        "Flan-T5 fallback loading successfully on first use. If that also "
        "fails, every verdict will be 'unknown' with no error surfaced to "
        "callers until call_llm() is invoked. Set OPENAI_API_KEY or verify "
        "local model availability before running this in production."
    )

# Local Flan-T5 lazy loader
_local_model = None
_local_tokenizer = None
def init_local_flan(model_name="google/flan-t5-large"):
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        logger.info("Loading local model %s (this may take a while)...", model_name)
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logger.info("Local model loaded.")
    except Exception as e:
        logger.warning("Local model init failed: %s", e)
        _local_model = None
        _local_tokenizer = None
    return _local_model, _local_tokenizer

def call_llm(prompt: str, use_openai_priority: bool = True, model_openai: str = "gpt-3.5-turbo", max_tokens: int = 256) -> str:
    """
    Tries OpenAI (if available), else local Flan-T5. Returns text or raises.
    """
    if use_openai_priority and _client is not None:
        try:
            resp = _client.chat.completions.create(
                model=model_openai,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            logger.info("Using OpenAI backend")
            logger.info("=" * 80)
            logger.info("LLM RESPONSE")
            logger.info("%r", resp.choices[0].message.content)
            logger.info("=" * 80)
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("OpenAI call failed: %s", e)

    # fallback to local model
    model, tokenizer = init_local_flan()
    if model is None or tokenizer is None:
        raise RuntimeError(
            "No LLM backend available: OpenAI call failed or OPENAI_API_KEY "
            "is unset, and the local Flan-T5 fallback failed to load. "
            "Set OPENAI_API_KEY or ensure the local model + its weights are "
            "reachable before running compliance checks."
        )

    tokens = tokenizer(prompt)["input_ids"]
    logger.debug("Input tokens: %d", len(tokens))
    if len(tokens) > 1024:
        logger.warning("Prompt exceeds 1024 tokens and will be truncated by the local model.")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    logger.debug("Raw local model output: %s", text)
    logger.info("Using local FLAN backend")
    logger.info("=" * 80)
    logger.info("LOCAL MODEL RESPONSE")
    logger.info("%r", text)
    logger.info("=" * 80)

    return text

# Prompt template used by call_llm(). The LLM is asked for a single-word
# PASS/FAIL verdict (see verdict parsing in app.py / this file's main()).
# Two earlier prompt variants (a tag-based "<Compliance Check ...>" format
# and a verbose few-shot format) were tried and discarded; removed here to
# avoid the confusion of three stacked redefinitions of the same name.
Tempt3 = """
You are a compliance classifier.

Business text:
{chunk}

Regulatory rules:
{triples_text}

Question:

Does the business text violate ANY of the regulatory rules?

Answer ONLY one word.

PASS = No violation.
FAIL = At least one violation.

Answer:
"""

Tempt3 = """
You are an expert compliance auditor.

Business statement:
{chunk}

Relevant regulatory rules:
{triples_text}

Task:

Determine whether the business statement CONTRADICTS any of the regulatory rules.

Rules:

- PASS = The statement is consistent with the regulations or does not violate them.
- FAIL = The statement explicitly violates or contradicts one or more regulations.
- Missing information is NOT a violation.
- If the statement simply doesn't mention a rule, do NOT assume it violates that rule.
- Judge only what is explicitly stated.

Answer ONLY one word:

PASS

or

FAIL
"""

Tempt3 = """
You are an expert compliance auditor.

Business statement:
{chunk}

Relevant regulatory rules:
{triples_text}

Task:

Determine whether the business statement explicitly violates or contradicts any of the regulatory rules.

Important:
- Missing information is NOT a violation.
- If the statement does not mention a rule, do NOT assume it is violated.
- Judge only what is explicitly stated.

First explain your reasoning in 2–3 sentences.

If you think there is a violation, explain:
- Which rule is violated.
- Which sentence in the business statement causes the violation.

If you think there is NO violation, explain why.

Finally, on the last line output exactly one word:

PASS

or

FAIL
"""

def load_graph(path: str) -> nx.DiGraph:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    logger.info("Loaded graph from %s (nodes=%d, edges=%d)", path, G.number_of_nodes(), G.number_of_edges())
    return G

def load_node_embeddings(npz_path: str) -> (List[str], np.ndarray):
    # print("LOADING:", npz_path)

    arr = np.load(npz_path, allow_pickle=True)

    # print("FILES:", arr.files)

    node_order = list(arr["node_order"])
    embeddings = arr["embeddings"]

    # print("NODES:", len(node_order))
    # print("EMB SHAPE:", embeddings.shape)

    if not os.path.exists(npz_path):
        logger.warning("Node embeddings file not found: %s", npz_path)
        return [], None
    arr = np.load(npz_path, allow_pickle=True)
    node_order = list(arr["node_order"])
    embeddings = arr["embeddings"]
    # ensure normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return node_order, embeddings

def compute_eventic_node_embeddings(G_eventic: nx.Graph, model: SentenceTransformer, cache_path: str = None):
    nodes = list(G_eventic.nodes())
    texts = []
    for n in nodes:
        attrs = G_eventic.nodes[n]
        label = attrs.get("label") or attrs.get("name") or n
        texts.append(str(label))
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return nodes, embs

def build_fused_subgraph(chunk_text: str, chunk_vec: np.ndarray, eventic_nodes: List[str], eventic_embs: np.ndarray,
                         eventic_graph: nx.Graph, static_graph: nx.Graph,
                         static_node_order,
                         static_embs,
                         model,
                         lambda_thresh: float = 0.75, 
                         hop_k: int = 1,
                         top_k_eventic: int = 10,
                         static_match_threshold: float = 0.45,
                         max_static_matches_per_hit: int = 3,
                         max_P_nodes: int = 5):
    """
    1. hits = top_k_eventic eventic nodes by cosine similarity to the chunk
       (top-k retrieval, NOT a hard threshold — lambda_thresh is accepted
       for API/CLI backward compatibility but is currently unused; the
       0.75 hard threshold was the root cause of zero retrieval hits in
       the original implementation and was replaced with top-k).
    2. For each hit, find the most similar static_graph nodes by embedding
       similarity (not exact string match — eventic and static node labels
       rarely match verbatim) above static_match_threshold.
    3. P = the resulting static node matches, deduplicated, capped at
       max_P_nodes.
    4. N = immediate predecessors/successors of P in static_graph
       (BFS-expanded if hop_k > 1).
    5. Gfus = union subgraph of eventic_graph + static_graph restricted to
       nodes in P ∪ N.
    """
    # sims = np.dot(eventic_embs, chunk_vec.reshape(-1))
    # topk = min(top_k_eventic, len(eventic_nodes))
    # hit_idx = np.argsort(sims)[-topk:][::-1]
    # hits = [eventic_nodes[i] for i in hit_idx]

    # Compute cosine similarity between chunk and every eventic node
    sims = np.dot(eventic_embs, chunk_vec.reshape(-1))
    logger.info(
        "Similarity range: max=%.4f min=%.4f",
        np.max(sims),
        np.min(sims),
    )

    # ------------------------------------------------------------
    # Primary retrieval: similarity threshold
    # ------------------------------------------------------------
    hit_idx = np.where(sims >= lambda_thresh)[0]

    # ------------------------------------------------------------
    # Fallback: if threshold retrieves nothing, use top-k
    # ------------------------------------------------------------
    fallback_used = False
    if len(hit_idx) == 0:
        fallback_used = True
        logger.warning(
            "No eventic nodes above lambda_thresh=%.2f. Falling back to top-%d retrieval.",
            lambda_thresh,
            top_k_eventic,
        )

        topk = min(top_k_eventic, len(eventic_nodes))
        hit_idx = np.argsort(sims)[-topk:][::-1]

    else:
        # Sort retrieved nodes by similarity (highest first)
        hit_idx = hit_idx[np.argsort(sims[hit_idx])[::-1]]

        # Keep at most top_k_eventic nodes
        hit_idx = hit_idx[:top_k_eventic]

    hits = [eventic_nodes[i] for i in hit_idx]

    logger.info("lambda_thresh = %.2f", lambda_thresh)
    logger.info("Retrieved %d hits", len(hits))

    logger.info(
        "Retrieved %d eventic hits (lambda_thresh=%.2f)",
        len(hits),
        lambda_thresh,
    )

    logger.debug(
        "Eventic hits: %s",
        [
            (eventic_nodes[i], round(float(sims[i]), 4))
            for i in hit_idx[:10]
        ],
    )

    logger.debug("Top eventic hits: %s", [(eventic_nodes[i], round(float(sims[i]), 4)) for i in hit_idx[:5]])

    P = []
    for hit in hits:
        hit_emb = model.encode([hit], convert_to_numpy=True)[0]
        hit_emb = hit_emb / (np.linalg.norm(hit_emb) + 1e-12)
        static_sims = np.dot(static_embs, hit_emb)

        top_matches = np.argsort(static_sims)[::-1][:max_static_matches_per_hit]
        for idx in top_matches:
            if static_sims[idx] >= static_match_threshold:
                P.append(static_node_order[idx])

    # Remove duplicates while preserving order, then cap to the top N
    # most-relevant static nodes to keep the fused subgraph (and therefore
    # the prompt) small.
    P = list(dict.fromkeys(P))[:max_P_nodes]

    logger.debug("Matched static nodes (P): %s", P)

    N = set()
    for p in P:
        if static_graph.has_node(p):
            for pred in static_graph.predecessors(p):
                N.add(pred)
            for succ in static_graph.successors(p):
                N.add(succ)

    if hop_k > 1 and N:
        frontier = set(N)
        for _ in range(hop_k - 1):
            newf = set()
            for n in frontier:
                for nb in static_graph.neighbors(n):
                    if nb not in N:
                        newf.add(nb)
            if not newf:
                break
            N.update(newf)
            frontier = newf

    fused_nodes = set(P) | set(N)
    Gfus = nx.DiGraph()
    for n in fused_nodes:
        if eventic_graph.has_node(n):
            Gfus.add_node(n, **dict(eventic_graph.nodes[n]))
        elif static_graph.has_node(n):
            Gfus.add_node(n, **dict(static_graph.nodes[n]))
        else:
            Gfus.add_node(n, label=n)

    for u, v, data in eventic_graph.edges(data=True):
        if u in fused_nodes and v in fused_nodes:
            Gfus.add_edge(u, v, **data)

    for u, v, data in static_graph.edges(data=True):
        if u in fused_nodes and v in fused_nodes:
            Gfus.add_edge(u, v, **data)

    return Gfus, hits, P, list(N), fallback_used


def triples_text_from_graph(G: nx.Graph, max_items: int = 50) -> str:
    """
    Convert graph edges into short triple statements for prompt.
    """
    lines = []
    count = 0
    for u, v, data in G.edges(data=True):
        pred = data.get("predicate") or data.get("rel") or data.get("label") or "relatedTo"
        u_label = G.nodes[u].get("label") or u
        v_label = G.nodes[v].get("label") or v
        lines.append(f"- {u_label} {pred} {v_label}")
        count += 1
        if count >= max_items:
            break
    if not lines:
        # fallback to listing nodes
        for n, data in G.nodes(data=True):
            lines.append(f"- {data.get('label') or n}")
    return "\n".join(lines)


def main(args):
    if not SBERT_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")

    # load static graph
    static_graph = load_graph(args.static_graph) if args.static_graph else nx.DiGraph()
    logger.info(
        "Static graph: %d nodes, %d edges",
        static_graph.number_of_nodes(), static_graph.number_of_edges(),
    )

    static_node_order, static_embs = ([], None)
    npz_path = os.path.join(
                os.path.dirname(args.static_graph),
                "static",
                "node_embeddings.npz"
            ) if args.static_graph else None

    if npz_path and os.path.exists(npz_path):
        static_node_order, static_embs = load_node_embeddings(npz_path)
        logger.info("Loaded %d static node embeddings from %s", len(static_node_order), npz_path)
    else:
        logger.warning(
            "Static node embeddings not found at %s — build_fused_subgraph "
            "will be unable to match eventic hits against the static graph.",
            npz_path,
        )

    # load eventic graph
    if not args.eventic_graph or not os.path.exists(args.eventic_graph):
        raise FileNotFoundError("Eventic graph not provided or not found. Generate it using your eventic_extractor and save with nx.write_gpickle(...).")
    eventic_graph = load_graph(args.eventic_graph)

    # load chunks metadata
    if not os.path.exists(args.chunks):
        raise FileNotFoundError(f"chunks.json not found: {args.chunks}")
    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks", len(chunks))

    # load faiss index if provided (optional)
    index = None
    if args.faiss:
        if not FAISS_AVAILABLE:
            logger.warning("faiss not available; skipping index loading. Install faiss-cpu.")
        else:
            if not os.path.exists(args.faiss):
                logger.warning("faiss index file not found: %s", args.faiss)
            else:
                index = faiss.read_index(args.faiss)
                logger.info("Faiss index loaded from %s", args.faiss)

    # sentence-transformer model
    model = SentenceTransformer(args.embed_model)

    # precompute eventic node embeddings
    eventic_nodes, eventic_embs = compute_eventic_node_embeddings(eventic_graph, model)
    logger.info("Computed eventic embeddings: %d nodes", len(eventic_nodes))

    # if user provided a saved eventic node embeddings file, could use that here (not implemented)
    # ensure eventic_embs normalized (already normalized in function)
    results = []

    # iterate chunks
    for idx, c in enumerate(chunks):
        fname = c.get("filename", f"doc_{c.get('doc_id', 0)}")
        text = c.get("text", "")
        if not text.strip():
            continue
        logger.info("Processing chunk %d/%d (doc: %s) ...", idx + 1, len(chunks), fname)
        # embed chunk
        cvec = model.encode([text], convert_to_numpy=True)[0]
        cvec = cvec / (np.linalg.norm(cvec) + 1e-12)

        Gfus, hits, P, N, _ = build_fused_subgraph(
                                                text,
                                                cvec,
                                                eventic_nodes,
                                                eventic_embs,
                                                eventic_graph,
                                                static_graph,
                                                static_node_order,
                                                static_embs,
                                                model,
                                                lambda_thresh=args.lambda_thresh,
                                                hop_k=args.hop_k
                                                )

        logger.debug(
            "Chunk %d graph stats — hits=%d P=%d N=%d fused_nodes=%d fused_edges=%d",
            idx, len(hits), len(P), len(N), Gfus.number_of_nodes(), Gfus.number_of_edges(),
        )

        triples_text = triples_text_from_graph(Gfus, max_items=args.max_triples)
        logger.debug("Rules injected into prompt: %d", len(triples_text.split("\n")) if triples_text else 0)
        prompt = Tempt3.format(chunk=text[:2000] + ("\n\n[TRUNCATED]" if len(text) > 2000 else ""), triples_text=triples_text)
        logger.debug("Prompt sent to LLM:\n%s", prompt)
        try:
            reply = call_llm(prompt, use_openai_priority=not args.prefer_local, model_openai=args.openai_model, max_tokens=256)
            logger.info("=" * 80)
            logger.info("FULL LLM RESPONSE")
            logger.info("\n%s", reply)
            logger.info("=" * 80)

            lines = [l.strip() for l in reply.strip().splitlines() if l.strip()]
            last = lines[-1].upper()

            if last == "PASS":
                verdict = "pass"
            elif last == "FAIL":
                verdict = "fail"
            else:
                verdict = "unknown"

            evidence = [{"raw": reply}]
        except Exception as e:
            logger.warning("LLM call failed for chunk %d: %s", idx, e)
            reply = None

        verdict = None
        evidence = None
        if reply:
            # NOTE: Tempt3 asks for a one-word PASS/FAIL answer. The legacy
            # tag-based format is checked first only as a defensive
            # fallback in case a future prompt revision reintroduces it.
            if "<Compliance Check Passed>" in reply:
                verdict = "pass"
                evidence = []
            elif "<Compliance Check Failed>" in reply:
                verdict = "fail"
                # attempt to parse JSON after the tag
                try:
                    suffix = reply.split("<Compliance Check Failed>")[-1].strip()
                    parsed = json.loads(suffix)
                    evidence = parsed
                except Exception:
                    # fallback: put the textual suffix as explanation
                    evidence = [ {"raw": reply.split("<Compliance Check Failed>")[-1].strip()} ]
            else:
                # heuristic classification
                lower = reply.strip().lower()

                if lower.startswith("fail"):
                    verdict = "fail"

                elif lower.startswith("pass"):
                    verdict = "pass"

                elif "failed" in lower or "violate" in lower or "violation" in lower:
                    verdict = "fail"

                else:
                    # Ambiguous reply — don't silently default to "pass".
                    # Defaulting to pass here would hide a parsing/model
                    # failure as a compliance pass, the worst possible
                    # failure mode for this tool.
                    verdict = "unknown"

                evidence = [{"raw": reply[:400]}]
        else:
            verdict = "unknown"
            evidence = []

        results.append({
            "chunk_id": idx,
            "filename": fname,
            "hits": hits,
            "P": P,
            "N": N,
            "verdict": verdict,
            "evidence": evidence,
            "llm_reply": reply,
            "triples_text": triples_text
        })

        # optional small checkpoint save every few chunks
        # if (idx + 1) % 10 == 0:
        #     tmp_out = args.out + ".partial.json"
        #     with open(tmp_out, "w", encoding="utf-8") as f:
        #         json.dump(results, f, ensure_ascii=False, indent=2)
        #     logger.info("Saved partial results to %s", tmp_out)

    # final save
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d results to %s", len(results), args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--static-graph", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "static_graph.gpickle"), help="Path to static graph gpickle")
    p.add_argument("--eventic", "--eventic-graph", dest="eventic_graph", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "eventic_graph.gpickle"), help="Path to eventic graph gpickle (generated from eventic_extractor)")
    p.add_argument("--chunks", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "chunks.json"), help="chunks metadata JSON (from build_chunk_index.py)")
    p.add_argument("--faiss", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "faiss.index"), help="Faiss index path (optional)")
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "preds.json"), help="Output JSON path")
    p.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model for embeddings")
    p.add_argument("--lambda_thresh", type=float, default=0.75, help="Cosine similarity threshold for matching eventic nodes")
    p.add_argument("--hop_k", type=int, default=1, help="Number of hops to expand neighbors in static graph")
    p.add_argument("--max_triples", type=int, default=60, help="Max number of triples to include in prompt")
    p.add_argument("--prefer_local", action="store_true", help="Prefer local model fallback instead of OpenAI")
    p.add_argument("--openai_model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use when available")
    args = p.parse_args()
    main(args)
