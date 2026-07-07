#!/usr/bin/env python3
"""
rag_runner.py — RAG pipeline with actual chunk retrieval.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import networkx as nx
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_runner")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
# _client = None
# if OPENAI_API_KEY:
#     try:
#         from openai import OpenAI
#         import httpx
#         _client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client())
#     except Exception as e:
#         logger.warning("OpenAI init failed: %s", e)
#         _client = None

# --- LLM Client: Groq (primary), OpenAI (fallback), Local (last resort) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

_groq_client = None
_openai_client = None

# Try Groq first
if GROQ_API_KEY:
    try:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized.")
    except Exception as e:
        logger.warning("Groq init failed: %s", e)
        _groq_client = None

# Fallback to OpenAI
if OPENAI_API_KEY and _groq_client is None:
    try:
        from openai import OpenAI
        import httpx
        _openai_client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client())
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.warning("OpenAI init failed: %s", e)
        _openai_client = None

if _groq_client is None and _openai_client is None:
    logger.warning("No API LLM available — will use local model fallback.")

_local_model = None
_local_tokenizer = None

def init_local_flan(model_name="google/flan-t5-large"):
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        logger.info("Loading local model %s ...", model_name)
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logger.info("Local model loaded.")
    except Exception as e:
        logger.error("Local model failed: %s", e)
        _local_model = None
        _local_tokenizer = None
    return _local_model, _local_tokenizer

# def call_llm(prompt, use_openai_priority=True, model_openai="gpt-3.5-turbo", max_tokens=256):
#     if use_openai_priority and _client is not None:
#         try:
#             resp = _client.chat.completions.create(
#                 model=model_openai,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0,
#                 max_tokens=max_tokens,
#             )
#             return resp.choices[0].message.content
#         except Exception as e:
#             logger.warning("OpenAI failed: %s", e)
#     model, tokenizer = init_local_flan()
#     if model is None or tokenizer is None:
#         raise RuntimeError("No LLM available.")
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
#     out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
#     return tokenizer.decode(out[0], skip_special_tokens=True)

def call_llm(prompt, use_api_priority=True, model="llama-3.3-70b-versatile", max_tokens=256):
    # Try Groq
    logger.info("call_llm: use_api_priority=%s, groq_client=%s, openai_client=%s",
                use_api_priority, _groq_client is not None, _openai_client is not None)
    if use_api_priority and _groq_client is not None:
        try:
            resp = _groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            logger.info("Groq call succeeded")
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("Groq call failed: %s", e)

    # Try OpenAI
    if use_api_priority and _openai_client is not None:
        try:
            resp = _openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            logger.info("OpenAI call succeeded")
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("OpenAI call failed: %s", e)

    # Fallback to local
    model_local, tokenizer = init_local_flan()
    if model_local is None or tokenizer is None:
        raise RuntimeError("No LLM available.")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = model_local.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

Tempt3 = """
You are a compliance classifier.

Business text:
{chunk}

Context:
{triples_text}

Question:
Does the business text violate ANY regulatory rules?

Answer ONLY one word: PASS or FAIL.
"""

Tempt3 = """
You are a compliance auditor. Compare the BUSINESS POLICY text below against the REGULATORY RULES.
Determine if the business policy violates any regulatory rule.

BUSINESS POLICY:
{chunk}

REGULATORY RULES:
{triples_text}

RELEVANT REGULATORY CONTEXT:
{retrieved_text}

Answer ONLY one word: PASS or FAIL.
PASS = The business policy complies with all rules.
FAIL = The business policy violates at least one rule.

Answer:
"""

def load_graph(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    logger.info("Loaded graph: %s (%d nodes, %d edges)", path, G.number_of_nodes(), G.number_of_edges())
    return G

def compute_eventic_node_embeddings(G_eventic, model):
    nodes = list(G_eventic.nodes())
    texts = [G_eventic.nodes[n].get("label") or n for n in nodes]
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return nodes, embs

def build_fused_subgraph(chunk_text, chunk_vec, eventic_nodes, eventic_embs,
                         eventic_graph, static_graph, static_node_order, static_embs,
                         model, lambda_thresh=0.75, hop_k=1):
    sims = np.dot(eventic_embs, chunk_vec.reshape(-1))
    topk = min(10, len(eventic_nodes))
    hit_idx = np.argsort(sims)[-topk:][::-1]
    hits = [eventic_nodes[i] for i in hit_idx]

    P = []
    for hit in hits:
        hit_emb = model.encode([hit], convert_to_numpy=True)[0]
        hit_emb = hit_emb / (np.linalg.norm(hit_emb) + 1e-12)
        static_sims = np.dot(static_embs, hit_emb)
        for idx in np.argsort(static_sims)[::-1][:3]:
            if static_sims[idx] >= 0.45:
                P.append(static_node_order[idx])

    P = list(dict.fromkeys(P))[:5]

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

    return Gfus, hits, P, list(N)

def triples_text_from_graph(G, max_items=50):
    lines = []
    count = 0
    for u, v, data in G.edges(data=True):
        pred = data.get("predicate") or data.get("rel") or "relatedTo"
        u_label = G.nodes[u].get("label") or u
        v_label = G.nodes[v].get("label") or v
        lines.append(f"- {u_label} {pred} {v_label}")
        count += 1
        if count >= max_items:
            break
    if not lines:
        for n, data in G.nodes(data=True):
            lines.append(f"- {data.get('label') or n}")
    return "\n".join(lines)

def main(args):
    if not SBERT_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed")

    sys.path.insert(0, str(Path(__file__).parent))
    from retriever import UnifiedRetriever
    retriever = UnifiedRetriever()

    static_graph = load_graph(args.static_graph) if args.static_graph else nx.DiGraph()
    static_node_order, static_embs = [], None
    npz_path = os.path.join(os.path.dirname(args.static_graph), "static", "node_embeddings.npz") if args.static_graph else None
    if npz_path and os.path.exists(npz_path):
        arr = np.load(npz_path, allow_pickle=True)
        static_node_order = list(arr["node_order"])
        static_embs = arr["embeddings"]
        norms = np.linalg.norm(static_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        static_embs = static_embs / norms

    eventic_graph = load_graph(args.eventic_graph)

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks", len(chunks))

    # Auto-detect business vs regulatory documents
    def is_regulatory_doc(filename, text_sample=""):
        lower_name = filename.lower()
        if any(kw in lower_name for kw in ["act", "statute", "law", "regulation", "dpdp"]):
            return True
        sample = text_sample[:500].lower()
        if "act" in sample and ("section" in sample or "chapter" in sample or "parliament" in sample):
            return True
        return False

    # # In main(), after loading chunks:
    # business_chunks = [c for c in chunks if "business" in c["filename"].lower() 
    #                 or c["filename"] in ["Indian Oil Priivacy policy.pdf", "Privacy_Policy_hdfc.pdf"]]
    # logger.info("Filtered to %d business chunks (excluded regulatory text)", len(business_chunks))

    business_chunks = [c for c in chunks if not is_regulatory_doc(c["filename"], c.get("text", ""))]
    regulatory_chunks = [c for c in chunks if is_regulatory_doc(c["filename"], c.get("text", ""))]
    logger.info("Auto-detected: %d business chunks, %d regulatory chunks", len(business_chunks), len(regulatory_chunks))

# # Then iterate business_chunks instead of chunks
# for idx, c in enumerate(business_chunks):
#     ...
    model = SentenceTransformer(args.embed_model)
    eventic_nodes, eventic_embs = compute_eventic_node_embeddings(eventic_graph, model)

    results = []
    for idx, c in enumerate(business_chunks):
        fname = c.get("filename", f"doc_{c.get('doc_id', 0)}")
        text = c.get("text", "")
        if not text.strip():
            continue

        logger.info("Processing chunk %d/%d: %s", idx + 1, len(business_chunks), fname[:50])

        # ACTUAL RETRIEVAL: get similar chunks from corpus
        retrieved = retriever.hybrid_search(text, k=3, alpha=0.6)
        retrieved_text = "\n\n".join([f"[{r['method']}] {r['text'][:300]}" for r in retrieved])

        cvec = model.encode([text], convert_to_numpy=True)[0]
        cvec = cvec / (np.linalg.norm(cvec) + 1e-12)

        Gfus, hits, P, N = build_fused_subgraph(
            text, cvec, eventic_nodes, eventic_embs,
            eventic_graph, static_graph,
            static_node_order, static_embs, model,
            lambda_thresh=args.lambda_thresh, hop_k=args.hop_k
        )

        triples_text = triples_text_from_graph(Gfus, max_items=args.max_triples)
        context = f"RETRIEVED CHUNKS:\n{retrieved_text}\n\nREGULATORY RULES:\n{triples_text}"
        # prompt = Tempt3.format(chunk=text[:2000], triples_text=context)
        prompt = Tempt3.format(
                chunk=text[:2000],
                triples_text=triples_text,
                retrieved_text=retrieved_text
            )


        try:
            # reply = call_llm(prompt, use_openai_priority=not args.prefer_local,
            #                model_openai=args.openai_model, max_tokens=256)
            reply = call_llm(prompt, use_api_priority=not args.prefer_local,
                 model="llama-3.3-70b-versatile", max_tokens=256)
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            reply = None

        verdict = "unknown"
        if reply:
            lower = reply.strip().lower()
            if lower.startswith("fail") or "violation" in lower:
                verdict = "fail"
            elif lower.startswith("pass"):
                verdict = "pass"
            else:
                verdict = "pass"

        results.append({
            "chunk_id": idx,
            "filename": fname,
            "retrieved_chunks": [r["id"] for r in retrieved],
            "hits": hits,
            "P": P,
            "N": N,
            "verdict": verdict,
            "llm_reply": reply,
            "triples_text": triples_text
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d results to %s", len(results), args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--static-graph", type=str, default="data/static_graph.gpickle")
    p.add_argument("--eventic-graph", type=str, default="data/eventic_graph.gpickle")
    p.add_argument("--chunks", type=str, default="data/chunks.json")
    p.add_argument("--out", type=str, default="data/preds.json")
    p.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--lambda_thresh", type=float, default=0.75)
    p.add_argument("--hop_k", type=int, default=1)
    p.add_argument("--max_triples", type=int, default=60)
    p.add_argument("--prefer_local", action="store_true")
    p.add_argument("--openai_model", type=str, default="gpt-3.5-turbo")
    args = p.parse_args()
    main(args)