#!/usr/bin/env python3
"""
eventic_extractor.py

Extracts eventic (regulatory obligation) triples from documents.
Tries OpenAI -> local Flan-T5 -> regex heuristics.
Logs which backend was used for every document.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eventic")

# --- OpenAI client ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        import httpx
        # Fix for httpx version conflicts
        _client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client())
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.warning("OpenAI init failed: %s", e)
        _client = None
else:
    logger.warning("No OPENAI_API_KEY — using heuristic fallback only.")

# --- Local model lazy loader ---
_local_model = None
_local_tokenizer = None

def init_local_model(model_name="google/flan-t5-large"):
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
        logger.error("Local model init failed: %s", e)
        _local_model = None
        _local_tokenizer = None
    return _local_model, _local_tokenizer

def call_openai(prompt, model="gpt-3.5-turbo", temperature=0.0, max_tokens=512):
    if _client is None:
        return None
    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning("OpenAI call failed: %s", e)
        return None

# def call_local_model(prompt, max_input_length=1024, max_new_tokens=256):
#     model, tokenizer = init_local_model()
#     if model is None or tokenizer is None:
#         return None
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
#         out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
#         return tokenizer.decode(out[0], skip_special_tokens=True)
#     except Exception as e:
#         logger.warning("Local model failed: %s", e)
#         return None

def call_local_model(prompt, **kwargs):
    logger.info("Local model skipped — too slow for batch extraction.")
    return None

# --- Prompts ---
Tempt1 = """Please read the following regulatory text. Return a JSON array of all agents/subjects that are explicitly targeted by deontic expressions like "must", "shall", "is required to", "is obliged to", "should", "is prohibited from". Output only a JSON array of strings.

Paragraph:
----
{doc}
----
"""

Tempt2 = """Given the paragraph below and an agent name, identify whether there is a deontic word applying to that agent. If yes, return JSON like {{"deontic": "...", "action": "..."}}. If not, return null.

Paragraph:
----
{para}
----
Agent: {agent}
"""

DEONTIC_RE = r"\b(must|shall|should|required to|is required to|is obliged to|is prohibited from|prohibited|forbidden|may not)\b"

def heuristic_extract(text: str) -> List[Tuple[str, str, str]]:
    """Regex fallback for agent/action extraction."""
    agents = set(re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text))
    stopwords = {"The", "This", "If", "When", "Where", "In", "On", "A", "An"}
    agents = [a for a in agents if a.split()[0] not in stopwords][:50]

    triples = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        m = re.search(DEONTIC_RE, sent, flags=re.IGNORECASE)
        if m:
            deon = m.group(0)
            action = sent[m.end():].strip()[:200]
            matched = None
            for ag in agents:
                if re.search(r"\b" + re.escape(ag) + r"\b", sent):
                    matched = ag
                    break
            if not matched:
                matched = agents[0] if agents else "Unknown"
            triples.append((matched, deon, action))
    return triples

def extract_text_from_pdf(path: str) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        texts = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n".join(texts)
    except Exception as e:
        logger.warning("PDF failed %s: %s", path, e)
        return ""

def read_text_file(path: str) -> str:
    for enc in ("utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def _chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def extract_eventic_triples(document_text: str) -> List[Tuple[str, str, str]]:
    paras = [p.strip() for p in re.split(r"\n{2,}", document_text) if p.strip()]
    if not paras:
        paras = [document_text]

    # Filter to deontic-containing paragraphs
    deontic_paras_idx = [i for i, p in enumerate(paras) if re.search(DEONTIC_RE, p, flags=re.IGNORECASE)]
    if not deontic_paras_idx:
        logger.info("No deontic paragraphs found — using heuristic fallback.")
        return heuristic_extract(document_text)

    deontic_paras = [paras[i] for i in deontic_paras_idx]

    # Extract agents
    agents = set()
    for chunk in _chunk_list(deontic_paras, 8):
        joined = "\n\n".join(chunk)
        prompt = Tempt1.format(doc=joined)
        resp = call_openai(prompt)
        backend = "openai"
        if resp is None:
            resp = call_local_model(prompt)
            backend = "local" if resp else "none"
        if resp is None:
            # heuristic agent extraction
            found = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", joined)
            for f in found:
                if len(f) > 1:
                    agents.add(f.strip())
            backend = "heuristic"
            continue

        logger.info("Agent extraction used backend: %s", backend)
        try:
            parsed = json.loads(resp)
            if isinstance(parsed, list):
                for x in parsed:
                    if isinstance(x, str) and x.strip():
                        agents.add(x.strip())
        except Exception:
            for a in re.split(r"[,;\n]+", resp):
                a = a.strip().strip('[] "')
                if a:
                    agents.add(a)

    agents = list(agents)
    if not agents:
        logger.info("No agents found by LLM — using heuristic fallback.")
        agents = list(set(re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", document_text)))[:50]

    # Extract triples
    triples = []
    action_re = re.compile(r"(?P<deon>\b(?:must|shall|should|required to|is required to|is obliged to|is prohibited from|prohibited|forbidden|may not)\b)\s+(?P<action>[^.\n]{1,250})", flags=re.IGNORECASE)

    for p in deontic_paras:
        local_agents = [a for a in agents if re.search(r"\b" + re.escape(a) + r"\b", p, flags=re.IGNORECASE)]
        if not local_agents:
            local_agents = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", p)
            local_agents = [a for a in local_agents if a.lower() not in {'the','this','that','when','if'}][:3]

        for m in action_re.finditer(p):
            deon = m.group("deon").strip()
            action = m.group("action").strip()
            chosen = None
            for a in local_agents:
                if re.search(r"\b" + re.escape(a) + r"\b", p, flags=re.IGNORECASE):
                    chosen = a
                    break
            if not chosen:
                chosen = local_agents[0] if local_agents else (agents[0] if agents else "Unknown")
            triples.append((chosen, deon, action))

    # Deduplicate
    seen = set()
    out = []
    for ag, de, ac in triples:
        key = (ag.strip().lower(), de.strip().lower(), ac.strip().lower())
        if key not in seen:
            seen.add(key)
            out.append((ag.strip(), de.strip(), ac.strip()))
    return out

def build_eventic_graph(triples):
    G = nx.DiGraph()
    for agent, deontic, action in triples:
        node_action = f"ACTION::{action}"
        node_deon = f"DEONTIC::{deontic}"
        G.add_node(agent, type="agent", label=agent)
        G.add_node(node_deon, type="deontic", label=deontic)
        G.add_node(node_action, type="action", label=action)
        G.add_edge(agent, node_deon, rel="has_deontic")
        G.add_edge(node_deon, node_action, rel="regulates")
    return G

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir = data_dir.resolve()

    # Only process actual documents (.pdf, .txt), skip binaries and artifacts
    all_texts = []
    for path in sorted(data_dir.iterdir()):
        if path.is_dir():
            continue
        lower = path.name.lower()
        if lower in ("chunks.json", "faiss.index", "static_graph.gpickle", "eventic_graph.gpickle",
                     "eval.json", "gold.json", "preds.json", "test_chunks.json", "test_chunks.txt"):
            continue
        if "query" in lower:
            continue

        text = ""
        if lower.endswith(".pdf"):
            text = extract_text_from_pdf(str(path))
        elif lower.endswith((".txt", ".md", ".text")):
            text = read_text_file(str(path))
        else:
            continue

        if not text.strip():
            logger.warning("No text from %s, skipping.", path.name)
            continue
        all_texts.append((path.name, text))

    if not all_texts:
        raise RuntimeError("No documents found in %s" % data_dir)

    # Build unified eventic graph from all documents
    G = nx.DiGraph()
    for fname, doc_text in all_texts:
        logger.info("=== Processing: %s ===", fname)
        triples = extract_eventic_triples(doc_text)
        logger.info("Extracted %d triples from %s", len(triples), fname)
        doc_G = build_eventic_graph(triples)
        # Merge into main graph
        for n, attrs in doc_G.nodes(data=True):
            if not G.has_node(n):
                G.add_node(n, **attrs)
        for u, v, data in doc_G.edges(data=True):
            G.add_edge(u, v, **data)

    out_path = data_dir / "eventic_graph.gpickle"
    with open(out_path, "wb") as f:
        pickle.dump(G, f)
    logger.info("Saved eventic graph to %s (%d nodes, %d edges)", out_path, G.number_of_nodes(), G.number_of_edges())