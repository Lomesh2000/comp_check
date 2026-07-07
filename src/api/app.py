#!/usr/bin/env python3
"""
FastAPI wrapper for the LexGuard RAG compliance engine.

Exposes REST endpoints to run compliance checks on policy documents.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import io
sys.path.insert(0, str(Path(__file__).parent.parent / "retriever_rag"))

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever_rag.rag_runner import (
    load_graph,
    load_node_embeddings,
    compute_eventic_node_embeddings,
    build_fused_subgraph,
    triples_text_from_graph,
    call_llm,
    Tempt3,
)

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import numpy as np
import networkx as nx

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")

# ------- Config -------
DATA_DIR = Path(__file__).parent.parent.parent / "data"
STATIC_GRAPH_PATH = DATA_DIR / "static_graph.gpickle"
EVENTIC_GRAPH_PATH = DATA_DIR / "eventic_graph.gpickle"
CHUNKS_PATH = DATA_DIR / "chunks.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"

_static_graph: Optional[nx.DiGraph] = None
_eventic_graph: Optional[nx.DiGraph] = None
_embeddings_model: Optional[SentenceTransformer] = None
_eventic_nodes: Optional[List[str]] = None
_eventic_embs: Optional[np.ndarray] = None
_static_node_order: Optional[List[str]] = None
_static_embs: Optional[np.ndarray] = None
_faiss_index: Optional[Any] = None

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------- Global state (lazy initialized) -------
_static_graph: Optional[nx.DiGraph] = None
_eventic_graph: Optional[nx.DiGraph] = None
_embeddings_model: Optional[SentenceTransformer] = None
_eventic_nodes: Optional[List[str]] = None
_eventic_embs: Optional[np.ndarray] = None
_faiss_index: Optional[Any] = None


# ------- Pydantic Models -------
class ComplianceCheckRequest(BaseModel):
    text: str
    lambda_thresh: float = 0.75
    hop_k: int = 1
    max_triples: int = 60
    prefer_local: bool = False
    openai_model: str = "gpt-3.5-turbo"


class ComplianceCheckResponse(BaseModel):
    verdict: str  # "pass", "fail", "unknown"
    evidence: List[Dict[str, Any]]
    triples_text: str
    llm_reply: Optional[str] = None
    hits: List[str]
    P: List[str]
    N: List[str]


class HealthCheckResponse(BaseModel):
    status: str
    loaded_models: Dict[str, bool]


class ChunkResult(BaseModel):
    """Result for a single chunk"""
    chunk_id: int
    text_preview: str  # First 100 chars of chunk
    verdict: str
    evidence: List[Dict[str, Any]]
    llm_reply: Optional[str] = None
    hits: List[str]
    P: List[str]
    N: List[str]


class ChunkAnalysisResponse(BaseModel):
    """Response for chunk-level PDF analysis"""
    filename: str
    total_chunks: int
    chunks: List[ChunkResult]
    summary_verdict: str  # "pass" if all pass, "fail" if any fail, "mixed" if combination


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text content from a PDF file supplied as bytes."""
    if not PDF_AVAILABLE:
        raise RuntimeError("PyPDF2 is not installed")
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            texts.append(page_text)
    return "\n\n".join(texts).strip()


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if not text or chunk_size <= 0:
        return [text] if text else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    
    return chunks if chunks else [text] if text else []


# def run_compliance_text(
#     text: str,
#     lambda_thresh: float = 0.75,
#     hop_k: int = 1,
#     max_triples: int = 60,
#     prefer_local: bool = False,
#     openai_model: str = "gpt-3.5-turbo",
# ) -> ComplianceCheckResponse:
#     if not _embeddings_model or not _eventic_graph or not _static_graph:
#         raise RuntimeError("Models not initialized")

#     text_vec = _embeddings_model.encode([text], convert_to_numpy=True)[0]
#     text_vec = text_vec / (np.linalg.norm(text_vec) + 1e-12)

#     Gfus, hits, P, N = build_fused_subgraph(
#         text,
#         text_vec,
#         _eventic_nodes,
#         _eventic_embs,
#         _eventic_graph,
#         _static_graph,
#         lambda_thresh=lambda_thresh,
#         hop_k=hop_k,
#     )

#     triples_text = triples_text_from_graph(Gfus, max_items=max_triples)
#     prompt = Tempt3.format(
#         chunk=text[:2000] + ("\n\n[TRUNCATED]" if len(text) > 2000 else ""),
#         triples_text=triples_text,
#     )

#     try:
#         reply = call_llm(
#             prompt,
#             use_openai_priority=not prefer_local,
#             model_openai=openai_model,
#             max_tokens=256,
#         )
#     except Exception as e:
#         logger.warning(f"LLM call failed: {e}")
#         reply = None

#     if reply:
#         if "<Compliance Check Passed>" in reply:
#             verdict = "pass"
#             evidence = []
#         elif "<Compliance Check Failed>" in reply:
#             verdict = "fail"
#             try:
#                 suffix = reply.split("<Compliance Check Failed>")[-1].strip()
#                 parsed = json.loads(suffix)
#                 evidence = parsed
#             except Exception:
#                 evidence = [{"raw": reply.split("<Compliance Check Failed>")[-1].strip()}]
#         else:
#             lower = reply.lower()
#             if "failed" in lower or "violate" in lower or "violation" in lower:
#                 verdict = "fail"
#             else:
#                 verdict = "pass"
#             evidence = [{"raw": reply[:400]}]
#     else:
#         verdict = "unknown"
#         evidence = []

#     return ComplianceCheckResponse(
#         verdict=verdict,
#         evidence=evidence,
#         triples_text=triples_text,
#         llm_reply=reply,
#         hits=hits,
#         P=P,
#         N=N,
#     )

def run_compliance_text(
    text: str,
    lambda_thresh: float = 0.75,
    hop_k: int = 1,
    max_triples: int = 60,
    prefer_local: bool = False,
    openai_model: str = "gpt-3.5-turbo",
) -> ComplianceCheckResponse:
    if not _embeddings_model or not _eventic_graph or not _static_graph:
        raise RuntimeError("Models not initialized")

    # Retrieve relevant chunks from corpus
    from retriever_rag.retriever import UnifiedRetriever
    retriever = UnifiedRetriever()
    retrieved = retriever.hybrid_search(text, k=5, alpha=0.6)
    retrieved_text = "\n\n".join([f"[{r['method']}] {r['text'][:300]}" for r in retrieved])

    text_vec = _embeddings_model.encode([text], convert_to_numpy=True)[0]
    text_vec = text_vec / (np.linalg.norm(text_vec) + 1e-12)

    Gfus, hits, P, N = build_fused_subgraph(
        text,
        text_vec,
        _eventic_nodes,
        _eventic_embs,
        _eventic_graph,
        _static_graph,
        _static_node_order,
        _static_embs,
        _embeddings_model,
        lambda_thresh=lambda_thresh,
        hop_k=hop_k,
    )

    triples_text = triples_text_from_graph(Gfus, max_items=max_triples)
    
    # Build prompt with retrieved chunks + graph triples
    context = f"RETRIEVED DOCUMENT CHUNKS:\n{retrieved_text}\n\nREGULATORY RULES:\n{triples_text}"
    prompt = Tempt3.format(
        chunk=text[:2000] + ("\n\n[TRUNCATED]" if len(text) > 2000 else ""),
        triples_text=context,
    )

    try:
        reply = call_llm(
            prompt,
            use_openai_priority=not prefer_local,
            model_openai=openai_model,
            max_tokens=256,
        )
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        reply = None

    if reply:
        lower = reply.strip().lower()
        if "pass" in lower and "fail" not in lower:
            verdict = "pass"
        elif "fail" in lower:
            verdict = "fail"
        else:
            verdict = "unknown"
        evidence = [{"raw": reply[:400]}]
    else:
        verdict = "unknown"
        evidence = []

    return ComplianceCheckResponse(
        verdict=verdict,
        evidence=evidence,
        triples_text=triples_text,
        llm_reply=reply,
        hits=[r["text"][:200] for r in retrieved],  # Return retrieved chunks as hits
        P=P,
        N=N,
    )


# def init_global_state():
#     """Initialize global state on startup."""
#     global _static_graph, _eventic_graph, _embeddings_model, _eventic_nodes, _eventic_embs, _faiss_index

#     logger.info("Initializing global state...")

#     if not SBERT_AVAILABLE:
#         raise RuntimeError("sentence-transformers not installed")

#     # Load graphs
#     try:
#         _static_graph = load_graph(str(STATIC_GRAPH_PATH))
#         logger.info(f"Loaded static graph: {_static_graph.number_of_nodes()} nodes, {_static_graph.number_of_edges()} edges")
#     except Exception as e:
#         logger.warning(f"Failed to load static graph: {e}")
#         _static_graph = nx.DiGraph()

#     try:
#         _eventic_graph = load_graph(str(EVENTIC_GRAPH_PATH))
#         logger.info(f"Loaded eventic graph: {_eventic_graph.number_of_nodes()} nodes, {_eventic_graph.number_of_edges()} edges")
#     except Exception as e:
#         logger.error(f"Failed to load eventic graph: {e}")
#         raise

#     # Load embeddings model
#     try:
#         _embeddings_model = SentenceTransformer(EMBED_MODEL)
#         logger.info(f"Loaded embeddings model: {EMBED_MODEL}")
#     except Exception as e:
#         logger.error(f"Failed to load embeddings model: {e}")
#         raise

#     # Precompute eventic node embeddings
#     try:
#         from retriever_rag.rag_runner import compute_eventic_node_embeddings
#         _eventic_nodes, _eventic_embs = compute_eventic_node_embeddings(_eventic_graph, _embeddings_model)
#         logger.info(f"Computed eventic embeddings: {len(_eventic_nodes)} nodes")
#     except Exception as e:
#         logger.error(f"Failed to compute eventic embeddings: {e}")
#         raise

#     # Load FAISS index (optional)
#     if FAISS_AVAILABLE and FAISS_INDEX_PATH.exists():
#         try:
#             _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
#             logger.info("Loaded FAISS index")
#         except Exception as e:
#             logger.warning(f"Failed to load FAISS index: {e}")
#     else:
#         logger.info("FAISS index not available")

def init_global_state():
    """Initialize global state on startup."""
    global _static_graph, _eventic_graph, _embeddings_model
    global _eventic_nodes, _eventic_embs, _static_node_order, _static_embs, _faiss_index

    logger.info("Initializing global state...")

    if not SBERT_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed")

    # Load static graph + embeddings
    try:
        _static_graph = load_graph(str(STATIC_GRAPH_PATH))
        logger.info(f"Loaded static graph: {_static_graph.number_of_nodes()} nodes, {_static_graph.number_of_edges()} edges")
        
        # Load static node embeddings
        npz_path = DATA_DIR / "static" / "node_embeddings.npz"
        if npz_path.exists():
            arr = np.load(str(npz_path), allow_pickle=True)
            _static_node_order = list(arr["node_order"])
            _static_embs = arr["embeddings"]
            norms = np.linalg.norm(_static_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            _static_embs = _static_embs / norms
            logger.info(f"Loaded static embeddings: {len(_static_node_order)} nodes")
        else:
            logger.warning("Static embeddings not found")
            _static_node_order = []
            _static_embs = None
    except Exception as e:
        logger.warning(f"Failed to load static graph: {e}")
        _static_graph = nx.DiGraph()
        _static_node_order = []
        _static_embs = None

    # Load eventic graph
    try:
        _eventic_graph = load_graph(str(EVENTIC_GRAPH_PATH))
        logger.info(f"Loaded eventic graph: {_eventic_graph.number_of_nodes()} nodes, {_eventic_graph.number_of_edges()} edges")
    except Exception as e:
        logger.error(f"Failed to load eventic graph: {e}")
        raise

    # Load embeddings model
    try:
        _embeddings_model = SentenceTransformer(EMBED_MODEL)
        logger.info(f"Loaded embeddings model: {EMBED_MODEL}")
    except Exception as e:
        logger.error(f"Failed to load embeddings model: {e}")
        raise

    # Precompute eventic node embeddings
    try:
        _eventic_nodes, _eventic_embs = compute_eventic_node_embeddings(_eventic_graph, _embeddings_model)
        logger.info(f"Computed eventic embeddings: {len(_eventic_nodes)} nodes")
    except Exception as e:
        logger.error(f"Failed to compute eventic embeddings: {e}")
        raise

    # Load FAISS index (optional)
    if FAISS_AVAILABLE and FAISS_INDEX_PATH.exists():
        try:
            _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
            logger.info("Loaded FAISS index")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
    else:
        logger.info("FAISS index not available")

# ------- FastAPI app -------
app = FastAPI(
    title="LexGuard Compliance Engine",
    description="Offline RAG-LLM service for policy compliance verification",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    try:
        init_global_state()
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise


# @app.get("/health", response_model=HealthCheckResponse)
# async def health_check():
#     """Health check endpoint."""
#     return HealthCheckResponse(
#         status="ok" if all(
#             [_static_graph, _eventic_graph, _embeddings_model, _eventic_nodes, _eventic_embs]
#         ) else "degraded",
#         loaded_models={
#             "static_graph": _static_graph is not None,
#             "eventic_graph": _eventic_graph is not None,
#             "embeddings_model": _embeddings_model is not None,
#             "faiss_index": _faiss_index is not None,
#         },
#     )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="ok" if all([
            _static_graph is not None,
            _eventic_graph is not None,
            _embeddings_model is not None,
            _eventic_nodes is not None,
            _eventic_embs is not None,
        ]) else "degraded",
        loaded_models={
            "static_graph": _static_graph is not None,
            "eventic_graph": _eventic_graph is not None,
            "embeddings_model": _embeddings_model is not None,
            "faiss_index": _faiss_index is not None,
        },
    )

@app.post("/compliance-check", response_model=ComplianceCheckResponse)
async def compliance_check(request: ComplianceCheckRequest):
    """
    Run a compliance check on a policy text chunk.
    """
    try:
        return run_compliance_text(
            request.text,
            lambda_thresh=request.lambda_thresh,
            hop_k=request.hop_k,
            max_triples=request.max_triples,
            prefer_local=request.prefer_local,
            openai_model=request.openai_model,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compliance check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-pdf", response_model=ComplianceCheckResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    lambda_thresh: float = Form(0.75),
    hop_k: int = Form(1),
    max_triples: int = Form(60),
    prefer_local: bool = Form(False),
    openai_model: str = Form("gpt-3.5-turbo"),
):
    """
    Upload a PDF and run compliance check on the entire document as a single unit.
    Returns a single verdict for the whole document.
    """
    if not PDF_AVAILABLE:
        raise HTTPException(status_code=500, detail="PyPDF2 is not installed for PDF extraction")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        contents = await file.read()
        text = extract_text_from_pdf_bytes(contents)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        return run_compliance_text(
            text,
            lambda_thresh=lambda_thresh,
            hop_k=hop_k,
            max_triples=max_triples,
            prefer_local=prefer_local,
            openai_model=openai_model,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-pdf-chunks", response_model=ChunkAnalysisResponse)
async def upload_pdf_chunks(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(100),
    lambda_thresh: float = Form(0.75),
    hop_k: int = Form(1),
    max_triples: int = Form(60),
    prefer_local: bool = Form(False),
    openai_model: str = Form("gpt-3.5-turbo"),
):
    """
    Upload a PDF and run compliance check on each chunk separately.
    Returns individual verdicts for each chunk (similar to preds.json).
    
    Parameters:
    - chunk_size: Number of characters per chunk (default: 500)
    - chunk_overlap: Overlap between consecutive chunks in characters (default: 100)
    """
    if not PDF_AVAILABLE:
        raise HTTPException(status_code=500, detail="PyPDF2 is not installed for PDF extraction")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        contents = await file.read()
        full_text = extract_text_from_pdf_bytes(contents)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        
        # Split into chunks
        text_chunks = split_text_into_chunks(full_text, chunk_size=chunk_size, overlap=chunk_overlap)
        chunk_results = []
        verdicts = []
        
        # Process each chunk
        for chunk_id, chunk_text in enumerate(text_chunks):
            logger.info(f"Processing chunk {chunk_id + 1}/{len(text_chunks)}")
            try:
                result = run_compliance_text(
                    chunk_text,
                    lambda_thresh=lambda_thresh,
                    hop_k=hop_k,
                    max_triples=max_triples,
                    prefer_local=prefer_local,
                    openai_model=openai_model,
                )
                verdicts.append(result.verdict)
                chunk_results.append(
                    ChunkResult(
                        chunk_id=chunk_id,
                        text_preview=chunk_text[:100],
                        verdict=result.verdict,
                        evidence=result.evidence,
                        llm_reply=result.llm_reply,
                        hits=result.hits,
                        P=result.P,
                        N=result.N,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process chunk {chunk_id}: {e}")
                chunk_results.append(
                    ChunkResult(
                        chunk_id=chunk_id,
                        text_preview=chunk_text[:100],
                        verdict="unknown",
                        evidence=[{"error": str(e)}],
                        llm_reply=None,
                        hits=[],
                        P=[],
                        N=[],
                    )
                )
        
        # Determine summary verdict
        if not verdicts:
            summary = "unknown"
        elif all(v == "pass" for v in verdicts):
            summary = "pass"
        elif all(v == "fail" for v in verdicts):
            summary = "fail"
        else:
            summary = "mixed"
        
        return ChunkAnalysisResponse(
            filename=file.filename or "unknown",
            total_chunks=len(text_chunks),
            chunks=chunk_results,
            summary_verdict=summary,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF chunk analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "service": "LexGuard Compliance Engine",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "compliance_check": "/compliance-check (POST) - Check single text",
            "upload_pdf": "/upload-pdf (POST) - Analyze PDF as single document",
            "upload_pdf_chunks": "/upload-pdf-chunks (POST) - Analyze PDF chunk-by-chunk",
            "docs": "/docs",
        },
        "description": {
            "upload_pdf": "Returns single verdict for entire PDF",
            "upload_pdf_chunks": "Returns individual verdicts for each chunk (configurable chunk size and overlap)",
        },
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
