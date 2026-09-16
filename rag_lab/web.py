import hmac
import logging
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import ingestion, runs, vector_store
from .config import (
    DEFAULT_COLLECTION, EMBEDDING_MODEL, LLM_BASE_URL, LLM_MODEL,
    LLM_VERIFIER_MODEL, MAX_INGEST_CHUNKS, MAX_UPLOAD_BYTES, RERANKER_MODEL,
    get_api_key, get_api_token,
)
from .parsers import PARSERS, as_result, pick_parser
from .retriever import RetrievalConfig, compact_chunk, retrieve

log = logging.getLogger("rag_lab.web")

# Host-header allowlist for no-token (loopback) mode. Browsers can be aimed
# at a loopback port via DNS rebinding (attacker.com -> 127.0.0.1); without
# this check a remote page could hit /api/query, /api/ingest, etc. same-origin.
# Token mode is exempt: the bearer gate is the real protection there and the
# server may legitimately be reached via a LAN hostname.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}

app = FastAPI(title="rag-lab test console")


@app.middleware("http")
async def validate_host_header(request: Request, call_next):
    if not get_api_token():
        host = (request.headers.get("host") or "").strip().lower()
        hostname = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        if hostname not in _LOOPBACK_HOSTS:
            log.warning("rejected request with Host %r (possible DNS rebinding)", host)
            return JSONResponse(status_code=403, content={"detail": "Host header not allowed"})
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


def require_api_token(authorization: str | None = Header(default=None)):
    """Bearer-token gate. No token configured = localhost mode, allow.
    Token configured = every /api route requires it. Fail-closed.
    """
    token = get_api_token()
    if not token:
        return
    if not hmac.compare_digest(authorization or "", f"Bearer {token}"):
        raise HTTPException(status_code=401, detail="Missing or invalid API token")


API_DEPENDENCIES = [Depends(require_api_token)]


@app.get("/health")
def health():
    try:
        return {
            "status": "ok",
            "chunks": vector_store.count(),
            "api_key_present": bool(get_api_key()),
            "collection": vector_store.default_collection_name(),
        }
    except Exception:
        log.exception("health check failed")
        return JSONResponse(status_code=503, content={"status": "degraded"})


@app.post("/api/ingest", dependencies=API_DEPENDENCIES)
def api_ingest(
    file: UploadFile = File(...),
    strategy: str = Form("sentence"),
    chunk_size: int = Form(512),
    overlap: int = Form(64),
):
    suffix = Path(file.filename or "").suffix
    if suffix.lower() not in PARSERS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or '(none)'}")
    if not (64 <= chunk_size <= 5000):
        raise HTTPException(status_code=400, detail="chunk_size must be between 64 and 5000")
    if not (0 <= overlap <= 1000):
        raise HTTPException(status_code=400, detail="overlap must be between 0 and 1000")
    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            parsed = as_result(pick_parser(tmp_path)(tmp_path))
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        quality = parsed.quality()
        chunks = ingestion.make_chunks(parsed.effective_text, strategy=strategy, chunk_size=chunk_size, overlap=overlap)

        if not chunks:
            raise HTTPException(status_code=400, detail=ingestion._no_text_error())
        if len(chunks) > MAX_INGEST_CHUNKS:
            raise HTTPException(
                status_code=413,
                detail=f"Too many chunks ({len(chunks)} > {MAX_INGEST_CHUNKS}) — raise chunk_size",
            )

        # Sanitize the stored source: a client filename must never become a
        # filesystem path (reingest later does Path(source).read_bytes()).
        source = Path(file.filename or "upload").name

        result = ingestion.ingest_text(
            source,
            content,
            parsed.effective_text,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            parse_quality=quality,
        )
        return {
            "status": "ok",
            "chunks": result["chunks"],
            "filename": file.filename,
            "doc_id": result["file_sha"],
            "warnings": quality["warnings"],
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("ingest failed: %s", file.filename)
        raise HTTPException(status_code=500, detail="Ingest failed")


class RetrievalParams(BaseModel):
    mode: str = Field("hybrid", pattern="^(vector|lexical|hybrid)$")
    top_k: int = Field(50, ge=1, le=200)
    rerank_top: int = Field(5, ge=1, le=50)
    use_reranker: bool = True
    small_to_big: bool = True
    parent_top_k: int = Field(5, ge=1, le=50)
    max_context_chars: int = Field(12000, ge=1, le=100000)

    def to_retrieval_config(self, **overrides) -> RetrievalConfig:
        return RetrievalConfig(
            mode=self.mode, top_k=self.top_k, rerank_top=self.rerank_top,
            use_reranker=self.use_reranker, small_to_big=self.small_to_big,
            parent_top_k=self.parent_top_k, max_context_chars=self.max_context_chars,
            **overrides,
        )


class QueryRequest(RetrievalParams):
    question: str
    min_score: int = Field(8, ge=1, le=10)


class EvalRequest(RetrievalParams):
    questions_file: str = "eval/questions.yaml"
    retrieval_only: bool = True
    variant: str = "dashboard"


class ReingestRequest(BaseModel):
    strategy: str = Field("sentence", pattern="^(fixed|sentence)$")
    chunk_size: int = Field(512, ge=64, le=5000)
    overlap: int = Field(64, ge=0, le=1000)
    parent_size: int = Field(4, ge=1, le=50)


@app.post("/api/query", dependencies=API_DEPENDENCIES)
def api_query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        result = retrieve(req.question, req.to_retrieval_config(min_score=req.min_score))
        return {
            "answer": result["answer"],
            "verifier": result["verifier"],
            "citation_validation": result.get("citation_validation"),
            "iterations": result["iterations"],
            "trace": result.get("trace", []),
            "partial": result.get("partial", False),
            "run_id": result.get("run_id"),
            "citations": [c.get("citation") for c in result.get("chunks", [])],
            "chunks": [compact_chunk(c) for c in result.get("chunks", [])],
            "usage": result.get("usage", {}),
            "latency_ms": result.get("latency_ms"),
        }
    except Exception:
        log.exception("query failed: %r", req.question)
        raise HTTPException(status_code=500, detail="Query failed")


@app.get("/api/config", dependencies=API_DEPENDENCIES)
def api_config():
    return {
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
        "llm_verifier_model": LLM_VERIFIER_MODEL,
        "api_key_present": bool(get_api_key()),
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "default_collection": DEFAULT_COLLECTION,
        "active_collection": vector_store.default_collection_name(),
    }


@app.get("/api/collections", dependencies=API_DEPENDENCIES)
def api_collections():
    return {"collections": vector_store.list_collections()}


@app.delete("/api/collections/{name}", dependencies=API_DEPENDENCIES)
def api_delete_collection(name: str):
    if name == vector_store.default_collection_name():
        raise HTTPException(status_code=400, detail="Refusing to delete the active collection")
    if not vector_store.delete_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection not found: {name}")
    return {"status": "ok", "collection": name}


@app.get("/api/docs", dependencies=API_DEPENDENCIES)
def api_docs():
    from . import manifest
    return {"documents": vector_store.list_documents(), "manifest": manifest.list_documents()}


@app.get("/api/docs/{doc_id}", dependencies=API_DEPENDENCIES)
def api_doc(doc_id: str):
    found = vector_store.find_document(doc_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return {"manifest": found["manifest"], "indexed": found["indexed"]}


@app.delete("/api/docs/{doc_id}", dependencies=API_DEPENDENCIES)
def api_delete_doc(doc_id: str):
    removed = vector_store.delete_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return {"status": "ok", "deleted_chunks": removed, "identifier": doc_id}


@app.post("/api/docs/{doc_id}/reingest", dependencies=API_DEPENDENCIES)
def api_reingest_doc(doc_id: str, req: ReingestRequest):
    source = vector_store.document_source(doc_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    path = Path(source)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Source file no longer exists: {path}")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    try:
        result = ingestion.ingest_file(
            str(path),
            strategy=req.strategy,
            chunk_size=req.chunk_size,
            overlap=req.overlap,
            parent_size=req.parent_size,
        )
        return {"status": "ok", "source": str(path), **result}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("reingest failed: %s", doc_id)
        raise HTTPException(status_code=500, detail="Reingest failed")


@app.get("/api/runs", dependencies=API_DEPENDENCIES)
def api_runs(limit: int = Query(50, ge=1, le=500)):
    return {"runs": runs.list_runs(limit=limit)}


@app.get("/api/runs/{run_id}", dependencies=API_DEPENDENCIES)
def api_run(run_id: int):
    run = runs.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


@app.get("/api/evals", dependencies=API_DEPENDENCIES)
def api_evals(limit: int = Query(50, ge=1, le=500)):
    return {"evals": runs.list_evals(limit=limit)}


@app.get("/api/evals/{eval_id}", dependencies=API_DEPENDENCIES)
def api_eval(eval_id: int):
    ev = runs.get_eval(eval_id)
    if ev is None:
        raise HTTPException(status_code=404, detail=f"Eval not found: {eval_id}")
    return ev


@app.post("/api/eval", dependencies=API_DEPENDENCIES)
def api_run_eval(req: EvalRequest):
    try:
        from . import evaluation
        report = evaluation.run_eval(
            req.questions_file,
            req.to_retrieval_config(),
            retrieval_only=req.retrieval_only,
            variant=req.variant,
        )
        return {"status": "ok", "eval_id": report["eval_id"], **report}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("eval failed: %s", req.questions_file)
        raise HTTPException(status_code=500, detail="Eval failed")


@app.get("/api/stats", dependencies=API_DEPENDENCIES)
def api_stats():
    meta = vector_store.collection_metadata()
    return {
        "chunk_count": vector_store.count(),
        "documents": vector_store.distinct_sources(),
        "collection": vector_store.default_collection_name(),
        "embedding_model": meta.get("embedding_model"),
        "chunking_version": meta.get("chunking_version"),
    }
