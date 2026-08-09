import json
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from . import ingestion, vector_store
from .config import (
    DEFAULT_COLLECTION, EMBEDDING_MODEL, LLM_BASE_URL, LLM_MODEL,
    LLM_VERIFIER_MODEL, RERANKER_MODEL, default_db_path, get_api_key,
    get_api_token, is_loopback_host,
)
from .retriever import RetrievalConfig, retrieve
from .web import app as web_app

app = typer.Typer(help="rag-lab CLI — standalone RAG learning project")
runs_app = typer.Typer(help="Inspect logged retrieval runs")
collections_app = typer.Typer(help="Inspect Chroma collections")
docs_app = typer.Typer(help="Inspect and delete indexed documents")
config_app = typer.Typer(help="Inspect runtime configuration")
mcp_app = typer.Typer(help="Run the rag-lab MCP server")
app.add_typer(runs_app, name="runs")
app.add_typer(collections_app, name="collections")
app.add_typer(docs_app, name="docs")
app.add_typer(config_app, name="config")
app.add_typer(mcp_app, name="mcp")
console = Console()

@app.callback()
def main(
    db_path: str = typer.Option(None, "--db-path", help="ChromaDB persist directory (default: $RAG_DB_PATH or ~/.local/share/rag-lab/chroma_db)"),
    collection: str = typer.Option(DEFAULT_COLLECTION, "--collection", help="ChromaDB collection name"),
):
    resolved = db_path or default_db_path()
    _warn_legacy_db(resolved)
    vector_store.init_store(resolved)
    vector_store.set_default_collection(collection)

def _warn_legacy_db(resolved: str):
    legacy = Path("./chroma_db")
    if legacy.is_dir() and legacy.resolve() != Path(resolved).resolve():
        console.print(
            f"[yellow]Note:[/yellow] ./chroma_db exists but the active DB is {resolved} — "
            f"migrate with: mv ./chroma_db {resolved}  (or pass --db-path ./chroma_db)"
        )

def _ingest_one(
    file_path: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    parent_size: int,
    force_model_mismatch: bool,
    allow_empty: bool = False,
    ocr: str = "auto",
) -> dict:
    p = Path(file_path)
    if not p.exists():
        raise ValueError(f"File not found: {file_path}")
    console.print(f"[blue]Parsing[/blue] {p.name}...")
    result = ingestion.ingest_file(
        file_path, ocr=ocr, strategy=strategy, chunk_size=chunk_size,
        overlap=overlap, parent_size=parent_size,
        force_model_mismatch=force_model_mismatch, allow_empty=allow_empty,
    )
    quality = result["quality"]
    console.print(
        f"[blue]Parsed[/blue] {quality['sections']} {quality['section_unit']}(s), "
        f"{quality['total_chars']} chars"
    )
    for w in quality["warnings"]:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    console.print(f"[blue]Chunking[/blue] strategy={strategy} size={chunk_size} overlap={overlap}...")
    if result["chunks"]:
        console.print(f"[blue]Embedding[/blue] {result['chunks']} chunks with {EMBEDDING_MODEL} (CPU)...")
    console.print(f"[green]✅ Ingested[/green] {result['chunks']} chunks from {p.name}")
    return result

@app.command()
def ingest(
    file_path: str = typer.Argument(..., help="Path to PDF / EPUB / Markdown file"),
    strategy: str = typer.Option("sentence", "--strategy", help="fixed | sentence"),
    chunk_size: int = typer.Option(512, "--size"),
    overlap: int = typer.Option(64, "--overlap"),
    parent_size: int = typer.Option(4, "--parent-size", help="Child chunks grouped into one parent context"),
    force_model_mismatch: bool = typer.Option(False, "--force-model-mismatch", help="Allow ingest into a legacy/mismatched collection"),
    allow_empty: bool = typer.Option(False, "--allow-empty", help="Record a zero-text file in the manifest instead of failing"),
    ocr: bool | None = typer.Option(
        None, "--ocr/--no-ocr",
        help="OCR scanned PDF pages (default: auto — on if tesseract is installed)",
    ),
):
    """Ingest one file into the vector store."""
    try:
        ocr_mode = "auto" if ocr is None else ("on" if ocr else "off")
        _ingest_one(file_path, strategy, chunk_size, overlap, parent_size, force_model_mismatch, allow_empty, ocr=ocr_mode)
    except ValueError as e:
        typer.echo(f"❌ {e}")
        raise typer.Exit(1)

@app.command()
def rebuild(
    file_paths: list[str] = typer.Argument(..., help="Files to ingest into the active collection"),
    strategy: str = typer.Option("sentence", "--strategy", help="fixed | sentence"),
    chunk_size: int = typer.Option(512, "--size"),
    overlap: int = typer.Option(64, "--overlap"),
    parent_size: int = typer.Option(4, "--parent-size"),
    force_model_mismatch: bool = typer.Option(False, "--force-model-mismatch"),
    ocr: bool | None = typer.Option(
        None, "--ocr/--no-ocr",
        help="OCR scanned PDF pages (default: auto — on if tesseract is installed)",
    ),
):
    """Batch-ingest files into the active collection using V3 metadata."""
    total = 0
    ocr_mode = "auto" if ocr is None else ("on" if ocr else "off")
    for file_path in file_paths:
        try:
            result = _ingest_one(file_path, strategy, chunk_size, overlap, parent_size, force_model_mismatch, ocr=ocr_mode)
            total += result["chunks"]
        except ValueError as e:
            typer.echo(f"❌ {file_path}: {e}")
            raise typer.Exit(1)
    console.print(f"[bold green]Rebuild complete[/bold green]: {total} chunks in {vector_store.default_collection_name()}")

@app.command()
def query(
    question: str = typer.Argument(...),
    mode: str = typer.Option("hybrid", "--mode", help="vector | lexical | hybrid"),
    top_k: int = typer.Option(50, "--top-k", help="Candidates fetched before reranking"),
    rerank: int = typer.Option(5, "--rerank", help="Child candidates selected after fusion/rerank"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip the cross-encoder reranker"),
    small_to_big: bool = typer.Option(True, "--small-to-big/--flat", help="Expand child hits into parent context"),
    parent_top_k: int = typer.Option(5, "--parent-top-k", help="Parent contexts passed to LLM"),
    max_context_chars: int = typer.Option(12000, "--max-context-chars", help="Context character budget"),
    min_score: int = typer.Option(8, "--min-score"),
    keyword: str = typer.Option(None, "--keyword", help="Literal substring for keyword retrieval (bypasses vector search)"),
    max_tokens: int = typer.Option(600, "--max-tokens", help="Max output tokens"),
    show_trace: bool = typer.Option(False, "--trace"),
):
    """Run the /goal retrieval loop on a question."""
    try:
        cfg = RetrievalConfig(
            mode=mode, top_k=top_k, rerank_top=rerank, use_reranker=not no_rerank,
            small_to_big=small_to_big, parent_top_k=parent_top_k,
            max_context_chars=max_context_chars, min_score=min_score,
            keyword=keyword, max_tokens=max_tokens,
        )
    except ValueError as e:
        typer.echo(f"❌ {e}")
        raise typer.Exit(1)
    console.print(f"[blue]Querying[/blue] '{question}' (mode={mode} min_score={min_score})...")
    result = retrieve(question, cfg)
    console.print(f"\n[bold green]Answer[/bold green] (after {result['iterations']} iteration(s)):")
    console.print(result["answer"])
    v = result["verifier"]
    console.print(f"\n[bold]Verifier[/bold]: score={v.get('score')}/10 verdict={v.get('verdict')}")
    cv = result.get("citation_validation") or {}
    if cv:
        status = "valid" if cv.get("citation_valid") else "invalid"
        console.print(f"[bold]Citations[/bold]: {status} ({cv.get('citation_count', 0)} cited)")
        if cv.get("citation_errors"):
            console.print(f"  [yellow]citation issues[/yellow]: {cv.get('citation_errors')}")
    if v.get("issues"):
        console.print(f"  [yellow]issues[/yellow]: {v.get('issues')}")
    if result.get("partial"):
        console.print("  [red]⚠ partial — max iters reached[/red]")
    if result.get("run_id"):
        console.print(f"[dim]Logged as run #{result['run_id']} — see `rag runs show {result['run_id']}`[/dim]")
    if result.get("log_error"):
        console.print(f"[yellow]⚠ {result['log_error']}[/yellow]")
    if show_trace:
        console.print("\n[bold]Trace[/bold]:")
        console.print_json(data=result["trace"])

@app.command()
def stats():
    """Show vector store stats."""
    n = vector_store.count()
    meta = vector_store.collection_metadata()
    t = Table(title="Vector store")
    t.add_column("Metric"); t.add_column("Value")
    t.add_row("Collection", vector_store.default_collection_name())
    t.add_row("Chunk count", str(n))
    t.add_row("Documents", str(vector_store.distinct_sources()))
    t.add_row("DB path", vector_store.persist_dir())
    t.add_row("Configured embedder", EMBEDDING_MODEL)
    t.add_row("Collection embedder", str(meta.get("embedding_model") or "unknown"))
    t.add_row("Reranker", RERANKER_MODEL)
    t.add_row("LLM", LLM_MODEL)
    t.add_row("Verifier LLM", LLM_VERIFIER_MODEL)
    console.print(t)

@config_app.command("show")
def config_show():
    """Show runtime model and collection configuration."""
    t = Table(title="Runtime config")
    t.add_column("Setting"); t.add_column("Value")
    t.add_row("LLM base URL", LLM_BASE_URL)
    t.add_row("Generator model", LLM_MODEL)
    t.add_row("Verifier model", LLM_VERIFIER_MODEL)
    t.add_row("API key present", "yes" if get_api_key() else "no")
    t.add_row("Embedding model", EMBEDDING_MODEL)
    t.add_row("Reranker model", RERANKER_MODEL)
    t.add_row("Default collection", DEFAULT_COLLECTION)
    t.add_row("Active collection", vector_store.default_collection_name())
    t.add_row("DB path", vector_store.persist_dir())
    console.print(t)

@mcp_app.command("serve")
def mcp_serve():
    """Run the MCP server on stdio."""
    from .mcp_server import main as mcp_main
    mcp_main()

def _fmt_pct(v) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"

def _fmt_num(v, suffix="") -> str:
    if v is None:
        return "—"
    return f"{v:.2f}{suffix}" if isinstance(v, float) else f"{v}{suffix}"

_METRIC_ROWS = [
    ("Hit rate @1", "hit_at_1", _fmt_pct, max),
    ("Hit rate @3", "hit_at_3", _fmt_pct, max),
    ("Hit rate @k", "hit_rate", _fmt_pct, max),
    ("Hit rate @10", "hit_at_10", _fmt_pct, max),
    ("MRR", "mrr", _fmt_num, max),
    ("Chunk hit rate @k", "chunk_hit_rate", _fmt_pct, max),
    ("Chunk MRR", "chunk_mrr", _fmt_num, max),
    ("Context hit rate", "context_hit_rate", _fmt_pct, max),
    ("Context chunk hit", "context_chunk_hit_rate", _fmt_pct, max),
    ("Fragment rate", "fragment_rate", _fmt_pct, max),
    ("Verifier mean", "verifier_mean", _fmt_num, max),
    ("Refusal accuracy", "refusal_accuracy", _fmt_pct, max),
    ("Citation validity", "citation_validity_rate", _fmt_pct, max),
    ("Retrieval latency (ms)", "retrieval_latency_ms_mean", _fmt_num, min),
    ("Mean latency (ms)", "latency_ms_mean", _fmt_num, min),
    ("Context chars", "context_chars_mean", _fmt_num, min),
    ("Prompt tokens", "prompt_tokens", _fmt_num, min),
    ("Completion tokens", "completion_tokens", _fmt_num, min),
]

def _render_eval(report: dict, variant: str):
    per = report["per_question"]
    t = Table(title=f"Eval — {variant} ({len(per)} questions)")
    for col in ("ID", "Hit@k", "Rank", "C-Hit", "C-Rank", "Frag", "Score", "Cite", "Refusal", "Latency"):
        t.add_column(col)
    def _mark(v):
        return "—" if v is None else ("[green]✓[/green]" if v else "[red]✗[/red]")
    for e in per:
        if e["expect_refusal"]:
            hit, rank, chit, crank, frag = "—", "—", "—", "—", "—"
            cite = _mark(e.get("citation_valid")) if "citation_valid" in e else "—"
            refusal = _mark(e.get("refusal_correct")) if "refusal_correct" in e else "—"
        else:
            hit = _mark(e.get("hit_at_k")) if "hit_at_k" in e else "—"
            rank = str(e.get("first_rank") or "—")
            chit = _mark(e.get("chunk_hit_at_k")) if "chunk_hit_at_k" in e else "—"
            crank = str(e.get("chunk_first_rank") or "—") if "chunk_hit_at_k" in e else "—"
            frag = _mark(e.get("fragment_matched"))
            cite = _mark(e.get("citation_valid")) if "citation_valid" in e else "—"
            refusal = "[red]refused[/red]" if e.get("refused") else "—"
        score = e.get("verifier_score")
        latency = f"{e['latency_ms']}ms" if e.get("latency_ms") is not None else "—"
        t.add_row(e["id"], hit, rank, chit, crank, frag, str(score) if score is not None else "—", cite, refusal, latency)
    console.print(t)
    s = report["summary"]
    console.print(
        f"[bold]Summary[/bold]: hit_rate={_fmt_pct(s['hit_rate'])} mrr={_fmt_num(s['mrr'])} "
        f"chunk_hit={_fmt_pct(s.get('chunk_hit_rate'))} chunk_mrr={_fmt_num(s.get('chunk_mrr'))} "
        f"fragments={_fmt_pct(s['fragment_rate'])} verifier={_fmt_num(s['verifier_mean'])} "
        f"refusals={_fmt_pct(s['refusal_accuracy'])} citations={_fmt_pct(s.get('citation_validity_rate'))} "
        f"tokens={s['prompt_tokens']}+{s['completion_tokens']}"
    )

@app.command("eval")
def eval_cmd(
    questions_file: str = typer.Option("eval/questions.yaml", "--questions", help="Golden question set YAML"),
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="Skip LLM calls; retrieval metrics only"),
    mode: str = typer.Option("hybrid", "--mode", help="vector | lexical | hybrid"),
    top_k: int = typer.Option(50, "--top-k", help="Candidates fetched before reranking"),
    rerank: int = typer.Option(5, "--rerank", help="Child candidates selected after fusion/rerank (the k in hit@k)"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip the cross-encoder reranker"),
    small_to_big: bool = typer.Option(True, "--small-to-big/--flat", help="Expand child hits into parent context"),
    parent_top_k: int = typer.Option(5, "--parent-top-k", help="Parent contexts passed to LLM"),
    max_context_chars: int = typer.Option(12000, "--max-context-chars"),
    min_score: int = typer.Option(8, "--min-score"),
    variant: str = typer.Option("default", "--variant", help="Label stored with this eval run"),
    gates_file: str = typer.Option(None, "--gate", help="Metric gates YAML; exits non-zero on failure"),
    json_out: bool = typer.Option(False, "--json", help="Print the full report as JSON"),
):
    """Run the golden question set and report retrieval/answer metrics."""
    from . import evaluation, runs
    try:
        questions = evaluation.load_questions(questions_file)
        cfg = RetrievalConfig(mode=mode, top_k=top_k, rerank_top=rerank,
                              use_reranker=not no_rerank, small_to_big=small_to_big,
                              parent_top_k=parent_top_k, max_context_chars=max_context_chars,
                              min_score=min_score)
    except ValueError as e:
        typer.echo(f"❌ {e}")
        raise typer.Exit(1)
    if not retrieval_only:
        console.print(
            f"[yellow]Running {len(questions)} questions through the full LLM loop with {LLM_MODEL} "
            f"and verifier {LLM_VERIFIER_MODEL} — expect roughly {len(questions) * 2} API calls plus retries.[/yellow]"
        )
    report = evaluation.evaluate(questions, cfg, retrieval_only=retrieval_only)
    report["config"]["collection"] = vector_store.default_collection_name()
    gate_result = None
    if gates_file:
        try:
            from . import gates
            gate_result = gates.evaluate_gates(
                report["summary"],
                gates.load_gates(gates_file),
                include_answer=not retrieval_only,
            )
            report["gates"] = gate_result
        except ValueError as e:
            typer.echo(f"❌ {e}")
            raise typer.Exit(1)
    eval_id = runs.log_eval(variant, report["config"], report["summary"], report["per_question"])
    if json_out:
        console.print_json(data=report)
    else:
        _render_eval(report, variant)
        if gate_result:
            if gate_result["passed"]:
                console.print("[green]Gates passed[/green]")
            else:
                console.print("[red]Gates failed[/red]:")
                for f in gate_result["failures"]:
                    console.print(f"  {f['section']}.{f['metric']}: {f['value']} < {f['threshold']} ({f['reason']})")
            if gate_result.get("skipped"):
                console.print(f"[dim]Skipped {len(gate_result['skipped'])} answer gate(s) in retrieval-only mode[/dim]")
        console.print(f"[dim]Saved as eval run #{eval_id}[/dim]")
    if gate_result and not gate_result["passed"]:
        raise typer.Exit(2)

@app.command()
def compare(
    names: list[str] = typer.Argument(None, help="Variant names to compare (default: all in the file)"),
    variants_file: str = typer.Option("eval/variants.yaml", "--variants", help="Variant definitions YAML"),
    questions_file: str = typer.Option("eval/questions.yaml", "--questions"),
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="Skip LLM calls; retrieval metrics only"),
    small_to_big: bool | None = typer.Option(None, "--small-to-big/--flat", help="Override variants' context expansion"),
):
    """Run the eval under multiple configs and show a side-by-side table."""
    from . import evaluation, runs
    try:
        questions = evaluation.load_questions(questions_file)
        variants = evaluation.load_variants(variants_file)
    except ValueError as e:
        typer.echo(f"❌ {e}")
        raise typer.Exit(1)
    if names:
        by_name = {v.name: v for v in variants}
        missing = [n for n in names if n not in by_name]
        if missing:
            typer.echo(f"❌ Unknown variant(s): {missing}. Available: {list(by_name)}")
            raise typer.Exit(1)
        variants = [by_name[n] for n in names]
    if len(variants) < 2:
        typer.echo("❌ Need at least 2 variants to compare")
        raise typer.Exit(1)
    if not retrieval_only:
        console.print(f"[yellow]Running {len(questions)} questions × {len(variants)} variants through the full LLM loop — this makes API calls.[/yellow]")

    original_collection = vector_store.default_collection_name()
    summaries = {}
    try:
        for v in variants:
            vector_store.set_default_collection(v.collection or original_collection)
            cfg = v.to_retrieval_config()
            if small_to_big is not None:
                cfg.small_to_big = small_to_big
            console.print(f"[blue]Evaluating[/blue] variant '{v.name}' "
                          f"(mode={v.mode} top_k={v.top_k} rerank={v.rerank_top} reranker={'on' if v.use_reranker else 'off'} "
                          f"small_to_big={'on' if cfg.small_to_big else 'off'} "
                          f"collection={vector_store.default_collection_name()})...")
            report = evaluation.evaluate(questions, cfg, retrieval_only=retrieval_only)
            summaries[v.name] = report["summary"]
            runs.log_eval(v.name, {**report["config"], **v.config()}, report["summary"], report["per_question"])
    finally:
        vector_store.set_default_collection(original_collection)

    t = Table(title=f"Compare — {len(questions)} questions")
    t.add_column("Metric")
    for v in variants:
        t.add_column(v.name)
    for label, key, fmt, best_fn in _METRIC_ROWS:
        values = {name: s.get(key) for name, s in summaries.items()}
        present = [x for x in values.values() if x is not None]
        best = best_fn(present) if len(present) > 1 and len(set(present)) > 1 else None
        cells = []
        for v in variants:
            val = values[v.name]
            text = fmt(val)
            cells.append(f"[green]{text}[/green]" if best is not None and val == best else text)
        t.add_row(label, *cells)
    console.print(t)

@collections_app.command("list")
def collections_list():
    """Show Chroma collections and their index metadata."""
    rows = vector_store.list_collections()
    if not rows:
        console.print("No collections found.")
        return
    t = Table(title="Collections")
    for col in ("Name", "Chunks", "Embedder", "Chunking", "Index"):
        t.add_column(col)
    for r in rows:
        meta = r.get("metadata") or {}
        t.add_row(
            r["name"], str(r["count"]),
            str(meta.get("embedding_model") or "unknown"),
            str(meta.get("chunking_version") or "unknown"),
            str(meta.get("index_version") or "unknown"),
        )
    console.print(t)

@collections_app.command("delete")
def collections_delete(name: str = typer.Argument(..., help="Collection name to delete")):
    """Delete a Chroma collection."""
    if name == vector_store.default_collection_name():
        typer.echo("❌ Refusing to delete the active collection. Switch --collection first.")
        raise typer.Exit(1)
    if not vector_store.delete_collection(name):
        typer.echo(f"❌ No collection named {name!r}")
        raise typer.Exit(1)
    console.print(f"[green]Deleted collection[/green] {name}")

@docs_app.command("list")
def docs_list():
    """Show indexed documents in the active collection."""
    rows = vector_store.list_documents()
    if not rows:
        console.print("No documents indexed.")
        return
    t = Table(title=f"Documents — {vector_store.default_collection_name()}")
    for col in ("Doc ID", "Source", "Chunks", "Strategy", "Embedder"):
        t.add_column(col)
    for r in rows:
        t.add_row(
            str(r.get("doc_id") or "—"),
            str(r.get("source") or "—"),
            str(r.get("chunks") or 0),
            str(r.get("strategy") or "—"),
            str(r.get("embedding_model") or "unknown"),
        )
    console.print(t)

@docs_app.command("show")
def docs_show(identifier: str = typer.Argument(..., help="doc_id, file_sha, source, or source basename")):
    """Show one manifest entry and current indexed document summary."""
    from . import manifest
    doc = manifest.get_document(identifier)
    indexed = next((
        d for d in vector_store.list_documents()
        if identifier in {d.get("doc_id"), d.get("file_sha"), d.get("source"), d.get("basename")}
    ), None)
    if not doc and not indexed:
        typer.echo(f"❌ No document matched {identifier!r}")
        raise typer.Exit(1)
    console.print("[bold]Manifest[/bold]:")
    console.print_json(data=doc or {})
    console.print("[bold]Indexed[/bold]:")
    console.print_json(data=indexed or {})

@docs_app.command("reingest")
def docs_reingest(
    identifier: str = typer.Argument(..., help="doc_id, file_sha, source, or source basename"),
    strategy: str = typer.Option("sentence", "--strategy", help="fixed | sentence"),
    chunk_size: int = typer.Option(512, "--size"),
    overlap: int = typer.Option(64, "--overlap"),
    parent_size: int = typer.Option(4, "--parent-size"),
):
    """Reingest a manifest document from its original source path."""
    from . import manifest
    doc = manifest.get_document(identifier)
    indexed = next((
        d for d in vector_store.list_documents()
        if identifier in {d.get("doc_id"), d.get("file_sha"), d.get("source"), d.get("basename")}
    ), None)
    source = (doc or indexed or {}).get("source")
    if not source:
        typer.echo(f"❌ No document matched {identifier!r}")
        raise typer.Exit(1)
    try:
        _ingest_one(source, strategy, chunk_size, overlap, parent_size, False)
    except ValueError as e:
        typer.echo(f"❌ {e}")
        raise typer.Exit(1)

@docs_app.command("delete")
def docs_delete(identifier: str = typer.Argument(..., help="doc_id, file_sha, source, or source basename")):
    """Delete one indexed document from the active collection."""
    removed = vector_store.delete_document(identifier)
    if not removed:
        typer.echo(f"❌ No indexed document matched {identifier!r}")
        raise typer.Exit(1)
    console.print(f"[green]Deleted[/green] {removed} chunks matching {identifier!r}")

@runs_app.command("list")
def runs_list(limit: int = typer.Option(20, "--limit")):
    """Show recent retrieval runs."""
    from . import runs
    rows = runs.list_runs(limit=limit)
    if not rows:
        console.print("No runs logged yet.")
        return
    t = Table(title="Retrieval runs")
    for col in ("ID", "Time", "Question", "Score", "Verdict", "Cite", "Iters", "Latency", "Tokens"):
        t.add_column(col)
    for r in rows:
        v = r.get("verifier") or {}
        cv = r.get("citation_validation") or {}
        t.add_row(
            str(r["id"]), r["timestamp"],
            (r["question"][:50] + "…") if len(r["question"]) > 50 else r["question"],
            str(v.get("score", "—")), str(v.get("verdict", "—")),
            "✓" if cv.get("citation_valid") else ("✗" if cv else "—"),
            str(r.get("iterations", "—")), f"{r.get('latency_ms', 0)}ms",
            f"{r.get('prompt_tokens', 0)}+{r.get('completion_tokens', 0)}",
        )
    console.print(t)

@runs_app.command("show")
def runs_show(run_id: int = typer.Argument(...)):
    """Show one run in full, including trace and retrieved chunks."""
    from . import runs
    r = runs.get_run(run_id)
    if r is None:
        typer.echo(f"❌ No run with id {run_id}")
        raise typer.Exit(1)
    console.print(f"[bold]Run #{r['id']}[/bold] — {r['timestamp']}")
    console.print(f"[bold]Question[/bold]: {r['question']}")
    console.print(f"[bold]Config[/bold]: {json.dumps(r['config'])}")
    console.print(f"[bold]Answer[/bold]:\n{r['answer']}")
    console.print(f"[bold]Verifier[/bold]: {json.dumps(r['verifier'])}")
    console.print(f"[bold]Citations[/bold]: {json.dumps(r.get('citation_validation') or {})}")
    console.print(f"[bold]Latency[/bold]: {r['latency_ms']}ms  [bold]Tokens[/bold]: {r['prompt_tokens']}+{r['completion_tokens']}")
    if r["chunks"]:
        t = Table(title="Retrieved chunks")
        for col in ("Rank", "Chunk", "Distance", "Source"):
            t.add_column(col)
        for c in r["chunks"]:
            dist = f"{c['distance']:.4f}" if c["distance"] is not None else "—"
            t.add_row(str(c["rank"]), c["chunk_id"], dist, c["source"] or "—")
        console.print(t)
    if r["trace"]:
        console.print("[bold]Trace[/bold]:")
        console.print_json(data=r["trace"])

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Start the web test console."""
    if not is_loopback_host(host) and not get_api_token():
        # Fail-closed: exposing the API beyond loopback without a token would
        # serve unauthenticated ingest/delete endpoints to the network.
        typer.echo(f"❌ Refusing to serve on {host} without RAG_API_TOKEN. Set it and retry.")
        raise typer.Exit(1)
    import uvicorn
    console.print(f"[green]Starting rag-lab test console at http://{host}:{port}[/green]")
    uvicorn.run(web_app, host=host, port=port, log_level="warning")

if __name__ == "__main__":
    app()
