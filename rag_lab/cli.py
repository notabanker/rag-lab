import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

from . import ingestion, vector_store
from .cli_eval import compare, eval_cmd
from .cli_inspect import collections_app, config_app, docs_app, runs_app
from .config import (
    DEFAULT_COLLECTION, EMBEDDING_MODEL, LLM_MODEL,
    LLM_VERIFIER_MODEL, RERANKER_MODEL, default_db_path,
    get_api_token, is_loopback_host,
)
from .retriever import RetrievalConfig, retrieve
from .web import app as web_app

app = typer.Typer(help="rag-lab CLI — standalone RAG learning project")
mcp_app = typer.Typer(help="Run the rag-lab MCP server")
app.add_typer(runs_app, name="runs")
app.add_typer(collections_app, name="collections")
app.add_typer(docs_app, name="docs")
app.add_typer(config_app, name="config")
app.add_typer(mcp_app, name="mcp")
app.command("eval")(eval_cmd)
app.command()(compare)
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


def _ocr_mode(ocr: bool | None) -> str:
    return "auto" if ocr is None else ("on" if ocr else "off")


_OCR_OPTION = typer.Option(None, "--ocr/--no-ocr", help="OCR scanned PDF pages (default: auto — on if tesseract is installed)")


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
    ocr: bool | None = _OCR_OPTION,
):
    """Ingest one file into the vector store."""
    try:
        _ingest_one(file_path, strategy, chunk_size, overlap, parent_size, force_model_mismatch, allow_empty, ocr=_ocr_mode(ocr))
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
    ocr: bool | None = _OCR_OPTION,
):
    """Batch-ingest files into the active collection using V3 metadata."""
    total = 0
    for file_path in file_paths:
        try:
            total += _ingest_one(file_path, strategy, chunk_size, overlap, parent_size, force_model_mismatch, ocr=_ocr_mode(ocr))["chunks"]
        except ValueError as e:
            typer.echo(f"❌ {file_path}: {e}")
            raise typer.Exit(1)
    console.print(f"[bold green]Rebuild complete[/bold green]: {total} chunks in {vector_store.default_collection_name()}")


@app.command()
def sync(
    dirs: list[str] = typer.Argument(..., help="Directories to scan recursively"),
    prune: bool = typer.Option(False, "--prune", help="Delete indexed docs whose source file is gone"),
    yes: bool = typer.Option(False, "--yes", help="Skip the prune confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change without changing anything"),
    ocr: bool | None = _OCR_OPTION,
    strategy: str = typer.Option("sentence", "--strategy", help="fixed | sentence"),
    chunk_size: int = typer.Option(512, "--size"),
    overlap: int = typer.Option(64, "--overlap"),
    parent_size: int = typer.Option(4, "--parent-size"),
):
    """Incrementally sync directories into the index (added/updated/unchanged/pruned)."""
    from . import manifest as mf
    from . import sync as sync_mod
    plan = sync_mod.build_plan(dirs, mf.list_documents())

    t = Table(title="Sync plan" + (" (dry run)" if dry_run else ""))
    t.add_column("Action")
    t.add_column("Count")
    t.add_row("added", str(len(plan.added)))
    t.add_row("updated", str(len(plan.updated)))
    t.add_row("unchanged", str(len(plan.unchanged)))
    t.add_row("pruned", str(len(plan.pruned)))
    if plan.failed:
        t.add_row("failed", str(len(plan.failed)))
    console.print(t)
    for src in plan.added + plan.updated + plan.pruned:
        console.print(f"  {src}")
    for f in plan.failed:
        console.print(f"[yellow]  failed: {f}[/yellow]")

    if not dry_run and plan.pruned:
        # --prune gates deletion itself, not just the confirmation: without
        # the flag a sync never deletes indexed docs (binding constraint).
        if not prune:
            plan.pruned = []
        elif not yes and not typer.confirm(f"Delete {len(plan.pruned)} indexed document(s) whose source is gone?"):
            plan.pruned = []

    counts = sync_mod.apply_plan(
        plan, dry_run=dry_run, ocr=_ocr_mode(ocr),
        strategy=strategy, chunk_size=chunk_size,
        overlap=overlap, parent_size=parent_size,
    )
    summary = (f"{counts['added']} added, {counts['updated']} updated, "
               f"{counts['unchanged']} unchanged, {counts['pruned']} pruned")
    if counts["failed"]:
        summary += f", {counts['failed']} failed"
    verb = "Would change" if dry_run else "Done"
    console.print(f"[green]{verb}[/green]: {summary}")


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
    """Run the retrieval loop on a question."""
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
    t.add_column("Metric")
    t.add_column("Value")
    for label, value in (
        ("Collection", vector_store.default_collection_name()),
        ("Chunk count", str(n)),
        ("Documents", str(vector_store.distinct_sources())),
        ("DB path", vector_store.persist_dir()),
        ("Configured embedder", EMBEDDING_MODEL),
        ("Collection embedder", str(meta.get("embedding_model") or "unknown")),
        ("Reranker", RERANKER_MODEL),
        ("LLM", LLM_MODEL),
        ("Verifier LLM", LLM_VERIFIER_MODEL),
    ):
        t.add_row(label, value)
    console.print(t)


@mcp_app.command("serve")
def mcp_serve():
    """Run the MCP server on stdio."""
    from .mcp_server import main as mcp_main
    mcp_main()



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
