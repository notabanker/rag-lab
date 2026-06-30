import hashlib
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from .config import DEFAULT_EMBED_MODEL, DEFAULT_PROVIDER, PROVIDERS, get_provider
from .logging import setup_logging
from .parsers import pick_parser
from . import chunker, embedder, vector_store
from .retriever import retrieve
from .web import app as web_app

app = typer.Typer(help="rag-lab CLI — standalone RAG learning project")
console = Console()


def _list_providers():
    return ", ".join(PROVIDERS)


@app.callback()
def main(
    db_path: str = typer.Option("./chroma_db", "--db-path", help="Path to ChromaDB persist directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    setup_logging(verbose=verbose)
    vector_store.init_store(db_path)


@app.command()
def ingest(
    file_path: str = typer.Argument(..., help="Path to PDF / EPUB / Markdown file"),
    strategy: str = typer.Option("sentence", "--strategy", help="fixed | sentence"),
    chunk_size: int = typer.Option(512, "--size"),
    overlap: int = typer.Option(64, "--overlap", help="Character overlap (fixed) or sentence count (sentence)"),
    embed_model: str = typer.Option(None, "--embed-model", help="SentenceTransformer model name"),
):
    """Ingest one file into the vector store."""
    p = Path(file_path)
    if not p.exists():
        typer.echo(f"❌ File not found: {file_path}")
        raise typer.Exit(1)
    console.print(f"[blue]Parsing[/blue] {p.name}...")
    parser = pick_parser(file_path)
    text = parser(file_path)
    console.print(f"[blue]Chunking[/blue] strategy={strategy} size={chunk_size} overlap={overlap}...")
    kwargs = {"text": text}
    if strategy == "fixed":
        kwargs["size"] = chunk_size
        kwargs["overlap"] = overlap
    else:
        kwargs["target_size"] = chunk_size
        kwargs["overlap"] = overlap
    chunks = chunker.chunk(**kwargs, strategy=strategy)
    console.print(f"[blue]Embedding[/blue] {len(chunks)} chunks (CPU)...")
    embed_name = embed_model or DEFAULT_EMBED_MODEL
    vecs = embedder.embed([c.text for c in chunks], model_name=embed_name)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()[:10]
    metadatas = [{"source": str(p), "chunk_idx": i, "strategy": strategy, "file_sha": sha} for i in range(len(chunks))]
    ids = [f"{sha}-{i}" for i in range(len(chunks))]
    vector_store.upsert(chunks, vecs, metadatas, ids)
    console.print(f"[green]✅ Ingested[/green] {len(chunks)} chunks from {p.name}")


@app.command()
def query(
    question: str = typer.Argument(...),
    top_k: int = typer.Option(20, "--top-k"),
    min_score: int = typer.Option(8, "--min-score"),
    show_trace: bool = typer.Option(False, "--trace"),
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", help=f"LLM provider ({_list_providers()})"),
    model: str = typer.Option(None, "--model", help="Override provider default model"),
    embed_model: str = typer.Option(None, "--embed-model", help="SentenceTransformer model for embeddings"),
):
    """Run the /goal retrieval loop on a question."""
    prov_cfg = get_provider(provider)
    info = f"provider={provider}, model={model or prov_cfg.model}, min_score={min_score}"
    console.print(f"[blue]Querying[/blue] '{question}' ({info})...")
    result = retrieve(
        question, top_k=top_k, min_score=min_score,
        provider=provider, model=model, embed_model=embed_model,
    )
    console.print(f"\n[bold green]Answer[/bold green] (after {result['iterations']} iteration(s)):")
    console.print(result["answer"])
    v = result["verifier"]
    console.print(f"\n[bold]Verifier[/bold]: score={v.get('score')}/10 verdict={v.get('verdict')}")
    if v.get("issues"):
        console.print(f"  [yellow]issues[/yellow]: {v.get('issues')}")
    if result.get("partial"):
        console.print("  [red]⚠ partial — max iters reached[/red]")
    if show_trace:
        console.print("\n[bold]Trace[/bold]:")
        console.print_json(data=result["trace"])


@app.command()
def index(
    vault_path: str = typer.Option(None, "--vault", help="Path to Obsidian vault root"),
    embed_model: str = typer.Option(None, "--embed-model", help="SentenceTransformer model for embeddings"),
):
    """Batch-index all markdown files from a vault directory."""
    from .indexer import index_vault
    vault_root = vault_path or vector_store._PERSIST_DIR
    try:
        result = index_vault(vault_root, embed_model=embed_model)
    except FileNotFoundError as e:
        typer.echo(f"❌ {e}")
        raise typer.Exit(1)
    console.print(f"[green]Indexed {result['files_indexed']} files, {result['chunks_total']} chunks[/green]")
    if result["by_collection"]:
        for coll, n in sorted(result["by_collection"].items()):
            console.print(f"  {coll}: {n} chunks")


@app.command()
def watch_cmd(
    vault_path: str = typer.Option(..., "--vault", help="Path to Obsidian vault root"),
    debounce: float = typer.Option(2.0, "--debounce", help="Seconds to wait before indexing changed file"),
    embed_model: str = typer.Option(None, "--embed-model", help="SentenceTransformer model for embeddings"),
):
    """Watch vault for changes and auto-index markdown files."""
    from .watcher import watch
    console.print(f"[green]Watching {vault_path} for changes... (Ctrl+C to stop)[/green]")
    watch(vault_path, embed_model=embed_model, debounce=debounce)


@app.command()
def delete(
    file_sha: str = typer.Argument(..., help="File SHA prefix to delete chunks for"),
):
    """Delete chunks from the store by file SHA."""
    n = vector_store.delete_by_source(file_sha)
    console.print(f"[green]Deleted {n} chunks[/green]")


@app.command()
def stats():
    """Show vector store stats."""
    n = vector_store.count()
    prov_cfg = get_provider()
    t = Table(title="Vector store")
    t.add_column("Metric")
    t.add_column("Value")
    t.add_row("Collection", "rag_lab")
    t.add_row("Chunk count", str(n))
    t.add_row("DB path", vector_store._PERSIST_DIR)
    t.add_row("Embedder", DEFAULT_EMBED_MODEL)
    t.add_row("LLM", f"{prov_cfg.name} {prov_cfg.model}")
    console.print(t)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Start the web test console."""
    import uvicorn
    console.print(f"[green]Starting rag-lab test console at http://{host}:{port}[/green]")
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
