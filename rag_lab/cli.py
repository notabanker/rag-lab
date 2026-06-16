import hashlib
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from .parsers import pick_parser
from . import chunker, embedder, vector_store
from .retriever import retrieve
from .web import app as web_app

app = typer.Typer(help="rag-lab CLI — standalone RAG learning project")
console = Console()

@app.callback()
def main(
    db_path: str = typer.Option("./chroma_db", "--db-path", help="Path to ChromaDB persist directory"),
):
    vector_store.init_store(db_path)

@app.command()
def ingest(
    file_path: str = typer.Argument(..., help="Path to PDF / EPUB / Markdown file"),
    strategy: str = typer.Option("sentence", "--strategy", help="fixed | sentence"),
    chunk_size: int = typer.Option(512, "--size"),
    overlap: int = typer.Option(64, "--overlap"),
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
    if strategy == "fixed":
        chunks = chunker.chunk_fixed(text, size=chunk_size, overlap=overlap)
    else:
        chunks = chunker.chunk_sentence(text, target_size=chunk_size, overlap=max(1, overlap // 64))
    console.print(f"[blue]Embedding[/blue] {len(chunks)} chunks (CPU)...")
    vecs = embedder.embed([c.text for c in chunks])
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
):
    """Run the /goal retrieval loop on a question."""
    console.print(f"[blue]Querying[/blue] '{question}' (min_score={min_score})...")
    result = retrieve(question, top_k=top_k, min_score=min_score)
    console.print(f"\n[bold green]Answer[/bold green] (after {result['iterations']} iteration(s)):")
    console.print(result["answer"])
    v = result["verifier"]
    console.print(f"\n[bold]Verifier[/bold]: score={v.get('score')}/10 verdict={v.get('verdict')}")
    if v.get("issues"):
        console.print(f"  [yellow]issues[/yellow]: {v.get('issues')}")
    if result.get("partial"):
        console.print(f"  [red]⚠ partial — max iters reached[/red]")
    if show_trace:
        console.print("\n[bold]Trace[/bold]:")
        console.print_json(data=result["trace"])

@app.command()
def stats():
    """Show vector store stats."""
    n = vector_store.count()
    t = Table(title="Vector store")
    t.add_column("Metric"); t.add_column("Value")
    t.add_row("Collection", "rag_lab")
    t.add_row("Chunk count", str(n))
    t.add_row("DB path", vector_store._PERSIST_DIR)
    t.add_row("Embedder", "all-MiniLM-L6-v2 (CPU)")
    t.add_row("LLM", "DeepSeek deepseek-chat")
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
