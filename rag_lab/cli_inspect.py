"""Inspection subcommand groups: runs, docs, collections, config."""
import json

import typer
from rich.console import Console
from rich.table import Table

from . import runs, vector_store

console = Console()

runs_app = typer.Typer(help="Inspect logged retrieval runs")
collections_app = typer.Typer(help="Inspect Chroma collections")
docs_app = typer.Typer(help="Inspect and delete indexed documents")
config_app = typer.Typer(help="Inspect runtime configuration")


def _doc_not_found(identifier: str):
    typer.echo(f"❌ No document matched {identifier!r}")
    raise typer.Exit(1)


@runs_app.command("list")
def runs_list(limit: int = typer.Option(20, "--limit")):
    """Show recent retrieval runs."""
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
    found = vector_store.find_document(identifier)
    if found is None:
        _doc_not_found(identifier)
    console.print("[bold]Manifest[/bold]:")
    console.print_json(data=found["manifest"] or {})
    console.print("[bold]Indexed[/bold]:")
    console.print_json(data=found["indexed"] or {})


@docs_app.command("reingest")
def docs_reingest(
    identifier: str = typer.Argument(..., help="doc_id, file_sha, source, or source basename"),
    strategy: str = typer.Option("sentence", "--strategy", help="fixed | sentence"),
    chunk_size: int = typer.Option(512, "--size"),
    overlap: int = typer.Option(64, "--overlap"),
    parent_size: int = typer.Option(4, "--parent-size"),
):
    """Reingest a manifest document from its original source path."""
    from .cli import _ingest_one
    source = vector_store.document_source(identifier)
    if not source:
        _doc_not_found(identifier)
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


@config_app.command("show")
def config_show():
    """Show runtime model and collection configuration."""
    from .config import (
        DEFAULT_COLLECTION, EMBEDDING_MODEL, LLM_BASE_URL, LLM_MODEL,
        LLM_VERIFIER_MODEL, RERANKER_MODEL, get_api_key,
    )
    t = Table(title="Runtime config")
    t.add_column("Setting")
    t.add_column("Value")
    for label, value in (
        ("LLM base URL", LLM_BASE_URL),
        ("Generator model", LLM_MODEL),
        ("Verifier model", LLM_VERIFIER_MODEL),
        ("API key present", "yes" if get_api_key() else "no"),
        ("Embedding model", EMBEDDING_MODEL),
        ("Reranker model", RERANKER_MODEL),
        ("Default collection", DEFAULT_COLLECTION),
        ("Active collection", vector_store.default_collection_name()),
        ("DB path", vector_store.persist_dir()),
    ):
        t.add_row(label, value)
    console.print(t)
