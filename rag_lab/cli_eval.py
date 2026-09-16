"""Eval commands: `rag eval` and `rag compare`."""
import typer
from rich.console import Console
from rich.table import Table

from . import vector_store
from .config import LLM_MODEL, LLM_VERIFIER_MODEL
from .retriever import RetrievalConfig

console = Console()


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


def _run_eval_with_gates(questions_file, cfg, retrieval_only, variant, gates_file):
    from . import evaluation, gates
    report = evaluation.run_eval(questions_file, cfg, retrieval_only=retrieval_only, variant=variant, secure=False)
    if gates_file:
        report["gates"] = gates.evaluate_gates(
            report["summary"],
            gates.load_gates(gates_file),
            include_answer=not retrieval_only,
        )
    return report


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
    try:
        cfg = RetrievalConfig(mode=mode, top_k=top_k, rerank_top=rerank,
                              use_reranker=not no_rerank, small_to_big=small_to_big,
                              parent_top_k=parent_top_k, max_context_chars=max_context_chars,
                              min_score=min_score)
    except ValueError as e:
        typer.echo(f"❌ {e}")
        raise typer.Exit(1)
    if not retrieval_only:
        console.print(
            f"[yellow]Running all questions through the full LLM loop with {LLM_MODEL} "
            f"and verifier {LLM_VERIFIER_MODEL} — expect roughly 2 API calls per question plus retries.[/yellow]"
        )
    try:
        report = _run_eval_with_gates(questions_file, cfg, retrieval_only, variant, gates_file)
    except ValueError as e:
        typer.echo(f"❌ {e}")
        raise typer.Exit(1)
    gate_result = report.get("gates")
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
        console.print(f"[dim]Saved as eval run #{report['eval_id']}[/dim]")
    if gate_result and not gate_result["passed"]:
        raise typer.Exit(2)


def compare(
    names: list[str] = typer.Argument(None, help="Variant names to compare (default: all in the file)"),
    variants_file: str = typer.Option("eval/variants.yaml", "--variants", help="Variant definitions YAML"),
    questions_file: str = typer.Option("eval/questions.yaml", "--questions"),
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="Skip LLM calls; retrieval metrics only"),
    small_to_big: bool | None = typer.Option(None, "--small-to-big/--flat", help="Override variants' context expansion"),
):
    """Run the eval under multiple configs and show a side-by-side table."""
    from . import evaluation
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
        console.print("[yellow]Running all questions × all variants through the full LLM loop — this makes API calls.[/yellow]")

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
            report = evaluation.run_eval(
                questions_file, cfg, retrieval_only=retrieval_only,
                variant=v.name, secure=False, extra_config=v.config(),
            )
            summaries[v.name] = report["summary"]
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
