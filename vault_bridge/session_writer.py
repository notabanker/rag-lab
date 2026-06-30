from .writer import append_l1_today, write_session_l2

_IMPORTANCE_KEYWORDS = {
    "project", "config", "agent", "persona", "vault", "system",
    "memory", "plutos", "stoa", "hermes", "francesca", "rag",
    "deploy", "build", "fix", "error", "decision", "strategy",
    "bitcoin", "portfolio", "risk", "trade", "market",
    "architecture", "pipeline", "mcp", "tool", "hippocampus",
    "compressor", "index", "search", "watcher", "gateway",
    "orchestrator", "quant", "macro", "equity", "crypto",
}


def _is_important(task: str, decisions: list[str] | None, lessons: list[str] | None,
                  messages: list[dict]) -> bool:
    if task:
        return True
    if decisions:
        return True
    if lessons:
        return True
    if len(messages) < 1:
        return False
    content = " ".join(m.get("content", "") for m in messages).lower()
    return any(kw in content for kw in _IMPORTANCE_KEYWORDS)


def write_session(
    date: str,
    agent: str,
    messages: list[dict],
    task: str = "",
    decisions: list[str] | None = None,
    lessons: list[str] | None = None,
) -> dict:
    dec_str = "; ".join(decisions) if decisions else ""
    les_str = "; ".join(lessons) if lessons else ""

    important = _is_important(task, decisions, lessons, messages)

    l2_result = write_session_l2(date, messages, agent, task=task, decisions=dec_str, lessons=les_str)

    if not important:
        import shutil
        from pathlib import Path
        from .config import get_vault_root
        vault = get_vault_root()
        drafts_dir = vault / "L2-episodic" / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        src = Path(l2_result["path"])
        dst = drafts_dir / src.name
        shutil.move(str(src), str(dst))
        l2_result["path"] = str(dst)

    summary_parts = [f"Session: {agent}"]
    if task:
        summary_parts.append(f"Task: {task}")
    if decisions:
        summary_parts.append(f"Decisions: {dec_str}")
    if lessons:
        summary_parts.append(f"Lessons: {les_str}")
    summary_parts.append(f"Messages: {len(messages)}")
    summary_parts.append(f"L2 log: {l2_result['path']}")

    l1_path = append_l1_today("\n".join(summary_parts))

    return {
        "l2_path": l2_result["path"],
        "l1_path": l1_path,
        "sha": l2_result["sha"],
        "important": important,
    }
