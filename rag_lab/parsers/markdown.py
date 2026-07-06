from markdown_it import MarkdownIt

from .base import ParseResult

def parse_markdown(path: str) -> ParseResult:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Strip frontmatter if present
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            raw = parts[2].lstrip("\n")
    # Render to plain-ish text via markdown-it (preserves structure)
    md = MarkdownIt().enable("table").enable("strikethrough")
    tokens = md.parse(raw)
    out = []
    for tok in tokens:
        if tok.type == "heading_open":
            out.append("\n\n")
        elif tok.type == "paragraph_open":
            out.append("")
        elif tok.type == "inline":
            out.append(tok.content)
        elif tok.type == "code_block" or tok.type == "fence":
            out.append(f"\n\n{tok.content}\n")
    text = "\n".join(out).strip()
    return ParseResult(text=text, section_unit="document", section_chars=[len(text)])
