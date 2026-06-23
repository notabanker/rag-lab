import re

from markdown_it import MarkdownIt

_FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)


def _walk_tokens(tokens, out: list[str]):
    for tok in tokens:
        if tok.type == "inline":
            for child in tok.children or []:
                if child.type == "text":
                    out.append(child.content)
                elif child.type == "code_inline":
                    out.append(child.content)
                elif child.type in ("softbreak", "hardbreak"):
                    out.append("\n")
        elif tok.type in ("code_block", "fence"):
            out.append(tok.content)
        elif tok.type in ("html_block", "html_inline"):
            continue
        elif tok.type == "heading_open":
            out.append("\n\n")
        elif tok.type == "paragraph_open":
            out.append("\n")
        elif tok.children:
            _walk_tokens(tok.children, out)


def parse_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    m = _FRONTMATTER_RE.match(raw)
    if m:
        raw = raw[m.end():].lstrip("\n")
    md = MarkdownIt().enable("table").enable("strikethrough")
    tokens = md.parse(raw)
    out: list[str] = []
    _walk_tokens(tokens, out)
    text = "".join(out).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text
