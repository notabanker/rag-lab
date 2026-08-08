"""DOCX parser: headings become section markers, tables flatten row-per-line."""
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .base import ParseResult


def _table_lines(table: Table) -> list[str]:
    lines = []
    for row in table.rows:
        cells = [" ".join(c.text.split()) for c in row.cells]
        line = " | ".join(c for c in cells if c)
        if line:
            lines.append(line)
    return lines


def parse_docx(path: str) -> ParseResult:
    doc = Document(path)
    # (heading label or None, body lines) per section; headings stay inline
    # in the body so chunks keep their section context.
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for item in doc.iter_inner_content():
        if isinstance(item, Paragraph):
            text = " ".join(item.text.split())
            if not text:
                continue
            style = item.style.name if item.style and item.style.name else ""
            if style.startswith("Heading"):
                sections.append((text, [text]))
            else:
                sections[-1][1].append(text)
        elif isinstance(item, Table):
            sections[-1][1].extend(_table_lines(item))

    parts: list[str] = []
    section_chars: list[int] = []
    for label, lines in sections:
        body = "\n".join(lines)
        if not body and label is None:
            continue  # empty preamble before the first heading
        section_chars.append(len(body.strip()))
        marker = f"\n\n--- {label} ---\n\n" if label else "\n\n"
        parts.append(f"{marker}{body}")
    return ParseResult(text="".join(parts), section_unit="section", section_chars=section_chars)
