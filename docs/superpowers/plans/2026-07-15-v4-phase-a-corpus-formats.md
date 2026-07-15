# V4 Phase A — Corpus Formats (DOCX, PPTX, OCR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** rag-lab can ingest DOCX and PPTX files with correct citations, and OCRs scanned PDFs instead of refusing them.

**Architecture:** Each new format is a parser module returning the existing `ParseResult` dataclass with `--- <marker> ---` section labels that `ingestion.build_metadatas` already converts to citations. OCR is a separate `rag_lab/ocr.py` module (optional dependency group) that re-parses low-yield PDFs through `ocrmypdf`, cached by content SHA.

**Tech Stack:** Python 3.11+, uv, pytest, python-docx, python-pptx, ocrmypdf (optional group), existing pypdf/Chroma pipeline.

**Plan 1 of 4** for the V4 Study Edition spec (`docs/superpowers/specs/2026-07-15-rag-lab-v4-study-edition-design.md`). Later plans: Phase B (sync/workspaces/MCP), Phase C2' (metadata filtering), Phase S (study layer).

## Global Constraints

- Run all commands from the repo root `~/projects/rag-lab` with `uv run …`.
- Work on branch `v4-phase-a` (created in Task 1, Step 1). One commit per task, conventional-commit messages (`feat:`, `test:`, `docs:`).
- After every task: `uv run pytest -q` must be green.
- End of plan: `uv run rag eval --retrieval-only --gate eval/gates.yaml` must pass (requires the bundled corpus DB; do not run it mid-plan).
- `documentation.md` must reflect user-facing changes — each task says exactly what to update.
- New required dependencies allowed: `python-docx>=1.1`, `python-pptx>=1.0`. `ocrmypdf>=16` goes in an **optional** dependency group `ocr`; core `uv run pytest` must pass **without** the ocr group installed.
- Parser contract (do not deviate): a parser is `parse_x(path: str) -> ParseResult` (`rag_lab/parsers/base.py:12`), emitting sections as `f"\n\n--- {label} ---\n\n{body}"`, with `section_chars` = stripped char count per section. Registration is one entry in `PARSERS` (`rag_lab/parsers/__init__.py:6`).

---

### Task 1: DOCX parser + marker-regex hyphen fix

**Files:**
- Modify: `pyproject.toml:6-22` (dependencies list)
- Create: `rag_lab/parsers/docx.py`
- Modify: `rag_lab/parsers/__init__.py`
- Modify: `rag_lab/parsers/base.py:14` (section_unit comment)
- Modify: `rag_lab/ingestion.py:9` (`_MARKER_RE`)
- Modify: `rag_lab/web.py:9,520` (derive allowed suffixes from `PARSERS`)
- Create: `tests/test_parsers_docx.py`

**Interfaces:**
- Consumes: `ParseResult` from `rag_lab/parsers/base.py`; `ingestion.make_chunks`, `ingestion.build_metadatas`, `ingestion._markers` (existing).
- Produces: `parse_docx(path: str) -> ParseResult` registered under `.docx`; `_MARKER_RE` that accepts hyphenated labels. Citations for DOCX look like `report.docx §2.3 Methods`.

- [ ] **Step 1: Create branch and add dependency**

```bash
git checkout -b v4-phase-a
```

In `pyproject.toml`, append to the `dependencies` list (after `"mcp>=1.0",`):

```toml
    "python-docx>=1.1",
```

Run: `uv sync`
Expected: resolves and installs python-docx.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_parsers_docx.py`:

```python
from docx import Document as DocxDocument

from rag_lab import ingestion
from rag_lab.parsers import ParseResult, pick_parser
from rag_lab.parsers.docx import parse_docx


def _make_docx(path):
    doc = DocxDocument()
    doc.add_heading("Introduction", level=1)
    doc.add_paragraph("Deep learning basics for the course.")
    doc.add_heading("2.3 Methods", level=2)
    doc.add_paragraph("We use gradient descent to train the model.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Accuracy"
    table.cell(1, 1).text = "0.95"
    doc.save(str(path))


def test_docx_is_registered():
    assert pick_parser("x.docx") is parse_docx


def test_parse_docx_headings_become_markers(tmp_path):
    f = tmp_path / "report.docx"
    _make_docx(f)
    result = parse_docx(str(f))
    assert isinstance(result, ParseResult)
    assert "--- Introduction ---" in result.text
    assert "--- 2.3 Methods ---" in result.text
    assert "gradient descent" in result.text
    assert result.section_unit == "section"
    assert len(result.section_chars) == 2
    assert result.quality()["warnings"] == []


def test_parse_docx_heading_text_stays_inline(tmp_path):
    # Heading text must appear in the chunkable body too, not only in the
    # marker scaffolding, so chunks keep their section context.
    f = tmp_path / "report.docx"
    _make_docx(f)
    text = parse_docx(str(f)).text
    body_after_marker = text.split("--- 2.3 Methods ---", 1)[1]
    assert "2.3 Methods" in body_after_marker


def test_parse_docx_flattens_tables_row_per_line(tmp_path):
    f = tmp_path / "report.docx"
    _make_docx(f)
    text = parse_docx(str(f)).text
    assert "Metric | Value" in text
    assert "Accuracy | 0.95" in text


def test_parse_docx_empty_doc_warns(tmp_path):
    f = tmp_path / "empty.docx"
    DocxDocument().save(str(f))
    q = parse_docx(str(f)).quality()
    assert q["total_chars"] == 0
    assert any("no text extracted" in w for w in q["warnings"])


def test_docx_citations_use_section_labels(tmp_path):
    f = tmp_path / "report.docx"
    _make_docx(f)
    parsed = parse_docx(str(f))
    chunks = ingestion.make_chunks(parsed.effective_text, "sentence", 512, 64)
    metas = ingestion.build_metadatas(
        str(f), parsed.effective_text, chunks, "sha1", "sentence", 512, 64
    )
    citations = [m["citation"] for m in metas]
    assert any(c == "report.docx §2.3 Methods" for c in citations)


def test_marker_regex_allows_hyphenated_headings():
    text = "\n\n--- State-of-the-art Models ---\n\nTransformers dominate."
    markers = ingestion._markers(text)
    assert markers and markers[0][1] == "State-of-the-art Models"


def test_marker_regex_page_priority_unchanged():
    text = "\n\n--- Page 12 ---\n\nSome page text."
    markers = ingestion._markers(text)
    assert markers[0][1] == "Page 12"
    assert ingestion._citation("a.pdf", markers[0][1], 0) == "a.pdf p.12"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_parsers_docx.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag_lab.parsers.docx'` (collection error).

- [ ] **Step 4: Implement the parser**

Create `rag_lab/parsers/docx.py`:

```python
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
```

Register it in `rag_lab/parsers/__init__.py` — replace the whole file with:

```python
from .base import ParseResult, as_result
from .pdf import parse_pdf
from .epub import parse_epub
from .markdown import parse_markdown
from .docx import parse_docx

PARSERS = {
    ".pdf": parse_pdf,
    ".epub": parse_epub,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".docx": parse_docx,
}

def pick_parser(path: str):
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return PARSERS[ext]
```

In `rag_lab/parsers/base.py:14`, update the comment on `section_unit`:

```python
    section_unit: str = "document"  # "page" | "chapter" | "section" | "slide" | "document"
```

- [ ] **Step 5: Fix the marker regex for hyphenated labels**

In `rag_lab/ingestion.py:9`, replace:

```python
_MARKER_RE = re.compile(r"---\s*(Page\s+\d+|[^-][^-]+?)\s*---", re.IGNORECASE)
```

with:

```python
# Label branch: starts with a non-hyphen, may contain single hyphens
# ("State-of-the-art") but never crosses a "---" or a newline.
_MARKER_RE = re.compile(r"---\s*(Page\s+\d+|[^-\n](?:(?!---)[^\n])*?)\s*---", re.IGNORECASE)
```

- [ ] **Step 6: Derive the web upload allowlist from PARSERS**

In `rag_lab/web.py:9`, extend the import:

```python
from .parsers import PARSERS, as_result, pick_parser
```

In `rag_lab/web.py:520`, replace:

```python
    if suffix.lower() not in {".pdf", ".epub", ".md", ".markdown"}:
```

with:

```python
    if suffix.lower() not in PARSERS:
```

- [ ] **Step 7: Run the new tests, then the full suite**

Run: `uv run pytest tests/test_parsers_docx.py -v`
Expected: all 8 PASS.

Run: `uv run pytest -q`
Expected: all pass (the regex change must not break `test_ingestion.py` / `test_citations.py`; if a failure appears there, the regex — not the old tests — is wrong).

- [ ] **Step 8: Update documentation and commit**

In `documentation.md`: find the supported-formats mention (search for "EPUB") and add DOCX with a note: headings → `§`-style citations (`report.docx §2.3 Methods`), tables flattened row-per-line. In `README.md:3` add DOCX to the format list, and add an example line `uv run rag ingest ~/uni/report.docx` near `README.md:48`.

```bash
git add pyproject.toml uv.lock rag_lab/parsers/ rag_lab/ingestion.py rag_lab/web.py tests/test_parsers_docx.py documentation.md README.md
git commit -m "feat: DOCX parser with heading-based section citations"
```

---

### Task 2: PPTX parser + slide citations

**Files:**
- Modify: `pyproject.toml` (dependencies list)
- Create: `rag_lab/parsers/pptx.py`
- Modify: `rag_lab/parsers/__init__.py`
- Modify: `rag_lab/ingestion.py:32-39` (`_citation` slide branch)
- Create: `tests/test_parsers_pptx.py`

**Interfaces:**
- Consumes: `ParseResult`; `ingestion._citation(source, marker, chunk_idx)` (existing).
- Produces: `parse_pptx(path: str) -> ParseResult` registered under `.pptx`; `_citation` returns `lecture.pptx slide 12` for markers matching `Slide N`.

- [ ] **Step 1: Add dependency**

In `pyproject.toml`, append to `dependencies` (after `"python-docx>=1.1",`):

```toml
    "python-pptx>=1.0",
```

Run: `uv sync`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_parsers_pptx.py`:

```python
from pptx import Presentation

from rag_lab import ingestion
from rag_lab.parsers import ParseResult, pick_parser
from rag_lab.parsers.pptx import parse_pptx


def _make_pptx(path):
    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])  # Title and Content
    s1.shapes.title.text = "Neural Networks"
    s1.placeholders[1].text = "Perceptrons\nActivation functions"
    s2 = prs.slides.add_slide(prs.slide_layouts[1])
    s2.shapes.title.text = "Limitations"
    s2.placeholders[1].text = "Linear separability"
    s2.notes_slide.notes_text_frame.text = "Mention the XOR problem here."
    prs.save(str(path))


def test_pptx_is_registered():
    assert pick_parser("x.pptx") is parse_pptx


def test_parse_pptx_one_section_per_slide(tmp_path):
    f = tmp_path / "deck.pptx"
    _make_pptx(f)
    result = parse_pptx(str(f))
    assert isinstance(result, ParseResult)
    assert "--- Slide 1 ---" in result.text
    assert "--- Slide 2 ---" in result.text
    assert result.section_unit == "slide"
    assert len(result.section_chars) == 2
    assert "Perceptrons" in result.text
    assert "Linear separability" in result.text


def test_parse_pptx_includes_speaker_notes(tmp_path):
    f = tmp_path / "deck.pptx"
    _make_pptx(f)
    text = parse_pptx(str(f)).text
    slide2 = text.split("--- Slide 2 ---", 1)[1]
    assert "Notes: Mention the XOR problem here." in slide2


def test_parse_pptx_empty_deck_warns(tmp_path):
    f = tmp_path / "empty.pptx"
    Presentation().save(str(f))
    q = parse_pptx(str(f)).quality()
    assert q["total_chars"] == 0
    assert any("no text extracted" in w for w in q["warnings"])


def test_citation_slide_label():
    assert ingestion._citation("deck.pptx", "Slide 12", 0) == "deck.pptx slide 12"


def test_pptx_end_to_end_citations(tmp_path):
    f = tmp_path / "deck.pptx"
    _make_pptx(f)
    parsed = parse_pptx(str(f))
    chunks = ingestion.make_chunks(parsed.effective_text, "fixed", 64, 8)
    metas = ingestion.build_metadatas(
        str(f), parsed.effective_text, chunks, "sha1", "fixed", 64, 8
    )
    citations = {m["citation"] for m in metas}
    assert "deck.pptx slide 1" in citations
    assert "deck.pptx slide 2" in citations
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_parsers_pptx.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag_lab.parsers.pptx'`.

- [ ] **Step 4: Implement the parser and the citation branch**

Create `rag_lab/parsers/pptx.py`:

```python
"""PPTX parser: one section per slide, text frames + speaker notes."""
from pptx import Presentation

from .base import ParseResult


def parse_pptx(path: str) -> ParseResult:
    prs = Presentation(path)
    parts: list[str] = []
    section_chars: list[int] = []
    for i, slide in enumerate(prs.slides, start=1):
        lines = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                lines.append(shape.text_frame.text.strip())
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                lines.append(f"Notes: {notes}")
        body = "\n".join(lines)
        section_chars.append(len(body.strip()))
        parts.append(f"\n\n--- Slide {i} ---\n\n{body}")
    return ParseResult(text="".join(parts), section_unit="slide", section_chars=section_chars)
```

Register in `rag_lab/parsers/__init__.py`: add `from .pptx import parse_pptx` after the docx import, and `".pptx": parse_pptx,` to `PARSERS`.

In `rag_lab/ingestion.py`, replace `_citation` (lines 32–39) with:

```python
def _citation(source: str, marker: str | None, chunk_idx: int) -> str:
    name = Path(source).name
    if marker:
        page = re.search(r"page\s+(\d+)", marker, re.IGNORECASE)
        if page:
            return f"{name} p.{page.group(1)}"
        slide = re.search(r"^slide\s+(\d+)$", marker, re.IGNORECASE)
        if slide:
            return f"{name} slide {slide.group(1)}"
        return f"{name} §{marker}"
    return f"{name} chunk {chunk_idx + 1}"
```

- [ ] **Step 5: Run the new tests, then the full suite**

Run: `uv run pytest tests/test_parsers_pptx.py -v` → all 6 PASS.
Run: `uv run pytest -q` → all pass.

- [ ] **Step 6: Update documentation and commit**

`documentation.md` + `README.md:3`: add PPTX (citations `deck.pptx slide 12`; speaker notes ingested with a `Notes:` prefix). Note for future: slide-as-parent-chunk is intentionally NOT implemented (TASKS.md A.4 "consider" item — parked until an eval demands it).

```bash
git add pyproject.toml uv.lock rag_lab/parsers/ rag_lab/ingestion.py tests/test_parsers_pptx.py documentation.md README.md
git commit -m "feat: PPTX parser with per-slide sections, notes, and slide citations"
```

---

### Task 3: OCR fallback for scanned PDFs

**Files:**
- Modify: `pyproject.toml:27-28` (`[dependency-groups]`)
- Modify: `rag_lab/config.py` (add `OCR_LANGS`)
- Create: `rag_lab/ocr.py`
- Modify: `rag_lab/parsers/base.py:31-33` (warning text)
- Modify: `rag_lab/cli.py:48-102` (`_ingest_one`, `ingest`, `rebuild`)
- Modify: `rag_lab/web.py:522-540` (`api_ingest` OCR wiring + message)
- Create: `tests/test_ocr.py`
- Create: `scripts/make_ocr_fixture.py` and `tests/fixtures/scanned.pdf`

**Interfaces:**
- Consumes: `ParseResult.quality()` dict (`section_unit`, `warnings` keys); `ingestion.content_sha(bytes) -> str`; `config.default_db_path()`; `parsers.pdf.parse_pdf`.
- Produces:
  - `ocr.ocr_available() -> bool`
  - `ocr.needs_ocr(quality: dict) -> bool`
  - `ocr.ocr_pdf(path: str, file_sha: str, langs: str | None = None) -> Path` (cached)
  - `ocr.maybe_ocr_parse(path: str, parsed: ParseResult, content: bytes, use_ocr: bool | None) -> tuple[ParseResult, dict, bool]` — the single wiring point for CLI and web. `use_ocr=None` means auto.
  - `config.OCR_LANGS` (env `RAG_OCR_LANGS`, default `"deu+eng"`).
  - `ocr.INSTALL_HINT` — the install one-liner used in error messages.

- [ ] **Step 1: Add the optional dependency group and config**

In `pyproject.toml`, replace:

```toml
[dependency-groups]
dev = ["pytest>=8.0"]
```

with:

```toml
[dependency-groups]
dev = ["pytest>=8.0"]
ocr = ["ocrmypdf>=16.0"]
```

In `rag_lab/config.py`, after the `RERANKER_BATCH_SIZE` line (line 12), add:

```python
OCR_LANGS = os.environ.get("RAG_OCR_LANGS", "").strip() or "deu+eng"
```

Do NOT run `uv sync --group ocr` as part of the tests — the suite must pass without it.

- [ ] **Step 2: Generate the scanned-PDF fixture**

Create `scripts/make_ocr_fixture.py`:

```python
"""Generate tests/fixtures/scanned.pdf — an image-only PDF for OCR tests.
Rerun only to regenerate; the output is committed."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

out = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "scanned.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
img = Image.new("RGB", (1224, 600), "white")
draw = ImageDraw.Draw(img)
font = ImageFont.load_default(size=48)
draw.text((60, 100), "HELLO OCR WORLD", fill="black", font=font)
draw.text((60, 220), "RAG LAB SCANNED FIXTURE", fill="black", font=font)
img.save(out, "PDF", resolution=144)
print(f"wrote {out}")
```

Run: `uv run --with pillow python scripts/make_ocr_fixture.py`
Expected: `wrote .../tests/fixtures/scanned.pdf` (file is a few KB).

- [ ] **Step 3: Write the failing tests**

Create `tests/test_ocr.py`:

```python
from pathlib import Path

import pytest

from rag_lab import ocr
from rag_lab.parsers import ParseResult
from rag_lab.parsers.pdf import parse_pdf

FIXTURE = Path(__file__).parent / "fixtures" / "scanned.pdf"


def _scanned_quality():
    return ParseResult(text="", section_unit="page", section_chars=[0]).quality()


def test_needs_ocr_on_scanned_warnings():
    assert ocr.needs_ocr(_scanned_quality()) is True
    low_yield = ParseResult(text="x" * 60, section_unit="page", section_chars=[20, 20, 20]).quality()
    assert ocr.needs_ocr(low_yield) is True


def test_needs_ocr_false_for_clean_and_non_page_docs():
    clean = ParseResult(text="x", section_unit="page", section_chars=[900, 1100]).quality()
    assert ocr.needs_ocr(clean) is False
    doc = ParseResult(text="", section_unit="document", section_chars=[0]).quality()
    assert ocr.needs_ocr(doc) is False


def test_ocr_pdf_caches_by_sha(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "data" / "chroma_db"))
    calls = []

    def fake_run(src, dst, langs):
        calls.append(langs)
        Path(dst).write_bytes(b"%PDF-fake")

    monkeypatch.setattr(ocr, "_run_ocrmypdf", fake_run)
    out1 = ocr.ocr_pdf(str(FIXTURE), "shaabc", langs="deu+eng")
    out2 = ocr.ocr_pdf(str(FIXTURE), "shaabc", langs="deu+eng")
    assert out1 == out2
    assert out1.exists()
    assert calls == ["deu+eng"]  # second call was a cache hit
    assert str(tmp_path / "data") in str(out1)


def test_maybe_ocr_parse_skips_when_unavailable(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)
    parsed = ParseResult(text="", section_unit="page", section_chars=[0])
    out, quality, used = ocr.maybe_ocr_parse("f.pdf", parsed, b"raw", use_ocr=None)
    assert out is parsed
    assert used is False
    assert "ocr" not in quality


def test_maybe_ocr_parse_forced_but_unavailable_errors(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)
    parsed = ParseResult(text="", section_unit="page", section_chars=[0])
    with pytest.raises(RuntimeError, match="uv sync --group ocr"):
        ocr.maybe_ocr_parse("f.pdf", parsed, b"raw", use_ocr=True)


def test_maybe_ocr_parse_disabled_skips_even_when_available(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    parsed = ParseResult(text="", section_unit="page", section_chars=[0])
    out, quality, used = ocr.maybe_ocr_parse("f.pdf", parsed, b"raw", use_ocr=False)
    assert used is False


def test_maybe_ocr_parse_reparses_via_ocr(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    fake_pdf = tmp_path / "ocred.pdf"
    fake_pdf.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(ocr, "ocr_pdf", lambda path, sha, langs=None: fake_pdf)
    good = ParseResult(
        text="\n\n--- Page 1 ---\n\nRecovered text from OCR, quite a lot of it. " * 5,
        section_unit="page",
        section_chars=[220],
    )
    monkeypatch.setattr(ocr, "parse_pdf", lambda p: good)
    scanned = ParseResult(text="", section_unit="page", section_chars=[0])
    out, quality, used = ocr.maybe_ocr_parse("f.pdf", scanned, b"raw", use_ocr=None)
    assert used is True
    assert out is good
    assert quality["ocr"] is True
    assert quality["total_chars"] > 0


@pytest.mark.skipif(not ocr.ocr_available(), reason="ocrmypdf/tesseract not installed")
def test_real_ocr_extracts_fixture_text(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "data" / "chroma_db"))
    scanned = parse_pdf(str(FIXTURE))
    assert scanned.quality()["total_chars"] == 0  # fixture really is image-only
    out, quality, used = ocr.maybe_ocr_parse(
        str(FIXTURE), scanned, FIXTURE.read_bytes(), use_ocr=True
    )
    assert used is True
    assert "HELLO OCR WORLD" in out.text.upper()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag_lab.ocr'` (collection error).

- [ ] **Step 5: Implement `rag_lab/ocr.py`**

```python
"""OCR fallback for scanned PDFs. Optional dependency group: `uv sync --group ocr`
plus the Tesseract binary (`brew install tesseract tesseract-lang`)."""
import shutil
from pathlib import Path

from .config import OCR_LANGS, default_db_path
from .ingestion import content_sha
from .parsers.base import ParseResult
from .parsers.pdf import parse_pdf

INSTALL_HINT = "uv sync --group ocr && brew install tesseract tesseract-lang"

# quality()["warnings"] substrings that mark a PDF as an OCR candidate
_SCANNED_MARKERS = ("no text extracted", "scanned", "contain no text")


def ocr_available() -> bool:
    try:
        import ocrmypdf  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def needs_ocr(quality: dict) -> bool:
    if quality.get("section_unit") != "page":
        return False
    return any(m in w for w in quality.get("warnings", []) for m in _SCANNED_MARKERS)


def _cache_dir() -> Path:
    d = Path(default_db_path()).parent / "ocr_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_ocrmypdf(src: str, dst: str, langs: str) -> None:
    import ocrmypdf

    # skip_text: never re-OCR pages that already have a text layer
    ocrmypdf.ocr(src, dst, language=langs, skip_text=True, progress_bar=False)


def ocr_pdf(path: str, file_sha: str, langs: str | None = None) -> Path:
    """OCR a PDF, cached by content SHA + languages. Returns the OCR'd PDF path."""
    langs = langs or OCR_LANGS
    out = _cache_dir() / f"{file_sha}-{langs.replace('+', '-')}.pdf"
    if out.exists():
        return out
    tmp = out.with_suffix(".tmp")
    _run_ocrmypdf(path, str(tmp), langs)
    tmp.replace(out)  # only ever cache a completed OCR run
    return out


def maybe_ocr_parse(
    path: str, parsed: ParseResult, content: bytes, use_ocr: bool | None
) -> tuple[ParseResult, dict, bool]:
    """Re-parse a PDF through OCR when it looks scanned.

    use_ocr: None = auto (OCR only when installed), True = require, False = never.
    Returns (parsed, quality, used_ocr); quality gains "ocr": True on the OCR path.
    """
    quality = parsed.quality()
    if use_ocr is False or not needs_ocr(quality):
        return parsed, quality, False
    if not ocr_available():
        if use_ocr is True:
            raise RuntimeError(f"--ocr requested but the OCR engine is missing — install: {INSTALL_HINT}")
        return parsed, quality, False
    ocred = ocr_pdf(path, content_sha(content))
    parsed = parse_pdf(str(ocred))
    quality = parsed.quality()
    quality["ocr"] = True
    return parsed, quality, True
```

- [ ] **Step 6: Run the OCR unit tests**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: 7 PASS, 1 SKIP (`test_real_ocr_extracts_fixture_text` skips unless ocrmypdf + tesseract are installed). If the ocr group IS installed locally, all 8 must pass.

- [ ] **Step 7: Wire into the CLI**

In `rag_lab/cli.py`, add to the imports at the top of the file:

```python
from . import ocr
```

Replace `_ingest_one` (lines 48–85) with:

```python
def _ingest_one(
    file_path: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    parent_size: int,
    force_model_mismatch: bool,
    allow_empty: bool = False,
    use_ocr: bool | None = None,
) -> dict:
    p = Path(file_path)
    if not p.exists():
        raise ValueError(f"File not found: {file_path}")
    console.print(f"[blue]Parsing[/blue] {p.name}...")
    parsed = as_result(pick_parser(file_path)(file_path))
    content = p.read_bytes()
    if p.suffix.lower() == ".pdf":
        parsed, quality, used_ocr = ocr.maybe_ocr_parse(file_path, parsed, content, use_ocr)
        if used_ocr:
            console.print("[blue]OCR[/blue] low text yield — re-parsed through OCR")
    else:
        quality = parsed.quality()
    console.print(
        f"[blue]Parsed[/blue] {quality['sections']} {quality['section_unit']}(s), "
        f"{quality['total_chars']} chars"
    )
    for w in quality["warnings"]:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    if quality["total_chars"] == 0 and not allow_empty:
        raise ValueError(
            "No text extracted — scanned PDF? Install OCR support: "
            f"{ocr.INSTALL_HINT}. Use --allow-empty to record the file in the manifest anyway."
        )
    console.print(f"[blue]Chunking[/blue] strategy={strategy} size={chunk_size} overlap={overlap}...")
    chunks = ingestion.make_chunks(parsed.effective_text, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
    if chunks:
        console.print(f"[blue]Embedding[/blue] {len(chunks)} chunks with {EMBEDDING_MODEL} (CPU)...")
    result = ingestion.ingest_text(
        str(p), content, parsed.effective_text,
        strategy=strategy, chunk_size=chunk_size, overlap=overlap,
        parent_size=parent_size, force_model_mismatch=force_model_mismatch,
        parse_quality=quality,
    )
    console.print(f"[green]✅ Ingested[/green] {result['chunks']} chunks from {p.name}")
    return result
```

In the `ingest` command (lines 87–102): add the option after `allow_empty`:

```python
    use_ocr: bool | None = typer.Option(None, "--ocr/--no-ocr", help="Force/disable OCR for scanned PDFs (default: auto when installed)"),
```

and pass it through:

```python
        _ingest_one(file_path, strategy, chunk_size, overlap, parent_size, force_model_mismatch, allow_empty, use_ocr)
```

In the `rebuild` command (lines 104–122): add the same `use_ocr` option and pass `use_ocr=use_ocr` in its `_ingest_one(...)` call (note: `rebuild`'s `_ingest_one` call has no `allow_empty` argument, so pass `use_ocr` by keyword).

- [ ] **Step 8: Wire into the web API**

In `rag_lab/web.py`, add `from . import ocr` to the module imports, then in `api_ingest` replace lines 522–531 (the tempfile/parse block plus the `quality = parsed.quality()` line):

```python
        content = file.file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            parser = pick_parser(tmp_path)
            parsed = as_result(parser(tmp_path))
            if suffix.lower() == ".pdf":
                # OCR must run while the temp file still exists
                parsed, quality, _ = ocr.maybe_ocr_parse(tmp_path, parsed, content, use_ocr=None)
            else:
                quality = parsed.quality()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
```

Replace the 400 detail at `rag_lab/web.py:539`:

```python
                detail=f"No text extracted — scanned PDF? Install OCR support: {ocr.INSTALL_HINT}",
```

Also update the stale warning text in `rag_lab/parsers/base.py:31-33` to:

```python
            warnings.append(
                "no text extracted — scanned/image-only file? install OCR: uv sync --group ocr"
            )
```

(keeps the `"no text extracted"` prefix that tests and `needs_ocr` match on).

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass, 1 skip (real-OCR test) when the ocr group is absent. `tests/test_parse_quality.py::test_cli_refuses_empty_pdf_by_default` still passes because the new message still starts with "No text extracted".

- [ ] **Step 10: Update documentation and commit**

`documentation.md`: remove "No OCR" from Known Limitations; add an OCR section — optional group install (`uv sync --group ocr`, `brew install tesseract tesseract-lang`), auto/`--ocr`/`--no-ocr` semantics, `RAG_OCR_LANGS` (default `deu+eng`), cache at `<data-dir>/ocr_cache/<sha>-<langs>.pdf`, `skip_text` guarantee (good text layers are never re-OCR'd), and the `"ocr": true` field in the persisted parse report. `README.md`: mention the OCR install one-liner in Setup.

```bash
git add pyproject.toml uv.lock rag_lab/config.py rag_lab/ocr.py rag_lab/cli.py rag_lab/web.py rag_lab/parsers/base.py tests/test_ocr.py tests/fixtures/scanned.pdf scripts/make_ocr_fixture.py documentation.md README.md
git commit -m "feat: OCR fallback for scanned PDFs (optional ocrmypdf group, SHA-cached)"
```

---

### Task 4: Surface updates — dashboard accept-list, MCP description

**Files:**
- Modify: `dashboard/src/App.tsx:316`
- Modify: `rag_lab/mcp_server.py:89` (add docstring to `rag_ingest`)

**Interfaces:**
- Consumes: `PARSERS` keys as the source of truth for supported formats.
- Produces: no code interfaces — user-facing metadata only.

- [ ] **Step 1: Restrict the dashboard file picker**

In `dashboard/src/App.tsx:316`, replace:

```tsx
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
```

with:

```tsx
      <input type="file" accept=".pdf,.epub,.md,.markdown,.docx,.pptx" onChange={(e) => setFile(e.target.files?.[0] || null)} />
```

- [ ] **Step 2: Describe formats on the MCP ingest tool**

In `rag_lab/mcp_server.py`, `rag_ingest` (line 89) currently has no docstring — FastMCP uses the docstring as the tool description. Add as the first line of the function body:

```python
    """Ingest a local file into the corpus. Supported: .pdf (OCR fallback for
    scanned PDFs when installed), .epub, .md/.markdown, .docx, .pptx.
    Returns chunk count, file_sha, and any parse-quality warnings."""
```

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest -q` → all pass (MCP smoke tests in `tests/test_mcp_server.py` must still pass with the docstring added).
Run: `cd dashboard && npx tsc --noEmit && cd ..` → no type errors (skip with a note in the commit message if `node_modules` is not installed).

```bash
git add dashboard/src/App.tsx rag_lab/mcp_server.py
git commit -m "docs: surface DOCX/PPTX/OCR support in dashboard picker and MCP tool description"
```

---

### Task 5: Eval fixtures + re-baseline

**Files:**
- Create: `scripts/make_office_fixtures.py`, `data/docx/study_methods.docx`, `data/pptx/lecture_neural_networks.pptx`
- Modify: `eval/questions.yaml` (append two questions)
- Modify: `README.md:101-115` (baseline table + date)
- Modify: `documentation.md` (eval section: new question tags `docx`, `pptx`, `slide-notes`)

**Interfaces:**
- Consumes: `rag rebuild`, `rag eval --retrieval-only --gate eval/gates.yaml` (existing CLI); question schema documented at `eval/questions.yaml:1-9`.
- Produces: two committed corpus fixtures and two golden questions targeting them.

- [ ] **Step 1: Create the fixture generator**

Create `scripts/make_office_fixtures.py`:

```python
"""Generate the DOCX/PPTX eval corpus fixtures. Rerun only to regenerate:
the outputs are committed. Content must stay in sync with the
docx-forgetting-curve / pptx-xor-notes questions in eval/questions.yaml."""
from pathlib import Path

from docx import Document
from pptx import Presentation

ROOT = Path(__file__).resolve().parent.parent


def make_docx():
    out = ROOT / "data" / "docx" / "study_methods.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    doc.add_heading("Study Methods", level=1)
    doc.add_paragraph("This note covers evidence-based study techniques for exam preparation.")
    doc.add_heading("1. Spaced Repetition", level=2)
    doc.add_paragraph(
        "The Ebbinghaus forgetting curve describes how memory retention declines "
        "exponentially over time unless information is reviewed. Spacing reviews at "
        "increasing intervals counteracts the forgetting curve and strengthens recall."
    )
    doc.add_heading("2. Active Recall", level=2)
    doc.add_paragraph(
        "Active recall means testing yourself instead of re-reading. Retrieval "
        "practice produces stronger long-term retention than passive review."
    )
    table = doc.add_table(rows=3, cols=2)
    for row, (a, b) in zip(table.rows, [("Technique", "Evidence"), ("Spaced repetition", "strong"), ("Highlighting", "weak")]):
        row.cells[0].text = a
        row.cells[1].text = b
    doc.save(out)
    print(f"wrote {out}")


def make_pptx():
    out = ROOT / "data" / "pptx" / "lecture_neural_networks.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "Lecture 3: Neural Networks"
    s1.placeholders[1].text = "Perceptron\nActivation functions\nBackpropagation"
    s2 = prs.slides.add_slide(prs.slide_layouts[1])
    s2.shapes.title.text = "Perceptron Limitations"
    s2.placeholders[1].text = "A single-layer perceptron can only separate linearly separable classes."
    s2.notes_slide.notes_text_frame.text = (
        "Exam hint: the XOR problem is the classic example — XOR is not linearly "
        "separable, so a single-layer perceptron cannot solve it."
    )
    prs.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_docx()
    make_pptx()
```

Run: `uv run python scripts/make_office_fixtures.py`
Expected: both `wrote …` lines; files exist under `data/docx/` and `data/pptx/`.

- [ ] **Step 2: Add golden questions**

Append to the `questions:` list in `eval/questions.yaml`:

```yaml
  - id: docx-forgetting-curve
    question: What does the Ebbinghaus forgetting curve describe?
    expected_sources: [study_methods.docx]
    expected_fragments: [retention, declines]
    tags: [factoid, en, docx]

  - id: pptx-xor-notes
    question: Why can a single-layer perceptron not solve the XOR problem?
    expected_sources: [lecture_neural_networks.pptx]
    expected_fragments: [linearly separable]
    tags: [factoid, en, pptx, slide-notes]
```

- [ ] **Step 3: Ingest the fixtures into the corpus**

Run: `uv run rag rebuild data/docx/study_methods.docx data/pptx/lecture_neural_networks.pptx`
Expected: both files ingest with chunks > 0 and no warnings.

- [ ] **Step 4: Re-baseline**

Run: `uv run rag eval --retrieval-only --gate eval/gates.yaml`
Expected: gate PASSES, now over 25 questions. If either new question misses: debug retrieval for that question with `uv run rag query "<question>" --trace` before touching gates — do not lower gate thresholds.

Update `README.md`: in the Current Baseline section (lines 101–113), set the measurement date to today, question count 23 → 25, and replace the metric numbers with the actual values from the eval output.

- [ ] **Step 5: Full green check and commit**

Run: `uv run pytest -q` → green.
Run: `scripts/check.sh` → exits 0.

Update `documentation.md`: eval section gains the new tags (`docx`, `pptx`, `slide-notes`) and the fixture generator script reference.

```bash
git add scripts/make_office_fixtures.py data/docx/ data/pptx/ eval/questions.yaml README.md documentation.md
git commit -m "feat: DOCX/PPTX eval fixtures and golden questions, re-baselined metrics"
```

---

## Completion

After Task 5: merge `v4-phase-a` into `master` (fast-forward or merge commit per repo habit), then start Plan 2 (Phase B). Phase A is done when:

1. `uv run pytest -q` green (with and without the ocr group installed)
2. `uv run rag eval --retrieval-only --gate eval/gates.yaml` passes with the two new questions
3. `documentation.md` and `README.md` list DOCX, PPTX, and OCR support
