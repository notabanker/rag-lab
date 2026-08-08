"""Parse result with per-section extraction stats for quality reporting."""
from dataclasses import dataclass, field

# Sections with fewer stripped chars than this count as empty (page headers,
# stray page numbers on otherwise image-only pages).
EMPTY_SECTION_CHARS = 5
# Below this average per page a PDF is almost certainly scanned.
LOW_YIELD_PAGE_CHARS = 50


@dataclass
class ParseResult:
    text: str
    section_unit: str = "document"  # "page" | "chapter" | "section" | "slide" | "document"
    section_chars: list[int] = field(default_factory=list)

    @property
    def effective_text(self) -> str:
        """Text for chunking; blank when no real content was extracted, so
        section-marker scaffolding (e.g. "--- Page 1 ---") never becomes chunks."""
        sections = self.section_chars or [len(self.text.strip())]
        return self.text if sum(sections) > 0 else ""

    def quality(self) -> dict:
        sections = self.section_chars or [len(self.text.strip())]
        total = sum(sections)
        empty = sum(1 for c in sections if c < EMPTY_SECTION_CHARS)
        avg = total / len(sections)
        warnings = []
        if total == 0:
            warnings.append(
                "no text extracted — scanned/image-only file? OCR is not supported yet"
            )
        elif self.section_unit == "page":
            if avg < LOW_YIELD_PAGE_CHARS:
                warnings.append(
                    f"low text yield (avg {avg:.0f} chars/page) — probably a scanned PDF"
                )
            elif empty / len(sections) > 0.5:
                warnings.append(
                    f"{empty} of {len(sections)} pages contain no text"
                )
        return {
            "section_unit": self.section_unit,
            "sections": len(sections),
            "empty_sections": empty,
            "total_chars": total,
            "avg_section_chars": round(avg, 1),
            "warnings": warnings,
        }


def as_result(raw) -> ParseResult:
    """Coerce a parser return value; plain-str parsers get a single-section result."""
    if isinstance(raw, ParseResult):
        return raw
    return ParseResult(text=raw, section_unit="document", section_chars=[len(raw.strip())])
