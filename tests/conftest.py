import pytest
from pypdf import PdfWriter

from rag_lab import vector_store


@pytest.fixture
def store(tmp_path):
    """Isolated Chroma store under tmp_path; returns the persist dir."""
    path = str(tmp_path / "db")
    vector_store.init_store(path)
    return path


@pytest.fixture
def blank_pdf(tmp_path):
    """A genuinely blank (zero-text) PDF; returns its path."""
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        writer.write(f)
    return path
