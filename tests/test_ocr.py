import importlib

from rag_lab import config, vector_store
from rag_lab.parsers import ocr


def test_ocr_langs_env_default(monkeypatch):
    monkeypatch.delenv("RAG_OCR_LANGS", raising=False)
    importlib.reload(config)
    try:
        assert config.RAG_OCR_LANGS == "deu+eng"
    finally:
        importlib.reload(config)


def test_ocr_langs_env_override(monkeypatch):
    monkeypatch.setenv("RAG_OCR_LANGS", "eng")
    importlib.reload(config)
    try:
        assert config.RAG_OCR_LANGS == "eng"
    finally:
        monkeypatch.delenv("RAG_OCR_LANGS")
        importlib.reload(config)


def test_ocr_available_false_when_engine_missing(monkeypatch):
    monkeypatch.setattr("rag_lab.parsers.ocr.import_module", _boom)
    assert ocr.ocr_available() is False


def _boom(name, **kw):
    raise ImportError(name)


class _FakeNamedTemp:
    def __init__(self, name, *a, **kw):
        self.name = name

    def write(self, data):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ocr_page_renders_and_ocrs(tmp_path, monkeypatch):
    calls = {}

    class FakePix:
        def tobytes(self, fmt):
            assert fmt == "png"
            return b"\x89PNG\r\n\x1a\nfakepngdata"

    class FakePixPage:
        def get_pixmap(self, dpi=72):
            calls["dpi"] = dpi
            return FakePix()

    class FakeDoc:
        def __init__(self, path):
            calls["open_path"] = path

        def __getitem__(self, i):
            calls["page"] = i
            return FakePixPage()

        def close(self):
            calls["closed"] = True

    class FakeTemp:
        NamedTemporaryFile = staticmethod(
            lambda *a, **kw: _FakeNamedTemp(str(tmp_path / "fake-ocr.png"), *a, **kw)
        )

    fake_fitz = type("fitz", (), {"open": FakeDoc})
    fake_tess = type("tess", (), {
        "image_to_string": lambda path, lang: calls.__setitem__("lang", lang) or "OCR TEXT",
        "get_tesseract_version": lambda: "5.0",
    })
    monkeypatch.setattr("rag_lab.parsers.ocr.fitz", fake_fitz)
    monkeypatch.setattr("rag_lab.parsers.ocr.pytesseract", fake_tess)
    monkeypatch.setattr("rag_lab.parsers.ocr.tempfile", FakeTemp)
    out = ocr.ocr_page("/tmp/x.pdf", 2, "deu+eng")
    assert out == "OCR TEXT"
    assert calls["page"] == 2
    assert calls["lang"] == "deu+eng"
    assert calls["open_path"] == "/tmp/x.pdf"
    assert calls["dpi"] == 300
    assert calls["closed"] is True


def test_ocr_pages_cache_hit_and_miss(tmp_path, monkeypatch):
    vector_store.init_store(str(tmp_path / "db"))
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"scan-bytes")
    monkeypatch.setattr(
        "rag_lab.parsers.ocr.ocr_page",
        lambda path, i, langs: f"page{i}",
    )
    first = ocr.ocr_pages(str(pdf), [0, 1], "deu+eng")
    assert first == {0: "page0", 1: "page1"}
    # Zweiter Aufruf: keine erneute OCR — Cache greift
    seen = []
    monkeypatch.setattr(
        "rag_lab.parsers.ocr.ocr_page",
        lambda path, i, langs: seen.append(i) or "SHOULD-NOT-RUN",
    )
    second = ocr.ocr_pages(str(pdf), [0, 1, 2], "deu+eng")
    assert second[0] == "page0" and second[1] == "page1"
    assert second[2] == "SHOULD-NOT-RUN" # nur Seite 2 neu OCR'd
    assert seen == [2]
    # Cache-Datei existiert neben dem persist_dir
    assert (tmp_path / "db" / "ocr_cache.json").exists()
    # Reset the module-global persist dir so other tests are unaffected.
    vector_store.init_store(None)
