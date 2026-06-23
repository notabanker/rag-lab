import json

import pytest

from rag_lab.verifier import _extract_json


class TestExtractJson:
    def test_plain_json(self):
        raw = '{"score": 8, "grounded": true}'
        result = _extract_json(raw)
        assert result["score"] == 8

    def test_fenced_json(self):
        raw = '```json\n{"score": 7, "grounded": false}\n```'
        result = _extract_json(raw)
        assert result["score"] == 7

    def test_fenced_no_tag(self):
        raw = '```\n{"score": 9}\n```'
        result = _extract_json(raw)
        assert result["score"] == 9

    def test_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("not json at all")

    def test_extra_text_around_json(self):
        raw = 'Some text before\n{"score": 5}\nSome text after'
        result = _extract_json(raw)
        assert result["score"] == 5
