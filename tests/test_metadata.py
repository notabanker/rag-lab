import json

from rag_lab.metadata import extract_frontmatter, frontmatter_to_metadata


class TestExtractFrontmatter:
    def test_with_frontmatter(self):
        fm, body = extract_frontmatter("---\nkey: val\n---\n\n# Body")
        assert fm == {"key": "val"}
        assert body == "# Body"

    def test_list_aliases(self):
        fm, body = extract_frontmatter("---\naliases: [a, b]\n---\n\nBody")
        assert fm == {"aliases": ["a", "b"]}
        assert body == "Body"

    def test_no_frontmatter(self):
        fm, body = extract_frontmatter("# Just a heading\n\nBody")
        assert fm == {}
        assert body == "# Just a heading\n\nBody"

    def test_invalid_yaml(self):
        fm, body = extract_frontmatter("---\nbad: [unclosed\n---\n\nBody")
        assert fm == {}
        assert body == "---\nbad: [unclosed\n---\n\nBody"

    def test_not_dict(self):
        fm, body = extract_frontmatter("---\n- list item\n---\n\nBody")
        assert fm == {}
        assert body == "---\n- list item\n---\n\nBody"

    def test_empty_frontmatter(self):
        fm, body = extract_frontmatter("---\n---\n\nBody")
        assert fm == {}
        assert body == "Body"

    def test_real_entity_frontmatter(self):
        raw = """---
canonical_id: node:6a16c2067054
type: agent
aliases: [hermes]
last_synced: 2026-05-22T19:41:06Z
node_count_inbound: 0
node_count_outbound: 9
---

# hermes"""
        fm, body = extract_frontmatter(raw)
        assert fm["canonical_id"] == "node:6a16c2067054"
        assert fm["type"] == "agent"
        assert fm["aliases"] == ["hermes"]
        assert fm["node_count_outbound"] == 9
        assert body == "# hermes"


class TestFrontmatterToMetadata:
    def test_canonical_id(self):
        fm = {"canonical_id": "node:abc"}
        meta = frontmatter_to_metadata(fm, "entities/test.md", "l3_semantic")
        assert meta["canonical_id"] == "node:abc"

    def test_node_type(self):
        fm = {"type": "agent"}
        meta = frontmatter_to_metadata(fm, "entities/test.md", "l3_semantic")
        assert meta["node_type"] == "agent"

    def test_aliases_list_to_json(self):
        fm = {"aliases": ["hermes", "gateway"]}
        meta = frontmatter_to_metadata(fm, "entities/test.md", "l3_semantic")
        assert meta["aliases"] == json.dumps(["hermes", "gateway"])

    def test_last_synced(self):
        fm = {"last_synced": "2026-05-22T19:41:06Z"}
        meta = frontmatter_to_metadata(fm, "entities/test.md", "l3_semantic")
        assert meta["last_synced"] == "2026-05-22T19:41:06Z"

    def test_extra_fields_in_frontmatter_blob(self):
        fm = {"type": "agent", "node_count_inbound": 5}
        meta = frontmatter_to_metadata(fm, "entities/test.md", "l3_semantic")
        assert "frontmatter" in meta
        extra = json.loads(meta["frontmatter"])
        assert extra["node_count_inbound"] == 5

    def test_basic_fields_always_present(self):
        meta = frontmatter_to_metadata({}, "entities/test.md", "l3_semantic")
        assert meta["file_path"] == "entities/test.md"
        assert meta["file_name"] == "test.md"
        assert meta["layer"] == "l3_semantic"

    def test_no_frontmatter_only_basic_fields(self):
        meta = frontmatter_to_metadata({}, "entities/test.md", "l3_semantic")
        assert set(meta.keys()) == {"file_path", "file_name", "layer"}
