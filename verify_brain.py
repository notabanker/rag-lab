"""Hermes Brain — startup verification. Run once after setup."""
import httpx
import os
import sys

def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return ok

print("Hermes Brain — Verification\n")

# 1. Environment
vault = os.environ.get("STOA_VAULT_ROOT", "")
ok = check("STOA_VAULT_ROOT set", bool(vault), vault)
if not ok:
    print("\n  Set it: export STOA_VAULT_ROOT=/Users/notabanker/vault")
    sys.exit(1)

# 2. Vault structure
import pathlib
ok = True
for d in ["L1-working", "L2-episodic", "L3-semantic"]:
    exists = (pathlib.Path(vault) / d).is_dir()
    ok &= check(f"Vault dir: {d}", exists)

# 3. vault_bridge imports
try:
    from vault_bridge import read_today, read_recent_sessions, write_session
    check("vault_bridge import", True)
except Exception as e:
    check("vault_bridge import", False, str(e))

# 4. Search API
try:
    r = httpx.get("http://127.0.0.1:8001/api/stats", timeout=5)
    ok = check("Search API reachable", r.status_code == 200, f"status={r.status_code}")
    if ok:
        data = r.json()
        check("Chunks indexed", data.get("chunk_count", 0) > 0, f"{data.get('chunk_count', 0)} chunks")
except Exception as e:
    check("Search API reachable", False, str(e))

# 5. Write test session
try:
    result = write_session(
        "test",
        "hermes",
        [{"role": "user", "content": "Brain verification test — hermes gateway config check"}],
        task="brain verification",
    )
    import pathlib
    written = pathlib.Path(result["l2_path"]).exists()
    check("Write test session", written, result["l2_path"])
    check("Session importance", result["important"], "routed to agent-logs" if result["important"] else "routed to drafts")
    if written:
        pathlib.Path(result["l2_path"]).unlink(missing_ok=True)
except Exception as e:
    check("Write test session", False, str(e))

# 6. Search for existing content
try:
    r = httpx.post("http://127.0.0.1:8001/api/search", json={
        "query": "hermes agent system architecture",
        "layers": ["l3_semantic"],
        "top_k": 3,
    }, timeout=30)
    ok = check("Semantic search", r.status_code == 200 and len(r.json().get("results", [])) > 0,
               f"{len(r.json().get('results', []))} results")
except Exception as e:
    check("Semantic search", False, str(e))

print("\nDone. If all PASS, your brain is online.")
