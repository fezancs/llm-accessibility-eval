"""E2 tool executor sandbox (harness/e2_tools.py). Synthetic workspaces in pytest tmp dirs."""
import os

from harness.e2_tools import EMPTY_FILE, LIMIT_MESSAGE, MAX_BYTES, Workspace, limit_result


def call(name, args=None, valid=True, cid="c1"):
    return {"id": cid, "name": name, "arguments": args, "valid": valid}


def ws_with_files(tmp_path):
    root = tmp_path / "ws"
    (root / "css").mkdir(parents=True)
    (root / "index.html").write_text("<p>hi</p>", encoding="utf-8")
    (root / "css" / "site.css").write_text("p{}", encoding="utf-8")
    (root / "app.js").write_text("", encoding="utf-8")
    return Workspace(root)


def test_list_read_write(tmp_path):
    ws = ws_with_files(tmp_path)
    r = ws.execute(call("list_files", {}))
    assert not r["is_error"]
    assert r["content"].splitlines() == ["app.js (0 bytes)", "css/site.css (3 bytes)", "index.html (9 bytes)"]
    assert ws.execute(call("read_file", {"path": "index.html"}))["content"] == "<p>hi</p>"
    assert ws.execute(call("read_file", {"path": "./css/site.css"}))["content"] == "p{}"
    assert ws.execute(call("read_file", {"path": "app.js"}))["content"] == EMPTY_FILE
    w = ws.execute(call("write_file", {"path": "pages/news.html", "content": "<h1>Né</h1>"}))
    assert not w["is_error"] and "Wrote 12 bytes to pages/news.html" in w["content"]
    assert (ws.root / "pages" / "news.html").read_text(encoding="utf-8") == "<h1>Né</h1>"
    assert ws.stats["executed"] == 5 and ws.stats["malformed"] == ws.stats["rejected"] == 0
    assert ws.stats["tool_time_s"] >= 0
    assert r["tool_call_id"] == "c1" and r["name"] == "list_files"


def test_empty_workspace_listing(tmp_path):
    (tmp_path / "e").mkdir()
    assert "no files" in Workspace(tmp_path / "e").execute(call("list_files", {}))["content"]


def test_sandbox_rejections(tmp_path):
    ws = ws_with_files(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    bad_paths = ["/etc/passwd", str(outside), "C:/x.html", "../secret.txt", "css/../../secret.txt", "a/../index.html",
                 "css\\site.css", "..", "", "   ", "a\x00b"]
    for p in bad_paths:
        for c in (call("read_file", {"path": p}), call("write_file", {"path": p, "content": "x"})):
            r = ws.execute(c)
            assert r["is_error"], p
            assert r["content"].startswith("Error:")
    assert ws.stats["rejected"] == 2 * len(bad_paths)
    assert ws.stats["malformed"] == 0
    assert outside.read_text(encoding="utf-8") == "secret"
    assert not (tmp_path / "x.html").exists()


def test_symlink_escape(tmp_path):
    ws = ws_with_files(tmp_path)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "target.txt").write_text("outside", encoding="utf-8")
    os.symlink(outside_dir, ws.root / "link")                       # folder link
    os.symlink(outside_dir / "target.txt", ws.root / "file.txt")    # file link
    os.symlink(tmp_path / "nonexistent.txt", ws.root / "dangling")  # dangling link pointing outside
    for p in ("link/target.txt", "file.txt", "dangling", "link/new.txt"):
        assert ws.execute(call("read_file", {"path": p}))["is_error"]
        assert ws.execute(call("write_file", {"path": p, "content": "pwned"}))["is_error"]
    assert (outside_dir / "target.txt").read_text(encoding="utf-8") == "outside"
    assert not (outside_dir / "new.txt").exists() and not (tmp_path / "nonexistent.txt").exists()
    assert ws.stats["rejected"] == 8
    listing = ws.execute(call("list_files", {}))["content"]
    assert "link" not in listing and "file.txt" not in listing and "dangling" not in listing
    os.symlink(ws.root / "index.html", ws.root / "alias.html")      # link inside the workspace is fine
    assert ws.execute(call("read_file", {"path": "alias.html"}))["content"] == "<p>hi</p>"


def test_size_and_encoding_limits(tmp_path):
    ws = ws_with_files(tmp_path)
    r = ws.execute(call("write_file", {"path": "big.html", "content": "x" * (MAX_BYTES + 1)}))
    assert r["is_error"] and "limit" in r["content"]
    assert not (ws.root / "big.html").exists()
    ok = ws.execute(call("write_file", {"path": "max.html", "content": "x" * MAX_BYTES}))
    assert not ok["is_error"]
    r = ws.execute(call("write_file", {"path": "s.html", "content": "bad \ud800 surrogate"}))
    assert r["is_error"] and "UTF-8" in r["content"]
    assert ws.stats["rejected"] == 2
    (ws.root / "bin.dat").write_bytes(b"\xff\xfe\x00")
    r = ws.execute(call("read_file", {"path": "bin.dat"}))
    assert r["is_error"] and "UTF-8" in r["content"] and ws.stats["errors"] == 1


def test_malformed_and_unknown(tmp_path):
    ws = ws_with_files(tmp_path)
    cases = [
        call("delete_file", {"path": "index.html"}),                 # unknown tool
        call("run_shell", {}),                                       # unknown tool
        call("write_file", None, valid=False),                       # invalid JSON
        call("read_file", None),                                     # missing arguments object
        call("read_file", {}),                                       # required argument missing
        call("read_file", {"file": "index.html"}),                   # wrong argument name
        call("write_file", {"path": "a.html"}),                      # content missing
        call("write_file", {"path": "a.html", "content": 5}),        # mistyped
        call("read_file", {"path": ["index.html"]}),                 # mistyped
    ]
    for c in cases:
        r = ws.execute(c)
        assert r["is_error"] and r["content"].startswith("Error:"), c
    assert ws.stats["malformed"] == len(cases)
    assert ws.stats["rejected"] == 0
    assert "Unknown tool 'delete_file'" in ws.execute(call("delete_file", {}))["content"]
    assert (ws.root / "index.html").exists() and not (ws.root / "a.html").exists()
    # an ordinary tool error is neither malformed nor a sandbox rejection
    r = ws.execute(call("read_file", {"path": "missing.html"}))
    assert r["is_error"] and "not found" in r["content"] and ws.stats["errors"] == 1
    r = ws.execute(call("read_file", {"path": "css"}))
    assert r["is_error"] and "folder" in r["content"]
    # extra arguments are allowed by the schema
    assert not ws.execute(call("list_files", {"verbose": True}))["is_error"]


def test_limit_result():
    r = limit_result({"id": "x", "name": "write_file"})
    assert r == {"tool_call_id": "x", "name": "write_file", "content": LIMIT_MESSAGE, "is_error": True}
