"""Sandboxed executor for the E2 tools (prompts/e2_tools.json) on one workspace folder.

    ws = Workspace(path)
    result = ws.execute({"id": ..., "name": "write_file", "arguments": {...}, "valid": True})
    # -> {"tool_call_id", "name", "content", "is_error"}; ws.stats counts what happened

Tools:
- list_files()             one line per file "relative/path (N bytes)", sorted by path
- read_file(path)          the file's full UTF-8 text ("(empty file)" for a 0-byte file)
- write_file(path, content) creates parent folders, overwrites the file

Sandbox: a path is rejected if it is empty, absolute (including "C:..."), contains a
backslash, a NUL byte or a '..' component, or resolves outside the workspace (symlinks
followed). write_file rejects content over MAX_BYTES (UTF-8) and text that cannot be
encoded as UTF-8. Symlinks are never listed.

Every problem becomes an is_error tool result with a clear message. Counted separately:
- malformed: unknown tool, arguments that are not valid JSON, or arguments that violate the
  tool's schema (a required argument missing or not a string)
- rejected: sandbox rejections (bad path, oversize or non-UTF-8 content)
- errors: other tool errors (file not found, not UTF-8 on read, path is a folder)
The executor is deterministic: the same workspace and calls give the same results, so a
replayed session sends byte-identical requests and is served from the response cache.
"""
from __future__ import annotations

import os
import pathlib
import re
import time

MAX_BYTES = 1024 * 1024
TOOL_NAMES = ("list_files", "read_file", "write_file")
SCHEMA = {"list_files": (), "read_file": ("path",), "write_file": ("path", "content")}
LIMIT_MESSAGE = "Tool limit for this request reached."
EMPTY_FILE = "(empty file)"
EMPTY_WORKSPACE = "(the workspace contains no files)"
_DRIVE = re.compile(r"^[A-Za-z]:")


class Rejected(Exception):
    """Sandbox rejection."""


class ToolError(Exception):
    """Ordinary tool failure (not a sandbox rejection)."""


class Workspace:
    def __init__(self, root: pathlib.Path | str):
        self.root = pathlib.Path(root)
        self.real_root = self.root.resolve()
        self.stats = {"executed": 0, "malformed": 0, "rejected": 0, "errors": 0, "tool_time_s": 0.0}

    # ------------------------------------------------------------------ paths
    def resolve(self, rel) -> pathlib.Path:
        if not isinstance(rel, str) or not rel.strip():
            raise Rejected("Path must be a non-empty relative path such as index.html.")
        if "\x00" in rel:
            raise Rejected("Path contains a NUL byte.")
        if "\\" in rel:
            raise Rejected(f"Path {rel!r} contains a backslash; use forward slashes.")
        if rel.startswith("/") or _DRIVE.match(rel):
            raise Rejected(f"Absolute path {rel!r} is not allowed; use a path relative to the workspace root.")
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise Rejected(f"Path {rel!r} contains '..'; paths must stay inside the workspace.")
        if not parts:
            raise Rejected(f"Path {rel!r} does not name a file.")
        target = self.root.joinpath(*parts)
        real = target.resolve()
        if not real.is_relative_to(self.real_root) or real == self.real_root:
            raise Rejected(f"Path {rel!r} resolves outside the workspace.")
        return target

    def rel(self, p: pathlib.Path) -> str:
        return p.relative_to(self.root).as_posix()

    # ------------------------------------------------------------------ tools
    def list_files(self) -> str:
        files = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = [d for d in dirnames if not (pathlib.Path(dirpath) / d).is_symlink()]
            for f in filenames:
                p = pathlib.Path(dirpath) / f
                if p.is_symlink() or not p.is_file():
                    continue
                files.append((self.rel(p), p.stat().st_size))
        files.sort()
        return "\n".join(f"{r} ({n} bytes)" for r, n in files) if files else EMPTY_WORKSPACE

    def read_file(self, path: str) -> str:
        p = self.resolve(path)
        if p.is_dir():
            raise ToolError(f"{path} is a folder, not a file.")
        if not p.is_file():
            raise ToolError(f"File not found: {path}. Use list_files to see the files in the workspace.")
        data = p.read_bytes()
        if len(data) > MAX_BYTES:
            raise ToolError(f"{path} is larger than {MAX_BYTES} bytes.")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{path} is not UTF-8 text.") from None
        return text if text else EMPTY_FILE

    def write_file(self, path: str, content: str) -> str:
        p = self.resolve(path)
        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError:
            raise Rejected("Content is not valid UTF-8 text.") from None
        if len(data) > MAX_BYTES:
            raise Rejected(f"Content is {len(data)} bytes; the limit is {MAX_BYTES} bytes per file.")
        if p.is_dir():
            raise ToolError(f"{path} is a folder, not a file.")
        for parent in reversed(p.relative_to(self.root).parents[:-1]):
            q = self.root / parent
            if q.exists() and not q.is_dir():
                raise ToolError(f"Cannot create {path}: {parent.as_posix()} is a file.")
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink():
            p.unlink()                     # replace the link itself, never write through it
        p.write_bytes(data)
        return f"Wrote {len(data)} bytes to {self.rel(p)}."

    # ------------------------------------------------------------------ dispatch
    def execute(self, call: dict) -> dict:
        """Execute one tool call {id, name, arguments, valid}; returns a tool result."""
        t0 = time.perf_counter()
        name = call.get("name")
        out = {"tool_call_id": call.get("id"), "name": name, "content": "", "is_error": False}
        try:
            out["content"] = self._dispatch(name, call)
        except _Malformed as e:
            self.stats["malformed"] += 1
            out.update(content=f"Error: {e}", is_error=True)
        except Rejected as e:
            self.stats["rejected"] += 1
            out.update(content=f"Error: {e}", is_error=True)
        except ToolError as e:
            self.stats["errors"] += 1
            out.update(content=f"Error: {e}", is_error=True)
        except OSError as e:
            self.stats["errors"] += 1
            out.update(content=f"Error: {type(e).__name__}: {e.strerror or e}", is_error=True)
        self.stats["executed"] += 1
        self.stats["tool_time_s"] += time.perf_counter() - t0
        return out

    def _dispatch(self, name, call: dict) -> str:
        if name not in TOOL_NAMES:
            raise _Malformed(f"Unknown tool {name!r}. Available tools: {', '.join(TOOL_NAMES)}.")
        args = call.get("arguments")
        if call.get("valid") is False or not isinstance(args, dict):
            raise _Malformed(f"The arguments for {name} are not a valid JSON object.")
        for key in SCHEMA[name]:
            if key not in args:
                raise _Malformed(f"{name} requires the argument {key!r}.")
            if not isinstance(args[key], str):
                raise _Malformed(f"The argument {key!r} of {name} must be a string.")
        if name == "list_files":
            return self.list_files()
        if name == "read_file":
            return self.read_file(args["path"])
        return self.write_file(args["path"], args["content"])


class _Malformed(Exception):
    pass


def limit_result(call: dict) -> dict:
    return {"tool_call_id": call.get("id"), "name": call.get("name"), "content": LIMIT_MESSAGE, "is_error": True}
