"""Rendering (SPEC 3): a local static HTTP server and Playwright/Chromium pages.

Pages are served over http://127.0.0.1 (not file://) so relative paths work. Requests to any
other host are blocked, so generated pages behave the same on every run and cannot reach
the network; missing local images simply 404. JavaScript dialogs (alert/confirm/prompt)
are accepted and recorded so a page cannot hang the run.

Playwright's sync API is not thread-safe: use one Renderer per process.
"""
from __future__ import annotations

import contextlib
import functools
import http.server
import pathlib
import socketserver
import threading
import urllib.parse

from playwright.sync_api import Page, sync_playwright

from . import config

VIEWPORT = {"width": 1280, "height": 800}
SETTLE_MS = 500

# Records timers and moving content so checks can be repeated at t = 6 s (SPEC 3).
INIT_SCRIPT = r"""
(() => {
  const rec = {intervals: 0, longTimeouts: 0, raf: 0};
  window.__se4allTimers = rec;
  const si = window.setInterval, st = window.setTimeout, raf = window.requestAnimationFrame;
  window.setInterval = function (fn, ms, ...a) { rec.intervals++; return si.call(this, fn, ms, ...a); };
  window.setTimeout = function (fn, ms, ...a) { if ((ms || 0) >= 1000) rec.longTimeouts++; return st.call(this, fn, ms, ...a); };
  window.requestAnimationFrame = function (fn) { rec.raf++; return raf.call(this, fn); };
})();
"""


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves files by absolute path, but only under the allowed roots (404 elsewhere)."""
    roots: tuple[pathlib.Path, ...] = ()

    def translate_path(self, path):
        fs = pathlib.Path(super().translate_path(path))
        try:
            real = fs.resolve()
        except OSError:
            return "/nonexistent"
        # Symlinks inside an allowed root (e.g. artifact folders linking the corpus images) are
        # allowed when they point into another allowed root.
        if any(real.is_relative_to(r) for r in self.roots) and any(fs.is_relative_to(r) for r in self.roots):
            return str(fs)
        return "/nonexistent"

    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class StaticServer:
    """Serves pages under the allowed roots on 127.0.0.1 at a free port, by absolute path."""

    def __init__(self, *roots: pathlib.Path):
        allowed = {pathlib.Path(r).resolve() for r in (*roots, config.ROOT, config.RUNS)}
        handler_cls = type("_RootedHandler", (_Handler,), {"roots": tuple(sorted(allowed))})
        handler = functools.partial(handler_cls, directory="/")
        self.httpd = _Server(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self, path: pathlib.Path | str) -> str:
        p = pathlib.Path(path).absolute()
        return f"http://127.0.0.1:{self.port}{urllib.parse.quote(p.as_posix())}"

    def close(self):
        self.httpd.shutdown()


class Renderer:
    def __init__(self, *roots: pathlib.Path):
        self.server = StaticServer(*roots)
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=True)
        self.dialogs: list[str] = []

    @property
    def chromium_version(self) -> str:
        return self.browser.version

    def close(self):
        with contextlib.suppress(Exception):
            self.browser.close()
        with contextlib.suppress(Exception):
            self._pw.stop()
        self.server.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextlib.contextmanager
    def page(self, path: pathlib.Path | str, settle: bool = True):
        """Open a fresh page (new browser context) for `path`; closed on exit."""
        ctx = self.browser.new_context(viewport=VIEWPORT)
        ctx.add_init_script(INIT_SCRIPT)
        ctx.route("**/*", _only_local)
        page = ctx.new_page()
        page.set_default_timeout(15_000)
        page.on("dialog", lambda d: (self.dialogs.append(f"{d.type}: {d.message}"), d.accept()))
        try:
            url = path if str(path).startswith("http") else self.server.url(path)
            try:
                page.goto(url, wait_until="networkidle", timeout=20_000)
            except Exception:              # noqa: BLE001 - pages with endless polling never go idle
                page.wait_for_load_state("load")
            if settle:
                page.wait_for_timeout(SETTLE_MS)
            yield page
        finally:
            with contextlib.suppress(Exception):
                ctx.close()


def _only_local(route):
    url = route.request.url
    if url.startswith("http://127.0.0.1:") or url.startswith("data:") or url.startswith("blob:"):
        route.continue_()
    else:
        route.abort()


def has_motion(page: Page) -> bool:
    """True if the page started timers/rAF loops or runs infinite CSS animations."""
    return bool(page.evaluate("""() => {
        const t = window.__se4allTimers || {};
        const anims = document.getAnimations ? document.getAnimations().filter(a =>
            a.playState === 'running' && a.effect && a.effect.getTiming &&
            a.effect.getTiming().iterations === Infinity) : [];
        return (t.intervals || 0) + (t.longTimeouts || 0) > 0 || anims.length > 0 || (t.raf || 0) > 30;
    }"""))
