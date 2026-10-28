"""Every probe passes its accessible fixture and fails its inaccessible one (SPEC 4.1)."""
import pytest
import yaml

from harness import config, probes
from harness.render import Renderer

FIX = config.HARNESS / "probe_fixtures"
MANIFEST = yaml.safe_load((FIX / "manifest.yaml").read_text())


@pytest.fixture(scope="module")
def renderer():
    with Renderer(FIX) as r:
        yield r


def test_manifest_covers_every_probe():
    assert set(MANIFEST) == set(probes.REGISTRY)


@pytest.mark.parametrize("name", sorted(MANIFEST))
@pytest.mark.parametrize("kind", ["pass", "fail"])
def test_probe_fixture(renderer, name, kind):
    spec = MANIFEST[name] or {}
    page = FIX / f"{name}__{kind}.html"
    out = probes.run_probe(name, lambda: renderer.page(page), spec.get("scope"), spec.get("anchors"))
    assert not out.get("error"), out
    assert out["applicable"], out
    assert out["passed"] is (kind == "pass"), out


def test_focus_threshold_calibration(renderer):
    """The focus_visible threshold sits far from both fixtures (SPEC 4.2 calibration)."""
    diffs = {}
    for kind in ("pass", "fail"):
        out = probes.run_probe("focus_visible", lambda: renderer.page(FIX / f"focus_visible__{kind}.html"))
        diffs[kind] = [s["diff"] for s in out["detail"]["sampled"]]
    assert max(diffs["fail"]) < 0.25 * probes.FOCUS_DIFF_THRESHOLD, diffs
    assert min(diffs["pass"]) > 10 * probes.FOCUS_DIFF_THRESHOLD, diffs
