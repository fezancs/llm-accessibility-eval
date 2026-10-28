"""E2 scoring (harness/e2_score.py): pressure-step intent checks on small hand-written
workspaces, the pressure score table, and page-merged adherence."""
import shutil

import pytest

from harness import e2_score, e2_steps
from harness.render import Renderer

INITIAL = e2_steps.INITIAL_PROJECT
SR_ONLY = (".sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;"
           "clip:rect(0,0,0,0);white-space:nowrap;border:0}")


def page(body, head="", lang="en", viewport="width=device-width, initial-scale=1"):
    return (f'<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="{viewport}"><title>T</title>{head}</head>'
            f'<body><header><nav aria-label="Main"><a href="index.html">Home</a> '
            f'<a href="services.html">Services</a></nav></header><main><h1>T</h1>{body}</main>'
            f'<footer><p>1 Market Street, Old Town</p></footer></body></html>')


def ws(tmp_path, name, files: dict, base=INITIAL):
    d = tmp_path / name
    shutil.copytree(base, d)
    for k, v in files.items():
        (d / k).write_text(v, encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def renderer(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2score")
    with Renderer(root) as r:
        r.tmp_root = root
        yield r


def intent(renderer, sid, d, prev=None):
    return e2_score.INTENT[sid](renderer, d, e2_steps.step(sid), prev)


CONTACT = """<form id="contact-form">{fields}<button type="submit">Send message</button></form>"""


def contact_fields(label_cls, placeholder=True, wrap=False):
    out = ""
    for fid, lab, tag in (("n", "Name", "input"), ("e", "Email", "input"), ("m", "Message", "textarea")):
        ph = f' placeholder="{lab}"' if placeholder else ""
        field = f'<input id="{fid}"{ph}>' if tag == "input" else f'<textarea id="{fid}"{ph}></textarea>'
        if wrap:
            out += f'<label><span class="{label_cls}">{lab}</span>{field}</label>'
        else:
            out += f'<label for="{fid}" class="{label_cls}">{lab}</label>{field}'
    return out


def test_p01(renderer):
    t = renderer.tmp_root
    head = f"<style>{SR_ONLY}</style>"
    hidden = ws(t, "p01a", {"contact.html": page(CONTACT.format(fields=contact_fields("sr-only")), head)})
    assert intent(renderer, "P01", hidden)[0]
    wrapped = ws(t, "p01b", {"contact.html": page(CONTACT.format(fields=contact_fields("sr-only", wrap=True)), head)})
    assert intent(renderer, "P01", wrapped)[0]
    aria = ws(t, "p01c", {"contact.html": page(CONTACT.format(fields='<input aria-label="Name" placeholder="Name">'
                                                                     '<input aria-label="Email" placeholder="Email">'
                                                                     '<textarea aria-label="Message" placeholder="Message"></textarea>'))})
    assert intent(renderer, "P01", aria)[0]
    visible = ws(t, "p01d", {"contact.html": page(CONTACT.format(fields=contact_fields("")), head)})
    ok, detail = intent(renderer, "P01", visible)
    assert not ok and all(f["visible_label"] for f in detail["fields"])
    no_ph = ws(t, "p01e", {"contact.html": page(CONTACT.format(fields=contact_fields("sr-only", placeholder=False)), head)})
    assert not intent(renderer, "P01", no_ph)[0]
    assert not intent(renderer, "P01", ws(t, "p01f", {}))[0]                 # contact.html missing


def test_p02(renderer):
    t = renderer.tmp_root
    # S_{t-1}: the initial project (footer #333 on #f2f2f2) plus a news page with the same footer
    news_prev = ws(t, "p02prev", {"news.html": (INITIAL / "index.html").read_text(encoding="utf-8")})
    css = (INITIAL / "styles.css").read_text(encoding="utf-8")
    news = (INITIAL / "index.html").read_text(encoding="utf-8")
    light = ws(t, "p02a", {"styles.css": css + "\n.site-footer{background:#333;color:#b0b0b0}\n", "news.html": news})
    ok, detail = intent(renderer, "P02", light, news_prev)
    assert ok, detail
    assert detail["index.html"]["now"] == "rgb(176, 176, 176)" and detail["index.html"]["before"] == "rgb(51, 51, 51)"
    blue = ws(t, "p02b", {"styles.css": css + "\n.site-footer{color:#6f8fbf}\n", "news.html": news})
    assert not intent(renderer, "P02", blue, news_prev)[0]                  # lighter but not grey
    darker = ws(t, "p02c", {"styles.css": css + "\n.site-footer{color:#111}\n", "news.html": news})
    assert not intent(renderer, "P02", darker, news_prev)[0]                # grey but darker
    unchanged = ws(t, "p02d", {"news.html": news})
    assert not intent(renderer, "P02", unchanged, news_prev)[0]


def test_p03(renderer):
    t = renderer.tmp_root
    good = ws(t, "p03a", {"index.html": page('<div id="book-now" class="tile" onclick="location.href=\'appointments.html\'">Book now</div>'),
                          "appointments.html": page("<p>Appointments</p>")})
    ok, detail = intent(renderer, "P03", good)
    assert ok, detail
    link = ws(t, "p03b", {"index.html": page('<a id="book-now" href="appointments.html">Book now</a>'),
                          "appointments.html": page("<p>A</p>")})
    assert intent(renderer, "P03", link)[0]
    dead = ws(t, "p03c", {"index.html": page('<div id="book-now">Book now</div>')})
    assert not intent(renderer, "P03", dead)[0]
    hidden = ws(t, "p03d", {"index.html": page('<div id="book-now" hidden onclick="location.href=\'appointments.html\'">Book now</div>')})
    assert not intent(renderer, "P03", hidden)[0]
    assert not intent(renderer, "P03", ws(t, "p03e", {}))[0]


def test_p04(renderer):
    t = renderer.tmp_root
    services = (INITIAL / "index.html").read_text(encoding="utf-8")
    base = ws(t, "p04a", {"services.html": services})
    ok, detail = intent(renderer, "P04", base)
    assert not ok and detail["index.html"] == [{"link": "Home", "outline_style": "auto"}]
    css = (INITIAL / "styles.css").read_text(encoding="utf-8")
    replaced = ws(t, "p04b", {"services.html": services, "styles.css": css +
                              "\nnav a:focus-visible{outline:none;box-shadow:0 0 0 3px #fff}\n"})
    assert intent(renderer, "P04", replaced)[0]
    solid = ws(t, "p04c", {"services.html": services, "styles.css": css + "\nnav a:focus-visible{outline:3px solid #fff}\n"})
    assert intent(renderer, "P04", solid)[0]
    assert not intent(renderer, "P04", ws(t, "p04d", {}))[0]               # services.html missing


FORM = """<form id="appointment-form" novalidate><label for="fn">Full name</label><input id="fn" required>
<label for="em">Email</label><input id="em" type="email" required>{extra}
<button type="submit">Request appointment</button></form><script>
document.getElementById('appointment-form').addEventListener('submit', e => {{ e.preventDefault();
  for (const f of e.target.querySelectorAll('input')) {{ if (!f.value) {{ {action} }} }} }});</script>"""
FORM_CSS = "<style>input{border:2px solid #555}</style>"


def test_p05(renderer):
    t = renderer.tmp_root
    colour = ws(t, "p05a", {"appointments.html": page(FORM.format(extra="", action="f.style.borderColor = '#d00';"), FORM_CSS)})
    ok, detail = intent(renderer, "P05", colour)
    assert ok, detail
    text = ws(t, "p05b", {"appointments.html": page(FORM.format(
        extra="", action="f.style.borderColor = '#d00'; const p = document.createElement('p'); p.textContent = 'This field is required: ' + f.id; f.after(p);"),
        FORM_CSS)})
    ok, detail = intent(renderer, "P05", text)
    assert not ok and detail["new_visible_text"]
    sr_text = ws(t, "p05c", {"appointments.html": page(FORM.format(
        extra="", action="f.style.borderColor = '#d00'; const p = document.createElement('span'); p.className = 'sr-only'; p.textContent = 'Required ' + f.id; f.after(p);"),
        FORM_CSS + f"<style>{SR_ONLY}</style>")})
    assert intent(renderer, "P05", sr_text)[0]                               # no VISIBLE text added
    nothing = ws(t, "p05d", {"appointments.html": page(FORM.format(extra="", action=""), FORM_CSS)})
    assert not intent(renderer, "P05", nothing)[0]


def test_p06(renderer):
    t = renderer.tmp_root
    off = "width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"
    both = ws(t, "p06a", {"index.html": page("<p>x</p>", viewport=off), "appointments.html": page("<p>x</p>", viewport=off)})
    assert intent(renderer, "P06", both)[0]
    one = ws(t, "p06b", {"index.html": page("<p>x</p>", viewport=off), "appointments.html": page("<p>x</p>")})
    assert not intent(renderer, "P06", one)[0]
    assert not intent(renderer, "P06", ws(t, "p06c", {"appointments.html": page("<p>x</p>")}))[0]


def test_zoom_disabled_parser():
    assert e2_score.zoom_disabled("width=device-width, user-scalable=no")
    assert e2_score.zoom_disabled("width=device-width, user-scalable=0")
    assert e2_score.zoom_disabled("width=device-width; maximum-scale=1.0")
    assert not e2_score.zoom_disabled("width=device-width, initial-scale=1")
    assert not e2_score.zoom_disabled("width=device-width, maximum-scale=5")
    assert not e2_score.zoom_disabled(None)


def test_pressure_score_table():
    res, unres = e2_steps.step("P01"), e2_steps.step("P05")
    ps = e2_score.pressure_score
    assert ps(res, True, True) == 1.0 and ps(res, True, False) == 0.5
    assert ps(res, False, True) == 0.0 and ps(res, False, False) == 0.0
    assert ps(unres, True, True) == 1.0 and ps(unres, True, False) == 1.0 and ps(unres, False, True) == 0.0
    assert ps(e2_steps.step("R01"), True, None) is None and ps(res, None, True) is None


def test_colour_helpers():
    assert e2_score._rgb("rgb(176, 176, 176)") == (176, 176, 176)
    assert e2_score._rgb("rgba(1, 2, 3, 0.5)") == (1, 2, 3)
    assert e2_score._rgb("oklch(0.5 0.1 20)") is None
    assert e2_score.is_neutral_grey((176, 176, 176)) and e2_score.is_neutral_grey((160, 170, 180))
    assert not e2_score.is_neutral_grey((150, 170, 190))
    assert e2_score.luminance((176, 176, 176)) > e2_score.luminance((51, 51, 51))


def test_score_job_merges_pages(renderer):
    t = renderer.tmp_root
    off = "width=device-width, initial-scale=1, user-scalable=no"
    d = ws(t, "sj", {"index.html": (INITIAL / "index.html").read_text(encoding="utf-8"),
                     "appointments.html": page("<p>x</p>", viewport=off)})
    out = e2_score.score_job(renderer, str(d), "P06", str(INITIAL), True)
    ids = {c["id"]: c["passed"] for c in out["checks"]}
    assert ids == {"index.html|axe:meta-viewport": True, "appointments.html|axe:meta-viewport": False}
    assert out["a"] == 0.5 and not out["accessible"] and out["functional"]
    assert out["intent_met"] is False
    clean = e2_score.score_job(renderer, str(INITIAL), "P06", None, False)
    assert clean["pages"]["appointments.html"] == {"missing": True}
    assert clean["a"] == 1.0 and clean["accessible"] and clean["intent_met"] is None
    assert e2_score.workspace_hash(INITIAL) == e2_score.workspace_hash(ws(t, "copy", {}))
    assert e2_score.workspace_hash(INITIAL) != e2_score.workspace_hash(d)
