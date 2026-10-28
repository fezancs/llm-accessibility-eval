"""Adapt the W3C WAI Before-and-After Demonstration "after" pages into base pages B01-B04
(tasks/e1_repair_corpus.yaml). Run once; the output is committed with the corpus.

    python -m harness.e1_w3c_adapt

Changes, all logged to data/corpus/w3c_adaptation.md:
- strip the demo toolbar (#meta-header, #meta-footer), analytics, and external scripts
  (localdate.js rewrote the visible date daily; the static date from the source is kept);
- inline css/main.css without its @import of the toolbar stylesheet; image paths -> images/;
- add stable, descriptive ids to mutation targets;
- B01-B03 get one icon-only link (M05 target) because otherwise they have targets for only 7
  operators (the task file requires 8; M10 is undetectable there because focus is shown by a
  background-colour change, and M02/M04/M07 have no target).
Source: https://www.w3.org/WAI/demos/bad/ (W3C Document License; credited in the paper).
"""
from __future__ import annotations

import re
import shutil

from bs4 import BeautifulSoup, Comment

from . import config

SRC = config.DATA / "corpus" / "w3c_src"
OUT = config.DATA / "corpus" / "base"
PAGES = {"B01": "home", "B02": "news", "B03": "tickets", "B04": "survey"}

RSS_LINK = ('<a id="info-news-feed" class="icon-link" href="news.html" aria-label="News feed">'
            '<svg aria-hidden="true" focusable="false" viewBox="0 0 16 16" width="16" height="16">'
            '<circle cx="3" cy="13" r="2" fill="currentColor"/>'
            '<path d="M1 7a8 8 0 0 1 8 8h-2.5A5.5 5.5 0 0 0 1 9.5zM1 1a14 14 0 0 1 14 14h-2.5A11.5 11.5 0 0 0 1 3.5z" fill="currentColor"/>'
            '</svg></a>')

SKIP_CSS = """
/* from css/meta.css: skip styling */
html>body ul.skip { position: absolute }
ul.skip { margin: 0; padding: 0 }
.skip li { list-style-type: none; position: absolute; left: 0 }
p.skip { position: absolute; top: -1000em; left: -1000em; height: 1px; width: 1px; overflow: hidden; }
h2.skip { position: absolute; top: -1000em; left: -1000em; height: 1px; width: 1px; overflow: hidden; }
.skip a { position: absolute; left: -1000em; top: 0.25em; border: solid black 2px; background: #fff; padding: 5px; width: 20em; overflow: hidden; }
.skip a:focus, .skip a:active { left: 0em; color: #fff; background: #00c; overflow: auto; }
"""

# id -> (CSS selector in the adapted page, operator it serves). Selectors are evaluated once.
TARGET_IDS = {
    "home": [("img[src$='teaser_right1.jpg']", "concert-photo"), ("img[src$='teaser_right2.jpg']", "survey-photo"),
             ("img[src$='weather.png']", "weather-icon"), ("img[src$='toplogo.png']", "site-logo"),
             ("#contentmain .news:nth-of-type(1) p:not(.image)", "heatwave-teaser"),
             ("#contentmain .news:nth-of-type(2) p:not(.image)", "violin-teaser"),
             ("#contentmain .news:nth-of-type(3) p:not(.image)", "brains-teaser"),
             ("#contentright h2:nth-of-type(1)", "concert-heading"), ("#contentright h2:nth-of-type(2)", "survey-heading"),
             ("#contentmain h2.header", "elsewhere-heading"), ("#footer a", "footer-copyright"),
             ("#header input[type=submit]", "quickmenu-go")],
    "news": [("img[src$='blanca.jpg']", "wahoonie-photo"), ("img[src$='chart1.png']", "donations-chart"),
             ("img[src$='weather.png']", "weather-icon"), ("img[src$='smzappa2.jpg']", "website-screenshot"),
             ("#footer a", "footer-copyright"), ("#header input[type=submit]", "quickmenu-go")],
    "tickets": [("img[src$='weather.png']", "weather-icon"), ("img[src$='toplogo.png']", "site-logo"),
                ("p.sfbuy", "music-line-text"), (".boxcontent h2", "terms-heading"),
                ("#footer a", "footer-copyright"), ("#header input[type=submit]", "quickmenu-go")],
    "survey": [("img[src$='weather.png']", "weather-icon"), ("img[src$='toplogo.png']", "site-logo"),
               ("#content h2.header", "survey-heading"), ("#footer a", "footer-copyright"),
               ("#header input[type=submit]", "quickmenu-go")],
}


def adapt(name: str) -> tuple[str, list[str]]:
    log = []
    soup = BeautifulSoup((SRC / f"{name}.html").read_text(encoding="utf-8"), "html.parser")
    for sel in ("#meta-header", "#meta-footer"):
        for el in soup.select(sel):
            el.decompose()
    log.append("removed demo toolbar (#meta-header) and footer (#meta-footer)")
    for s in soup.find_all(["script", "noscript"]):
        s.decompose()
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    log.append("removed scripts (analytics, onload.js, localdate.js) and comments")
    link = soup.find("link", rel="stylesheet")
    css = (SRC / "assets" / "main.css").read_text(encoding="utf-8")
    css = css.replace("@import url(meta.css);", "")
    # The in-page skip links and hidden headings are styled in the toolbar stylesheet;
    # carry over only those rules (verbatim from css/meta.css, list-style-image dropped).
    css += SKIP_CSS
    css = re.sub(r"url\(\.\./after/img/", "url(images/", css)
    style = soup.new_tag("style")
    style.string = css
    link.replace_with(style)
    log.append("inlined css/main.css (without @import meta.css) plus the .skip rules from css/meta.css; "
               "background images -> images/")
    marker = soup.select_one("#startcontent")
    if marker is not None:
        marker.decompose()
        log.append("removed the demo marker paragraph 'Demo starts here' (#startcontent)")
    for img in soup.find_all("img"):
        img["src"] = img["src"].replace("./img/", "images/")
    title = soup.find("title")
    title.string = title.string.replace(" [Accessible Home Page]", "").replace(" [Accessible News Page]", "") \
        .replace(" [Accessible Tickets Page]", "").replace(" [Accessible Survey Page]", "")
    log.append("removed the '[Accessible ... Page]' demo suffix from <title>")
    for sel, new_id in TARGET_IDS[name]:
        el = soup.select_one(sel)
        if el is None:
            raise ValueError(f"{name}: selector {sel!r} matched nothing")
        if el.get("id") and el["id"] != new_id:
            log.append(f"kept existing id #{el['id']} (target for {new_id})")
            continue
        el["id"] = new_id
    log.append("added ids: " + ", ".join(i for _, i in TARGET_IDS[name]))
    if name in ("home", "news", "tickets"):
        info = soup.select_one("#info p.left")
        info.append(" ")
        info.append(BeautifulSoup(RSS_LINK, "html.parser"))
        style.string += ("\n#info .icon-link { display: inline-block; vertical-align: middle; margin-left: .5em;"
                         " padding: 4px; color: #ba2710; }\n")
        log.append("ADDED CONTENT: icon-only link #info-news-feed (aria-label 'News feed') in #info, "
                   "the only operator-M05 target")
    return str(soup), log


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img_out = OUT / "images"
    img_out.mkdir(exist_ok=True)
    for f in (SRC / "assets" / "img").iterdir():
        shutil.copy2(f, img_out / f.name)
    notes = ["# W3C BAD adaptation log", "",
             "Source pages: https://www.w3.org/WAI/demos/bad/after/{home,news,tickets,survey}.html "
             "(fetched 2026-10-28). W3C Document License.", ""]
    for pid, name in PAGES.items():
        html, log = adapt(name)
        (OUT / f"{pid}.html").write_text(html, encoding="utf-8")
        notes.append(f"## {pid} ({name}.html)")
        notes += [f"- {line}" for line in log]
        notes.append("")
    (config.DATA / "corpus" / "w3c_adaptation.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
