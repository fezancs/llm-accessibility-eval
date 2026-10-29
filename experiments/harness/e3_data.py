"""E3 data (SPEC 7.1): D_BLV and D_CR items, their context fields, and their images.

    python -m harness.e3_data                     # download sources + images, write items.jsonl
    python -m harness.e3_data --no-download       # rebuild items.jsonl from data/e3/raw
    python -m harness.e3_data --check-originals   # also HEAD-check the original Wikimedia URLs

Sources (pinned commits; raw files land in data/e3/raw/, images in data/e3/images/):
- D_BLV (Kreiss et al., EMNLP 2022): github.com/elisakreiss/contextual-description-evaluation,
  behavioral_data/blv_data_criticaltrials.csv (68 descriptions x 4 BLV raters, the trials the
  paper analyses) and all_descriptions.csv (image file per description).
- D_CR (ContextRef, Kreiss et al., ICLR 2024): github.com/elisakreiss/contextref plus the
  repository's linked Google Drive data folder: model_input_data/df_metric_gold.csv (204
  descriptions, one per image, crowd ratings after the authors' exclusions) and
  data_chunk0-{1..5}_anon.csv (original Wikimedia URL per image).

Images. Each item's image is the stimulus file the human participants were shown, downloaded
from the dataset's own repository (ContextRef's interface loads stimuli/data/images/<img_id>;
D_BLV's descriptions were written for the files in behavioral_experiments/.../stimuli/). The
original web URL (Wikimedia) is recorded in `image_url` for provenance. Images are not
modified, converted or downscaled, and are not redistributed (data/**/images/ is git-ignored).
An image that cannot be retrieved marks its item `retrieved: false`; nothing is substituted.

Context fields (prompt placeholders):
- D_BLV: article title = the Wikipedia article (`context`); section title "none" and caption
  "none" (the BLV raters saw only the article title and its opening paragraph);
  excerpt = that paragraph (`article_text`).
- D_CR: article title, section title ("none" for the lead section), caption, and excerpt =
  the section text around the image, or the page's opening text when the image is in the
  lead section or the section text repeats it (the same rule the ContextRef interface used).
- Excerpts are cut to 250 words; a cut is marked with " ...".

Human ratings (1-5 scales): D_BLV `overall_mean` = mean q_overall of the 4 BLV raters.
D_CR `overall_mean` = mean q_overall.postimg (ContextRef's headline rating, given after the
rater saw the image); the pre-image ratings and the other scales are kept alongside.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import pathlib
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config

E3 = config.DATA / "e3"
RAW = E3 / "raw"
IMAGES = E3 / "images"
ITEMS = E3 / "items.jsonl"
SOURCES = E3 / "sources.json"
README = E3 / "README.md"

USER_AGENT = ("se4all-2027-research-harness/1.0 (academic accessibility study; "
              "https://github.com/elisakreiss datasets; contact via repository owner)")
THROTTLE_S = 0.5
MAX_IMAGE_BYTES_TOTAL = 500 * 1024 * 1024
EXCERPT_WORDS = 250

BLV_REPO = "elisakreiss/contextual-description-evaluation"
BLV_SHA = "46407c28cb41efc7bb28936b86351726547ee154"
BLV_FILES = {
    "blv_data_criticaltrials.csv": "behavioral_data/blv_data_criticaltrials.csv",
    "all_descriptions.csv": "behavioral_data/all_descriptions.csv",
    "LICENSE": "LICENSE",
}
BLV_STIMULI_DIR = "behavioral_experiments/evaluation_sighted/stimuli"

CR_REPO = "elisakreiss/contextref"
CR_SHA = "306cad8d2c81a6ef73d688c50a5a1db740c5b871"
CR_STIMULI_DIR = "dataset/human_subject_study/stimuli/data/images"
CR_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1l_7jfsXZX99ZEGO0fIuaSbxorDmB4Gpc"
CR_DRIVE_FILES = {
    "df_metric_gold.csv": "1tHbgk5kVj-ILg9hPfRM8dKJ7Wd1wM_cJ",
    "data_chunk0-1_anon.csv": "1WSwTRA_twj-osfmLZnfDN23VNNpKU-1-",
    "data_chunk0-2_anon.csv": "1xi28HnI6FD9LKJ0l2n9lRVxiy0pZdZUu",
    "data_chunk0-3_anon.csv": "1gsCZtUbvtMJB6wJ2yPmhdVE6-L408qrA",
    "data_chunk0-4_anon.csv": "1r9M3DgsE2o5plTr2NBpT9BfJZjCnxa-M",
    "data_chunk0-5_anon.csv": "1Xuz50CHnIbPb3M55Tes75EvtrxAYMV6Q",
}

BLV_SCALES = {"overall": "q_overall", "imaginability": "q_reconstructivity", "relevance": "q_relevance",
              "irrelevance": "q_irrelevance", "imgfit": "q_imgfit"}
CR_SCALES = {"overall_postimg": "q_overall.postimg", "overall_preimg": "q_overall.preimg",
             "imaginability_preimg": "q_reconstructivity.preimg",
             "relevance_postimg": "q_relevance.postimg", "relevance_preimg": "q_relevance.preimg",
             "irrelevance_postimg": "q_irrelevance.postimg", "irrelevance_preimg": "q_irrelevance.preimg",
             "imgfit_postimg": "q_imgfit.postimg", "imgfit_preimg": "q_imgfit.preimg",
             "addedinfo_postimg": "q_addedinfo.postimg", "addedinfo_preimg": "q_addedinfo.preimg"}

_NA = {"", "na", "n/a", "null", "none", "nan"}


# ------------------------------------------------------------------ helpers

def clean(value) -> str | None:
    """Missing-value markers used by the datasets (R's NA, empty, null) become None."""
    if value is None:
        return None
    s = str(value).strip()
    return None if s.lower() in _NA else s


def truncate_words(text: str | None, n: int = EXCERPT_WORDS) -> str:
    """First n whitespace-separated words, whitespace collapsed; ' ...' marks a cut."""
    words = (text or "").split()
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]) + " ..."


def media_type(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_size(data: bytes) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:                       # noqa: BLE001 - size is diagnostic only
        return None, None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def raw_url(repo: str, sha: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{sha}/" + urllib.parse.quote(path)


_last_request = [0.0]


def fetch(url: str, timeout: float = 60.0, method: str = "GET") -> tuple[int, bytes, dict]:
    """HTTP request with a descriptive User-Agent, throttling and 3 retries on 5xx/network errors."""
    err = None
    for attempt in range(4):
        wait = THROTTLE_S - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, (resp.read() if method == "GET" else b""), \
                    {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                return e.code, b"", {k.lower(): v for k, v in (e.headers or {}).items()}
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {err}")


# ------------------------------------------------------------------ sources

def download_sources() -> dict:
    """Fetch the raw rating files into data/e3/raw/ and record their hashes."""
    manifest = {"retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    (RAW / "blv").mkdir(parents=True, exist_ok=True)
    (RAW / "cr" / "drive").mkdir(parents=True, exist_ok=True)
    jobs = [(RAW / "blv" / name, raw_url(BLV_REPO, BLV_SHA, path)) for name, path in BLV_FILES.items()]
    jobs += [(RAW / "cr" / "drive" / name,
              f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t")
             for name, fid in CR_DRIVE_FILES.items()]
    for dest, url in jobs:
        status, data, _ = fetch(url, timeout=180)
        if status != 200 or not data:
            raise RuntimeError(f"could not download {url} (HTTP {status})")
        if data[:15].lower().startswith(b"<!doctype html") or data[:5].lower() == b"<html":
            raise RuntimeError(f"{url} returned an HTML page instead of the data file")
        dest.write_bytes(data)
        manifest["files"][str(dest.relative_to(E3))] = {"url": url, "bytes": len(data),
                                                         "sha256": hashlib.sha256(data).hexdigest()}
    manifest["blv"] = {"repo": BLV_REPO, "commit": BLV_SHA}
    manifest["cr"] = {"repo": CR_REPO, "commit": CR_SHA, "drive_folder": CR_DRIVE_FOLDER}
    return manifest


# ------------------------------------------------------------------ parsing

def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def parse_blv(raw_dir: pathlib.Path) -> list[dict]:
    """One item per (image, context, description) of the BLV critical trials."""
    ratings = _read_csv(raw_dir / "blv_data_criticaltrials.csv")
    files = {}
    for r in _read_csv(raw_dir / "all_descriptions.csv"):
        files.setdefault(r["img_id"], set()).add(r["img_file"])
    groups: dict[tuple, list[dict]] = collections.OrderedDict()
    for r in ratings:
        groups.setdefault((r["img_id"], r["context"], r["description"]), []).append(r)
    per_image = collections.defaultdict(list)
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        per_image[key[0]].append(key)
    items = []
    for img_id in sorted(per_image):
        if len(files.get(img_id, ())) != 1:
            raise ValueError(f"D_BLV image {img_id}: expected one image file, got {files.get(img_id)}")
        img_file = next(iter(files[img_id]))
        for k, key in enumerate(per_image[img_id]):
            rows = groups[key]
            human = {f"{name}_mean": _mean([float(r[col]) for r in rows if clean(r.get(col)) is not None])
                     for name, col in BLV_SCALES.items()}
            human.update({"n_raters": len(rows), "raters": "blv", "scale": "1-5"})
            items.append({
                "item_id": f"blv-{img_id}-{k:02d}", "dataset": "blv", "image_id": f"blv-{img_id}",
                "image_file": img_file, "description": key[2].strip(),
                "context": {"article_title": key[1].strip(), "section_title": "none",
                            "excerpt": truncate_words(rows[0]["article_text"]), "caption": "none"},
                "human": human, "source_ids": {"img_id": img_id, "context": key[1]},
            })
    return items


def cr_excerpt(page_context: str | None, section_context: str | None) -> str:
    """Section text around the image; the page's opening text for lead-section images or when
    the section text repeats it (ContextRef's interface hid the section text in that case)."""
    sec, page = clean(section_context), clean(page_context)
    if sec and not (page and page[:200] == sec[:200]):
        return truncate_words(sec)
    return truncate_words(page or "")


def parse_cr(raw_dir: pathlib.Path) -> list[dict]:
    """One item per ContextRef description (one per image) with crowd-rating means."""
    gold = _read_csv(raw_dir / "df_metric_gold.csv")
    urls = collections.defaultdict(set)
    for chunk in sorted(raw_dir.glob("data_chunk0-*_anon.csv")):
        for r in _read_csv(chunk):
            if clean(r.get("img_file")):
                urls[r["img_id"]].add(r["img_file"].strip().replace("http://", "https://", 1))
    groups: dict[tuple, list[dict]] = collections.OrderedDict()
    for r in gold:
        groups.setdefault((r["img_id"], r["description"]), []).append(r)
    items = []
    for (img_id, description), rows in sorted(groups.items()):
        first = rows[0]
        human = {f"{name}_mean": _mean([float(r[col]) for r in rows if clean(r.get(col)) is not None])
                 for name, col in CR_SCALES.items()}
        human["overall_mean"] = human["overall_postimg_mean"]
        human.update({"n_raters": len(rows), "raters": "crowd", "scale": "1-5",
                      "overall_definition": "q_overall.postimg"})
        stem = img_id.rsplit(".", 1)[0]
        u = sorted(urls.get(img_id, ()))
        items.append({
            "item_id": f"cr-{stem}", "dataset": "cr", "image_id": f"cr-{stem}", "image_file": img_id,
            "image_url": u[0] if u else None, "description": description.strip(),
            "context": {"article_title": (clean(first["context"]) or "none"),
                        "section_title": clean(first["section_title"]) or "none",
                        "excerpt": cr_excerpt(first["page_context"], first["section_context"]),
                        "caption": clean(first["caption"]) or "none"},
            "human": human, "split": clean(first.get("split")),
            "source_ids": {"img_id": img_id, "description_caption_copy": first.get("descr_caption_copy")},
        })
    return items


# ------------------------------------------------------------------ images

def commons_url(file_name: str) -> str | None:
    """Original Wikimedia Commons URL for a D_BLV stimulus file name (provenance only)."""
    name = re.sub(r"^\d+px-", "", file_name)
    # en.wikipedia.org answers for Commons-hosted ("shared") files; commons.wikimedia.org resets
    # connections from some networks.
    api = ("https://en.wikipedia.org/w/api.php?action=query&format=json&prop=imageinfo"
           "&iiprop=url&titles=" + urllib.parse.quote("File:" + name))
    try:
        status, data, _ = fetch(api)
    except RuntimeError:
        return None
    if status != 200:
        return None
    pages = json.loads(data.decode("utf-8")).get("query", {}).get("pages", {})
    for p in pages.values():
        info = p.get("imageinfo")
        if info and info[0].get("url"):
            return info[0]["url"].split("?", 1)[0]
    return None


def fetch_images(items: list[dict], resolve_blv_originals: bool = True) -> dict:
    """Download each distinct image once; annotate items in place. Returns a summary."""
    by_image = collections.OrderedDict()
    for it in items:
        by_image.setdefault(it["image_id"], []).append(it)
    total = 0
    failures = {}
    for image_id, its in by_image.items():
        it0 = its[0]
        if it0["dataset"] == "blv":
            url = raw_url(BLV_REPO, BLV_SHA, f"{BLV_STIMULI_DIR}/{it0['image_file']}")
        else:
            url = raw_url(CR_REPO, CR_SHA, f"{CR_STIMULI_DIR}/{it0['image_file']}")
        ext = pathlib.Path(it0["image_file"]).suffix.lower() or ".img"
        dest = IMAGES / it0["dataset"] / f"{image_id}{ext}"
        reason, data = None, None
        if dest.exists() and dest.stat().st_size > 0:
            data = dest.read_bytes()
        else:
            try:
                status, data, _ = fetch(url)
                if status != 200 or not data:
                    reason, data = f"HTTP {status}", None
            except RuntimeError as e:
                reason = f"network error: {e}"
        mt = media_type(data) if data else None
        if data and mt is None:
            reason, data = "not a JPEG/PNG/GIF/WebP file", None
        if data:
            total += len(data)
            if total > MAX_IMAGE_BYTES_TOTAL:
                raise RuntimeError("images would exceed 500 MB; stopping (see task instructions)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(data)
            w, h = image_size(data)
        orig = it0.get("image_url")
        if it0["dataset"] == "blv" and resolve_blv_originals and orig is None:
            orig = commons_url(it0["image_file"])
        for it in its:
            it["image_url"] = orig
            it["image_source_url"] = url
            it["retrieved"] = bool(data)
            it["image_path"] = str(dest.relative_to(config.ROOT)) if data else None
            it["media_type"] = mt if data else None
            it["image_bytes"] = len(data) if data else None
            it["image_width"], it["image_height"] = (w, h) if data else (None, None)
            it["failure_reason"] = reason
        if reason:
            failures[image_id] = reason
    return {"bytes": total, "images": len(by_image), "failed": failures}


def check_originals(items: list[dict]) -> dict:
    """HEAD-check the original web URLs (reported only; images come from the dataset repos)."""
    seen, live, dead, sizes = set(), 0, {}, []
    for it in items:
        u = it.get("image_url")
        if not u or u in seen:
            continue
        seen.add(u)
        try:
            status, _, headers = fetch(u, method="HEAD")
        except RuntimeError as e:
            dead[u] = str(e)
            continue
        if status == 200:
            live += 1
            if headers.get("content-length"):
                sizes.append(int(headers["content-length"]))
        else:
            dead[u] = f"HTTP {status}"
    return {"checked": len(seen), "live": live, "dead": dead, "total_bytes_live": sum(sizes),
            "max_bytes": max(sizes) if sizes else None, "over_5MB": sum(1 for s in sizes if s > 5 * 2 ** 20)}


# ------------------------------------------------------------------ outputs

ITEM_KEYS = ["item_id", "dataset", "image_id", "image_path", "image_url", "retrieved", "description",
             "context", "human", "media_type", "image_bytes", "image_width", "image_height",
             "image_source_url", "image_file", "failure_reason", "split", "source_ids"]


def write_items(items: list[dict], path: pathlib.Path = ITEMS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({k: it.get(k) for k in ITEM_KEYS}, ensure_ascii=False) + "\n")


def load_items(path: pathlib.Path = ITEMS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def blv_pair_groups(items: list[dict]) -> dict:
    """Descriptions per (image, article) in D_BLV, the unit within which BLV ratings compare."""
    g = collections.defaultdict(list)
    for it in items:
        if it["dataset"] == "blv":
            g[(it["image_id"], it["context"]["article_title"])].append(it)
    return g


def summary(items: list[dict], img: dict, originals: dict | None) -> dict:
    out = {}
    for ds in ("blv", "cr"):
        its = [i for i in items if i["dataset"] == ds]
        imgs = {i["image_id"] for i in its}
        out[ds] = {"items": len(its), "images": len(imgs),
                   "items_retrieved": sum(1 for i in its if i["retrieved"]),
                   "images_retrieved": len({i["image_id"] for i in its if i["retrieved"]}),
                   "raters_per_item_mean": round(statistics.fmean(i["human"]["n_raters"] for i in its), 3)
                   if its else None}
    per_img = collections.Counter(i["image_id"] for i in items if i["dataset"] == "blv")
    out["blv"]["descriptions_per_image"] = dict(sorted(collections.Counter(per_img.values()).items()))
    groups = blv_pair_groups(items)
    out["blv"]["descriptions_per_image_and_article"] = dict(sorted(collections.Counter(
        len(v) for v in groups.values()).items()))
    n_pairs = n_pairs_05 = 0
    for its in groups.values():
        for a in range(len(its)):
            for b in range(a + 1, len(its)):
                n_pairs += 1
                n_pairs_05 += abs(its[a]["human"]["overall_mean"] - its[b]["human"]["overall_mean"]) >= 0.5
    out["blv"]["within_image_article_pairs"] = n_pairs
    out["blv"]["within_image_article_pairs_diff_ge_0_5"] = n_pairs_05
    out["image_bytes_total"] = img["bytes"]
    out["image_failures"] = img["failed"]
    out["originals_check"] = originals
    return out


def write_readme(s: dict, manifest: dict) -> None:
    b, c = s["blv"], s["cr"]
    lines = [
        "# E3 data (generated by harness/e3_data.py; do not edit by hand)",
        "",
        f"Retrieved: {manifest.get('retrieved_at', 'n/a')}. Sources are pinned in `sources.json`.",
        "",
        "| Dataset | Items | Images | Items with image | Images retrieved | Raters per item (mean) |",
        "|---|---|---|---|---|---|",
        f"| D_BLV (Kreiss et al. 2022, BLV raters) | {b['items']} | {b['images']} | {b['items_retrieved']} | "
        f"{b['images_retrieved']} | {b['raters_per_item_mean']} |",
        f"| D_CR (ContextRef, crowd raters) | {c['items']} | {c['images']} | {c['items_retrieved']} | "
        f"{c['images_retrieved']} | {c['raters_per_item_mean']} |",
        "",
        f"Image failures: {len(s['image_failures'])}.",
    ]
    for k, v in s["image_failures"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        f"Total image bytes: {s['image_bytes_total']:,}.",
        "",
        "## D_BLV descriptions per image (decides E3a pairwise)",
        "",
        f"- Descriptions per image: {b['descriptions_per_image']} (count of images with that many).",
        f"- Descriptions per (image, article): {b['descriptions_per_image_and_article']}.",
        f"- Within-(image, article) pairs: {b['within_image_article_pairs']}; with a BLV mean-overall "
        f"difference >= 0.5: {b['within_image_article_pairs_diff_ge_0_5']}.",
        "- Each image's descriptions were written for, and rated in, different articles; only pairs",
        "  rated in the same article have a comparable human preference, so E3a uses those pairs.",
        "",
        "## Notes",
        "",
        "- Images are the exact stimulus files shown to the human participants, from the datasets'",
        "  repositories; `image_url` keeps the original Wikimedia URL. Images are not modified and",
        "  not redistributed.",
        "- D_CR `overall_mean` is the post-image overall rating (ContextRef's headline measure);",
        "  `overall_preimg_mean` is kept alongside.",
        "- D_BLV has no section title or caption (raters saw the article title and opening paragraph):",
        "  both are \"none\" in prompts.",
    ]
    if s.get("originals_check"):
        o = s["originals_check"]
        lines += ["", "## Original Wikimedia URLs (HEAD check, informational)", "",
                  f"Checked {o['checked']}; live {o['live']}; dead {len(o['dead'])}; total size of live "
                  f"originals {o['total_bytes_live']:,} bytes; largest {o['max_bytes']}; over 5 MB: {o['over_5MB']}."]
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-download", action="store_true", help="reuse files already in data/e3/raw")
    ap.add_argument("--check-originals", action="store_true", help="HEAD-check original Wikimedia URLs")
    args = ap.parse_args(argv)
    if args.no_download and SOURCES.exists():
        manifest = json.loads(SOURCES.read_text(encoding="utf-8"))
    else:
        manifest = download_sources()
    items = parse_blv(RAW / "blv") + parse_cr(RAW / "cr" / "drive")
    img = fetch_images(items)
    originals = check_originals([i for i in items if i["dataset"] == "cr"]) if args.check_originals else None
    write_items(items)
    s = summary(items, img, originals)
    manifest["summary"] = s
    SOURCES.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(s, manifest)
    print(json.dumps({k: v for k, v in s.items() if k != "originals_check"}, indent=2))
    if originals:
        print(json.dumps({k: v for k, v in originals.items() if k != "dead"}, indent=2), f"dead={len(originals['dead'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
