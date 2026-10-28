"""E3 data parsing on format-identical synthetic fixtures (no network)."""
import json

import pytest

from harness import config, e3_data

FIX = config.DATA / "e3" / "test_fixtures"


@pytest.fixture(scope="module")
def blv():
    return e3_data.parse_blv(FIX / "blv")


@pytest.fixture(scope="module")
def cr():
    return e3_data.parse_cr(FIX / "cr")


def test_truncate_words():
    assert e3_data.truncate_words("a  b\nc") == "a b c"
    words = " ".join(f"w{i}" for i in range(260))
    out = e3_data.truncate_words(words)
    assert out.endswith(" ...") and len(out[:-4].split()) == 250
    exact = " ".join(f"w{i}" for i in range(250))
    assert e3_data.truncate_words(exact) == exact
    assert e3_data.truncate_words(None) == ""


def test_clean_missing_markers():
    for v in ["", " NA ", "null", "None", "nan", None]:
        assert e3_data.clean(v) is None
    assert e3_data.clean(" Roof ") == "Roof"


def test_parse_blv_items(blv):
    assert [i["item_id"] for i in blv] == ["blv-church2-00", "blv-table1-00", "blv-table1-01", "blv-table1-02"]
    by_desc = {i["description"]: i for i in blv}
    good = by_desc["A table set for dinner with candles."]
    assert good["dataset"] == "blv" and good["image_id"] == "blv-table1"
    assert good["human"]["overall_mean"] == pytest.approx(4.5)
    assert good["human"]["n_raters"] == 4 and good["human"]["raters"] == "blv"
    assert good["human"]["imaginability_mean"] == pytest.approx(3.0)
    assert good["context"] == {"article_title": "Dinner", "section_title": "none",
                               "excerpt": "Dinner is the main meal of the day.", "caption": "none"}
    assert good["image_file"] == "1600px-Table_photo.jpg"
    long_ctx = by_desc["A long wooden dining table. Six chairs surround it."]["context"]["excerpt"]
    assert long_ctx.endswith(" ...") and len(long_ctx.split()) == 251


def test_parse_cr_items(cr):
    assert [i["item_id"] for i in cr] == ["cr-11", "cr-22", "cr-33"]
    a, b, c = cr
    assert a["human"]["overall_mean"] == pytest.approx(4.0)            # post-image ratings 5, 4, 3
    assert a["human"]["overall_preimg_mean"] == pytest.approx(2.0)
    assert a["human"]["n_raters"] == 3 and a["human"]["overall_definition"] == "q_overall.postimg"
    assert a["image_url"] == "https://upload.wikimedia.org/wikipedia/commons/a/ab/Bridge.jpg"
    # lead-section image: no section title, excerpt = page text
    assert a["context"]["section_title"] == "none"
    assert a["context"]["excerpt"].startswith("Riverside is a town")
    assert a["context"]["caption"] == "The old bridge"
    # section text repeating the page text -> page text; caption NA -> "none"
    assert b["context"]["section_title"] == "Geography"
    assert not b["context"]["excerpt"].endswith("More.")
    assert b["context"]["caption"] == "none"
    # distinct section text -> section text
    assert c["context"]["excerpt"] == "The festival is held every June."
    assert c["image_url"] is None                                       # no chunk row for 33.jpg


def test_cr_excerpt_rule():
    page = "p " * 120
    assert e3_data.cr_excerpt(page, "NA") == e3_data.truncate_words(page)
    assert e3_data.cr_excerpt(page, page + "extra") == e3_data.truncate_words(page)
    assert e3_data.cr_excerpt(page, "Section text.") == "Section text."
    assert e3_data.cr_excerpt(None, None) == ""


def test_media_type_sniffing():
    assert e3_data.media_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert e3_data.media_type(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert e3_data.media_type(b"GIF89a....") == "image/gif"
    assert e3_data.media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert e3_data.media_type(b"<svg xmlns") is None


def test_write_items_follows_schema(tmp_path, blv, cr):
    items = [dict(i, retrieved=False, image_path=None) for i in blv + cr]
    path = tmp_path / "items.jsonl"
    e3_data.write_items(items, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    required = {"item_id", "dataset", "image_id", "image_path", "image_url", "retrieved", "description",
                "context", "human"}
    assert all(required <= set(r) for r in rows)
    assert all({"article_title", "section_title", "excerpt", "caption"} == set(r["context"]) for r in rows)
    assert all("overall_mean" in r["human"] and "n_raters" in r["human"] for r in rows)
    assert e3_data.load_items(path) == rows


def test_blv_pair_groups(blv):
    groups = e3_data.blv_pair_groups(blv)
    assert sorted(len(v) for v in groups.values()) == [1, 1, 2]       # Dinner has two descriptions
