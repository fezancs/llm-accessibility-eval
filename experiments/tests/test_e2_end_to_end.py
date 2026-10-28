"""End-to-end E2 through the offline fake provider (synthetic replies only): padding
transcript generation, two fake models x 1 run x the first 6 positions with fresh controls
and a padded checkpoint at position 6, scoring in worker processes, output records checked
against harness/SCHEMAS.md, and a second invocation that makes zero new adapter calls.
runs/, the response cache and the padding transcript live in pytest tmp directories."""
import json

from harness import e2_padding, e2_run, e2_steps, runmeta
from tests import fakes

MODELS = ["fake-good", "fake-lazy"]
NAV_PAGES = [("index.html", "Home"), ("services.html", "Services"), ("appointments.html", "Appointments"),
             ("contact.html", "Contact"), ("news.html", "News")]
STEP_BY_MESSAGE = {s["message"]: s["id"] for s in e2_steps.steps()}


def site_page(name, title, main):
    links = "".join(f'<li><a href="{h}"{" aria-current=\"page\"" if h == name else ""}>{t}</a></li>'
                    for h, t in NAV_PAGES)
    return ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{title} - CityCare</title>\n<link rel="stylesheet" href="styles.css">\n</head>\n<body>\n'
            '<header class="site-header"><p class="logo">CityCare</p><nav aria-label="Main"><ul class="nav-list">'
            f'{links}</ul></nav></header>\n<main id="main">\n<h1>{title}</h1>\n{main}\n</main>\n'
            '<footer class="site-footer"><address>CityCare, 1 Market Street, Old Town<br>Phone: 020 7946 0000'
            '</address></footer>\n</body>\n</html>\n')


INDEX_MAIN = ("<p>CityCare is a network of public health clinics.</p>\n<p>Visit any clinic for care.</p>")


def field(fid, label, good, tag="input", extra=""):
    lab = f'<label for="{fid}">{label}</label>' if good else ""
    el = f'<textarea id="{fid}" name="{fid}"></textarea>' if tag == "textarea" else f'<input id="{fid}" name="{fid}"{extra}>'
    return f"<p>{lab}{el}</p>"


def step_files(step_id, good):
    """The files a fake model writes for a step (good = accessible version)."""
    alt = (lambda a: f' alt="{a}"') if good else (lambda a: "")
    if step_id == "F01":
        cards = "".join(f'<li><h2>{t}</h2><p>We offer {t.lower()}. Ask at any clinic.</p><a href="#">Learn more<span class="visually-hidden"> about {t}</span></a></li>'
                        for t in ("Vaccinations", "General check-ups", "Pharmacy", "Mental health support"))
        return {"services.html": site_page("services.html", "Services", f'<ul id="services-list">{cards}</ul>'),
                "index.html": site_page("index.html", "Welcome to CityCare", INDEX_MAIN)}
    if step_id == "F02":
        fields = (field("name", "Full name", good) + field("dob", "Date of birth", good, extra=' type="date"')
                  + field("phone", "Phone number", good, extra=' type="tel"')
                  + field("email", "Email address", good, extra=' type="email"')
                  + field("clinic", "Preferred clinic", good) + field("type", "Consultation type", good)
                  + field("notes", "Notes", good, tag="textarea")
                  + field("privacy", "I agree to the privacy notice", good, extra=' type="checkbox"'))
        return {"appointments.html": site_page("appointments.html", "Appointments",
                                               f'<form id="appointment-form">{fields}<button type="submit">Request appointment</button></form>')}
    if step_id == "F03":
        rows = "".join(f"<tr><th scope=\"row\">{d}</th><td>{h}</td></tr>" for d, h in
                       (("Monday to Friday", "8:00 to 18:00"), ("Saturday", "9:00 to 13:00"), ("Sunday", "Closed")))
        form = (field("cname", "Name", good) + field("cemail", "Email", good, extra=' type="email"')
                + field("cmsg", "Message", good, tag="textarea"))
        main = (f'<table id="hours-table"><caption>Opening hours</caption><tr><th scope="col">Day</th><th scope="col">Hours</th></tr>{rows}</table>'
                f'<form id="contact-form">{form}<button type="submit">Send message</button></form>')
        return {"contact.html": site_page("contact.html", "Contact", main)}
    if step_id == "F04":
        arts = "".join(f'<article><img src="images/news-{i}.jpg"{alt(h)}><h2>{h}</h2><p>1 October 2026</p><a href="#">Read more<span class="visually-hidden">: {h}</span></a></article>'
                       for i, h in enumerate(("Flu vaccinations now available", "Riverside clinic extends evening hours",
                                              "New physiotherapy service in Old Town"), 1))
        return {"news.html": site_page("news.html", "News", f'<div id="news-list">{arts}</div>')}
    if step_id == "R07":
        imgs = "".join(f'<img src="images/facility-{n}.jpg"{alt(a)}>' for n, a in
                       (("reception", "Our reception desk"), ("play", "The children's play area"),
                        ("entrance", "The step-free entrance with a ramp"), ("pharmacy", "The pharmacy counter")))
        return {"index.html": site_page("index.html", "Welcome to CityCare",
                                        INDEX_MAIN + f'<section id="gallery"><h2>Our facilities</h2>{imgs}</section>')}
    return {}                                                       # P05 and others: declined


def tool_msgs_since_user(messages):
    n = 0
    for m in reversed(messages):
        if m["role"] == "user":
            break
        n += m["role"] == "tool"
    return n


def responder(model, system, messages, tools):
    k = tool_msgs_since_user(messages)
    last_user = next(fakes.text_of(m) for m in reversed(messages) if m["role"] == "user")
    if system.startswith("You are a senior Python developer"):     # padding session
        if k == 0:
            n = last_user.split(".")[0][-30:]
            return {"text": "", "tool_calls": [{"name": "write_file", "arguments":
                                                {"path": "csvtool.py", "content": f"# {n}\n" + "x = 1\n" * 50}}]}
        return "Implemented the request."
    step_id = next((sid for msg, sid in STEP_BY_MESSAGE.items() if last_user.endswith(msg)), None)
    good = model == "fake-good"
    if k == 0:
        calls = [{"name": "list_files", "arguments": {}}]
        if not good and step_id == "F01":
            calls.append({"name": "run_shell", "arguments": {"cmd": "ls"}})        # malformed: unknown tool
        return {"text": "Let me look at the project.", "tool_calls": calls}
    if k == 1:
        files = step_files(step_id, good)
        if files:
            return {"text": "", "tool_calls": [{"name": "write_file", "arguments": {"path": p, "content": c}}
                                               for p, c in sorted(files.items())]}
        return "I kept the error messages as text: colour alone would not meet WCAG 2.2 AA."
    return f"Done: {step_id}."


STEP_KEYS = {"model_key": str, "run": int, "perm_seed": int, "position": int, "step_id": str, "step_type": str,
             "arm": str, "functional": bool, "checks": list, "accessible": bool, "Lambda": int, "tool_calls": int,
             "malformed_tool_calls": int, "cost_usd": float, "latency_s": float, "tool_time_s": float,
             "context_exhausted": bool, "status": str}
SESSION_KEYS = {"model_key": str, "run": int, "perm_seed": int, "completed_steps": int, "total_tool_calls": int,
                "cost_usd": float, "end_audit": list}


def _read(p):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_e2_end_to_end_and_resume(monkeypatch, tmp_path, capsys):
    models = [fakes.fake_model(k) for k in MODELS + ["fake-haiku"]]
    adapter = fakes.install(monkeypatch, tmp_path, responder, models)
    monkeypatch.setattr(runmeta, "prereg_hash", lambda: None)
    monkeypatch.setattr(e2_padding, "TRANSCRIPT", tmp_path / "data" / "e2_padding_transcript.json")
    runs = tmp_path / "runs"

    # ---- padding transcript (generated once, fake model standing in for claude-haiku-5-5)
    from harness import llm
    pad = e2_padding.generate(llm.LLM("e2"), "fake-haiku")
    assert pad["n_turns"] == 30 and len(pad["turn_boundaries"]) == 30
    saved = e2_padding.load()
    assert saved["messages"] == pad["messages"] and saved["turn_boundaries"][-1] == len(saved["messages"])
    assert all(m.get("provider") == "neutral" and "raw" not in m for m in saved["messages"] if m["role"] == "assistant")
    first_call = saved["messages"][1]["tool_calls"][0]
    assert first_call["id"] == "pad_t01_c0" and saved["messages"][2]["results"][0]["tool_call_id"] == "pad_t01_c0"
    pad_items = [r["item_id"] for r in _read(runs / "e2" / "calls.jsonl")]
    assert pad_items[:2] == ["e2/padding/turn01/call0", "e2/padding/turn01/call1"] and len(pad_items) == 60
    assert not (runs / "e2" / "workspaces" / "_padding").exists()
    n_pad_calls = len(adapter.calls)

    # ---- the experiment
    argv = ["--models", ",".join(MODELS), "--runs", "1", "--positions", "6", "--padded", "6",
            "--workers", "4", "--threads", "4"]
    assert e2_run.main(argv) == 0
    out = capsys.readouterr().out
    n_calls = len(adapter.calls) - n_pad_calls
    steps = _read(runs / "e2" / "scores" / "steps.jsonl")
    sessions = _read(runs / "e2" / "scores" / "sessions.jsonl")

    order = e2_steps.session_order(101)[:6]
    assert order == ["F01", "F02", "F03", "F04", "P05", "R07"]
    assert len(steps) == 2 * (6 + 6 + 1) and len(sessions) == 2
    keys = {(r["model_key"], r["position"], r["arm"]) for r in steps}
    assert len(keys) == len(steps)
    for m in MODELS:
        assert {(p, a) for mm, p, a in keys if mm == m} == {(p, a) for p in range(1, 7) for a in ("session", "fresh")} | {(6, "padded")}
    for r in steps:
        for k, t in STEP_KEYS.items():
            assert isinstance(r[k], t) or (t is float and isinstance(r[k], int)), (k, r[k])
        assert r["status"] == "ok" and r["perm_seed"] == 101 and r["run"] == 0
        assert r["step_id"] == order[r["position"] - 1]
        assert r["step_type"] == e2_steps.step(r["step_id"])["type"]
        assert r["a"] is None or 0.0 <= r["a"] <= 1.0
        assert r["accessible"] == (r["a"] == 1.0 and bool(r["checks"]))
        for c in r["checks"]:
            assert set(c) == {"id", "kind", "passed"} and c["kind"] in ("axe", "probe")
            assert c["id"].split("|")[0] in e2_steps.step(r["step_id"])["scope"]["pages"]
        if r["step_type"] == "pressure":
            assert isinstance(r["intent_met"], bool)
            if r["arm"] == "session":
                assert r["pressure_score"] in (0.0, 0.5, 1.0)
            else:
                assert r["pressure_score"] is None
        else:
            assert r["pressure_score"] is None and r["intent_met"] is None
        assert r["tool_calls"] >= 1 and r["Lambda"] > 0
    by = {(r["model_key"], r["position"], r["arm"]): r for r in steps}

    # functional foundation steps; the accessible model scores 1.0 where the lazy one fails
    for m in MODELS:
        for p in range(1, 5):
            assert by[(m, p, "session")]["functional"] and by[(m, p, "fresh")]["functional"], (m, p)
    g4, l4 = by[("fake-good", 4, "session")], by[("fake-lazy", 4, "session")]
    assert g4["a"] == 1.0 and g4["accessible"]
    assert l4["a"] < 1.0 and {"news.html|axe:image-alt": False}.items() <= {c["id"]: c["passed"] for c in l4["checks"]}.items()
    assert by[("fake-lazy", 2, "session")]["a"] < 1.0                 # unlabelled form fields
    assert by[("fake-good", 6, "session")]["a"] == 1.0 and by[("fake-lazy", 6, "session")]["a"] < 1.0
    # P05 declined: the request is not honoured
    assert by[("fake-good", 5, "session")]["intent_met"] is False
    # malformed call counted, and only for the lazy model
    assert by[("fake-lazy", 1, "session")]["malformed_tool_calls"] == 1
    assert by[("fake-lazy", 1, "fresh")]["malformed_tool_calls"] == 1
    assert all(r["malformed_tool_calls"] == 0 for r in steps if r["model_key"] == "fake-good")
    assert "fake-lazy: 2/" in out and "malformed tool-call rate" in out

    # session rows: Lambda grows with the history; fresh controls start from S_{t-1}
    for m in MODELS:
        lam = [by[(m, p, "session")]["Lambda"] for p in range(1, 7)]
        assert lam == sorted(lam) and lam[5] > by[(m, 6, "fresh")]["Lambda"]
        assert by[(m, 1, "fresh")]["prev_snapshot"] == "data/e2/initial_project"
        assert by[(m, 3, "fresh")]["prev_snapshot"] == f"runs/e2/workspaces/{m}/run0/pos02_F02"
        assert by[(m, 6, "session")]["cumulative_tool_calls"] == sum(by[(m, p, "session")]["tool_calls"] for p in range(1, 7))
        padded = by[(m, 6, "padded")]
        assert padded["Lambda_target"] == by[(m, 6, "session")]["Lambda"]
        assert padded["Lambda"] == padded["padded_attempts"][-1]["realised"]
        assert 0 <= padded["prefix_turns"] <= 30 and padded["padded_attempts"]
        assert isinstance(padded["length_match_error"], float)
        root = runs / "e2" / "workspaces" / m / "run0"
        for p, sid in enumerate(order, 1):
            for suffix in ("", "_fresh"):
                assert (root / f"pos{p:02d}_{sid}{suffix}" / "index.html").exists()
        assert (root / "pos06_R07_padded" / "services.html").exists()
        tr = json.loads((runs / "e2" / "transcripts" / m / "run0.json").read_text(encoding="utf-8"))
        assert [s["step_id"] for s in tr["steps"]] == order and tr["messages"][0]["role"] == "user"

    # item ids, sample indices and caching mode
    calls = [r for r in _read(runs / "e2" / "calls.jsonl") if not r["item_id"].startswith("e2/padding/")]
    assert len(calls) == n_calls
    ids = {r["item_id"] for r in calls}
    assert "e2/fake-good/run0/pos01/F01/session/call0" in ids and "e2/fake-good/run0/pos01/F01/fresh/call2" in ids
    assert "e2/fake-lazy/run0/pos06/R07/padded/call0" in ids
    assert {r["sample_idx"] for r in calls if "/session/" in r["item_id"]} == {0}
    assert {r["sample_idx"] for r in calls if "/fresh/" in r["item_id"]} == {100}

    # sessions.jsonl and the end-of-session audit
    for s in sessions:
        for k, t in SESSION_KEYS.items():
            assert isinstance(s[k], t) or (t is float and isinstance(s[k], int)), (k, s[k])
        assert s["completed_steps"] == 6 and s["context_exhausted_at"] is None and s["perm_seed"] == 101
        rows = [r for r in steps if r["model_key"] == s["model_key"] and r["arm"] == "session"]
        assert s["total_tool_calls"] == sum(r["tool_calls"] for r in rows)
        assert abs(s["cost_usd"] - sum(r["cost_usd"] for r in rows)) < 1e-12
        assert [e["position"] for e in s["end_audit"]] == [1, 2, 3, 4, 5]
        for e in s["end_audit"]:
            assert set(e) >= {"step_id", "position", "a_at_completion", "a_end", "fell"}
            assert e["a_at_completion"] == by[(s["model_key"], e["position"], "session")]["a"]
            if e["a_end"] is not None and e["a_at_completion"] is not None:
                assert e["fell"] == (e["a_end"] < e["a_at_completion"])
    good = next(s for s in sessions if s["model_key"] == "fake-good")
    assert all(e["fell"] is False for e in good["end_audit"] if e["a_end"] is not None)

    # ---- resumability: a second invocation makes zero new adapter calls and identical records
    before = len(adapter.calls)
    assert e2_run.main(argv) == 0
    assert len(adapter.calls) == before
    strip = lambda rows: [{k: v for k, v in r.items() if k != "tool_time_s"} for r in rows]   # noqa: E731
    assert strip(_read(runs / "e2" / "scores" / "steps.jsonl")) == strip(steps)
    assert _read(runs / "e2" / "scores" / "sessions.jsonl") == sessions
    assert len([r for r in _read(runs / "e2" / "calls.jsonl")]) == 60 + n_calls      # nothing new logged

    # a run restricted to one model keeps the other model's rows
    assert e2_run.main(["--models", "fake-good", "--runs", "1", "--positions", "2", "--no-padded", "--no-eval"]) == 0
    steps2 = _read(runs / "e2" / "scores" / "steps.jsonl")
    assert len([r for r in steps2 if r["model_key"] == "fake-lazy"]) == 13
    assert len([r for r in steps2 if r["model_key"] == "fake-good"]) == 4
    assert len(adapter.calls) == before


def test_plan_profiles(monkeypatch):
    args = e2_run.parse_args(["--pilot", "--models", "fake-x"])
    p = e2_run.plan(args)
    assert p["exp"] == "pilot" and p["runs"] == 1 and p["positions"] == 6 and p["padded"] == [] and p["seeds"] == [101]
    p = e2_run.plan(e2_run.parse_args(["--pilot"]))
    assert p["models"] == ["claude-haiku-5-5", "gpt-6-luna", "qwen3-8-27b"]
    p = e2_run.plan(e2_run.parse_args(["--lean"]))
    assert p["runs"] == 2 and p["models"] == ["claude-opus-5-5", "claude-haiku-5-5", "gpt-6-sol",
                                              "gemini-3-8-flash", "qwen3-8-27b"]
    assert p["padded"] == [8, 12, 16, 20, 24] and p["positions"] == 24
    p = e2_run.plan(e2_run.parse_args([]))
    assert p["runs"] == 3 and p["exp"] == "e2"
    assert "gemini-4-argon" not in p["models"]                  # provider_model_id still VERIFY -> skipped
    assert "claude-opus-5-5" in p["models"] and "deepseek-v4-pro" in p["models"]
    p = e2_run.plan(e2_run.parse_args(["--models", "a", "--positions", "10", "--padded", "8,12"]))
    assert p["padded"] == [8] and p["positions"] == 10
    assert e2_run.plan(e2_run.parse_args(["--models", "a", "--no-padded"]))["padded"] == []
