# Runbook: from API keys to results in the paper

All commands run from `experiments/` with the project venv: `.venv/bin/python -m ...`
(the venv reuses the conda base packages and adds Playwright, pytest, krippendorff and current
anthropic/openai/google-genai SDKs; Chromium headless shell and axe-core 4.14.0 are installed).

Status on 2026-10-28: the harness, corpus, E3 data and analysis are built and tested
offline (fake provider; no model has been called). No result exists yet.

## 0. Before the first API call

1. **Disk.** About 2.5 GB is free. A full run needs roughly 1-2 GB for logs, raw
   responses, workspaces and reference screenshots. Free 10 GB if you can.
2. **Keys.** Copy `.env.example` to `.env` and fill in the five keys (never commit `.env`).
3. **Open-weight models.** In `config/models.yaml`, set `openrouter_provider` (the pinned
   host) and `price` for `qwen3-8-27b`, `gemma-4-31b`, `llama-4-maverick` (and
   `qwen3-8-flagship` if used). The harness refuses to call a model without a price.
4. **Check every model:** `python -m harness.check_keys` (add `--optional` for the optional
   tier). Each model gets a text call; judges a judge-settings call; vision models an image
   call; E2 models a tool round trip. Read `runs/setup/model_check.md`; fix model ids or
   `judge_params` that fail, or mark a model `available: false` (ask yourself whether to drop
   it: CLAUDE.md rule 5). If Gemini 4 Argon fails, apply the SPEC 12 fallback and log it.
5. **Review** (CLAUDE.md Phase 1): skim three base pages, e.g. `data/corpus/base/B05.html`,
   `B06.html`, `B11.html` (open them in a browser), read SPEC.md section 13 (implementation
   clarifications) and `data/corpus/w3c_adaptation.md`.
6. **Freeze.** Commit everything, then
   `git tag prereg-v1`, push the tag to a public GitHub repository (or upload a zip of the
   tag to OSF/Zenodo), and write the commit hash into the header of `DEVIATIONS.md`.
   This must happen before the pilot's first E1/E2/E3 call. If no public timestamp exists,
   the paper must say "frozen before data collection" instead of "pre-registered".

## 1. Pilot (cap $25)

```
python -m harness.e2_padding            # padding transcript for the E2 padded control (Claude Haiku 5.5, once)
python -m harness.e1_gen --pilot
python -m harness.e1_repair --pilot
python -m harness.e1_audit --pilot
python -m harness.e2_run --pilot
python -m harness.e3_generate --pilot
python -m harness.e3_judge --pilot
python -m harness.pilot_report           # -> runs/pilot_cost_projection.md
```

Read `runs/pilot_cost_projection.md`: parse failures, tool-call validity (a model with more than
20 % malformed tool calls is excluded from E2, SPEC 12), latency capture, cache hits, and the
projected cost of the full and lean profiles against the caps. Approve a profile before
Phase 3. Pilot data never enter the paper.

## 2. Full runs (order: E1, E3, E2)

```
python -m harness.e1_gen        [--lean] [--optional]
python -m harness.e1_repair     [--lean] [--optional]
python -m harness.e1_audit      [--lean] [--optional]
python -m harness.e3_generate   [--lean]
python -m harness.e3_judge      [--lean]
python -m harness.e2_run        [--lean]
```

Every script is resumable: re-running it replays cached responses and continues where it
stopped. Spend is tracked per call in `runs/<exp>/calls.jsonl`; a script stops with
`BudgetExceeded` at its cap in `config/budget.yaml` (then decide: raise the cap or switch to the
lean profile). `--no-eval` makes only the API calls; a later run without it scores them.

## 3. Analysis and paper

```
python -m analysis.run_all               # paper/results/*.tex, paper/figures/*.pdf, analysis/summary.md
cd ../paper && python3 check_placeholders.py
```

Then Claude Code writes the results paragraphs (CLAUDE.md Phase 5), and you compile with
`\finaltrue` (Overleaf works if LaTeX is not installed locally).

## Decisions only you can make

- Budget profile (full, about $250-450, or lean, about $120) after the pilot projection.
- Which GPT-6 Sol snapshot to pin (GPT-6.1 Sol also exists); whether to run the optional tier.
- Dropping or replacing any model that fails the key check or the tool-call rule.
- The public timestamp for the pre-registration (GitHub/OSF/Zenodo account).
- H3's decision rule: the analysis treats H3 as supported only if pi >= 0.5 and the Holm-
  corrected bootstrap p < 0.05 (SPEC 13.5); you may prefer to report pi >= 0.5 descriptively.
