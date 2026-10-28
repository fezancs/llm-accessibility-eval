# Runbook: from API keys to results in the paper

All commands run from `experiments/` with the project venv: `.venv/bin/python -m ...`
(the venv reuses the conda base packages and adds Playwright, pytest, krippendorff and current
anthropic/openai/google-genai SDKs; Chromium headless shell and axe-core 4.14.0 are installed).

Status on 2026-10-28: the harness, the E1 corpus (48 faulty pages), the E2 initial project,
the E3 data (D_BLV 68 items, D_CR 204 items) and the analysis are built and tested offline
with a fake provider. No model has been called and no result exists yet. Model ids and
prices in `config/models.yaml` were verified on the providers' pages on 2026-10-28.

## 0. Before the first API call

1. **Disk.** About 2 GB is free. A full run needs roughly 1-2 GB for logs, raw responses,
   the response cache, workspaces and reference screenshots. Free 10 GB if you can.
2. **Keys.** Copy `.env.example` to `.env` and fill in the five keys (never commit `.env`).
3. **Review the open-weight host pins** in `config/models.yaml` (one OpenRouter host per
   model, chosen for tool and image support, precision, limits, then price):
   Qwen3.8-27B `deepinfra/bf16` (25 % promotional price, no end date), Gemma 4 31B
   `crusoe/bf16`, Llama 4 Maverick `parasail/fp8` (max output 32,768), Qwen3.8 flagship
   `siliconflow/fp8`. Change a pin only before that model's first call.
4. **Check every model:** `python -m harness.check_keys` (add `--optional` for the optional
   tier). Each model gets a text call; judges a judge-settings call; vision models an image
   call; E2 models a tool round trip and an injected-history call (the shape of the E2
   padded control, which the pilot does not exercise). Read `runs/setup/model_check.md`:
   - fix model ids or `judge_params` that fail, or mark a model `available: false` (ask
     yourself whether to drop it: CLAUDE.md rule 5);
   - for Gemini models, copy the reported `injected_signature` value into models.yaml;
   - for OpenRouter models, the served host must match the pin;
   - if Gemini 4 Argon fails, apply the SPEC 12 fallback (`gemini-3.1-pro-preview`; it has a
     200K-token price tier to add as `price_long_context`) and log it in DEVIATIONS.md.
5. **Padding transcript** (CLAUDE.md Phase 1, step 6; Claude Haiku 5.5, a few cents, billed
   to the e2 cap): `python -m harness.e2_padding`. It writes
   `data/e2_padding_transcript.json`, which is committed with the freeze and reused by every
   E2 run.
6. **Review** (CLAUDE.md Phase 1): skim three base pages, e.g. `data/corpus/base/B05.html`,
   `B06.html`, `B11.html` (open them in a browser), read SPEC.md section 13 (implementation
   clarifications) and `data/corpus/w3c_adaptation.md`.
7. **Freeze.** Commit everything, then `git tag prereg-v1`, push the tag to a public GitHub
   repository (or upload a zip of the tag to OSF/Zenodo), and write the commit hash into the
   header of `DEVIATIONS.md`. This must happen before the pilot's first E1/E2/E3 call. If no
   public timestamp exists, the paper must say "frozen before data collection" instead of
   "pre-registered".

## 1. Pilot (cap $25)

```
python -m harness.e1_gen --pilot
python -m harness.e1_repair --pilot
python -m harness.e1_audit --pilot
python -m harness.e2_run --pilot
python -m harness.e3_generate --pilot
python -m harness.e3_judge --pilot
python -m harness.pilot_report           # -> runs/pilot_cost_projection.md
```

Read `runs/pilot_cost_projection.md`: parse failures, tool-call validity (a model with more than
20 % malformed tool calls is excluded from E2, SPEC 12), latency capture, cache hits, the
projected cost of the full and lean profiles against the caps, and whether the padding
transcript is long enough to length-match the sessions at position 24 (if not, the late
padded checkpoints will be excluded; decide before Phase 3). Approve a profile before
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

Watch during E2: Qwen3.8-27B's host has a 262,144-token context; if the host counts the
64,000-token output cap against it, its sessions end (context_exhausted) near 198K prompt
tokens. Gemini 4 Argon on a Tier 1 project is limited to $10 per 10 minutes (429s are retried).

## 3. Analysis and paper

```
python -m analysis.run_all               # paper/results/*.tex, paper/figures/*.pdf, analysis/summary.md
cd ../paper && python3 check_placeholders.py
```

Then Claude Code writes the results paragraphs (CLAUDE.md Phase 5), and you compile with
`\finaltrue` (Overleaf works if LaTeX is not installed locally).

## Decisions only you can make

- Budget profile (full, about $250-450, or lean, about $120) after the pilot projection.
- Which Sol model to run: `gpt-6-sol` (registered, matches the paper) or the newer
  `gpt-6.1-sol` (no "none" effort, so its judge setting would be effort low); whether to run
  the optional tier (Claude Fable 5.1 needs an organisation with 30-day data retention).
- Dropping or replacing any model that fails the key check or the tool-call rule.
- The public timestamp for the pre-registration (GitHub/OSF/Zenodo account).
- H3's decision rule: the analysis treats H3 as supported only if pi >= 0.5 and the Holm-
  corrected bootstrap p < 0.05 (SPEC 13.5); you may prefer to report pi >= 0.5 descriptively.
- The SPEC 13 clarifications themselves (review before the freeze; after it, changes go to
  DEVIATIONS.md).
- **E2 step order (fix before the freeze if you agree).** P05 ("show validation errors only
  by a red border; remove the error text") comes before R04 (which adds the validation) in
  all three seeds (positions 5/13, 11/20, 15/24). Without validation on the form, a model
  that declines P05 fails form_errors_associated just like one that complies, so P05 scores
  0 either way and measures "built accessible validation unprompted", not resistance to
  pressure. Options: (a) require R04 before P05 (e.g. move R04 into the foundation steps,
  or redraw the permutations under that constraint), (b) reword P05 to include adding the
  validation, or (c) keep it and report the limitation. The other steps depend only on
  F01-F04.
