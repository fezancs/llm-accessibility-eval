# LLM Accessibility Evaluation

A framework and test harness for evaluating large language models (LLMs) on **accessible
software engineering**: writing, repairing and auditing user-interface code against WCAG 2.2,
keeping accessibility requirements during long agentic coding sessions, and judging alt text
for blind and low-vision users.

Most comparisons of LLMs use static leaderboards that ignore accessibility. Studies of LLMs and
accessibility usually test one or two models with one-shot prompts, sometimes let a model
judge its own output, and rarely report cost. This project measures accessibility the way it
matters in practice: with real browser checks, over long sessions, with bias-controlled
judging, and per dollar and second.

> **Status:** the harness, task corpora and analysis code are complete and tested offline
> against a scripted fake model. No model has been run yet, so this repository contains no
> results. Every result will come only from logged API responses.

## What the framework measures

| Part | What it is |
|---|---|
| **Capability taxonomy** | Six dimensions that map general LLM capabilities to WCAG 2.2-anchored tasks and automated oracles: conformance auditing (C1), accessible UI generation and repair (C2), agentic execution of a multi-page build (C3), text alternatives for images (C4), keeping adherence as the context grows (C5), and handling requests that conflict with accessibility, plus honest conformance claims (C6). |
| **DCDI** (Dynamic Contextual Drift Index) | The share of accessibility adherence an agent loses over a long, tool-using session, compared with a counterfactual fresh-context control. A padded control separates the effect of context *length* from the effect of the session's *history*. |
| **CMBPE** (Cross-Model Blind Peer Evaluation) | An anonymised jury of models from different families that judges every pair in both orders. Judges never score their own model family. A Bradley-Terry model adjusts for position and length bias, and the jury is calibrated against ratings by blind and low-vision people. |
| **CPDS** (Capability-per-Dollar-Second) | A scenario-weighted efficiency score (batch remediation, interactive assistant, balanced) with bootstrap Pareto-membership probabilities over capability, cost and latency. |

A composite capability score combines the six dimension scores (weighted geometric mean).

Research questions:
1. How do current LLMs perform across the six capability dimensions?
2. Do LLM agents keep honouring accessibility requirements as a long, tool-using session grows?
3. Does a blind, family-recused, order-permuted, length-adjusted jury judge accessibility
   content more reliably than single LLM judges?
4. Which models are Pareto-efficient in capability, cost and latency, and what does
   AI-assisted remediation cost?

## Experiments

**E1: capability tasks** (`tasks/e1_gen_tasks.yaml`, `data/corpus/`)
- *Generation:* 20 UI tasks (forms, dialogs, tabs, accordions, carousels, comboboxes, sliders,
  data tables and more), each with and
  without a one-sentence accessibility nudge, 3 samples each. The "default gap" between the
  two conditions is reported. After each page, the model is asked whether its page conforms
  to WCAG 2.2 AA, which measures over-claiming.
- *Repair:* 48 faulty pages built from 12 accessible base pages (4 adapted W3C "Before and
  After" demo pages and 8 authored pages). 12 mutation operators inject 4 known faults into
  each page. A repair scores the faults it fixes minus any new failures it introduces, and
  counts only if the page's text and structure are preserved.
- *Audit:* the model lists the problems on the same 48 pages, and the list is scored against
  the 192 known faults (micro-F1).

**E2: long agentic sessions** (`tasks/e2_steps.yaml`)
- The model builds a small clinic website over 24 steps, using file tools (`list_files`,
  `read_file`, `write_file`) in a sandboxed workspace. The system prompt states the WCAG 2.2
  AA requirement once; no user message mentions accessibility.
- Steps: 4 foundation steps, 14 regular feature steps and 6 "pressure" steps. Pressure steps
  ask for things that conflict with accessibility, such as placeholder-only labels or
  removing focus outlines. The 20 non-foundation steps run in three fixed random orders.
- Controls: a *fresh* control repeats every step in a new conversation on the same workspace
  snapshot. A *padded* control repeats checkpoint steps after unrelated history of the same
  length.

**E3: alt-text judging** (`prompts/e3_*`)
- Calibration against human ratings: alt texts rated by blind and low-vision participants, and
  crowd-rated alt texts with article context.
- Known-answer pairs: each original description is paired with degraded versions (appended,
  repeated, swapped, truncated).
- A generator comparison: several models write alt text for the same images, and the jury
  ranks them.

## How accessibility is scored

Pages are rendered in headless Chromium (Playwright). Two kinds of checks run on each page:
- **axe-core** with the WCAG 2.x Level A and AA rules;
- **27 keyboard and semantics probes** for things axe cannot see, for example tab
  reachability, visible focus, keyboard traps, the behaviour of dialogs, tabs, accordions,
  comboboxes, sliders and switches, error messages tied to their fields and not shown by
  colour alone, pause controls for moving content, and contrast after a theme toggle. Each probe has a passing and a failing test fixture
  (`harness/probe_fixtures/`).

Each page gets a rule-level pass rate `a(x)`: the share of applicable checks that pass. Task
functional checks (for example "the form blocks an empty submit") confirm that the page does
what was asked.

## Principles

- **No invented numbers.** Every reported value is computed by committed code from logged API
  responses. If something fails or does not run, the report says so.
- **Plan frozen before data collection.** Metrics, prompts, tasks, exclusion rules and
  hypotheses are fixed and tagged (`prereg-v1`) before the first model call. Any later
  change is recorded in `experiments/DEVIATIONS.md` with its reason.
- **Honest statistics.** 95% bootstrap confidence intervals, sample sizes and exclusions are
  reported, and Holm correction is used across hypothesis tests.
- **Cost is tracked on every call** with the provider's price card (long-context tiers,
  caching, peak and off-peak hours). Spending caps stop a run.
- **Resumable and reproducible.** Every request is cached by its exact content, so an
  interrupted run resumes without new calls, and every API attempt is logged.

## Repository layout

```
experiments/
  harness/      LLM client (Anthropic, OpenAI, Gemini and OpenAI-compatible APIs), cost
                tracking, cache, renderer, axe runner, probes, E1/E2/E3 runners
  analysis/     metrics, bootstrap, DCDI, Bradley-Terry jury model, CPDS/Pareto,
                composite score, hypothesis tests, figures, table export
  tests/        unit and end-to-end tests (synthetic data only)
  tasks/        E1 generation tasks, mutation operators, E2 step script
  prompts/      every prompt, verbatim
  config/       models.yaml (models, ids, price cards, roles), budget.yaml (spend caps)
  data/         repair corpus (base pages, variants, ground truth), E2 starter project,
                E3 item lists (images are downloaded, not stored)
  SPEC.md       full method: metrics, procedures, statistics
  RUNBOOK.md    step-by-step commands from API keys to results
  DEVIATIONS.md changes made after the plan was frozen
```

## Quick start

Requirements: Python 3.12 and Node.js 18 or newer.

```bash
cd experiments
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
npm ci                                  # pinned axe-core

.venv/bin/python -m pytest              # offline tests (~10 min, browser tests included): no API key, no cost
```

To run real models, copy `.env.example` to `.env` and add the API keys for the providers you
use (`.env` is git-ignored). Models, ids and prices live in `config/models.yaml`, and spend
caps in `config/budget.yaml`. Then follow `RUNBOOK.md`: check keys
(`python -m harness.check_keys`), run a small pilot (`--pilot`) and its cost projection
(`python -m harness.pilot_report`), then the full runs and `python -m analysis.run_all`.

### Running for free on a local GPU

No API budget? You can run a smaller version entirely on your own hardware at zero token
cost, using self-hosted open-weight models through [Ollama](https://ollama.com). An 8 GB
NVIDIA GPU runs 7–8B models that cover E1 (generation/repair/audit) and E3 (alt text), and
E2 (agentic sessions) more slowly. See **`docs/LOCAL_SETUP.md`** for the full Ubuntu setup.
This evaluates small open-weight models, not the frontier models, and is recorded as a
deviation in `experiments/DEVIATIONS.md`.

## Data sources

- **W3C "Before and After" demonstration pages** (the four "after" pages), adapted under the
  W3C Document License. The changes are listed in `data/corpus/w3c_adaptation.md`.
- **Alt-text ratings by blind and low-vision participants** (Kreiss et al., 2022) and the
  **ContextRef** alt-text ratings. The scripts download images and source files from the
  original repositories (`harness/e3_data.py`, sources pinned in `data/e3/sources.json`).
  Images are not redistributed.
