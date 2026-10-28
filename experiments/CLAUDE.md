# CLAUDE.md — operating instructions for running the SE4All 2027 study

You (Claude Code) are implementing and running the experiments for the paper in `../paper/main.tex`, then writing the real results into it. The full method is in `SPEC.md`; task data in `tasks/`; prompts in `prompts/`; models and prices in `config/models.yaml`; spend caps in `config/budget.yaml`.

Submission deadline: **13 November 2026, 23:59 AoE** (= 14 November 2026, 16:59 in Pakistan). Target: results in the paper by 8 November, leaving time for the author's review.

## Hard rules (never break these)

1. **Never fabricate, estimate, impute or "placeholder-fill" a result.** Every number written into `paper/` must be produced by committed code from logged API responses. If something did not run or failed, the paper must say so plainly. Synthetic data is allowed only inside unit tests and must never reach `paper/` or `results/`.
2. **Do not change metrics, exclusion rules, prompts or hypotheses after seeing results.** If a change is unavoidable (a bug, an unavailable model), write it in `DEVIATIONS.md` (date, what, why, expected effect) before re-running, and make sure the paper's threats section reports it.
3. **Respect the budget.** Track cost per call (SPEC §2.2). Stop when a cap in `config/budget.yaml` is reached and ask the author. Before Phase 3, show the author the pilot cost projection and wait for a "go".
4. **Secrets.** API keys live only in `.env` (git-ignored). Never print, log or commit them.
5. **The author is the decision-maker.** Ask before: spending beyond a cap, dropping a model, switching to the lean profile, or changing any paper text outside the results/abstract/discussion placeholders.
6. **Be honest in the paper's text.** Report CIs, n and exclusions; do not call a difference real unless its CI excludes 0; do not describe hypotheses as confirmed when Holm-corrected tests fail.

## Repository layout to create

```
experiments/
  harness/            llm.py, budget.py, cache.py, render.py, axe_runner.py, probes.py,
                      probe_fixtures/, e1_gen.py, e1_corpus.py, e1_repair.py, e1_audit.py,
                      e2_agent.py, e2_controls.py, e3_data.py, e3_generate.py, e3_judge.py
  analysis/           metrics.py, bootstrap.py, dcdi.py, cmbpe_bt.py, cpds.py, ccs.py,
                      hypotheses.py, figures.py, export_latex.py
  tests/              unit tests (synthetic inputs allowed here only)
  data/               downloaded datasets, corpora, padding transcript (git-ignore images)
  runs/               call logs (JSONL) and raw responses, per experiment/model
  DEVIATIONS.md       starts with "No deviations yet."
  .env.example        ANTHROPIC_API_KEY=, OPENAI_API_KEY=, GOOGLE_API_KEY=, DEEPSEEK_API_KEY=, OPENROUTER_API_KEY=
```

## Phase 0 — environment (half a day)

- Python 3.11+ virtualenv. Install: `anthropic openai google-genai httpx pyyaml pandas numpy scipy statsmodels scikit-learn matplotlib rapidfuzz scikit-image playwright pytest python-dotenv krippendorff`.
- `python -m playwright install chromium`. Install `axe-core` with npm and load `node_modules/axe-core/axe.min.js` into pages. Record the axe-core and Chromium versions.
- `git init` (if not already), add `.gitignore` (`.env`, `data/**/images/`, `runs/**/raw/` if large).
- Check each provider key with a 1-token call per model in `models.yaml`; record the model name each API returns. Verify every price on the provider's pricing page and update `models.yaml` (commit with the date). Resolve every "VERIFY" field or mark the model unavailable.
- If Gemini 4 Argon is not available to this account, apply the fallback in `models.yaml` and log it.

## Phase 1 — build and validate (1.5–2 days)

1. LLM client, cache, call log, budget tracker (SPEC §2). Unit-test cost computation against hand-computed examples, including Haiku 5.5's >100k-token tier and DeepSeek's off-peak pricing.
2. Renderer + axe runner + rule-level pass rate (SPEC §3).
3. All probes (SPEC §4) with accessible/inaccessible fixtures and tests. All must pass before continuing.
4. E1 repair corpus (SPEC §5.2): base pages, operators, 48 variants, ground truth; confirm base pages are clean and every injected fault is detected. Have the author skim 3 base pages.
5. E2 initial project and the scripted steps loader; tool executor sandboxed to the workspace directory (reject paths with `..` or absolute paths).
6. E2 padding transcript (`prompts/e2_padding.txt`).
7. E3 data download and context extraction (SPEC §7.1); report how many items could be retrieved.
8. Analysis code for every metric, tested on synthetic inputs (DCDI, BT with nuisance terms, recusal, CPDS, Pareto, CCS, bootstrap).
9. **Freeze:** commit everything, tag `prereg-v1`, and write the commit hash into `DEVIATIONS.md`'s header. Then create a **public timestamp before any Phase 3 call**: push the tag to a public GitHub repository, or upload a zip of the tag to OSF or Zenodo. The paper calls the study pre-registered; if no public timestamp exists before data collection, change that wording to "frozen before data collection" and tell the author.

## Phase 2 — pilot (half a day, cap $25)

- Models: `claude-haiku-5-5`, `gpt-6-luna`, plus one open model.
- E1: 3 tasks x 2 conditions x 1 sample; 4 repair pages; 4 audit pages. E2: one run, first 6 steps, with fresh controls. E3: 10 items of each procedure.
- Check: parse-failure rates, tool-call validity per model (SPEC §12 threshold), latency capture, cache hits, cost per call.
- Produce `runs/pilot_cost_projection.md`: projected cost per experiment and model for the full and lean profiles. **Show it to the author and wait for approval.**
- Pilot data is not used in the paper.

## Phase 3 — full runs (3–4 days)

Order: E1 (all models) → E3 → E2. E2 is the longest: start it as soon as E1 is stable, run models in parallel within rate limits, and keep everything resumable. Re-check prices if a run spans the 30 November Gemini 4 Argon price change (it should not). Watch spend; stop at caps.

## Phase 4 — analysis (1 day)

Run `analysis/` end to end from the logs. Produce everything listed in SPEC §11, including both figures (Okabe–Ito palette plus line styles or markers, so colour is never the only cue; text >= 7 pt at final size; vector PDF). Write `analysis/summary.md` with every number the paper will use, its CI and n.

## Phase 5 — write results into the paper (1 day)

Edit only these parts of `paper/main.tex`:
- Every `\resultpara{...}{...}` block: replace the whole macro with prose that follows its instructions.
- Every `\todo{...}` in the abstract, introduction (last paragraph), discussion and conclusion.
- The study-design counts that depend on what actually ran (use `\res{...}` keys). If the lean profile or any fallback was used, also correct the fixed design numbers in Section V (runs per model, images, pairs per image, models per experiment) and the model rows in Table II.
- The threats section: add any deviation from `DEVIATIONS.md`.

Writing rules:
- Use `\res{key}` for every number, so a re-run updates the text automatically.
- State point estimates with 95% CIs, e.g., "DCDI was \res{dcdi-x} [\res{dcdi-x-lo}, \res{dcdi-x-hi}]".
- Describe what the data show, not why, unless the analysis tested the why.
- Keep each replacement about the length of the placeholder it replaces; the paper must stay within **8 pages including references**.
- Then, from `paper/`: compile (`latexmk -pdf main.tex`), run `python3 check_placeholders.py` (it lists every leftover `\todo`, `\tbd`, `\resultpara`, figure placeholder and undefined `\res{}` key, and the page count; it must report nothing left), then set `\finaltrue` in `main.tex` and compile once more. In final mode any leftover placeholder stops the build with an error.
- If over 8 pages, cut in this order: Section II detail, Table II columns, Fig. 1, discussion bullets. Never cut threats to validity.
- If LaTeX is not installed locally, the author can upload `paper/` to Overleaf (IEEEtran is built in).

## Phase 6 — artifact

Create a replication package: harness, analysis, tasks, prompts, corpora, the E2 padding transcript, call logs (without keys), and a README with exact commands. Publish on GitHub and archive on Zenodo for a DOI (the author creates the accounts). Put the link in the paper's "Data availability" paragraph. Do not include dataset images whose licence does not allow redistribution; include the scripts that fetch them.

## Fallbacks

See SPEC §12. Every fallback is logged in `DEVIATIONS.md` and reported in the paper.

## If time runs out

If by 8 November the full E2 or E3 cannot finish, stop and tell the author. Two honest options: (a) report what finished, clearly labelled, with the remaining parts as future work; or (b) convert the submission to a 5-page vision paper (Sections I–IV plus a pilot description, with every pilot number labelled as pilot data). The author chooses.
