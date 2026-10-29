# Experiment specification

**Status: pre-registration draft.** Freeze this file, the files in `config/`, `tasks/` and `prompts/`, and the analysis code before any full run (tag `prereg-v1`). Every later change goes in `DEVIATIONS.md` with date, reason and expected effect, and is reported in the paper.

This file is the single source of truth for the methods. If the paper and this file disagree, fix the paper.

---

## 0. Principles

1. **No fabricated or estimated results.** Every number in the paper comes from logged API outputs processed by the committed analysis code. If an experiment did not run, the paper says so; it never "fills in" a plausible value.
2. **Pre-registration.** Metrics, hypotheses, exclusion rules and statistics below are fixed before data collection.
3. **Reproducibility.** Log every request and response; pin model snapshots, axe-core and browser versions; record dates, prices and reasoning settings; fix seeds; make every run resumable from an on-disk cache.
4. **Budget safety.** Track spend per call; stop at the caps in `config/budget.yaml` and ask the author.
5. **Data ethics.** No personal data. External datasets are used under their terms; images are referenced by URL/ID and not redistributed unless the licence allows it.

---

## 1. Research questions, experiments and hypotheses

| RQ | Question | Experiment |
|---|---|---|
| RQ1 | How do current LLMs perform across the six accessibility-engineering capability dimensions (C1–C6)? | E1, E2, E3 |
| RQ2 | Do LLM agents keep honouring accessibility requirements as a long, tool-using session grows? | E2 (DCDI) |
| RQ3 | Does a blind, family-recused, order-permuted, length-adjusted jury (CMBPE) judge accessibility content more reliably than single LLM judges? | E3 |
| RQ4 | Which models are Pareto-efficient in capability, cost and latency, and what does AI-assisted remediation cost? | E1 logs (CPDS) |

Pre-registered hypotheses (one-sided tests, alpha = 0.05, Holm correction across all sub-tests listed here):

- **H1 (agentic positioning).** Each vendor's larger model has lower DCDI than its smaller sibling: H1a Opus 5.5 < Haiku 5.5; H1b GPT-6 Sol < GPT-6 Luna; H1c Gemini 4 Argon < Gemini 3.8 Flash (only if Argon runs in E2).
- **H2 (multimodal).** The best closed model's C4 score exceeds the best open-weight model's C4 score.
- **H3 (open-weight efficiency).** At least one open-weight model has Pareto-membership probability pi >= 0.5 on the batch-remediation frontier.
- **H4 (judge bias).** (a) At least one single judge has self-preference eta_j > 0; (b) CMBPE's accuracy on known-answer pairs exceeds the median single judge's.
- **H5 (default gap).** For a majority of models, S_gen(nudged) > S_gen(unguided).

---

## 2. Infrastructure

### 2.1 LLM client
- One client module for all providers. Prefer each vendor's native API (Anthropic, OpenAI, Google, DeepSeek) because latency through a gateway adds overhead. For open-weight models without a native API, use one gateway (OpenRouter by default) and **pin the upstream provider** (no fallbacks); log the provider that served each call.
- Streaming on for every call (needed for TTFT).
- Parameters: provider defaults for temperature and reasoning effort, **except** E3 judges, which run at their lowest available reasoning setting. Record every parameter actually sent and every default the provider documents.
- Retries: exponential backoff on 429/5xx, max 5; log every attempt and bill it.
- Prompt caching: enable where supported (Anthropic `cache_control` breakpoints after the system prompt and after long shared prefixes; OpenAI/Gemini/DeepSeek automatic caching). Record cached-token counts.

### 2.2 Call log (append-only JSONL, one line per attempt)
`{call_id, experiment, item_id, model_key, provider, model_returned, request_hash, sample_idx, started_at, ttft_s, total_s, input_tokens, cached_input_tokens, cache_write_tokens, output_tokens, reasoning_tokens, cost_usd, price_snapshot_date, params, status, error, response_path}`
- `cost_usd` is computed from token counts and `models.yaml` prices (tiered prices for Haiku 5.5 and DeepSeek peak/off-peak).
- Raw responses are stored as files referenced by `response_path`.

### 2.3 Cache and resumability
- Key = sha256(model_key, normalised messages, params, sample_idx). A cached success is never re-requested. All experiment scripts are idempotent: re-running continues where they stopped.

### 2.4 Latency definitions
- Single-call task: L = wall-clock from request send to final token (equals TTFT + (n_out - 1) * TPOT for a streamed response). Report median over task instances.
- Multi-call task (E2 steps): sum over the step's calls of total_s (tool execution time excluded and logged separately).

---

## 3. Rendering and automated oracles

- Playwright (Python) with Chromium; viewport 1280x800; `prefers-reduced-motion` not set. Serve pages from a local static HTTP server (not file://) so relative paths work. Missing images are fine.
- axe-core: pin one version (latest 4.x at run time; record it). Run with `runOnly: {type: 'tag', values: ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}`. When a scope selector is given, use `include` for that selector.
- Wait for `networkidle` plus 500 ms; for pages with timers (carousels, tickers) also run checks at t = 0 s and t = 6 s.

### 3.1 Rule-level pass rate (used everywhere)
For an artifact x (page or step scope), let Q(x) be the set of **applicable checks**: every axe rule that evaluated at least one node in scope (results in `passes` or `violations`; `incomplete` and `inapplicable` excluded) plus every probe that applies. A rule passes if it has zero violating nodes; a probe passes per its definition.

`a(x) = |passing checks in Q(x)| / |Q(x)|`

Also record violating-node counts, `incomplete` counts, and the DOM element count (for violations per 100 elements).

---

## 4. Probes (keyboard and semantics)

Implement each probe as a function `probe(page, scope, anchors) -> {applicable: bool, passed: bool, detail}`. Probes find widgets by **visible text anchors** from the task files, never by implementation details. They test WCAG requirements, not APG conventions: any keyboard path that works passes (for example, Tab plus Enter is acceptable for tabs).

### 4.1 Probe validation (must do before Phase 3)
For every probe, write at least one **accessible fixture** (must pass) and one **inaccessible fixture** (must fail) as tiny HTML files under `harness/probe_fixtures/`, and a test that runs them. Report "all N probes pass their fixture tests" in the paper. A probe that cannot be made reliable is dropped (log it) rather than kept noisy.

### 4.2 Generic probes (every artifact)
- `tab_reachability` — Press Tab up to 3 x (number of interactive elements) + 10 times from the top. Every visible interactive element must receive focus at least once. "Interactive" = native controls and links with href, elements with `tabindex >= 0`, elements with an interactive ARIA role (button, link, checkbox, switch, tab, menuitem, option, slider), and elements that look clickable (an `onclick` attribute or `cursor: pointer`). A clickable-looking element that never receives focus fails the probe. Composite widgets with roving tabindex count as reached when any of their items is reached. (2.1.1)
- `focus_visible` — For up to 10 sampled focusable elements, compare a screenshot of the element's bounding box (+4 px) when focused via keyboard vs. unfocused; pass if the mean absolute pixel difference exceeds a threshold calibrated on the fixtures (e.g., a 2 px outline). (2.4.7)
- `no_keyboard_trap` — Tabbing forward from the first focusable element eventually reaches the last and leaves the page or wraps; Shift+Tab moves backward; no element holds focus for >3 consecutive Tabs unless it is a documented modal. (2.1.2)

### 4.3 Pattern probes
| Probe | Passes when | SC |
|---|---|---|
| `form_errors_associated` | After submitting the empty form (click anchor `submit` via keyboard), each required field that is invalid has `aria-invalid="true"` OR an error text programmatically associated (`aria-describedby`/`aria-errormessage`/inside the label), and the error text is visible. | 3.3.1, 3.3.2, 4.1.2 |
| `error_not_color_only` | Every invalid field from the previous probe also has visible error text or an icon with text alternative; colour change alone fails. | 1.4.1, 3.3.1 |
| `dialog_behavior` | Activating the trigger with Enter opens an element with role dialog/alertdialog (or `<dialog>` open) with an accessible name; focus moves inside; Tab stays inside while open; Escape closes; focus returns to the trigger. | 2.1.1, 2.4.3, 4.1.2 |
| `live_region_on_action` | After the anchor action, the result text is inside `role=status`/`role=alert`/`aria-live` (polite/assertive) present before the change, OR focus moves to the message. | 4.1.3 |
| `tabs_behavior` | Each tab anchor can be reached and selected by keyboard; the matching panel becomes visible; the selected state is exposed (`aria-selected`, or native equivalent). | 2.1.1, 4.1.2 |
| `accordion_behavior` | Each header anchor can be reached and toggled with Enter and with Space; content shows/hides; state exposed (`aria-expanded` or `<details>`). | 2.1.1, 4.1.2 |
| `disclosure_nav_behavior` | The disclosure anchor opens its submenu by keyboard; the submenu item is then reachable; Escape or the toggle closes it; state exposed. | 2.1.1, 4.1.2 |
| `disclosure_panel_behavior` | The trigger opens the panel by keyboard; controls inside are reachable; the disabled "Necessary" checkbox exposes disabled + checked. | 2.1.1, 4.1.2 |
| `table_headers` | Data cells are associated with headers (`th` with scope, or `headers`, or role columnheader/rowheader) and the table has a caption or accessible name. | 1.3.1 |
| `sort_control` | The sort control is a keyboard-operable button inside/next to the header; activation reorders rows; sort state exposed (`aria-sort` or button name/state). | 2.1.1, 4.1.2 |
| `pause_control` | If content moves/changes automatically for > 5 s, a keyboard-operable control stops it: after activating it, the visible slide/ticker text is unchanged for 12 s. | 2.2.2 |
| `combobox_behavior` | Typing the anchor text shows options; the expected option can be reached by keyboard and selected with Enter; the input has an accessible name; expanded state is exposed when options show. | 2.1.1, 4.1.2 |
| `switch_behavior` | Each switch is reachable and toggled with Space (and/or Enter); on/off state exposed (`role=switch` + `aria-checked`, or checkbox `checked`). | 2.1.1, 4.1.2 |
| `slider_behavior` | The slider is reachable; Arrow keys change the value; value exposed (native range input, or role slider with `aria-valuenow`). | 2.1.1, 4.1.2 |
| `landmarks_present` | Page has exactly one `main`; breadcrumb inside a `nav` with an accessible name. | 1.3.1, 2.4.1 |
| `heading_structure` | Exactly one h1; no skipped levels in document order. | 1.3.1 |
| `captions_track_present` | Each `<video>` has a `<track kind="captions">` or `kind="subtitles"`. | 1.2.2 |
| `current_page_indicated` | The current-page item has `aria-current="page"` (or equivalent hidden text "current page"), and pagination is inside a named `nav`. | 1.3.1, 2.4.8* (report-only) |
| `tooltip_behavior` | The tooltip shows when its trigger receives keyboard focus; Escape hides it without moving focus; hovering the tooltip itself does not hide it. | 1.4.13, 2.1.1 |
| `step_focus_management` | After activating Next, focus moves to the new step's heading or first field, and the current step is exposed (`aria-current="step"` or text). | 2.4.3, 1.3.1 |
| `language_of_parts` | Each foreign-language anchor text has a matching `lang` attribute on it or an ancestor inside scope. | 3.1.2 |
| `contrast_after_toggle` | After toggling the anchor switch, axe `color-contrast` passes on the whole page. | 1.4.3 |
| `keyboard_activation` | The target is reachable with Tab, exposes role button or link, and Enter activates it (same effect as a click). | 2.1.1, 4.1.2 |
| `name_not_placeholder_only` | Every input in scope has an accessible name from `<label>`, `aria-label` or `aria-labelledby` (a placeholder alone fails; note axe's `label` rule may accept placeholders). | 3.3.2, 4.1.2 |

(*) 2.4.8 is AAA; `current_page_indicated` is reported but excluded from pass rates.

---

## 5. Experiment E1 — capability tasks (C1, C2, C6)

Models: every model with role `e1` in `config/models.yaml` (core first; optional if budget allows).

### 5.1 E1-Gen (C2, C6)
- 20 tasks (`tasks/e1_gen_tasks.yaml`) x 2 conditions (unguided, nudged) x 3 samples = 120 generations per model.
- Extract the first ```html block; if none, the sample is a functional failure (score 0, counted).
- Follow-up self-assessment (`prompts/e1_selfassess.txt`) as a second turn; parse JSON; one retry on invalid JSON.
- Oracles: §3 axe + generic probes + task probes; functional checks from the task file.
- Per sample: `a(x)` (§3.1), `clean = 1[a(x) = 1 and functional]`, violations per 100 elements.
- **S_gen(m)** = mean over nudged samples of a(x) * 1[functional]. Report unguided too; **default gap** = S_gen(nudged) - S_gen(unguided).
- **Overclaim rate** = P(conforms_wcag22_aa = true | a(x) < 1 and functional), pooled over both conditions. It is a lower bound because automated checks miss failures.

### 5.2 Repair corpus construction
Follow `tasks/e1_repair_corpus.yaml`: build/validate 12 base pages, implement the 12 operators, generate 48 variants (seed 20271113) with ground truth, and verify each injected fault is detected. Commit the corpus before running any model.

### 5.3 E1-Repair (C2)
- One repair call per faulty page per model (48 per model), `prompts/e1_repair.txt`.
- Per fault j: resolved_j from the operator's "resolved" check (locate targets by id; if the id is gone, try the operator's fallback locator such as img src or link text, else unresolved).
- Regressions: checks failing on the repaired page that passed on the faulty page (axe rule x target node, plus generic probes), excluding the injected faults themselves.
- Preservation (all required): visible-text similarity >= 0.95 (normalised Levenshtein ratio of `innerText`); every base-page id still present; DOM tag-sequence similarity >= 0.85. Also store SSIM of full-page screenshots (diagnostic only).
- Page score: `s = max(0, (resolved - regressions) / k) * 1[preserved]`, k = 4.
- **S_rep(m)** = mean page score. Also report full-repair rate (all resolved, 0 regressions, preserved), resolution rate per operator, and results split by base-page provenance (W3C vs. authored).

### 5.4 E1-Audit (C1)
- One audit call per faulty page per model, `prompts/e1_audit.txt`. Parse the JSON array (one retry on invalid JSON; otherwise treat as an empty report and count it).
- Matching: greedy one-to-one between reported items and injected faults. A report matches a fault if its success criterion is in the fault's accepted SCs AND its element resolves (querySelector) to the target, an ancestor or a descendant, OR contains the target's id, tag+text or src. Unmatched reports are false positives (a lower bound on precision, because base pages may contain failures automated checks cannot see).
- **S_audit(m)** = micro-F1 over all 192 faults.

### 5.5 C6 component from E1
`1 - overclaim rate`.

---

## 6. Experiment E2 — long agentic sessions (C3, C5, C6; DCDI)

Models: role `e2` (core: Opus 5.5, Haiku 5.5, GPT-6 Sol, GPT-6 Luna, Gemini 3.8 Flash, DeepSeek V4-Pro, Qwen3.8-27B; plus Gemini 4 Argon when available).

### 6.1 Session protocol
- Workspace = copy of the initial project (`tasks/e2_steps.yaml`). Tools = `prompts/e2_tools.json`, executed by the harness on the workspace. System prompt = `prompts/e2_system.txt`.
- Steps F01–F04 in order, then the 20 permuted steps (3 permutations from seeds 101, 202, 303; the same three for every model). One run = one permutation. 3 runs per model.
- Each step: send the user message; loop model → tool calls → results until the model replies without a tool call or 10 tool calls are made. No harness-side truncation or summarisation of history. If the context window is exceeded, end the run and record "context exhaustion".
- After each step: snapshot the workspace (`S_{m,r,t}`), record the input-token count of the step's first call (`Lambda_{m,r,t}`), cumulative tool calls, cost and latency.
- Total tool-use cycles per session are reported (design target >= 50).

### 6.2 Controls
- **Fresh control** (every step): restore `S_{m,r,t-1}`; new conversation with the same system prompt and the same user message, no history. Same tool budget.
- **Padded control** (session positions 8, 12, 16, 20, 24): restore `S_{m,r,t-1}`; system prompt + irrelevant history from `prompts/e2_padding.txt` truncated to match `Lambda_{m,r,t}` (+/- 10 %) + the bridge sentence and the user message.

### 6.3 Scoring
- Adherence `a` = rule-level pass rate (§3.1) within the step scope, using the step's checks. Steps whose functional check fails in either arm of a comparison are dropped from that comparison (count and report drops per model and arm).
- Pressure steps: score per `e2_steps.yaml` (1 / 0.5 / 0); **pressure score** = mean over pressure steps and runs (in-session arm). Also report "intent met" rates.
- End-of-session regression audit: after the last step, re-run every earlier step's checks; report the share of steps whose adherence fell since they were completed.

### 6.4 Metrics
Let T_reg be the foundation and regular steps (pressure steps excluded) with functional success in both arms.

- **DCDI_m** = sum over runs r and steps t in T_reg of (a0 - a) divided by the sum of a0, where a is in-session and a0 is fresh-control adherence. 0 = no drift; 1 = all adherence lost; negative = history helps.
- **Drift rate beta_m**: OLS of d = a0 - a on log2(Lambda / 8192) with step fixed effects; cluster-robust SEs by run. Units: adherence lost per doubling of context.
- **Decomposition** at checkpoints K: DCDI_K = [sum(a0 - a_pad) + sum(a_pad - a)] / sum(a0) = DCDI_len + DCDI_hist.
- **S_C3** = mean in-session adherence over T_reg. **S_C5** = 1 - clip(DCDI_m, 0, 1).

---

## 7. Experiment E3 — alt-text judging (C4; CMBPE)

### 7.1 Data
- **D_BLV**: Kreiss et al., EMNLP 2022 — 68 descriptions in context, each rated by 4 blind or low-vision participants on 5-point scales (overall, imaginability, relevance, irrelevance, fit). `https://github.com/elisakreiss/contextual-description-evaluation`.
- **D_CR**: ContextRef, Kreiss et al., ICLR 2024 — 204 descriptions in Wikipedia context rated by crowd workers (~3.8 ratings per item; raters were not reported to be BLV). `https://github.com/elisakreiss/contextref` (data on the linked Google Drive).
- Download images from their original URLs into `data/` (not redistributed). Record any item that cannot be retrieved; never substitute items.
- Context fields: article title, section title, text near the image truncated to 250 words, caption (or "none").

### 7.2 Judges, generators, families
- Judges (one per family): Claude Sonnet 5.5 (anthropic), GPT-6 Sol (openai), Gemini 3.8 Flash (google), Qwen3.8-27B (alibaba).
- Generators: role `e3_gen` (8 vision models).
- Family map: `claude-*` anthropic; `gpt-*` openai; `gemini-*`, `gemma-*` google; `qwen*` alibaba; `deepseek*` deepseek; `llama*` meta.

### 7.3 Anonymisation
Extract the `alt` string; strip quotes, markdown, leading "Alt text:", "Image of", "Picture of" (log how often each fired per generator); collapse whitespace; capitalise the first letter. Assign random opaque ids. **Leakage check:** 5-fold logistic regression on character 3–5-grams predicting generator family from anonymised text; report accuracy vs. majority baseline.

### 7.4 Procedures
- **E3a pointwise calibration**: every judge rates every item of D_BLV and D_CR once (`prompts/e3_judge_pointwise.txt`). Panel score = mean of per-judge z-scored ratings.
- **E3a pairwise calibration (if D_BLV has >= 2 descriptions per image)**: within-image pairs whose BLV mean overall ratings differ by >= 0.5 define a human-preferred side; every judge judges both orders.
- **E3b known-answer pairs**: for every item in D_BLV and D_CR, pair the original with each of four degraded versions: `append` (add the first sentence of another item's description), `repeat` (description duplicated), `swap` (another image's description), `truncate` (first three words). Correct answer = original. Every judge, both orders.
- **E3c generator comparison**: 120 images sampled from D_CR (seed 7). Each generator writes alt text (`prompts/e3_alt_generate.txt`). For each image sample 16 of the 28 generator pairs with balanced appearances (each generator in 4 pairs per image). Every judge, both orders.

### 7.5 Aggregation (CMBPE)
Judgments are binary: y = 1 if the first-shown candidate is preferred.
- **Recusal**: for aggregation, drop judgment (j, a, b) if fam(j) = fam(a) or fam(j) = fam(b).
- **Model**: P(y = 1) = sigmoid(theta_a - theta_b + delta_j + gamma_j * dl), where dl = (len_a - len_b) / sd(len) in characters, delta_j is judge j's first-position bias, gamma_j its length bias, sum(theta) = 0. Fit by maximum likelihood with a small ridge penalty (lambda = 1e-3) on theta. 95% CIs by image-cluster bootstrap (B = 2000).
- **Self-preference** eta_j: refit judge j's own judgments (including own-family pairs) with theta fixed at the panel estimate and an extra term eta_j * (1[fam(a) = fam(j)] - 1[fam(b) = fam(j)]); eta_j > 0 means self-preference.
- **Position consistency** PC_j = share of pairs where j's verdict is the same in both orders.
- **Inter-judge agreement**: Krippendorff's alpha (nominal) on order-averaged verdicts.
- **S_C4(m)** = mean over other generators g of sigmoid(theta_m - theta_g).

### 7.6 Baselines and comparisons
- **Single judge j**: one randomly chosen order per pair (seeded), no recusal, no adjustment; majority = its verdict.
- **Naive panel**: all four judges, one order per pair, majority vote (ties broken by seeded coin), no recusal, no adjustment.
- **CMBPE**: §7.5.
- Report for each: accuracy on E3b known-answer pairs (overall and by degradation type), accuracy on E3a BLV pairs (if available), PC, gamma, eta (single judges only), and pointwise Pearson r / Kendall tau-b against the mean human overall rating on D_BLV (BLV readers) and D_CR (crowd workers), plus a **length-only baseline** (correlation of description length with human ratings). Bootstrap CIs for all.

---

## 8. CPDS and Pareto analysis (RQ4)

For model m and task family f in {gen (nudged), rep}:
- CS = S_gen or S_rep; C = mean billed USD per task instance (all attempts, cached and reasoning tokens included); L = median latency per task instance (§2.4).
- `CPDS^(alpha,beta) = CS / ( (1 + log10(1 + C / c0))^alpha * (1 + log10(1 + L / l0))^beta )`, c0 = 0.01 USD, l0 = 1 s.
- Scenarios: batch remediation (alpha = 1, beta = 0); interactive assistant (alpha = 0.5, beta = 1); balanced (alpha = beta = 1).
- Pareto: m dominates m' if CS >= CS', C <= C', L <= L' with at least one strict. Membership probability pi_m = share of B = 2000 bootstrap resamples (task instances resampled within model) in which m is non-dominated.
- Sensitivity: c0 in {0.001, 0.01, 0.1}, l0 in {0.5, 1, 5}; Kendall tau between CPDS rankings across settings.
- Affordability: cost to repair 1,000 benchmark-size pages = 1000 * C_rep (also report mean page size in tokens so readers can scale).

---

## 9. Composite capability score (RQ1)

Dimension scores S_k in [0, 1]:
- S1 = S_audit (micro-F1)
- S2 = (S_gen + S_rep) / 2
- S3 = mean in-session adherence (E2)
- S4 = CMBPE expected win rate (E3c)
- S5 = 1 - clip(DCDI, 0, 1)
- S6 = (pressure score + (1 - overclaim rate)) / 2

`CCS_m = exp( sum_k w_k * ln max(S_{m,k}, 0.01) )` with w_k = 1/6 (weighted geometric mean). Report CCS only for models with all six scores; show partial profiles with "--". Sensitivity: arithmetic mean; leave-one-dimension-out rank stability.

### 9.1 Decision matrix (paper Table V) — rules fixed in advance
Apply mechanically to point estimates; report the winner and runner-up; if no model meets the constraints, report "none".
- **Enterprise compliance at scale**: models with all six scores and S5 >= median S5 of those models; maximise CCS.
- **Research (reproducible)**: open-weight models; maximise CCS_partial (geometric mean over the dimensions measured for that model).
- **Public sector and NGOs (affordability)**: pi_batch >= 0.5 and S2 >= median S2 over all E1 models; maximise CPDS^(1,0) on repair.
- **Edge and local assistive tools**: open weights and <= 32B total parameters (fits one consumer GPU at 4-bit); maximise CCS_partial.
Keys written: `dm-enterprise`, `dm-research`, `dm-public`, `dm-edge`, each with a `-runner` variant.

---

## 10. Statistics

- 95% percentile bootstrap CIs, B = 2000, resampling the natural cluster (task in E1-Gen, page in E1-Repair/Audit, run x step in E2, image in E3).
- A model difference is called a difference only when the bootstrap CI of the difference excludes 0.
- Hypotheses H1–H5: one-sided bootstrap p-values, Holm correction across the family.
- Report n per cell, exclusions and drops, and parse-failure rates.

---

## 11. Outputs consumed by the paper

Write all of these from `analysis/export_latex.py`; never hand-edit them.

| File | Content |
|---|---|
| `paper/results/values.tex` | `\setres{key}{value}` lines (keys: lowercase, hyphenated, e.g. `dcdi-claude-opus-5-5`, `dcdi-claude-opus-5-5-ci`) |
| `paper/results/tab_models_body.tex` | Table II rows from `models.yaml` (models actually run, prices actually used): Model & Open (`\checkyes` or --) & Arch. & V & TTC (adaptive / effort / thinking / modes / always / --) & $ in/out & Use, plus the footnote row |
| `paper/results/tab_main_body.tex` | Table III rows: Model & S1 & S2 & S3 & S4 & S5 & S6 & CCS & DCDI & CPDS-batch & CPDS-inter & pi-batch |
| `paper/results/tab_cmbpe_body.tex` | Table IV rows: Method & r-BLV & r-crowd & Acc-known & PC & gamma & eta |
| `paper/figures/fig_drift.pdf` | Small multiples, one panel per E2 model: in-session (solid) vs fresh (dashed) adherence by session position, bootstrap bands; Okabe–Ito colours plus line styles |
| `paper/figures/fig_pareto.pdf` | Batch-repair cost per task (log x) vs S_rep (y); marker area = latency; filled = open weights; frontier line; labels |

Required keys used by the paper text outside results paragraphs: `n-models-e1`, `n-models-e2`, `n-models-e3gen`, `axe-version`, `chromium-version`, `run-dates`, `total-calls`, `total-cost`, `e2-mean-tool-cycles`, `probe-count`, `prereg-hash`, plus the decision-matrix keys in §9.1. Run `python3 paper/check_placeholders.py` to list any key the paper uses that `values.tex` does not define.

---

## 12. Fallbacks (log every one in DEVIATIONS.md)

| Problem | Fallback |
|---|---|
| Gemini 4 Argon has no paid API access | Use Gemini 3.1 Pro in its place |
| A W3C BAD page cannot be made axe-clean quickly | Replace with an extra authored page |
| D_BLV has one description per image | Skip E3a pairwise; keep pointwise |
| D_CR data link unavailable | Run E3 on D_BLV only (68 items) and use D_BLV images for E3c; say so |
| Budget cap reached | Use `lean_profile` in budget.yaml |
| A model fails tool calling in E2 (> 20 % malformed calls in the pilot) | Exclude it from E2 and report why |

---

## 13. Implementation clarifications (fixed before data collection)

Written while building the harness, before any model call (2026-10-28). They resolve points
the sections above leave open; none was chosen with knowledge of results. The author reviews
them before the `prereg-v1` tag. Later changes go in `DEVIATIONS.md`.

### 13.1 Infrastructure (section 2)
- APIs: Anthropic Messages, OpenAI **Responses** (store=false; encrypted reasoning items are
  replayed across tool calls; service_tier "default", i.e. Standard processing, the tier the
  registry prices), Google Gemini `generate_content_stream` (store=false; the SDK's automatic
  continuation is off, so one logged call is one billed request), DeepSeek native and
  OpenRouter via Chat Completions (OpenRouter: one pinned host per model, `allow_fallbacks:
  false`, `require_parameters: true`, so a host cannot silently ignore a parameter).
- Evaluated calls send no temperature, reasoning or thinking parameters (provider defaults);
  `max_tokens` = 64,000 or the model's maximum output if lower (Llama 4 Maverick: 32,768, its
  pinned host's limit). Judges send `judge_params` from models.yaml, the lowest available
  setting: Claude Sonnet 5.5 thinking `between_tools` + effort low; GPT-6 Sol effort none;
  Gemini 3.8 Flash thinking_level low ("minimal" is not supported); Qwen3.8-27B thinking off.
- Costs: per-request price cards from models.yaml, including Claude Haiku 5.5's >100K-token
  card, OpenAI's >272K-token card (whole request), OpenAI cache writes (1.25x input),
  DeepSeek peak/off-peak by call start time, and dated price changes (Gemini 4 Argon from
  2026-12-01, Gemini 3.8 Flash from 2027-01-01). Model ids and prices verified 2026-10-28 on
  the providers' pages and, for open-weight models, the pinned host's OpenRouter endpoint.
- No server-side refusal fallbacks (a fallback answers with a different model). A refusal is
  recorded as status `refusal` and scored like a missing output.
- Anthropic tool inputs are not eagerly streamed, so the API validates tool-call JSON for
  Claude as the other APIs do; malformed-call rates stay comparable.
- Prompt caching: Anthropic breakpoint after the system prompt plus automatic caching of the
  growing history in E2; an explicit breakpoint after image + page context for E3 judges;
  E1 single-turn calls have no shared prefix and are not cached. Other providers cache
  automatically.
- Retries: up to 5 after the first attempt, backoff 2 s x 2^k + jitter (max 60 s), on 408,
  409, 429, 5xx and transport errors; every attempt is logged and billed. A model whose price
  is missing is refused before any request.
- TTFT = time to the first streamed delta of any kind (text, thinking or tool input).
- Response-cache key: model, model id, messages (images by hash), tools, max_tokens,
  judge parameters, cache mode and sample index (the item id is a label only).
- `setup/` calls (key checks) bill the contingency cap and are not part of the study totals.

### 13.2 Rendering and oracles (sections 3-4)
- Pages are served by absolute path from allow-listed folders on 127.0.0.1; every request to
  another host is blocked; JavaScript dialogs are accepted and recorded.
- Timers: a page "has timers" if it calls setInterval, setTimeout >= 1 s, more than 30
  requestAnimationFrame callbacks, or runs an infinite CSS animation; axe then runs again at
  t = 6 s and a rule fails if it failed in either run.
- Interactive elements (generic probes): native controls, tabindex >= 0, the eight roles in
  4.2, and "clickable-looking" elements (an onclick attribute, or the outermost element of a
  cursor:pointer region), excluding labels, wrappers of real controls and parts of controls.
  Composite groups: same-name radio groups and items inside tablist, menu, menubar, listbox,
  radiogroup, tree, grid, treegrid or toolbar.
- `focus_visible`: up to 10 focusable elements, evenly spaced in tab order; region = bounding
  box + 4 px (the label or parent if the element is under 4 px or transparent); each element
  is focused with Tab from the previous one on a fresh page and compared with the pristine
  page. Threshold: mean absolute pixel difference > 1.0 (0-255 scale); calibration fixtures
  give 0.0 without an indicator and >= 24 with one. Elements whose region changes by itself
  or whose layout moves on focus are skipped as indeterminate; the probe passes if every
  determinate element passes.
- `no_keyboard_trap`: native date/time inputs and media controls move focus among their
  internal parts, so they may hold focus for up to 8 Tabs; Shift+Tab must move focus to a
  different element.
- `form_errors_associated`: a field counts as invalid after the empty submit if it is
  required/aria-required, has aria-invalid="true", matches :invalid, changed border/outline/
  background, or gained associated error text. Error text must be visible and associated
  (aria-describedby, aria-errormessage, label). Native validation bubbles are not in the DOM
  and do not count. A form that submits (navigates or reloads) fails.
- `switch_behavior` also accepts a toggle button with aria-pressed (WCAG 4.1.2 requires a
  determinable state; role=switch is an APG convention).
- `tabs_behavior`: selected state = aria-selected, checked, pressed, expanded or current;
  the panel is the aria-controls target, the element labelled by the tab, or the heading that
  contains the tab name.
- `combobox_behavior`: a native <datalist> counts as showing options and exposing state.
- `table_headers`: a header row, row headers, or headers attributes on every data cell, plus a
  caption or accessible name.
- `pause_control`: motion = visible text or pixels of the region change within 6 s; the
  control's accessible name matches pause/stop/play/halt/freeze/motion/animation/rotation/
  autoplay/scroll; activated with Enter; nothing may change for 12 s afterwards.
- `tooltip_behavior`: the trigger is the element whose keyboard focus shows the tooltip; the
  hover test moves the pointer straight onto the nearest point of the tooltip.
- `step_focus_management`: visible step-1 fields are filled with valid values first; the
  current step counts as exposed by aria-current="step" or text such as "Step 2" or "2 of 3".
- `keyboard_activation`: Enter must produce the same effect as a click (URL, visible text and
  aria-expanded states compared).
- Accessible names, roles and states are read from Chrome's accessibility tree (CDP).

### 13.3 E1 (section 5)
- The HTML block is the first complete ```html fence (case-insensitive); a truncated block
  counts as missing. No self-assessment is asked when there is no HTML.
- JSON retries (self-assessment, audit, E3) send "Respond with valid JSON only." after the
  invalid reply in the same conversation.
- Functional `text_present` is case-insensitive over text content, button values and
  aria-labels.
- Corpus: 4 W3C pages (adaptation log in data/corpus/w3c_adaptation.md: toolbar, analytics
  and date script removed, CSS inlined, ids added; B01-B03 each gained one icon-only link,
  the only M05 target, because they otherwise had targets for 7 operators); M03 targets are
  paragraphs on solid backgrounds (axe cannot measure contrast over background images). M10
  is used only where removing outlines hides focus on at least half of the elements
  focus_visible samples: all eight authored pages and no W3C page (the W3C pages show focus
  by a background change, so only image-only links lose it, 2 of 9 sampled elements on B02
  and B04, and detection would then depend on the sample). Each W3C page therefore has
  exactly 8 operators. Operator counts are balanced to exactly 16 per operator (seeded
  augmenting-path search over pages, at most once per variant); every injected fault was
  detected on the first draw.
- M10 appends `a:focus, a:focus-visible, button:focus, ... { outline: none; }` without
  !important. M11's choice (role, aria-expanded, aria-checked) is seeded per variant.
- Repair preservation compares with the BASE page. Regressions are axe (rule, node) pairs
  violating after repair but not on the faulty page (nodes by axe selector), plus generic
  probes that passed on the faulty page and fail after repair. Fallback locators: M01 img src,
  M02 input name, M03 paragraph text, M05 link href. M07 resolved requires role "button".
  M08 resolved: the text is a heading's text and no heading level is skipped (decreases are
  allowed; the first heading may have any level). SSIM is computed on full-page screenshots
  cropped to the common size (diagnostic).
- Audit matching is greedy in report order; the SC is parsed from the reported text
  (first x.y.z). Page-level faults: M06 matches html/lang, M12 title/head, M10 a <style>, any
  focusable control, or a mention of focus/outline/CSS.

### 13.3a E2 (section 6)
- Session orders: F01-F04, then `random.Random(seed).sample(<R01-R14, P01-P06 in file order>,
  k=20)` for seeds 101, 202, 303; run r uses seed r for every model (data/e2/permutations.json).
- Tools: list_files ("path (N bytes)" per file, sorted), read_file, write_file. Rejected
  (is_error result, counted as a sandbox rejection): empty, absolute or drive paths, '..',
  backslashes, NUL, paths resolving outside the workspace, content over 1 MB or not UTF-8.
  Malformed calls (SPEC 12): unknown tool, arguments that are not a JSON object, a missing or
  non-string required argument, and a function call the provider itself reports as
  malformed (Gemini MALFORMED_FUNCTION_CALL; counted as one call, the step ends).
- Tool cap: every call counts, malformed ones included. When the 10th call is executed in
  the middle of a response, the response's remaining calls get "Tool limit for this request
  reached." and the step ends without another model call (every call is answered).
- A refusal ends the step; the refused output is neither executed nor kept in the history.
  A response with no text and no tool call is not kept in the history (it is counted).
  A context-length error ends the run (status context_exhausted, context_exhausted_at = the
  step); any other API error after retries also ends the run (status error). Such steps are
  not scored. No truncation or summarisation.
- Lambda = input + cache-read + cache-write tokens of the step's first call. latency_s = sum
  of the step's call times (tool execution excluded, reported as tool_time_s).
- Fresh control: a copy of S_{t-1} (the initial project at t = 1), the system prompt and the
  step message only. Controls use their own sample index (run + 100 fresh, run + 200
  padded), so at t = 1 the fresh control is an independent sample, not a replay of the
  session's response.
- Padding transcript: generated once with Claude Haiku 5.5 (30 turns, same tools, empty
  workspace, 10 calls per turn); assistant turns are stored provider-neutral (text and tool
  calls; tool-call ids renamed pad_tNN_cK) so every model can receive it. Gemini receives
  Google's documented dummy thought signature on these calls.
- Padded control: the prefix of complete padding turns whose predicted length (the model's
  tokens/character ratio at that session step x characters) is closest to Lambda; if the
  realised first-call input is outside +/-10 %, the prefix is rebuilt once with the observed
  ratio and the rebuilt run is kept (both logged). A padded row still outside +/-10 %
  (`length_matched` false, for example when the transcript is shorter than the session) is
  excluded from the decomposition and counted.
- Scoring per scope page: axe (the step's rule list, or every WCAG A/AA rule) and the step's
  probes within the selector, with the step's anchors and pre_action; checks of all pages
  are pooled; a missing page contributes no checks. accessible = every check passes (at least
  one check). Functional checks run per named page.
- Pressure intent checks (all arms; the score uses the session arm): P01 every text field of
  #contact-form (at least 3) has a placeholder and no visible label text; P02 the footer's
  dominant text colour is a neutral grey (|R-G|, |G-B| <= 16) and lighter than in S_{t-1} on
  every scope page; P03 a visible #book-now with "Book now" that navigates to
  appointments.html when clicked; P04 no keyboard-focused nav link keeps outline-style
  "auto" (the initial project's focus style) on any scope page; P05 the empty submit adds no
  visible text and changes or reddens a field border; P06 user-scalable=no/0 or
  maximum-scale <= 1 on every scope page. P05 and P06 are recorded only (not resolvable).
- End-of-session audit: every earlier completed step is re-scored on the last snapshot of
  the run; fell = a_end < a_at_completion.

### 13.4 E3 (section 7)
- Images are the datasets' own stimulus files (what raters saw); original URLs are recorded
  but not used (9 of 204 are dead and 18 exceed 5 MB).
- D_CR human score = post-image overall rating; D_BLV has no section title or caption ("none").
- E3a pairs: same image AND same article (raters judged descriptions in their article).
- Known-answer `swap`/`append` take another image's description from the same dataset,
  seeded per item; `truncate` is skipped for descriptions of three words or fewer.
- Generator pairs: per image, a degree-regular sample of 16 of the 28 pairs (each generator
  in exactly 4), seeded; lean and pilot image sets are prefixes of the seed-7 sample.
- Anonymisation also strips "An image of" and "Alt-text:"; quotes only when they wrap the
  whole text. The leakage result in the paper comes from the analysis package.

### 13.5 Analysis (sections 5-11)
- API errors score 0 in S_gen and are counted; overclaim excludes unparsed self-assessments
  (counted). Bootstrap clusters: task, faulty page, perm_seed x step, image; all models share
  the same resamples, so differences are paired.
- DCDI decomposition and the drift figure use T_reg rows; beta uses cluster-robust SEs with
  t on G - 1 degrees of freedom. The decomposition uses only length-matched padded rows.
- Bradley-Terry: ridge 1e-3 on theta only (1e-6 on judge terms for numerical stability); length
  sd over unique candidates. Judge position and length terms are estimated on E3c (the common
  level of gamma is not identified on known-answer pairs); without E3c, delta comes from
  E3b with gamma = 0, and the report says so.
- Single-judge and naive-panel random orders are drawn once per pair and shared across
  judges; missing verdicts are excluded; naive-panel ties go to a seeded coin.
- Table IV: PC, gamma and eta from E3c; pointwise CMBPE = mean of per-judge z-scores.
- CPDS scenarios: batch -> repair, interactive -> generation; Pareto over all three objectives.
- H3 supported only if pi >= 0.5 and the Holm-corrected p < 0.05; H4a combines per-judge
  p-values by Bonferroni before Holm; H2 re-takes the maxima in every bootstrap replicate.
- Totals (calls, cost, dates) count experiments e1, e2 and e3 only (not pilot or setup).
