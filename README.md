# SE4All 2027 submission package

**Paper:** *Beyond Static Leaderboards: Drift-Aware, Bias-Controlled, and Cost-Normalized Evaluation of LLMs for Accessible Software Engineering*
**Venue:** SE4All 2027 (ICSE 2027 workshop on software engineering and AI for accessibility and inclusion), Dublin, 1 May 2027
**Format:** IEEE conference (IEEEtran, 10 pt), regular paper, **max 8 pages including references**, single-blind (your name goes on the paper)
**Deadline:** **13 November 2026, 23:59 AoE** = **14 November 2026, 16:59 Pakistan time**. Submit at https://icse2027-se4all.hotcrp.com/

## What is in this folder

```
paper/                      the paper (compiles to main.pdf, currently 7 pages)
  main.tex                  full text; results parts are placeholders
  refs.bib                  47 references, venue details checked on 28 Oct 2026
  macros.tex                placeholder and results macros (\res, \todo, \tbd, \resultpara)
  results/                  table bodies and values.tex, overwritten by the analysis
  figures/                  empty until the analysis draws the two figures
  check_placeholders.py     pre-submission checker
  IEEEtran.cls / .bst       IEEE template v1.8b
experiments/                everything Claude Code needs to run the study
  CLAUDE.md                 step-by-step instructions Claude Code reads automatically
  SPEC.md                   the method in full: metrics, procedures, statistics
  config/models.yaml        models, IDs, prices (28 Oct 2026; re-check on run day)
  config/budget.yaml        spend caps, full and lean profiles
  tasks/                    20 generation tasks, repair corpus and mutation operators, 24-step agent script
  prompts/                  every prompt, verbatim
```

## How the draft handles results

No numbers were invented. Every result in the paper is a placeholder that fills itself in when the analysis runs:
- numbers in the text use `\res{key}` and print **TBD** until `results/values.tex` defines them;
- the three result tables read their rows from `results/*.tex`;
- the two figures show grey boxes until `figures/*.pdf` exist;
- red boxes and **[TODO: ...]** mark the few paragraphs that must be written after the results exist.

If you set `\finaltrue` in `main.tex`, any leftover placeholder stops the build with an error, so a draft cannot be submitted by accident.

## How to run the experiments with Claude Code

1. **Get API keys** for Anthropic, OpenAI, Google (Gemini API), DeepSeek, and OpenRouter (for Qwen, Gemma and Llama). Put them in `experiments/.env` (Claude Code creates `.env.example` for you).
2. **Run Claude Code on your own computer** (recommended): open a terminal in `experiments/`, start `claude`, and say: *"Read CLAUDE.md and SPEC.md, then start Phase 0."* Claude Code works through the phases and stops for your approval after the pilot, before the expensive runs.
   - Cloud sessions also work, but their default network access only reaches package registries and GitHub. You would need Custom network access for the model API hosts and the API keys added as network secrets, and long commands time out after 2–10 minutes by default. Running locally avoids all of that.
3. **Plan** (each phase is described in `CLAUDE.md`):

| Dates | Phase |
|---|---|
| 29–30 Oct | Set up, build the harness, validate probes and the repair corpus, freeze the plan (git tag `prereg-v1`) |
| 31 Oct | Pilot on cheap models; you approve the projected cost |
| 1–5 Nov | Full runs: E1, then E3, then E2 (the longest) |
| 6 Nov | Analysis, tables and figures |
| 7–8 Nov | Claude Code writes the results text into the paper |
| 9–11 Nov | Your review and edits; final build; page check |
| 12 Nov | Submit (one day of slack before the deadline) |

4. **Budget:** about **$250–450** for the full study including a pilot and re-runs, or **about $120** with the lean profile in `config/budget.yaml`. The pilot produces a real per-call cost table before you commit.

## Before you submit

- [ ] Author names, affiliations and email in `main.tex` (two `\todo`s) and `pdfauthor`
- [ ] `python3 check_placeholders.py` reports nothing left
- [ ] `\finaltrue` build succeeds; `main.pdf` is 8 pages or fewer, references included
- [ ] The AI-use statement in the Acknowledgment matches what you actually did (IEEE requires disclosing AI-generated text, sections and level of use)
- [ ] Every deviation in `experiments/DEVIATIONS.md` is reported in the threats paragraph
- [ ] Data-availability link (GitHub + Zenodo DOI) filled in
- [ ] Figures use the colour-blind-safe palette with line styles, and every caption says what the figure shows (SE4All is an accessibility venue; reviewers will notice)

## What changed from the original master prompt, and why

- **Scope.** SE4All accepts only work on SE and AI for accessibility and inclusion, so the three contributions now evaluate LLMs on accessibility engineering (WCAG 2.2 code, repair, auditing, alt text). The six-dimension taxonomy is kept and mapped onto accessibility tasks (Table I).
- **Models.** GPT-4o, Claude 3.5 and Gemini 1.5 are two years old by November 2026; the study uses the October 2026 lineup (Claude 5.5 family, GPT-6 Sol/Luna, Gemini 4 Argon / 3.8 Flash, DeepSeek V4-Pro, Qwen3.8, Gemma 4, Llama 4).
- **CPDS formula.** The original divided by log10(price per 1M tokens), which is zero at $1 and negative below it, and added seconds to seconds-per-token. The paper explains the problem and uses per-task cost and latency with a positive, log-compressed penalty and scenario weights.
- **Novelty claims** are positioned against the closest prior work (juries of LLM judges, length-controlled evaluation, cost-accuracy Pareto analysis, and 2025–2026 accessibility-repair studies) rather than claimed from scratch.
- **Length.** The venue allows 8 pages including references, so the paper is written to fit with room for the results text.

## If the experiments cannot finish in time

`CLAUDE.md` ("If time runs out") gives two honest options: report what finished with the rest as future work, or convert to a 5-page vision paper (allowed by the call; such papers are judged on the strength of the idea).
