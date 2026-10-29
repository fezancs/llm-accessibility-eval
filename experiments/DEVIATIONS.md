# Deviations from the pre-registered plan

Pre-registration tag: `prereg-v1` (commit: _fill in when tagged_)

Record every change to metrics, prompts, tasks, models, exclusion rules or analysis
made after the tag, BEFORE re-running anything. Each entry is reported in the paper's
threats-to-validity paragraph.

| Date | What changed | Why | Expected effect on results |
|---|---|---|---|
| 2026-10-29 | Added a `local` provider and three self-hosted models (`local-qwen3-8b`, `local-qwen25vl-7b`, `local-gemma3-4b`) that run on an 8 GB consumer GPU via Ollama, at $0. See `docs/LOCAL_SETUP.md`. | No budget for paid API tokens; the only free way to run E2 (whose long contexts exceed every free cloud tier's tokens-per-minute cap) is on local hardware. | A local-only run evaluates small open-weight models instead of the frontier closed models. It cannot test GPT-6, Gemini 4, Claude 5.5 or DeepSeek V4-Pro, so cross-vendor and "best closed vs. open" claims (e.g. H2) are out of scope for that run. Local vision models give weaker alt text, lowering C4/E3 scores. The paper must state that results are for 7–8B open-weight models self-hostable on 8 GB hardware, not the frontier. This is an added capability, not a change to any metric, prompt, task or scoring rule. |
