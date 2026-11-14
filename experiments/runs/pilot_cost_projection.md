# Pilot report and cost projection

Pilot calls (attempts): 40; billed: $0.00 (cap $25). Price snapshot: 2026-10-28.

## Checks

| Model | E1 no html block | self-assess unparsed | repair no html | audit unparsed | API errors (attempts) |
|---|---|---|---|---|---|
| local-qwen3-8b | -- | -- | -- | -- | 0 |

E2 tool-call validity (session and fresh arms; SPEC 12: more than 20 % malformed -> exclude from E2):

| Model | tool calls | malformed | rate | rule |
|---|---|---|---|---|
| local-qwen3-8b | 34 | 0 | 0.0% | ok |

E2 padded control: padding transcript length vs projected session length at position 24 (rough):

| Model | tokens/char | padding (tokens) | projected Lambda | covered |
|---|---|---|---|---|
| local-qwen3-8b | 0.295 | 33,624 | 31,856 | yes |

Latency captured: total_s on 40/40 successful calls; ttft_s on 40/40.

| Provider | prompt tokens | cached share | cache writes |
|---|---|---|---|
| local | 94925 | 76.9% | 0 |

Mean cost per call by component and model:

| Component | Model | units | $ per unit |
|---|---|---|---|
| e2_fresh | local-qwen3-8b | 6 | 0.0000 |
| e2_session | local-qwen3-8b | 6 | 0.0000 |

## Projection: full profile

| Exp. | Model | Unit | n | Projected $ | Basis |
|---|---|---|---|---|---|
| E1 | claude-opus-5-5 | e1_gen | 120 | -- | no pilot data |
| E1 | claude-opus-5-5 | e1_selfassess | 120 | -- | no pilot data |
| E1 | claude-opus-5-5 | e1_repair | 48 | -- | no pilot data |
| E1 | claude-opus-5-5 | e1_audit | 48 | -- | no pilot data |
| E1 | claude-sonnet-5-5 | e1_gen | 120 | -- | no pilot data |
| E1 | claude-sonnet-5-5 | e1_selfassess | 120 | -- | no pilot data |
| E1 | claude-sonnet-5-5 | e1_repair | 48 | -- | no pilot data |
| E1 | claude-sonnet-5-5 | e1_audit | 48 | -- | no pilot data |
| E1 | claude-haiku-5-5 | e1_gen | 120 | -- | no pilot data |
| E1 | claude-haiku-5-5 | e1_selfassess | 120 | -- | no pilot data |
| E1 | claude-haiku-5-5 | e1_repair | 48 | -- | no pilot data |
| E1 | claude-haiku-5-5 | e1_audit | 48 | -- | no pilot data |
| E1 | gpt-6-sol | e1_gen | 120 | -- | no pilot data |
| E1 | gpt-6-sol | e1_selfassess | 120 | -- | no pilot data |
| E1 | gpt-6-sol | e1_repair | 48 | -- | no pilot data |
| E1 | gpt-6-sol | e1_audit | 48 | -- | no pilot data |
| E1 | gpt-6-luna | e1_gen | 120 | -- | no pilot data |
| E1 | gpt-6-luna | e1_selfassess | 120 | -- | no pilot data |
| E1 | gpt-6-luna | e1_repair | 48 | -- | no pilot data |
| E1 | gpt-6-luna | e1_audit | 48 | -- | no pilot data |
| E1 | gemini-4-argon | e1_gen | 120 | -- | no pilot data |
| E1 | gemini-4-argon | e1_selfassess | 120 | -- | no pilot data |
| E1 | gemini-4-argon | e1_repair | 48 | -- | no pilot data |
| E1 | gemini-4-argon | e1_audit | 48 | -- | no pilot data |
| E1 | gemini-3-8-flash | e1_gen | 120 | -- | no pilot data |
| E1 | gemini-3-8-flash | e1_selfassess | 120 | -- | no pilot data |
| E1 | gemini-3-8-flash | e1_repair | 48 | -- | no pilot data |
| E1 | gemini-3-8-flash | e1_audit | 48 | -- | no pilot data |
| E1 | deepseek-v4-pro | e1_gen | 120 | -- | no pilot data |
| E1 | deepseek-v4-pro | e1_selfassess | 120 | -- | no pilot data |
| E1 | deepseek-v4-pro | e1_repair | 48 | -- | no pilot data |
| E1 | deepseek-v4-pro | e1_audit | 48 | -- | no pilot data |
| E1 | qwen3-8-27b | e1_gen | 120 | -- | no pilot data |
| E1 | qwen3-8-27b | e1_selfassess | 120 | -- | no pilot data |
| E1 | qwen3-8-27b | e1_repair | 48 | -- | no pilot data |
| E1 | qwen3-8-27b | e1_audit | 48 | -- | no pilot data |
| E1 | gemma-4-31b | e1_gen | 120 | -- | no pilot data |
| E1 | gemma-4-31b | e1_selfassess | 120 | -- | no pilot data |
| E1 | gemma-4-31b | e1_repair | 48 | -- | no pilot data |
| E1 | gemma-4-31b | e1_audit | 48 | -- | no pilot data |
| E1 | llama-4-maverick | e1_gen | 120 | -- | no pilot data |
| E1 | llama-4-maverick | e1_selfassess | 120 | -- | no pilot data |
| E1 | llama-4-maverick | e1_repair | 48 | -- | no pilot data |
| E1 | llama-4-maverick | e1_audit | 48 | -- | no pilot data |
| E1 | local-qwen3-8b | e1_gen | 120 | -- | no pilot data |
| E1 | local-qwen3-8b | e1_selfassess | 120 | -- | no pilot data |
| E1 | local-qwen3-8b | e1_repair | 48 | -- | no pilot data |
| E1 | local-qwen3-8b | e1_audit | 48 | -- | no pilot data |
| E1 | local-qwen25vl-7b | e1_gen | 120 | -- | no pilot data |
| E1 | local-qwen25vl-7b | e1_selfassess | 120 | -- | no pilot data |
| E1 | local-qwen25vl-7b | e1_repair | 48 | -- | no pilot data |
| E1 | local-qwen25vl-7b | e1_audit | 48 | -- | no pilot data |
| E2 | claude-opus-5-5 | session x3 runs | 72 | 32.83 | token-projected (all pilot models); x2 for longer sessions |
| E2 | claude-opus-5-5 | fresh x3 runs | 72 | 20.44 | token-projected (all pilot models); x2 for longer sessions |
| E2 | claude-opus-5-5 | padded x3 runs | 15 | 6.84 | token-projected (all pilot models); x2 for longer sessions |
| E2 | claude-haiku-5-5 | session x3 runs | 72 | 0.83 | token-projected (all pilot models); x2 for longer sessions |
| E2 | claude-haiku-5-5 | fresh x3 runs | 72 | 0.51 | token-projected (all pilot models); x2 for longer sessions |
| E2 | claude-haiku-5-5 | padded x3 runs | 15 | 0.17 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gpt-6-sol | session x3 runs | 72 | 16.54 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gpt-6-sol | fresh x3 runs | 72 | 10.26 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gpt-6-sol | padded x3 runs | 15 | 3.45 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gpt-6-luna | session x3 runs | 72 | 0.83 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gpt-6-luna | fresh x3 runs | 72 | 0.51 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gpt-6-luna | padded x3 runs | 15 | 0.17 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gemini-4-argon | session x3 runs | 72 | 16.41 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gemini-4-argon | fresh x3 runs | 72 | 10.22 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gemini-4-argon | padded x3 runs | 15 | 3.42 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gemini-3-8-flash | session x3 runs | 72 | 6.20 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gemini-3-8-flash | fresh x3 runs | 72 | 3.85 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gemini-3-8-flash | padded x3 runs | 15 | 1.29 | token-projected (all pilot models); x2 for longer sessions |
| E2 | deepseek-v4-pro | session x3 runs | 72 | 3.33 | token-projected (all pilot models); x2 for longer sessions |
| E2 | deepseek-v4-pro | fresh x3 runs | 72 | 2.09 | token-projected (all pilot models); x2 for longer sessions |
| E2 | deepseek-v4-pro | padded x3 runs | 15 | 0.69 | token-projected (all pilot models); x2 for longer sessions |
| E2 | qwen3-8-27b | session x3 runs | 72 | 3.04 | token-projected (all pilot models); x2 for longer sessions |
| E2 | qwen3-8-27b | fresh x3 runs | 72 | 1.87 | token-projected (all pilot models); x2 for longer sessions |
| E2 | qwen3-8-27b | padded x3 runs | 15 | 0.63 | token-projected (all pilot models); x2 for longer sessions |
| E2 | local-qwen3-8b | session x3 runs | 72 | 0.00 | measured, linear in position |
| E2 | local-qwen3-8b | fresh x3 runs | 72 | 0.00 | measured, linear in position |
| E2 | local-qwen3-8b | padded x3 runs | 15 | 0.00 | measured, linear in position |
| E3 | local-qwen25vl-7b | e3_pointwise | 272 | -- | no pilot data |
| E3 | local-gemma3-4b | e3_pointwise | 272 | -- | no pilot data |
| E3 | local-ministral3-8b | e3_pointwise | 272 | -- | no pilot data |
| E3 | local-qwen25vl-7b | e3_known | 2116 | -- | no pilot data |
| E3 | local-gemma3-4b | e3_known | 2116 | -- | no pilot data |
| E3 | local-ministral3-8b | e3_known | 2116 | -- | no pilot data |
| E3 | claude-opus-5-5 | e3_alt | 120 | -- | no pilot data |
| E3 | claude-haiku-5-5 | e3_alt | 120 | -- | no pilot data |
| E3 | gpt-6-sol | e3_alt | 120 | -- | no pilot data |
| E3 | gpt-6-luna | e3_alt | 120 | -- | no pilot data |
| E3 | gemini-3-8-flash | e3_alt | 120 | -- | no pilot data |
| E3 | qwen3-8-27b | e3_alt | 120 | -- | no pilot data |
| E3 | gemma-4-31b | e3_alt | 120 | -- | no pilot data |
| E3 | llama-4-maverick | e3_alt | 120 | -- | no pilot data |
| E3 | local-qwen25vl-7b | e3_alt | 120 | -- | no pilot data |

| Experiment | Projected $ | Cap $ |
|---|---|---|
| E1 | 0.00 | 120 |
| E2 | 146.43 | 160 |
| E3 | 0.00 | 110 |
| Total | 146.43 | 450 (incl. contingency) |

## Projection: lean profile

| Exp. | Model | Unit | n | Projected $ | Basis |
|---|---|---|---|---|---|
| E1 | claude-opus-5-5 | e1_gen | 120 | -- | no pilot data |
| E1 | claude-opus-5-5 | e1_selfassess | 120 | -- | no pilot data |
| E1 | claude-opus-5-5 | e1_repair | 48 | -- | no pilot data |
| E1 | claude-opus-5-5 | e1_audit | 48 | -- | no pilot data |
| E1 | claude-haiku-5-5 | e1_gen | 120 | -- | no pilot data |
| E1 | claude-haiku-5-5 | e1_selfassess | 120 | -- | no pilot data |
| E1 | claude-haiku-5-5 | e1_repair | 48 | -- | no pilot data |
| E1 | claude-haiku-5-5 | e1_audit | 48 | -- | no pilot data |
| E1 | gpt-6-sol | e1_gen | 120 | -- | no pilot data |
| E1 | gpt-6-sol | e1_selfassess | 120 | -- | no pilot data |
| E1 | gpt-6-sol | e1_repair | 48 | -- | no pilot data |
| E1 | gpt-6-sol | e1_audit | 48 | -- | no pilot data |
| E1 | gpt-6-luna | e1_gen | 120 | -- | no pilot data |
| E1 | gpt-6-luna | e1_selfassess | 120 | -- | no pilot data |
| E1 | gpt-6-luna | e1_repair | 48 | -- | no pilot data |
| E1 | gpt-6-luna | e1_audit | 48 | -- | no pilot data |
| E1 | gemini-3-8-flash | e1_gen | 120 | -- | no pilot data |
| E1 | gemini-3-8-flash | e1_selfassess | 120 | -- | no pilot data |
| E1 | gemini-3-8-flash | e1_repair | 48 | -- | no pilot data |
| E1 | gemini-3-8-flash | e1_audit | 48 | -- | no pilot data |
| E1 | deepseek-v4-pro | e1_gen | 120 | -- | no pilot data |
| E1 | deepseek-v4-pro | e1_selfassess | 120 | -- | no pilot data |
| E1 | deepseek-v4-pro | e1_repair | 48 | -- | no pilot data |
| E1 | deepseek-v4-pro | e1_audit | 48 | -- | no pilot data |
| E1 | qwen3-8-27b | e1_gen | 120 | -- | no pilot data |
| E1 | qwen3-8-27b | e1_selfassess | 120 | -- | no pilot data |
| E1 | qwen3-8-27b | e1_repair | 48 | -- | no pilot data |
| E1 | qwen3-8-27b | e1_audit | 48 | -- | no pilot data |
| E1 | llama-4-maverick | e1_gen | 120 | -- | no pilot data |
| E1 | llama-4-maverick | e1_selfassess | 120 | -- | no pilot data |
| E1 | llama-4-maverick | e1_repair | 48 | -- | no pilot data |
| E1 | llama-4-maverick | e1_audit | 48 | -- | no pilot data |
| E2 | claude-opus-5-5 | session x2 runs | 48 | 21.88 | token-projected (all pilot models); x2 for longer sessions |
| E2 | claude-opus-5-5 | fresh x2 runs | 48 | 13.63 | token-projected (all pilot models); x2 for longer sessions |
| E2 | claude-opus-5-5 | padded x2 runs | 10 | 4.56 | token-projected (all pilot models); x2 for longer sessions |
| E2 | claude-haiku-5-5 | session x2 runs | 48 | 0.55 | token-projected (all pilot models); x2 for longer sessions |
| E2 | claude-haiku-5-5 | fresh x2 runs | 48 | 0.34 | token-projected (all pilot models); x2 for longer sessions |
| E2 | claude-haiku-5-5 | padded x2 runs | 10 | 0.11 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gpt-6-sol | session x2 runs | 48 | 11.03 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gpt-6-sol | fresh x2 runs | 48 | 6.84 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gpt-6-sol | padded x2 runs | 10 | 2.30 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gemini-3-8-flash | session x2 runs | 48 | 4.14 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gemini-3-8-flash | fresh x2 runs | 48 | 2.57 | token-projected (all pilot models); x2 for longer sessions |
| E2 | gemini-3-8-flash | padded x2 runs | 10 | 0.86 | token-projected (all pilot models); x2 for longer sessions |
| E2 | qwen3-8-27b | session x2 runs | 48 | 2.03 | token-projected (all pilot models); x2 for longer sessions |
| E2 | qwen3-8-27b | fresh x2 runs | 48 | 1.25 | token-projected (all pilot models); x2 for longer sessions |
| E2 | qwen3-8-27b | padded x2 runs | 10 | 0.42 | token-projected (all pilot models); x2 for longer sessions |
| E3 | local-qwen25vl-7b | e3_pointwise | 272 | -- | no pilot data |
| E3 | local-gemma3-4b | e3_pointwise | 272 | -- | no pilot data |
| E3 | local-ministral3-8b | e3_pointwise | 272 | -- | no pilot data |
| E3 | local-qwen25vl-7b | e3_known | 2116 | -- | no pilot data |
| E3 | local-gemma3-4b | e3_known | 2116 | -- | no pilot data |
| E3 | local-ministral3-8b | e3_known | 2116 | -- | no pilot data |
| E3 | claude-opus-5-5 | e3_alt | 80 | -- | no pilot data |
| E3 | claude-haiku-5-5 | e3_alt | 80 | -- | no pilot data |
| E3 | gpt-6-sol | e3_alt | 80 | -- | no pilot data |
| E3 | gpt-6-luna | e3_alt | 80 | -- | no pilot data |
| E3 | gemini-3-8-flash | e3_alt | 80 | -- | no pilot data |
| E3 | qwen3-8-27b | e3_alt | 80 | -- | no pilot data |
| E3 | gemma-4-31b | e3_alt | 80 | -- | no pilot data |
| E3 | llama-4-maverick | e3_alt | 80 | -- | no pilot data |
| E3 | local-qwen25vl-7b | e3_alt | 80 | -- | no pilot data |

| Experiment | Projected $ | Cap $ |
|---|---|---|
| E1 | 0.00 | 120 |
| E2 | 72.51 | 160 |
| E3 | 0.00 | 110 |
| Total | 72.51 | 450 (incl. contingency) |

Rows marked token-projected are estimates for models the pilot did not run; check them before approving.
Pilot data are not used in the paper (CLAUDE.md Phase 2).
