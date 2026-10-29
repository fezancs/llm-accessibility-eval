# Data contracts between the harness and the analysis

Every experiment script writes scored records as JSONL; `analysis/` reads only these files
and `runs/*/calls.jsonl` (SPEC 2.2). One JSON object per line. `null` means "not measured".
`cost_usd` is the billed cost of the request(s) that produced the record, all retry attempts
included; `latency_s` follows SPEC 2.4 (single call: total_s of the successful attempt;
multi-call step: sum of the step's calls, tool execution excluded).

A `checks` list holds `{"id": str, "kind": "axe" | "probe", "passed": bool}` for the applicable
checks Q(x) of SPEC 3.1 (report-only probes are omitted from `checks` and listed in
`report_only`). `a` = share of passing checks, or `null` when Q(x) is empty.

## E1 (runs/e1/scores/)

`gen.jsonl` — one line per (model, task, condition, sample):
```
model_key, task_id, condition ("unguided" | "nudged"), sample_idx (0-2),
status ("ok" | "refusal" | "error"), html_found (bool), functional (bool),
a (float | null), checks, report_only, n_elements (int | null), violating_nodes (int),
violations_per_100 (float | null), incomplete (int),
selfassess: {conforms (bool | null), parse_ok (bool), retried (bool)},
cost_usd, latency_s, ttft_s, input_tokens, output_tokens, call_ids
```
Sample score for S_gen = `a * functional` (0 when `html_found` is false, status is not ok,
or `a` is null).

`repair.jsonl` — one line per (model, faulty page):
```
model_key, page_id ("B05-V2"), base_id ("B05"), provenance ("w3c" | "authored"), k (4),
faults: [{operator ("M01"), target_id, resolved (bool)}],
regressions (int), regression_list ([str]), preserved (bool),
preservation: {text_sim, ids_ok, tag_sim, ssim},
score (float), full_repair (bool), status, html_found,
cost_usd, latency_s, ttft_s, input_tokens, output_tokens, page_tokens, call_ids
```

`audit.jsonl` — one line per (model, faulty page):
```
model_key, page_id, base_id, provenance, n_reported (int), parse_ok (bool), retried (bool),
faults: [{operator, target_id, matched (bool)}], tp (int), fp (int), fn (int),
status, cost_usd, latency_s, call_ids
```

## E2 (runs/e2/scores/)

`steps.jsonl` — one line per (model, run, session position, arm):
```
model_key, run (0-2), perm_seed (101 | 202 | 303), position (1-24), step_id ("R07"),
step_type ("foundation" | "regular" | "pressure"), arm ("session" | "fresh" | "padded"),
functional (bool), a (float | null), checks, accessible (bool: all checks pass),
Lambda (int: input tokens of the step's first call in this arm),
tool_calls (int), malformed_tool_calls (int), cost_usd, latency_s, tool_time_s,
pressure_score (1 | 0.5 | 0 | null), intent_met (bool | null),
context_exhausted (bool), status ("ok" | "error" | "context_exhausted")
```

`sessions.jsonl` — one line per (model, run):
```
model_key, run, perm_seed, completed_steps (int), total_tool_calls (int),
context_exhausted_at (int | null), cost_usd,
end_audit: [{step_id, position, a_at_completion, a_end, fell (bool)}]
```

## E3 (data/e3/ and runs/e3/judgments/)

`data/e3/items.jsonl` — one line per human-rated description:
```
item_id, dataset ("blv" | "cr"), image_id, image_path (relative to experiments/ | null),
image_url, retrieved (bool), description, context: {article_title, section_title, excerpt, caption},
human: {overall_mean, n_raters, <other scale>_mean ...}
```

`runs/e3/judgments/pointwise.jsonl`: `judge, dataset, item_id, image_id, score (1-5 | null), parse_ok, retried`

`runs/e3/judgments/known_pairs.jsonl`:
```
judge, dataset, item_id, image_id, degradation ("append" | "repeat" | "swap" | "truncate"),
original_first (bool), len_first, len_second, better (1 | 2 | null),
y (1 if the first-shown candidate was preferred | null), correct (bool | null)
```

`runs/e3/judgments/blv_pairs.jsonl` (only if D_BLV has >= 2 descriptions per image):
```
judge, image_id, item_a, item_b, human_prefers ("a" | "b"), a_first (bool), y, correct
```

`runs/e3/generated.jsonl`:
```
generator, image_id, status, alt_raw, alt_clean, transforms (list of fired rules),
opaque_id, length (chars), cost_usd, latency_s
```

`runs/e3/judgments/gen_pairs.jsonl`:
```
judge, image_id, gen_first, gen_second, id_first, id_second, len_first, len_second,
better (1 | 2 | null), y (1 if first preferred | null)
```

Families come from `config/models.yaml` (`family`); the judge panel from its `judges` map.
