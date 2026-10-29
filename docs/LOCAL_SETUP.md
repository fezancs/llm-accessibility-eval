# Running the evaluation locally on an 8 GB NVIDIA GPU (Ubuntu + Ollama)

This guide runs the harness end to end with **free, self-hosted models** on a machine with an
8 GB NVIDIA GPU (tested target: RTX 2060 SUPER, 8 GB VRAM, 32 GB RAM, Ubuntu). Every model
call is logged at **$0** and never counts against a budget cap.

## What runs locally, and what does not

| Experiment | Local model | Runs locally? |
|---|---|---|
| **E1** generate / repair / audit | `local-qwen3-8b` (text), `local-qwen25vl-7b` (vision) | Yes — fast enough |
| **E3** alt-text write + judge | `local-qwen25vl-7b`, `local-gemma3-4b` | Yes — short calls |
| **E2** 24-step agentic sessions | `local-qwen3-8b` | Yes, **but slow**: contexts grow past 100K tokens and spill into system RAM on 8 GB. Expect many hours per run. Start with `--runs 1 --positions 6` to measure. |

**Honest scope.** 8 GB VRAM fits only 7–8B models at 4-bit. These substitute for the larger
study models (Qwen3.8-27B, Gemma 4 31B) and cannot run the frontier closed models (GPT-6,
Gemini 4, Claude 5.5, DeepSeek V4-Pro). This is a real but **narrower** study of
"small open-weight models a small organisation can self-host for free". It is recorded as a
deviation in `experiments/DEVIATIONS.md`; reflect it in the paper's model table and claims.

Local vision models are weaker at alt text than cloud models, so E3 quality will be lower —
report that as a finding, not a bug.

## 1. System prerequisites (Ubuntu)

```bash
# NVIDIA driver + confirm the GPU is visible
nvidia-smi                      # should list the GPU and its 8192 MiB

# System packages
sudo apt update
sudo apt install -y python3.12 python3.12-venv git curl
# Node.js 18+ (for the pinned axe-core used by the page checks)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

## 2. Get the code

```bash
git clone https://github.com/fezancs/llm-accessibility-eval.git
cd llm-accessibility-eval/experiments
```

## 3. Python + browser + axe-core

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m playwright install --with-deps chromium   # headless Chromium for page checks
npm ci                                                         # pinned axe-core + prettier
```

Confirm the harness works with **no model and no API key**:

```bash
.venv/bin/python -m pytest -q        # ~10 min; browser tests included; must pass
```

## 4. Install Ollama and pull the models

```bash
curl -fsSL https://ollama.com/install.sh | sh    # installs and starts the ollama service

ollama pull qwen3:8b          # ~5.2 GB  — E1, E2 (tool calling)
ollama pull qwen2.5vl:7b      # ~6 GB    — E1, E3 (vision)
ollama pull gemma3:4b         # ~3.3 GB  — optional second E3 family
```

Only one model is resident in VRAM at a time; Ollama swaps them as the harness switches
models, which is fine for this study (experiments run one model at a time).

### Raise the context window (important for E2)

Ollama defaults to a **4096-token** context regardless of the model's real limit. E2 needs
much more. Set it before running:

```bash
export OLLAMA_CONTEXT_LENGTH=32768      # or create per-model Modelfiles with `PARAMETER num_ctx`
# restart the service so it takes effect
sudo systemctl restart ollama           # or: pkill ollama && ollama serve &
```

8 GB VRAM cannot hold a 100K+ token KV-cache, so very long E2 steps will offload to CPU/RAM
and slow down. That is expected; it is why E2 is the slow part.

## 5. Point the harness at the local server

The `local` provider reads its endpoint from an environment variable (default is Ollama's):

```bash
export LOCAL_BASE_URL="http://localhost:11434/v1"   # default; set only if different
# no API key needed for Ollama
```

Quick connectivity check:

```bash
.venv/bin/python - <<'PY'
from harness import llm
r = llm.LLM("pilot").call("local-qwen3-8b",
        [llm.user("Reply with the single word OK.")], item_id="smoke/1", max_tokens=64)
print("reply:", r.text, "| cost $", r.cost_usd)
PY
```

You should see `OK` and `cost $ 0.0`.

## 6. Run the experiments (all $0)

Pass the local model keys explicitly with `--models`:

```bash
# E1 — generation, repair, audit
.venv/bin/python -m harness.e1_gen    --models local-qwen3-8b,local-qwen25vl-7b
.venv/bin/python -m harness.e1_repair --models local-qwen3-8b
.venv/bin/python -m harness.e1_audit  --models local-qwen3-8b

# E3 — alt-text generation and judging (vision models)
.venv/bin/python -m harness.e3_generate --models local-qwen25vl-7b,local-gemma3-4b
# E3 judging needs the judges map in config/models.yaml set to local families (see that file).

# E2 — start SMALL to measure speed before committing to a full run
.venv/bin/python -m harness.e2_run --models local-qwen3-8b --runs 1 --positions 6
```

Everything is resumable: re-running replays cached responses and continues where it stopped.
Watch GPU memory with `watch -n2 nvidia-smi` during E2.

## 7. Analysis and paper

```bash
.venv/bin/python -m analysis.run_all
cd ../paper && python3 check_placeholders.py
```

## Troubleshooting

- **`connection refused` on port 11434** — the Ollama service is not running: `ollama serve &`
  or `sudo systemctl start ollama`.
- **Tool calls ignored in E2** — the context is still 4096; set `OLLAMA_CONTEXT_LENGTH` and
  restart Ollama (step 4).
- **Out-of-memory / very slow E2** — expected on 8 GB for long steps; reduce `--positions`,
  or run E2 only on the smallest model.
- **E3 judging won't form a jury** — CMBPE needs >= 3 model families. Locally you have two
  (`alibaba-local`, `google-local`); add a third small vision model from another family, or
  run E3 generation locally and the judging with a free cloud tier.
