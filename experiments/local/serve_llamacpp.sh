#!/usr/bin/env bash
# Serve one local study model with llama.cpp's llama-server on 127.0.0.1:11434, the harness's
# default LOCAL_BASE_URL (http://localhost:11434/v1). Used for the 2026-11 local run
# (DEVIATIONS.md 2026-11-12; docs/LOCAL_SETUP.md, "Serving with llama.cpp").
#
#   local/serve_llamacpp.sh <profile> [extra llama-server args...]
#
# Profiles (one model resident at a time on an 8 GB GPU):
#   local-qwen3-8b          E1 + E2   ctx 32768, KV cache q8_0 (f16 does not fit next to the weights)
#   local-qwen25vl-7b-text  E1        ctx 32768, KV f16, no vision projector (E1 is text-only)
#   local-qwen25vl-7b       E3        ctx 8192,  KV f16, vision projector, <= 1280 tokens per image
#   local-gemma3-4b         E3        ctx 8192,  KV f16, vision projector (fixed 256 tokens per image)
#   local-ministral3-8b     E3        ctx 8192,  KV q8_0, vision projector, <= 1280 tokens per image
#
# The image cap (Qwen2.5-VL's documented max_pixels example, 1280*28*28): uncapped, the Ministral
# server crashed (bus error) on large images and Qwen2.5-VL took ~57 s per large image on NVK.
#
# Sampling defaults are each model's Ollama-library defaults (the provider defaults of the
# documented Ollama setup); unspecified values use Ollama's defaults (top_k 40, top_p 0.9,
# min_p 0) and no repeat penalty. The harness sends no sampling parameters. The alias is the
# model's provider_model_id in config/models.yaml; the adapter refuses answers from any other.
set -euo pipefail
LLAMA_SERVER=${LLAMA_SERVER:-$HOME/opt/llama.cpp/llama-b12004/llama-server}
MODELS_DIR=${MODELS_DIR:-$HOME/models}
PROFILE=${1:?profile}
shift
M=$MODELS_DIR

case "$PROFILE" in
  local-qwen3-8b)
    ARGS=(-m "$M/Qwen3-8B-Q4_K_M.gguf" --alias Qwen3-8B-Q4_K_M -c 32768 -ctk q8_0 -ctv q8_0
          --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0) ;;
  local-qwen25vl-7b-text)
    ARGS=(-m "$M/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" --alias Qwen2.5-VL-7B-Instruct-Q4_K_M -c 32768 --no-mmproj
          --temp 0.0001 --top-k 40 --top-p 0.9 --min-p 0) ;;
  local-qwen25vl-7b)
    ARGS=(-m "$M/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" --alias Qwen2.5-VL-7B-Instruct-Q4_K_M -c 8192
          --mmproj "$M/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" --image-max-tokens 1280
          --temp 0.0001 --top-k 40 --top-p 0.9 --min-p 0) ;;
  local-gemma3-4b)
    ARGS=(-m "$M/gemma-3-4b-it-Q4_K_M.gguf" --alias gemma-3-4b-it-Q4_K_M -c 8192
          --mmproj "$M/mmproj-gemma-3-4b-it-f16.gguf"
          --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0) ;;
  local-ministral3-8b)
    ARGS=(-m "$M/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf" --alias Ministral-3-8B-Instruct-2512-Q4_K_M -c 8192
          -ctk q8_0 -ctv q8_0 --mmproj "$M/mmproj-Ministral-3-8B-Instruct-2512-BF16.gguf" --image-max-tokens 1280
          --temp 0.15 --top-k 40 --top-p 0.9 --min-p 0) ;;
  *) echo "unknown profile $PROFILE" >&2; exit 1 ;;
esac

# -ngl all: every layer on the GPU; --fit off: never shrink the context or offload silently;
# -np 1: one slot (the harness serialises local calls); --no-context-shift: a full context
# ends generation instead of dropping tokens; a prompt over the context is rejected (HTTP 400),
# which the harness records as context exhaustion.
exec "$LLAMA_SERVER" "${ARGS[@]}" --repeat-penalty 1.0 --host 127.0.0.1 --port 11434 -np 1 -ngl all \
  --fit off -fa on --jinja --no-context-shift "$@"
