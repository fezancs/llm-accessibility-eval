#!/usr/bin/env bash
# Unattended local run of E1, E3 and E2 on one 8 GB GPU (docs/LOCAL_SETUP.md local profile;
# DEVIATIONS.md 2026-11-12). Switches llama-server between models, runs every experiment step,
# scores, runs the analysis and pushes results to the results branch at each milestone.
#
#   nohup setsid experiments/local/run_local.sh > /dev/null 2>&1 &
#
# Resumable: a finished step leaves runs/local_pipeline/<step>.done and is skipped on a re-run;
# an unfinished step re-runs and replays every cached response (harness/cache.py).
# Each API step runs twice: the second pass only retries requests that failed (successes are
# cache hits), as RUNBOOK.md prescribes for interrupted runs.
set -uo pipefail
EXP=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$EXP/.." && pwd)
PY="$EXP/.venv/bin/python"
SERVE="$EXP/local/serve_llamacpp.sh"
STATE="$EXP/runs/local_pipeline"
LOGS=${LOCAL_LOG_DIR:-$HOME/opt/logs}
BRANCH=${RESULTS_BRANCH:-local-run-2026-11}
RESULTS="$EXP/results/local-run-2026-11"
GIT_NAME=${GIT_NAME:-"Fezan Rasool"}
GIT_EMAIL=${GIT_EMAIL:-"52468414+fezancs@users.noreply.github.com"}
mkdir -p "$STATE/logs" "$LOGS" "$RESULTS"
cd "$EXP"

Q3=local-qwen3-8b; QV=local-qwen25vl-7b; G3=local-gemma3-4b; MI=local-ministral3-8b
declare -A ALIAS=([local-qwen3-8b]=Qwen3-8B-Q4_K_M [local-qwen25vl-7b-text]=Qwen2.5-VL-7B-Instruct-Q4_K_M
                  [local-qwen25vl-7b]=Qwen2.5-VL-7B-Instruct-Q4_K_M [local-gemma3-4b]=gemma-3-4b-it-Q4_K_M
                  [local-ministral3-8b]=Ministral-3-8B-Instruct-2512-Q4_K_M)
FAILED=()

log() { echo "$(date -Is) $*" | tee -a "$STATE/pipeline.log"; }

# ------------------------------------------------------------------ server management
healthy() {   # alias -> 0 if the server is up and serves that alias
  "$PY" - "$1" <<'PYEOF' 2>/dev/null
import json, sys, urllib.request
try:
    ok = urllib.request.urlopen("http://127.0.0.1:11434/health", timeout=5).status == 200
    ids = [m["id"] for m in json.load(urllib.request.urlopen("http://127.0.0.1:11434/v1/models", timeout=5))["data"]]
    sys.exit(0 if ok and sys.argv[1] in ids else 1)
except Exception:
    sys.exit(1)
PYEOF
}

launch() {    # profile (the watchdog leaves the server alone while .launching exists)
  touch "$STATE/.launching"
  pkill -f "llama-server.*--port 11434" 2>/dev/null; sleep 5
  nohup "$SERVE" "$1" >> "$LOGS/server_$1.log" 2>&1 &
  local ok=1
  for _ in $(seq 1 120); do
    if healthy "${ALIAS[$1]}"; then ok=0; break; fi
    pgrep -f "llama-server.*--port 11434" > /dev/null || break      # died while loading
    sleep 3
  done
  rm -f "$STATE/.launching"
  return $ok
}

serve() {     # profile: start the server and keep it alive (watchdog restarts a crashed server)
  echo "$1" > "$STATE/.server"
  if healthy "${ALIAS[$1]}"; then return 0; fi
  log "server: start $1"
  if ! launch "$1"; then log "server: $1 failed to start (see $LOGS/server_$1.log)"; return 1; fi
}

unserve() {
  : > "$STATE/.server"
  rm -f "$STATE/.launching"
  pkill -f "llama-server.*--port 11434" 2>/dev/null; sleep 5
}

watchdog() {  # restart a crashed server; after 3 failed restarts, stop the running step
  local fails=0
  while true; do
    sleep 30
    [ -e "$STATE/.launching" ] && continue
    p=$(cat "$STATE/.server" 2>/dev/null)
    [ -z "$p" ] && { fails=0; continue; }
    healthy "${ALIAS[$p]}" && { fails=0; continue; }
    sleep 20
    [ -e "$STATE/.launching" ] && continue
    [ "$(cat "$STATE/.server" 2>/dev/null)" = "$p" ] || continue
    healthy "${ALIAS[$p]}" && continue
    log "watchdog: $p not healthy, restarting"
    if launch "$p"; then fails=0; log "watchdog: $p restarted"; continue; fi
    fails=$((fails + 1))
    log "watchdog: restart $fails of $p failed (see $LOGS/server_$p.log)"
    if [ $fails -ge 3 ]; then
      log "watchdog: giving up on $p; stopping the running step so the pipeline can move on"
      : > "$STATE/.server"
      pkill -f "$EXP/.venv/bin/python -m harness" 2>/dev/null
      fails=0
    fi
  done
}

# ------------------------------------------------------------------ steps
step() {      # name cmd... : run once unless done
  local name=$1; shift
  if [ -e "$STATE/$name.done" ]; then return 0; fi
  log "start $name: ${*#$PY }"
  local t0; t0=$(date +%s)
  "$@" >> "$LOGS/$name.log" 2>&1
  local rc=$?
  log "end   $name rc=$rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if [ $rc -eq 0 ]; then touch "$STATE/$name.done"; else FAILED+=("$name"); tail -15 "$LOGS/$name.log" >> "$STATE/pipeline.log"; fi
  return $rc
}

api_step() {  # name cmd... : two passes (the second retries failed requests only)
  local name=$1; shift
  if [ -e "$STATE/$name.done" ]; then return 0; fi
  step "$name.pass1" "$@"
  step "$name" "$@"
}

needed() {    # step names... -> 0 if any is not done
  for s in "$@"; do [ -e "$STATE/$s.done" ] || return 0; done
  return 1
}

# ------------------------------------------------------------------ results
publish() {   # message
  for f in "$LOGS"/*.log; do
    case "$(basename "$f")" in server_*) continue ;; esac
    [ "$(stat -c %s "$f")" -lt 5000000 ] && cp "$f" "$STATE/logs/"
  done
  cd "$REPO" || return
  for p in experiments/runs experiments/results experiments/data/e2_padding_transcript.json; do
    [ -e "$p" ] && git add -- "$p"
  done
  if ! git diff --cached --quiet; then
    git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" commit -q -m "Local run: $1" \
      && log "committed: $1"
    for i in 1 2 3 4 5; do
      if GIT_TERMINAL_PROMPT=0 git push -q origin "$BRANCH" >> "$LOGS/git_push.log" 2>&1; then
        log "pushed to origin/$BRANCH"; break
      fi
      log "push attempt $i failed"; sleep 120
    done
  fi
  cd "$EXP" || exit 1
}

milestone() { # message
  log "milestone: $1"
  "$PY" -m analysis.run_all --paper "$RESULTS" --summary "$RESULTS/summary.md" \
    >> "$LOGS/analysis.log" 2>&1 || log "analysis failed (see logs/analysis.log)"
  publish "$1"
}

e1_score() {  # models: score E1 from cached responses (no server needed)
  step "e1score_gen_$2" "$PY" -m harness.e1_gen --models "$1"
  step "e1score_rep_$2" "$PY" -m harness.e1_repair --models "$1"
  step "e1score_aud_$2" "$PY" -m harness.e1_audit --models "$1"
}

# ================================================================== run
log "pipeline start (pid $$) on branch $(git -C "$REPO" rev-parse --abbrev-ref HEAD) at $(git -C "$REPO" rev-parse --short HEAD)"
watchdog & WD=$!
trap 'kill $WD 2>/dev/null; unserve' EXIT

# --- 1. E1 with Qwen2.5-VL 7B (text only) -------------------------------------------------
if needed e1gen_qv e1rep_qv e1aud_qv; then
  serve local-qwen25vl-7b-text && {
    api_step e1gen_qv "$PY" -m harness.e1_gen --models "$QV" --no-eval
    api_step e1rep_qv "$PY" -m harness.e1_repair --models "$QV" --no-eval
    api_step e1aud_qv "$PY" -m harness.e1_audit --models "$QV" --no-eval
  }
  unserve
fi
e1_score "$QV" qv
[ -e "$STATE/m1.done" ] || { milestone "E1 for Qwen2.5-VL 7B"; touch "$STATE/m1.done"; }

# --- 2. E3: generation and judging, one vision model at a time ---------------------------
for M in "$QV" "$G3" "$MI"; do
  if needed "check_$M" "e3gen_$M" "e3judge_$M"; then
    serve "$M" && {
      step "check_$M" "$PY" -m harness.check_keys --models "$M"
      api_step "e3gen_$M" "$PY" -m harness.e3_generate --models "$M"
      api_step "e3judge_$M" "$PY" -m harness.e3_judge --models "$M" --procedures pointwise,known,blv_pairs
    }
  fi
done
unserve
step e3gen_all "$PY" -m harness.e3_generate --models "$QV,$G3,$MI"
step e3leak "$PY" -m harness.e3_generate --models "$QV,$G3,$MI" --leakage
for M in "$QV" "$G3" "$MI"; do
  if needed "e3pairs_$M"; then
    serve "$M" && api_step "e3pairs_$M" "$PY" -m harness.e3_judge --models "$M" --procedures gen_pairs
  fi
done
unserve
step e3judge_all "$PY" -m harness.e3_judge --models "$QV,$G3,$MI"
[ -e "$STATE/m2.done" ] || { milestone "E3 (alt-text generation and CMBPE judging)"; touch "$STATE/m2.done"; }

# --- 3. E2 with Qwen3 8B --------------------------------------------------------------------
if needed e2pilot e2full; then
  serve "$Q3" && {
    step check_q3 "$PY" -m harness.check_keys --models "$Q3"
    api_step e2pilot "$PY" -m harness.e2_run --pilot --models "$Q3"
    rate=$("$PY" -c "import json; d=json.load(open('runs/pilot/scores/e2_tool_calls.json')); print(d['$Q3']['all']['rate'] or 0)" 2>/dev/null || echo 1)
    log "E2 pilot malformed tool-call rate for $Q3: $rate (SPEC 12 exclusion above 0.20)"
    if "$PY" -c "import sys; sys.exit(0 if float('$rate') <= 0.20 else 1)"; then
      if [ -e "$EXP/data/e2_padding_transcript.json" ] || step e2pad "$PY" -m harness.e2_padding --model "$Q3"; then
        step pilot_report "$PY" -m harness.pilot_report
        api_step e2full "$PY" -m harness.e2_run --models "$Q3" --runs 1
      else
        log "padding transcript unavailable: E2 runs without the padded control"
        api_step e2full "$PY" -m harness.e2_run --models "$Q3" --runs 1 --no-padded
      fi
    else
      log "SPEC 12: $Q3 excluded from E2 (malformed tool calls above 20 % in the pilot)"
      touch "$STATE/e2full.done"
    fi
  }
fi
[ -e "$STATE/m3.done" ] || { milestone "E2 (long agentic sessions, DCDI)"; touch "$STATE/m3.done"; }

# --- 4. E1 with Qwen3 8B, then E1 scoring for both models ----------------------------------
if needed e1gen_q3 e1rep_q3 e1aud_q3; then
  serve "$Q3" && {
    api_step e1gen_q3 "$PY" -m harness.e1_gen --models "$Q3" --no-eval
    api_step e1rep_q3 "$PY" -m harness.e1_repair --models "$Q3" --no-eval
    api_step e1aud_q3 "$PY" -m harness.e1_audit --models "$Q3" --no-eval
  }
  unserve
fi
e1_score "$Q3,$QV" all
step checks_all "$PY" -m harness.check_keys --models "$Q3,$QV,$G3,$MI"
log "failed steps this pass: ${FAILED[*]:-none}"
[ -e "$STATE/m4.done" ] || { milestone "E1 complete; all experiments done"; touch "$STATE/m4.done"; }
log "pipeline end"
