#!/usr/bin/env bash
# Unattended local run of E1, E3 and E2 on one 8 GB GPU (docs/LOCAL_SETUP.md local profile;
# DEVIATIONS.md 2026-11-12). Switches llama-server between models, runs every experiment step,
# scores, runs the analysis and pushes results to the results branch at each milestone.
#
#   nohup setsid experiments/local/run_local.sh > /dev/null 2>&1 &
# (or run a copy, local/run_local.active.sh, so that editing this file cannot disturb a running pass)
#
# Resumable: a finished step leaves runs/local_pipeline/<step>.done and is skipped on a re-run;
# an unfinished step re-runs and replays every cached response (harness/cache.py).
# Each API step runs up to four passes: a later pass only retries requests that failed (successes
# are cache hits), as RUNBOOK.md prescribes for interrupted runs. A step is done only when a pass
# exits 0 and leaves no failed request (local/step_errors.py).
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

# Vision warm-up: one image of each large E3 image shape, largest first, then a small check image.
# The vision encoder allocates GPU buffers per image shape; with ~1.3 GB of VRAM left next to
# Qwen2.5-VL and the desktop, an allocation that fails mid-run leaves the encoder unusable until
# the server restarts (2026-11-13, DEVIATIONS.md). Warming up right after loading allocates them
# while memory is free. The requests are live and uncached and never reach the call logs.
WARMUP_IMAGES=(e3/images/blv/blv-living_room6.jpg e3/images/blv/blv-meeting_in_progress7.jpg
  e3/images/blv/blv-church19.jpeg e3/images/blv/blv-laid_table3.jpg e3/images/blv/blv-meeting_in_progress2.jpg
  e3/images/blv/blv-laid_table14.jpg e3/images/blv/blv-church4.jpg corpus/base/images/cc-reception.png)

vision_ok() { # alias -> 0 if the loaded server encodes every warm-up image (live, uncached requests)
  local paths=()
  for f in "${WARMUP_IMAGES[@]}"; do [ -f "$EXP/data/$f" ] && paths+=("$EXP/data/$f"); done
  "$PY" - "$1" "${paths[@]}" <<'PYEOF' 2>/dev/null
import base64, json, sys, urllib.request
for path in sys.argv[2:]:
    img = base64.b64encode(open(path, "rb").read()).decode()
    mt = "image/png" if path.endswith(".png") else "image/jpeg"
    body = {"model": sys.argv[1], "max_tokens": 16, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{img}"}},
        {"type": "text", "text": "Reply with one word."}]}]}
    req = urllib.request.Request("http://127.0.0.1:11434/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        if urllib.request.urlopen(req, timeout=300).status != 200:
            sys.exit(1)
    except Exception:
        sys.exit(1)
PYEOF
}

VISION_ERRORS="ErrorOutOfDeviceMemory|failed to process mtmd|ErrorDeviceLost|DeviceLostError"
vision_broken() { # profile -> 0 if its server log shows a vision-encoder failure since the last launch
  local f="$LOGS/server_$1.log" n0
  n0=$(cat "$STATE/.logline" 2>/dev/null || echo 0)
  [ -f "$f" ] && tail -n +"$((n0 + 1))" "$f" | grep -qE "$VISION_ERRORS"
}

kill_server() {
  pkill -f "local/serve_llamacpp.sh" 2>/dev/null
  pkill -f "llama-server.*--port 11434" 2>/dev/null
  : > "$STATE/.loaded"
  sleep 5
}

# The E1 (text-only) and E3 (vision) profiles of Qwen2.5-VL share one alias, so the alias alone
# cannot tell them apart: .loaded records the profile the running server was launched with.
# (Before this check, the E3 steps of 2026-11-12 ran against the text-only E1 server.)
launch() {    # profile (the watchdog leaves the server alone while .launching exists)
  touch "$STATE/.launching"
  kill_server
  if [ -f "$LOGS/server_$1.log" ]; then wc -l < "$LOGS/server_$1.log"; else echo 0; fi > "$STATE/.logline"
  nohup "$SERVE" "$1" >> "$LOGS/server_$1.log" 2>&1 &
  local pid=$! ok=1                             # the serve script execs llama-server: same pid
  for _ in $(seq 1 120); do
    if healthy "${ALIAS[$1]}"; then ok=0; break; fi
    kill -0 "$pid" 2>/dev/null || break         # died while loading
    sleep 3
  done
  if [ $ok -eq 0 ]; then
    case "$1" in
      local-qwen25vl-7b|local-gemma3-4b|local-ministral3-8b)
        if ! vision_ok "${ALIAS[$1]}"; then log "server: $1 is up but refuses image input"; ok=1; fi ;;
    esac
  fi
  [ $ok -eq 0 ] && echo "$1" > "$STATE/.loaded"
  rm -f "$STATE/.launching"
  return $ok
}

serve() {     # profile: start the server and keep it alive (watchdog restarts a crashed server)
  echo "$1" > "$STATE/.server"
  if [ "$(cat "$STATE/.loaded" 2>/dev/null)" = "$1" ] && healthy "${ALIAS[$1]}"; then return 0; fi
  log "server: start $1"
  if ! launch "$1"; then log "server: $1 failed to start (see $LOGS/server_$1.log)"; return 1; fi
}

unserve() {
  : > "$STATE/.server"
  rm -f "$STATE/.launching"
  kill_server
}

watchdog() {  # restart a crashed server, or one whose vision encoder failed (its /health stays
              # ok); after 3 failed restarts, stop the running step
  local fails=0
  while true; do
    sleep 15
    [ -e "$STATE/.launching" ] && continue
    p=$(cat "$STATE/.server" 2>/dev/null)
    [ -z "$p" ] && { fails=0; continue; }
    if healthy "${ALIAS[$p]}"; then
      vision_broken "$p" || { fails=0; continue; }
      log "watchdog: $p vision encoder failed ($(tail -n +"$(( $(cat "$STATE/.logline" 2>/dev/null || echo 0) + 1 ))" "$LOGS/server_$p.log" | grep -m1 -oE "$VISION_ERRORS")), restarting"
    else
      sleep 20
      [ -e "$STATE/.launching" ] && continue
      [ "$(cat "$STATE/.server" 2>/dev/null)" = "$p" ] || continue
      healthy "${ALIAS[$p]}" && continue
      log "watchdog: $p not healthy, restarting"
    fi
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
# A step counts as done only if its command exits 0 AND no request it made ended in failure
# (local/step_errors.py; context exhaustion is a legitimate outcome, not a failure). The harness
# scripts record failed requests and still exit 0, so rc alone marked steps done that had run
# against a broken server (2026-11-12/13, DEVIATIONS.md).
run_pass() {  # name cmd... : run once, then gate; 0 only if both pass (never writes .done)
  local name=$1; shift
  local since; since=$(date -u +%Y-%m-%dT%H:%M:%S)
  log "start $name: ${*#$PY }"
  local t0; t0=$(date +%s)
  "$@" >> "$LOGS/$name.log" 2>&1
  local rc=$?
  local gate; gate=$("$PY" "$EXP/local/step_errors.py" "$since" 2>&1); local g=$?
  log "end   $name rc=$rc ($(( ($(date +%s) - t0) / 60 )) min); $gate"
  [ $rc -eq 0 ] || tail -15 "$LOGS/$name.log" >> "$STATE/pipeline.log"
  [ $rc -eq 0 ] && [ $g -eq 0 ]
}

step() {      # name cmd... : one pass unless done
  local name=$1; shift
  if [ -e "$STATE/$name.done" ]; then return 0; fi
  if run_pass "$name" "$@"; then touch "$STATE/$name.done"; return 0; fi
  FAILED+=("$name"); return 1
}

api_step() {  # name cmd... : up to 4 passes; a later pass replays successes from the cache and
              # retries only the failed requests
  local name=$1; shift
  if [ -e "$STATE/$name.done" ]; then return 0; fi
  local k
  for k in 1 2 3 4; do
    if run_pass "$name.pass$k" "$@"; then touch "$STATE/$name.done"; return 0; fi
    [ $k -lt 4 ] && sleep 60
  done
  FAILED+=("$name"); log "$name: not done after 4 passes"
  return 1
}

needed() {    # step names... -> 0 if any is not done
  for s in "$@"; do [ -e "$STATE/$s.done" ] || return 0; done
  return 1
}

all_done() {  # step names... -> 0 if every one is done (steps that replay the cache need this)
  for s in "$@"; do
    [ -e "$STATE/$s.done" ] || { log "skip: $s not done, so the steps that depend on it wait for a re-run"; return 1; }
  done
  return 0
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
  "$PY" local/extra_values.py >> "$LOGS/analysis.log" 2>&1 || log "extra values failed (see logs/analysis.log)"
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

# A server stays loaded between phases: steps that replay the cache for several models then
# get a fast refusal (wrong model, chat_adapter.py) for any request that is not cached, instead
# of retrying against a dead endpoint. Cache-replay steps run only after their inputs are done.

# --- 1. E1 with Qwen2.5-VL 7B (text only), scored while its server is still loaded ---------
if needed e1gen_qv e1rep_qv e1aud_qv e1score_gen_qv e1score_rep_qv e1score_aud_qv; then
  serve local-qwen25vl-7b-text && {
    api_step e1gen_qv "$PY" -m harness.e1_gen --models "$QV" --no-eval
    api_step e1rep_qv "$PY" -m harness.e1_repair --models "$QV" --no-eval
    api_step e1aud_qv "$PY" -m harness.e1_audit --models "$QV" --no-eval
    all_done e1gen_qv e1rep_qv e1aud_qv && e1_score "$QV" qv
  }
fi
if all_done e1score_gen_qv e1score_rep_qv e1score_aud_qv && [ ! -e "$STATE/m1.done" ]; then
  milestone "E1 for Qwen2.5-VL 7B"; touch "$STATE/m1.done"
fi

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
if all_done "e3gen_$QV" "e3gen_$G3" "e3gen_$MI"; then
  step e3gen_all "$PY" -m harness.e3_generate --models "$QV,$G3,$MI"
  all_done e3gen_all && step e3leak "$PY" -m harness.e3_generate --models "$QV,$G3,$MI" --leakage
fi
if all_done e3gen_all; then
  for M in "$QV" "$G3" "$MI"; do
    if needed "e3pairs_$M"; then
      serve "$M" && api_step "e3pairs_$M" "$PY" -m harness.e3_judge --models "$M" --procedures gen_pairs
    fi
  done
fi
if all_done "e3judge_$QV" "e3judge_$G3" "e3judge_$MI" "e3pairs_$QV" "e3pairs_$G3" "e3pairs_$MI"; then
  step e3judge_all "$PY" -m harness.e3_judge --models "$QV,$G3,$MI"
  if all_done e3judge_all && [ ! -e "$STATE/m2.done" ]; then
    milestone "E3 (alt-text generation and CMBPE judging)"; touch "$STATE/m2.done"
  fi
fi

# --- 3. E2 with Qwen3 8B --------------------------------------------------------------------
if needed e2pilot e2full; then
  serve "$Q3" && {
    step check_q3 "$PY" -m harness.check_keys --models "$Q3"
    # SPEC 12 gate on the pilot. A pilot that did not finish, or made no tool call, decides
    # nothing: E2 then waits for a re-run instead of being recorded as an exclusion (2026-11-14).
    if api_step e2pilot "$PY" -m harness.e2_run --pilot --models "$Q3"; then
      rate=$("$PY" -c "import json; a=json.load(open('runs/pilot/scores/e2_tool_calls.json'))['$Q3']['all']; print(a['rate'] if a['tool_calls'] else 'none')" 2>/dev/null || echo missing)
    else
      rate=missing
    fi
    log "E2 pilot malformed tool-call rate for $Q3: $rate (SPEC 12 exclusion above 0.20)"
    if [ "$rate" = missing ] || [ "$rate" = none ]; then
      log "E2 pilot incomplete or without tool calls ($rate): E2 not started; re-run the pipeline"
      FAILED+=(e2pilot)
    elif "$PY" -c "import sys; sys.exit(0 if float('$rate') <= 0.20 else 1)"; then
      if [ -e "$EXP/data/e2_padding_transcript.json" ] || api_step e2pad "$PY" -m harness.e2_padding --model "$Q3"; then
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
if all_done e2full && [ ! -e "$STATE/m3.done" ]; then
  milestone "E2 (long agentic sessions, DCDI)"; touch "$STATE/m3.done"
fi

# --- 4. E1 with Qwen3 8B, then E1 scoring for both models (Qwen3 server still loaded) -------
if needed e1gen_q3 e1rep_q3 e1aud_q3 e1score_gen_all e1score_rep_all e1score_aud_all; then
  serve "$Q3" && {
    api_step e1gen_q3 "$PY" -m harness.e1_gen --models "$Q3" --no-eval
    api_step e1rep_q3 "$PY" -m harness.e1_repair --models "$Q3" --no-eval
    api_step e1aud_q3 "$PY" -m harness.e1_audit --models "$Q3" --no-eval
    all_done e1gen_q3 e1rep_q3 e1aud_q3 e1gen_qv e1rep_qv e1aud_qv && e1_score "$Q3,$QV" all
  }
fi
all_done check_q3 "check_$QV" "check_$G3" "check_$MI" \
  && step checks_all "$PY" -m harness.check_keys --models "$Q3,$QV,$G3,$MI"
unserve
log "failed steps this pass: ${FAILED[*]:-none}"
if all_done m1 m2 m3 e1score_gen_all e1score_rep_all e1score_aud_all checks_all; then
  [ -e "$STATE/m4.done" ] || { milestone "E1 complete; all experiments done"; touch "$STATE/m4.done"; }
else
  milestone "partial (pipeline pass ended; see runs/local_pipeline/pipeline.log)"
fi
log "pipeline end"
