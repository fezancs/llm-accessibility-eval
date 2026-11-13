#!/usr/bin/env python3
"""Live GPU view for the local run on the open nouveau/NVK driver, where nvidia-smi does not exist.

    .venv/bin/python local/gpu_monitor.py                 # refresh every 2 s (Ctrl+C to stop)
    .venv/bin/python local/gpu_monitor.py --once          # one sample
    .venv/bin/python local/gpu_monitor.py --csv runs/local_pipeline/gpu_usage.csv   # also log

What it shows and where each number comes from:
- VRAM used / total: the Vulkan memory budget of the GPU (VK_EXT_memory_budget, read with
  `llama-server --list-devices`). It counts every process, the desktop included.
- Model: the model the llama.cpp server holds (/props), and its context size.
- State: "generating", "reading prompt" or "idle" (/slots), with live tokens per second.
  Every layer runs on the GPU (-ngl all), so a working server means a working GPU.
- GPU busy: share of the last 60 samples in which the server was working. nouveau gives
  unprivileged users no hardware utilisation counter, so this is the closest honest measure.
- Server CPU: CPU use of llama-server in % of one core (low when the GPU does the work;
  a CPU-only run of a 7-8B model keeps all cores busy).
- Pipeline: the step local/run_local.sh is running.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

EXP = pathlib.Path(__file__).resolve().parent.parent
LLAMA = os.environ.get("LLAMA_SERVER", str(pathlib.Path.home() / "opt/llama.cpp/llama-b12004/llama-server"))
BASE = os.environ.get("LOCAL_SERVER", "http://127.0.0.1:11434")
PIPELINE_LOG = EXP / "runs" / "local_pipeline" / "pipeline.log"
CLK = os.sysconf("SC_CLK_TCK")


def vram() -> tuple[int | None, int | None, str | None]:
    """(total MiB, free MiB, device name) from the Vulkan memory budget."""
    try:
        out = subprocess.run([LLAMA, "--list-devices"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, None, None
    m = re.search(r"Vulkan0: (.+?) \((\d+) MiB, (\d+) MiB free\)", out)
    return (int(m.group(2)), int(m.group(3)), m.group(1)) if m else (None, None, None)


def get(path: str):
    try:
        return json.load(urllib.request.urlopen(BASE + path, timeout=3))
    except Exception:  # noqa: BLE001 - server down or loading
        return None


def server_pid() -> int | None:
    for p in pathlib.Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                cmd = (p / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if cmd and cmd[0].endswith(b"llama-server") and b"11434" in cmd:
                return int(p.name)
    return None


def cpu_ticks(pid: int) -> int | None:
    try:
        f = (pathlib.Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])          # utime + stime
    except (OSError, IndexError, ValueError):
        return None


def pipeline_step() -> str:
    try:
        lines = PIPELINE_LOG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "-"
    for line in reversed(lines):
        m = re.match(r"\S+ (start|end  ) (\S+?):?(?:\s|$)", line)
        if m:
            return m.group(2) + (" (running)" if m.group(1) == "start" else " (finished)")
    return lines[-1][26:] if lines else "-"


def bar(frac: float, width: int = 30) -> str:
    n = max(0, min(width, round(frac * width)))
    return "[" + "#" * n + "." * (width - n) + "]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--csv", type=pathlib.Path, help="append one row per sample to this CSV file")
    args = ap.parse_args()

    busy_hist = collections.deque(maxlen=60)
    prev = {}
    writer = None
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        new = not args.csv.exists()
        fh = open(args.csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        if new:
            writer.writerow(["time", "vram_used_mib", "vram_total_mib", "model", "state", "tok_s",
                             "prompt_tok_s", "server_cpu_pct", "step"])
    try:
        while True:
            t = time.monotonic()
            total, free, dev = vram()
            props, slots = get("/props"), get("/slots")
            model = (props or {}).get("model_alias") or ("loading / not running" if server_pid() else "no server")
            n_ctx = ((props or {}).get("default_generation_settings") or {}).get("n_ctx")
            s = slots[0] if slots else {}
            nt = (s.get("next_token") or [{}])
            nt = nt[0] if isinstance(nt, list) and nt else (nt if isinstance(nt, dict) else {})
            task, decoded = s.get("id_task"), nt.get("n_decoded") or 0
            n_prompt, n_done = s.get("n_prompt_tokens") or 0, s.get("n_prompt_tokens_processed") or 0
            working = bool(s.get("is_processing"))
            tok_s = ptok_s = None
            if prev.get("task") == task and prev.get("t"):
                dtt = t - prev["t"]
                if decoded >= prev["decoded"] and dtt > 0:
                    tok_s = (decoded - prev["decoded"]) / dtt
                if n_done >= prev["n_done"] and dtt > 0:
                    ptok_s = (n_done - prev["n_done"]) / dtt
            state = "idle"
            if working:
                state = "generating" if decoded > 0 else "reading prompt"
            busy_hist.append(1 if working else 0)
            pid = server_pid()
            ticks = cpu_ticks(pid) if pid else None
            cpu = None
            if ticks is not None and prev.get("ticks") is not None and prev.get("pid") == pid:
                cpu = 100.0 * (ticks - prev["ticks"]) / CLK / max(t - prev["t"], 1e-6)
            prev = {"task": task, "decoded": decoded, "n_done": n_done, "t": t, "ticks": ticks, "pid": pid}
            used = (total - free) if total is not None else None
            step = pipeline_step()

            lines = [f"GPU monitor  {dt.datetime.now():%Y-%m-%d %H:%M:%S}   (Ctrl+C to quit)", ""]
            if total:
                lines.append(f"GPU          {dev}")
                lines.append(f"VRAM         {bar(used / total)} {used:5d} / {total} MiB used ({free} MiB free)")
            else:
                lines.append("VRAM         unavailable (Vulkan query failed)")
            lines.append(f"Model        {model}" + (f"   context {n_ctx}" if n_ctx else ""))
            rate = ""
            if state == "generating" and tok_s is not None:
                rate = f"   {tok_s:5.1f} tokens/s   ({decoded} tokens so far)"
            elif state == "reading prompt" and ptok_s is not None:
                rate = f"   {ptok_s:6.1f} prompt tokens/s   ({n_done}/{n_prompt})"
            lines.append(f"State        {state}{rate}")
            busy = sum(busy_hist) / len(busy_hist)
            lines.append(f"GPU busy     {bar(busy)} {100 * busy:5.1f} % of the last {len(busy_hist)} samples")
            lines.append(f"Server CPU   {cpu:5.1f} % of one core" if cpu is not None else "Server CPU   -")
            lines.append(f"Pipeline     {step}")
            if args.once:
                print("\n".join(lines))
            else:
                sys.stdout.write("\033[2J\033[H" + "\n".join(lines) + "\n")
                sys.stdout.flush()
            if writer:
                writer.writerow([dt.datetime.now().isoformat(timespec="seconds"), used, total, model, state,
                                 f"{tok_s:.1f}" if tok_s is not None else "",
                                 f"{ptok_s:.1f}" if ptok_s is not None else "",
                                 f"{cpu:.1f}" if cpu is not None else "", step])
                fh.flush()
            if args.once:
                return 0
            time.sleep(max(0.0, args.interval - (time.monotonic() - t)))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
