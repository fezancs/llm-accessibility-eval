"""Figures (SPEC 11): fig_drift.pdf and fig_pareto.pdf, vector PDF at final print size.

Accessibility of the figures themselves: two Okabe-Ito hues only - blue #0072B2 and
vermillion #D55E00, validated for colour-vision deficiency with the dataviz palette checker
(CVD dE 21.9 on white) - and every colour difference is doubled by a non-colour cue:
in-session = solid line, fresh control = dashed line; closed = hollow circle, open weights =
filled square. Text is >= 7 pt at final size and uses neutral ink, never a series colour.
Gridlines are solid hairlines. Numbers behind each figure are in Table III / summary.md.
"""
from __future__ import annotations

import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter   # noqa: E402

BLUE, VERMILLION = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e3e3e3"
FONT = 7.0
TEXT_W = 7.16          # IEEE two-column text width, inches
COL_W = 3.5            # IEEE column width, inches

RC = {"font.size": FONT, "axes.titlesize": FONT + 0.5, "axes.labelsize": FONT + 0.5,
      "xtick.labelsize": FONT, "ytick.labelsize": FONT, "legend.fontsize": FONT,
      "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
      "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
      "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
      "font.family": "DejaVu Sans"}


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def drift(curves: dict, names: dict, out_dir: pathlib.Path) -> pathlib.Path | None:
    """Small multiples: in-session (solid) vs fresh-control (dashed) adherence by position.

    Panels follow the registry order in `names` (models.yaml), so each vendor's larger and
    smaller model sit side by side (H1)."""
    rank = {m: i for i, m in enumerate(names)}
    models = sorted((m for m in curves if len(curves[m])), key=lambda m: (rank.get(m, len(rank)), m))
    if not models:
        return None
    n = len(models)
    cols = n if n <= 4 else math.ceil(n / 2)
    rows = 1 if n <= 4 else 2
    lo = min(min(c["session_lo"].min(), c["fresh_lo"].min()) for c in curves.values() if len(c))
    ymin = max(0.0, math.floor((lo - 0.05) * 10) / 10)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(rows, cols, figsize=(TEXT_W, 1.55 * rows + 0.35), sharex=True, sharey=True,
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i >= n:
                ax.set_visible(False)
                continue
            m = models[i]
            c = curves[m]
            x = c["position"].to_numpy()
            ax.fill_between(x, c["fresh_lo"], c["fresh_hi"], color=VERMILLION, alpha=0.12, linewidth=0)
            ax.fill_between(x, c["session_lo"], c["session_hi"], color=BLUE, alpha=0.12, linewidth=0)
            ax.plot(x, c["fresh"], color=VERMILLION, linestyle=(0, (4, 2)), linewidth=1.2)
            ax.plot(x, c["session"], color=BLUE, linestyle="-", linewidth=1.2)
            ax.set_title(names.get(m, m), loc="left", pad=2)
            ax.set_xlim(1, 24)
            ax.set_ylim(ymin, 1.0)
            ax.set_xticks([1, 8, 16, 24])
            _style(ax)
        for col in range(cols):
            visible = [axes[r, col] for r in range(rows) if axes[r, col].get_visible()]
            if visible:                                  # the lowest visible panel of each column
                visible[-1].set_xlabel("Position in session")
                visible[-1].xaxis.set_tick_params(labelbottom=True)
        for ax in axes[:, 0]:
            ax.set_ylabel("Adherence")
        handles = [Line2D([], [], color=BLUE, linestyle="-", linewidth=1.2, label="In session (solid)"),
                   Line2D([], [], color=VERMILLION, linestyle=(0, (4, 2)), linewidth=1.2,
                          label="Fresh-context control (dashed)")]
        fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.6, h_pad=0.6)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "fig_drift.pdf"
        fig.savefig(path, format="pdf")
        plt.close(fig)
    return path


def _place_labels(fig, ax, pts, labels, sizes, line_pts=()):
    """Greedy label placement next to each marker, avoiding markers, placed labels and the line.

    `sizes` are scatter areas (pt^2); each label starts just outside its own marker.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    obstacles = []
    for (x, y), s in zip(pts, sizes):
        px, py = ax.transData.transform((x, y))
        r = math.sqrt(s) / 2 * px_per_pt + 1
        obstacles.append((px - r, py - r, px + r, py + r))
    for (x0, y0), (x1, y1) in zip(line_pts, line_pts[1:]):
        a = ax.transData.transform((x0, y0))
        b = ax.transData.transform((x1, y1))
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 4))
        for t in range(n + 1):
            cx, cy = a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n
            obstacles.append((cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5))
    axb = ax.get_window_extent(renderer)
    placed = []
    for (x, y), text, s in zip(pts, labels, sizes):
        r = math.sqrt(s) / 2 + 1.5                                 # marker radius + gap, in points
        cands = [(r, 0, "left", "center"), (-r, 0, "right", "center"), (0, r, "center", "bottom"),
                 (0, -r, "center", "top"), (r * .7, r * .7, "left", "bottom"), (r * .7, -r * .7, "left", "top"),
                 (-r * .7, r * .7, "right", "bottom"), (-r * .7, -r * .7, "right", "top")]
        best = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=FONT, color=INK, zorder=4)
            bb = t.get_window_extent(renderer)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            outside = box[0] < axb.x0 or box[2] > axb.x1 or box[1] < axb.y0 or box[3] > axb.y1
            cost = sum(_ov(box, o) for o in placed + obstacles) + (1e6 if outside else 0)
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, t, box)
            else:
                t.remove()
            if cost == 0:
                break
        placed.append(best[2])


def _ov(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def pareto(summary: dict, frontier: list[str], names: dict, open_models: set, out_dir: pathlib.Path) -> pathlib.Path | None:
    """Batch repair: mean cost per page (log x) vs S_rep; marker area = median latency."""
    pts = {m: s for m, s in summary.items() if s.get("c") and s.get("l") is not None and s["c"] > 0}
    if not pts:
        return None
    lmax = max(s["l"] for s in pts.values()) or 1.0

    def area(l):                       # proportional to latency (floor keeps tiny markers visible)
        return max(8.0, 130.0 * l / lmax)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(COL_W, 2.7))
        ax.set_xscale("log")
        front = sorted([m for m in frontier if m in pts], key=lambda m: pts[m]["c"])
        if len(front) > 1:
            ax.plot([pts[m]["c"] for m in front], [pts[m]["cs"] for m in front], color=INK, linewidth=0.8,
                    zorder=1)
        order = sorted(pts, key=lambda m: pts[m]["c"])
        for m in order:
            s = pts[m]
            if m in open_models:
                ax.scatter(s["c"], s["cs"], s=area(s["l"]), marker="s", facecolor=VERMILLION,
                           edgecolor="white", linewidth=0.8, zorder=3)
            else:
                ax.scatter(s["c"], s["cs"], s=area(s["l"]), marker="o", facecolor="white",
                           edgecolor=BLUE, linewidth=1.2, zorder=3)
        ax.set_xlabel("Mean cost per page (USD, log scale)")
        ax.set_ylabel("Repair score $S_{rep}$")
        ys = [s["cs"] for s in pts.values()]
        ax.set_ylim(max(0.0, min(ys) - 0.1), min(1.0, max(ys) + 0.1))
        xs = [s["c"] for s in pts.values()]
        ax.set_xlim(min(xs) / 2.2, max(xs) * 2.2)
        _style(ax)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(1, 10))))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"\\${v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        _place_labels(fig, ax, [(pts[m]["c"], pts[m]["cs"]) for m in order], [names.get(m, m) for m in order],
                      [area(pts[m]["l"]) for m in order], [(pts[m]["c"], pts[m]["cs"]) for m in front])
        nice = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600]
        lmin = min(s["l"] for s in pts.values())
        ref = [v for v in nice if lmin <= v <= lmax]
        if len(ref) >= 3:
            ref = [ref[0], ref[len(ref) // 2], ref[-1]]
        else:                              # narrow range: show its ends, 2 significant digits
            ref = sorted({float(f"{lmin:.2g}"), float(f"{lmax:.2g}")})
        handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=BLUE,
                          markeredgewidth=1.2, markersize=5, label="Closed (hollow circle)"),
                   Line2D([], [], marker="s", linestyle="", markerfacecolor=VERMILLION, markeredgecolor="white",
                          markersize=5, label="Open weights (filled square)")]
        if len(front) > 1:
            handles.append(Line2D([], [], color=INK, linewidth=0.8, label="Pareto set"))
        handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=MUTED,
                           markersize=math.sqrt(area(v)), label=f"{v:g} s median latency") for v in ref]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False,
                  handletextpad=0.4, columnspacing=1.0)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "fig_pareto.pdf"
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
    return path
