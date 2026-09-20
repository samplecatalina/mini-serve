"""The ablation figures, drawn from the result rows and nothing else.

Every line and bar here reads a row out of ``results/<device>/*.csv``; nothing is
typed in. The reference line on the latency panels is the floor a decode step has
on this GPU -- the bytes it must move divided by the copy bandwidth measured on
that same GPU (``roofline.csv``) -- so a bar's distance from it is the part of
the step that is not memory traffic.

    python -m bench.charts --device l40s          # -> results/l40s/figures/*.png

Runs in the load generator's image, which has matplotlib; the engine's image does
not need it.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Qwen3-0.6B, BF16. A decode step reads every layer weight and the output
# projection once (28 layers 440,466,432 params + a tied lm_head of 155,582,464),
# and the whole KV of every running sequence: 2 x 28 layers x 8 KV heads x 128 x 2 B.
DECODE_WEIGHT_BYTES = 596_049_920 * 2
KV_BYTES_PER_TOKEN = 2 * 28 * 8 * 128 * 2

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdcd8"
# Categorical slots 1-3, in fixed order. Validated as a set for every pair, in
# light mode, against this surface. Aqua sits below 3:1 on it, so every mark that
# uses these carries a visible label rather than relying on the fill alone.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
# A second encoding beside colour: lines that coincide stay tellable apart, and
# the pair that is hardest under colour-vision deficiency is separated by shape.
MARKERS = ("o", "s", "^")


def read(device: str, name: str) -> list[dict]:
    path = f"results/{device}/{name}"
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def copy_bandwidth(device: str) -> float:
    """Bytes per second this GPU reached on the largest copy measured on it."""
    rows = [r for r in read(device, "roofline.csv") if r["kernel"] == "copy"]
    if not rows:
        raise SystemExit(f"results/{device}/roofline.csv has no copy rows: no floor to draw against")
    return float(max(rows, key=lambda r: int(r["bytes"]))["gb_s"]) * 1e9


def floor_ms(batch: float, ctx: float, bw: float) -> float:
    return (DECODE_WEIGHT_BYTES + batch * ctx * KV_BYTES_PER_TOKEN) / bw * 1e3


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def figure(title: str, subtitle: str, panels: int = 2):
    """A figure whose title block is a fixed band above the axes, so a two-line
    subtitle never lands on the plot or runs off the edge."""
    width = 9.0 if panels == 1 else 5.6 * panels
    lines = subtitle.count("\n") + 1
    # The band holds the title, the subtitle and the panel titles that sit just
    # above the axes; without the last term a panel title lands on the subtitle.
    head = 0.62 + 0.20 * lines + 0.34
    fig, axes = plt.subplots(1, panels, figsize=(width, 4.0 + head), facecolor=SURFACE)
    fig.subplots_adjust(top=1 - head / (4.0 + head))
    fig.text(0.011, 1 - 0.30 / (4.0 + head), title, color=INK, fontsize=13, ha="left", va="top")
    fig.text(0.011, 1 - 0.62 / (4.0 + head), subtitle, color=INK_2, fontsize=9.5, ha="left", va="top",
             linespacing=1.45)
    return fig, (axes if panels > 1 else [axes])


def bars(ax, groups: list[str], series: dict[str, list[float]], fmt="{:.0f}") -> None:
    """Grouped bars, one colour per series, every bar labelled (identity is never colour alone)."""
    n = len(series)
    span = 0.78              # of the unit between groups, so groups never touch
    width = span / n - 0.02  # a 2px-ish surface gap between adjacent bars
    for i, (label, values) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * (span / n) for g in range(len(groups))]
        ax.bar(xs, values, width, label=label, color=SERIES[i], edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, values):
            ax.annotate(fmt.format(v), (x, v), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, color=INK_2)


def dumbbell(ax, groups: list[str], series: dict[str, list[float]], fmt="{:.0f}") -> None:
    """Two values per group as dots joined by a line.

    Bars would be wrong here: this panel is on a log axis, and a bar's length
    only means something measured from zero. A dot's position is the value.
    """
    keys = list(series)
    for g in range(len(groups)):
        lo, hi = sorted(series[k][g] for k in keys)
        ax.plot([g, g], [lo, hi], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
    for i, (label, values) in enumerate(series.items()):
        ax.plot(range(len(groups)), values, linestyle="none", marker="o", markersize=11,
                color=SERIES[i], markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        for g, v in enumerate(values):
            ax.annotate(fmt.format(v), (g, v), textcoords="offset points", xytext=(12, -3),
                        fontsize=8.5, color=INK_2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, color=INK_2)
    ax.set_xlim(-0.5, len(groups) - 0.5)


def save(fig, device: str, name: str) -> str:
    out = f"results/{device}/figures"
    os.makedirs(out, exist_ok=True)
    path = f"{out}/{name}.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    return path


def pick(rows, arm, conc):
    for r in rows:
        if r["arm"] == arm and int(r["concurrency"]) == conc:
            return r
    return None


def on_off_figure(device: str, csv_name: str, on: str, off: str, names: tuple[str, str],
                  title: str, subtitle: str,
                  latency_key: str, latency_label: str, log_latency: bool, bw: float):
    rows = read(device, csv_name)
    if not rows:
        return None
    concs = sorted({int(r["concurrency"]) for r in rows})
    if not all(pick(rows, a, c) for a in (on, off) for c in concs):
        return None
    labels = [f"{c} in flight" for c in concs]

    fig, (ax1, ax2) = figure(title, subtitle)
    bars(ax1, labels, {
        names[0]: [float(pick(rows, on, c)["output_tok_s"]) for c in concs],
        names[1]: [float(pick(rows, off, c)["output_tok_s"]) for c in concs],
    })
    style(ax1, "Output throughput", "tokens/s")
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

    pairs = {
        names[0]: [float(pick(rows, on, c)[latency_key]) for c in concs],
        names[1]: [float(pick(rows, off, c)[latency_key]) for c in concs],
    }
    if log_latency:
        dumbbell(ax2, labels, pairs)
        style(ax2, latency_label, "ms")
        ax2.set_yscale("log")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax2.grid(axis="y", which="minor", color=GRID, linewidth=0.4)
        ax2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    else:
        bars(ax2, labels, pairs, fmt="{:.1f}")
        style(ax2, latency_label, "ms")
        # What the memory system alone would take, at the batch each bar ran at.
        for i, c in enumerate(concs):
            r = pick(rows, on, c)
            f = floor_ms(float(r["running_batch_mean"]), 1013 + 128, bw)
            ax2.plot([i - 0.40, i + 0.40], [f, f], color=INK_2, linewidth=1.6, linestyle=(0, (4, 3)),
                     zorder=5, label="memory-traffic floor" if i == 0 else None)
            ax2.annotate(f"floor {f:.1f}", (i + 0.40, f), textcoords="offset points", xytext=(2, 2),
                         fontsize=8, color=INK_2)
        ax2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    return fig


def host_cost_figure(device: str, bw: float):
    """One picture of where a decode step's time goes, at both ends of the batch range."""
    graph, overlap, main = (read(device, n) for n in
                            ("m4_2_cuda_graph.csv", "m4_2_overlap.csv", "m4_2_main.csv"))
    if not (graph and overlap and main):
        return None
    concs = sorted({int(r["concurrency"]) for r in graph})
    labels = [f"{c} in flight" for c in concs]
    series = {
        "graphs + overlap": [float(pick(main, "miniserve", c)["tpot_p50_ms"]) for c in concs],
        "without graphs": [float(pick(graph, "graph_off", c)["tpot_p50_ms"]) for c in concs],
        "without overlap": [float(pick(overlap, "overlap_off", c)["tpot_p50_ms"]) for c in concs],
    }
    fig, (ax,) = figure(
        "What a decode step spends its time on",
        "Time between tokens with each host-side mechanism disabled in turn; the dashed line is the "
        "memory traffic alone.\nGraphs matter where a step is shorter than launching it. Overlap "
        "matters where the host work grows with the batch.",
        panels=1,
    )
    bars(ax, labels, series, fmt="{:.1f}")
    style(ax, "", "ms between tokens")
    for i, c in enumerate(concs):
        f = floor_ms(float(pick(main, "miniserve", c)["running_batch_mean"]), 1013 + 128, bw)
        ax.plot([i - 0.42, i + 0.42], [f, f], color=INK_2, linewidth=1.6, linestyle=(0, (4, 3)), zorder=5,
                label="memory-traffic floor" if i == 0 else None)
        ax.annotate(f"floor {f:.1f}", (i + 0.42, f), textcoords="offset points", xytext=(3, 1),
                    fontsize=8, color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    return fig


def policy_figure(device: str):
    """Throughput against pool size, one line per policy, one panel per workload.

    The two workloads are not on one axis: they differ in how many requests they
    offer, so a point from one says nothing about a point from the other. What is
    comparable is the shape of each panel.
    """
    panels = [(n, t) for n, t in
              (("m4_2_policy_small.csv", "64 requests"), ("m4_2_policy_large.csv", "512 requests"))
              if read(device, n)]
    if not panels:
        return None
    names = {"policy_fcfs": "first come, first served", "policy_sjf": "shortest job first",
             "policy_cache": "longest cached prefix first"}

    fig, axes = figure(
        "Scheduling policy against KV pool size",
        "Requests in groups sharing a 512-token prompt prefix - the thing a cache-aware policy exists "
        "to exploit.\nEvery point is the median of three runs. First-come and shortest-job coincide "
        "wherever the pool is not the binding constraint.",
        panels=len(panels),
    )
    for ax, (name, workload) in zip(axes, panels):
        rows = read(device, name)
        pools = sorted({int(r["kv_pool_tokens"]) for r in rows})
        for i, arm in enumerate(("policy_fcfs", "policy_sjf", "policy_cache")):
            xs, ys = [], []
            for pool in pools:
                vals = sorted(float(r["output_tok_s"]) for r in rows
                              if r["arm"] == arm and int(r["kv_pool_tokens"]) == pool)
                if vals:
                    xs.append(pool)
                    ys.append(vals[len(vals) // 2])
            if not xs:
                continue
            ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=names[arm])
            # Staggered, because two policies that land on the same number would
            # otherwise print their labels on top of each other.
            ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(9, [-11, 1, 6][i]), fontsize=8.5, color=INK_2)
        style(ax, workload, "output tokens/s")
        ax.set_xscale("log")
        ax.set_xlabel("KV pool (tokens)", color=INK_2, fontsize=9)
        ax.set_xticks(pools)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{round(v / 1000):g}k"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_ylim(0, None)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="l40s")
    args = ap.parse_args()
    bw = copy_bandwidth(args.device)
    made = []

    f = on_off_figure(
        args.device, "m4_2_overlap.csv", "overlap_on", "overlap_off", ("overlap on", "overlap off"),
        "Overlapping the host with the GPU",
        "Launching each step before reading back the previous one. Decode CUDA Graphs are on in both "
        "arms,\nso what is left to hide is scheduling, sampling and request bookkeeping - and that grows "
        "with the batch.",
        "tpot_p50_ms", "Time between tokens", False, bw)
    if f:
        made.append(save(f, args.device, "m4_2_overlap"))

    f = on_off_figure(
        args.device, "m4_2_chunked.csv", "chunk_2048", "chunk_off",
        ("chunks of 2048 tokens", "whole prefills"),
        "Cutting prefills into chunks",
        "Without it a long prompt is computed in a step of its own, and every running sequence waits for "
        "it.\nThe cost lands on the wait for a first token, and on how many sequences the engine can keep "
        "running.",
        "ttft_p50_ms", "Time to first token (log scale)", True, bw)
    if f:
        made.append(save(f, args.device, "m4_2_chunked"))

    f = policy_figure(args.device)
    if f:
        made.append(save(f, args.device, "m4_2_policy"))

    f = host_cost_figure(args.device, bw)
    if f:
        made.append(save(f, args.device, "m4_2_host_cost"))

    if not made:
        raise SystemExit(f"no result rows under results/{args.device}: nothing to draw")
    for p in made:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
