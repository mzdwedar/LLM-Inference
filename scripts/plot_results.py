"""Render the README charts from the raw exp2 benchmark JSONs.

    uv run --with matplotlib python scripts/plot_results.py

Writes results/figures/*.png. Every number plotted is read from results/exp2/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "results" / "exp2"
OUT = ROOT / "results" / "figures"

CONFIGS = [
    ("Baseline\n(bf16)", "baseline-fp16-cache", "baseline"),
    ("+ Chunked prefill\n& prefix caching", "optimized-fp16-cache", "optimized"),
    ("+ FP8 weights", "quantized-8fp-cache", "quantized"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # series 1
ORANGE = "#eb6834"  # series 2
MUTED = "#86b6ef"  # blue ramp step 250, for de-emphasised bars

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "font.family": "DejaVu Sans",
    }
)


def load(name):
    return json.loads((EXP / name).read_text())


def bench(cfg):
    return load(f"vllm-rag-{cfg}.json")["benchmarks"][0]["metrics"]


def stat(metric, key="median"):
    s = metric["successful"]
    return s["percentiles"][key] if key.startswith("p") else s[key]


def bar_panel(ax, values, title, fmt, highlight_last=True):
    labels = [c[0] for c in CONFIGS]
    colors = [MUTED] * (len(values) - 1) + [BLUE] if highlight_last else [BLUE] * len(values)
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_title(title, pad=10)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, labelsize=8.5)
    ax.set_ylim(0, max(values) * 1.18)
    for b, v in zip(bars, values):
        ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9.5, color=INK, fontweight="bold")


def headline():
    m = [bench(c[1]) for c in CONFIGS]
    ttft_p95 = [stat(x["time_to_first_token_ms"], "p95") / 1000 for x in m]
    out_tps = [x["output_tokens_per_second"]["successful"]["mean"] for x in m]
    rps = [x["requests_per_second"]["successful"]["mean"] for x in m]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    bar_panel(axes[0], ttft_p95, "p95 time to first token (s)  ↓",
              lambda v: f"{v:.1f} s" if v >= 1 else f"{v * 1000:.0f} ms")
    bar_panel(axes[1], out_tps, "Output throughput (tok/s)  ↑", lambda v: f"{v:,.0f}")
    bar_panel(axes[2], rps, "Achieved RPS of 8.0 offered  ↑", lambda v: f"{v:.2f}")
    axes[2].axhline(8.0, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    axes[2].set_ylim(0, 9.2)
    axes[2].annotate("offered load", (2.3, 8.0), xytext=(0, 4), textcoords="offset points",
                     ha="right", fontsize=8.5, color=INK_2)
    fig.suptitle("Llama-3.1-8B on one NVIDIA L4 (24 GB), RAG workload, Poisson 8 RPS",
                 x=0.012, y=0.99, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=3)
    fig.savefig(OUT / "headline.png", dpi=160)
    plt.close(fig)
    return ttft_p95, out_tps, rps


def queueing():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (label, _, scrape) in zip(axes, CONFIGS):
        rows = load(f"vllm_metrics_{scrape}.json")
        t0 = rows[0]["timestamp"]
        t = [r["timestamp"] - t0 for r in rows]
        ax.plot(t, [r["num_requests_running"] for r in rows], color=BLUE, linewidth=2, label="running")
        ax.plot(t, [r["num_requests_waiting"] for r in rows], color=ORANGE, linewidth=2, label="waiting")
        ax.set_title(label.replace("\n", " "), pad=8)
        ax.set_xlabel("seconds into run")
    axes[0].set_ylabel("requests")
    axes[2].legend(frameon=False, loc="center right", labelcolor=INK)
    fig.suptitle("Requests resident on the GPU vs queued (vLLM /metrics)",
                 x=0.012, y=0.99, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2)
    fig.savefig(OUT / "queueing.png", dpi=160)
    plt.close(fig)


def sweep():
    runs = load("vllm-rag-optimized-8fp-concurrent.json")["benchmarks"]
    streams = [r["config"]["strategy"]["streams"] for r in runs]
    ttft95 = [stat(r["metrics"]["time_to_first_token_ms"], "p95") for r in runs]
    # "total" counts tokens from requests cut off by the 30 s step, matching the README table
    tps = [r["metrics"]["output_tokens_per_second"]["total"]["mean"] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(streams)
        ax.set_xlabel("concurrent streams")
        ax.axvline(128, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].plot(streams, ttft95, color=BLUE, linewidth=2, marker="o", markersize=5)
    axes[0].set_yscale("log")
    axes[0].set_yticks([200, 500, 1000, 2000, 5000, 10000])
    axes[0].minorticks_off()
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms"))
    axes[0].axhline(1000, color=ORANGE, linewidth=1, linestyle=(0, (2, 2)))
    axes[0].annotate("1 s SLA", (1, 1000), xytext=(2, 4), textcoords="offset points", fontsize=8.5, color=INK_2)
    axes[0].set_title("p95 time to first token (log scale)", pad=8)
    axes[1].plot(streams, tps, color=BLUE, linewidth=2, marker="o", markersize=5)
    axes[1].set_title("Output throughput (tok/s)", pad=8)
    axes[1].set_ylim(0, max(tps) * 1.12)
    for ax in axes:
        ax.annotate("knee: 128", (128, ax.get_ylim()[1]), xytext=(-4, -14), textcoords="offset points",
                    ha="right", fontsize=8.5, color=INK_2)
    fig.suptitle("FP8 server, fixed-concurrency sweep (30 s per step)",
                 x=0.012, y=0.99, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=3)
    fig.savefig(OUT / "concurrency_sweep.png", dpi=160)
    plt.close(fig)
    return list(zip(streams, ttft95, tps))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("headline (ttft_p95_s, out_tok_s, rps):", headline())
    queueing()
    print("sweep (streams, ttft_p95_ms, out_tok_s):", sweep())
