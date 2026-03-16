"""
Create a teaching chart for the 1-minute spot stop-loss scenarios discussed in
the ORB short breakout example.

Usage:
    python backtesting/plot_spot_sl_examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "results" / "spot_sl_touch_examples.png"

OR_LOW = 17950

THEME = {
    "bg": "#0F172A",
    "panel": "#111827",
    "border": "#334155",
    "grid": "#1F2937",
    "text": "#E5E7EB",
    "muted": "#94A3B8",
    "up": "#22C55E",
    "down": "#EF4444",
    "line": "#F59E0B",
    "close_logic": "#38BDF8",
    "wick_logic": "#F97316",
    "highlight": "#FACC15",
}


SCENARIOS = [
    {
        "title": "1. Clean breach, close above",
        "candles": [
            {"open": 17900, "high": 17960, "low": 17895, "close": 17955},
        ],
        "note": "Both checks stop out.",
    },
    {
        "title": "2. Wick touch, close below",
        "candles": [
            {"open": 17920, "high": 17952, "low": 17915, "close": 17930},
        ],
        "note": "This is the miss in the current close-based logic.",
    },
    {
        "title": "3. Gap open above",
        "candles": [
            {"open": 17960, "high": 17970, "low": 17955, "close": 17965},
        ],
        "prev_close": 17920,
        "note": "Both checks stop out, but the fill is already worse.",
    },
    {
        "title": "4. Gradual approach, no touch",
        "candles": [
            {"open": 17910, "high": 17940, "low": 17905, "close": 17930},
            {"open": 17930, "high": 17945, "low": 17925, "close": 17940},
            {"open": 17940, "high": 17948, "low": 17935, "close": 17944},
            {"open": 17944, "high": 17944, "low": 17924, "close": 17932},
        ],
        "note": "No candle touches the stop, so the trade stays open.",
    },
    {
        "title": "5. Exact touch",
        "candles": [
            {"open": 17935, "high": 17950, "low": 17930, "close": 17940},
        ],
        "note": "A touch should count, so the wick rule must use >=.",
    },
]


def style_axis(ax):
    ax.set_facecolor(THEME["panel"])
    ax.grid(axis="y", color=THEME["grid"], linewidth=0.7)
    ax.tick_params(colors=THEME["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(THEME["border"])


def draw_candle(ax, x, candle):
    color = THEME["up"] if candle["close"] >= candle["open"] else THEME["down"]
    ax.plot([x, x], [candle["low"], candle["high"]], color=color, lw=2, zorder=2)
    body_low = min(candle["open"], candle["close"])
    body_height = max(abs(candle["close"] - candle["open"]), 0.5)
    ax.add_patch(
        Rectangle(
            (x - 0.28, body_low),
            0.56,
            body_height,
            facecolor=color,
            edgecolor=color,
            zorder=3,
        )
    )


def evaluate_stop(candles, stop_level):
    close_hit_idx = next(
        (idx for idx, candle in enumerate(candles) if candle["close"] >= stop_level),
        None,
    )
    wick_hit_idx = next(
        (idx for idx, candle in enumerate(candles) if candle["high"] >= stop_level),
        None,
    )
    return close_hit_idx, wick_hit_idx


def add_logic_labels(ax, candles, close_hit_idx, wick_hit_idx):
    close_label = (
        f"Close-based: STOP on candle {close_hit_idx + 1}"
        if close_hit_idx is not None
        else "Close-based: NO STOP"
    )
    wick_label = (
        f"Wick-based: STOP on candle {wick_hit_idx + 1}"
        if wick_hit_idx is not None
        else "Wick-based: NO STOP"
    )
    ax.text(
        0.02,
        0.96,
        close_label,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=THEME["close_logic"],
        bbox={"facecolor": THEME["bg"], "edgecolor": THEME["close_logic"], "pad": 4},
    )
    ax.text(
        0.02,
        0.83,
        wick_label,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=THEME["wick_logic"],
        bbox={"facecolor": THEME["bg"], "edgecolor": THEME["wick_logic"], "pad": 4},
    )


def plot_scenario(ax, scenario):
    style_axis(ax)
    candles = scenario["candles"]
    close_hit_idx, wick_hit_idx = evaluate_stop(candles, OR_LOW)

    for idx, candle in enumerate(candles):
        draw_candle(ax, idx, candle)
        ax.text(
            idx,
            candle["low"] - 4,
            f"{idx + 1}",
            color=THEME["muted"],
            fontsize=8,
            ha="center",
            va="top",
        )

    if "prev_close" in scenario:
        ax.scatter(
            -0.7,
            scenario["prev_close"],
            s=50,
            color=THEME["highlight"],
            zorder=4,
            label="Prev close",
        )
        ax.plot(
            [-0.7, 0],
            [scenario["prev_close"], candles[0]["open"]],
            color=THEME["highlight"],
            lw=2,
            ls="--",
            zorder=1,
        )
        ax.text(
            -0.7,
            scenario["prev_close"] - 6,
            "prev close",
            color=THEME["highlight"],
            fontsize=8,
            ha="center",
        )

    ax.axhline(OR_LOW, color=THEME["line"], lw=2, ls="--")
    ax.text(
        0.98,
        OR_LOW + 1.5,
        f"or_low / stop = {OR_LOW}",
        color=THEME["line"],
        fontsize=9,
        ha="right",
        va="bottom",
    )

    if wick_hit_idx is not None:
        ax.scatter(
            wick_hit_idx,
            candles[wick_hit_idx]["high"],
            s=80,
            color=THEME["wick_logic"],
            edgecolors=THEME["text"],
            zorder=5,
        )

    if close_hit_idx is not None:
        ax.scatter(
            close_hit_idx,
            candles[close_hit_idx]["close"],
            s=70,
            marker="s",
            color=THEME["close_logic"],
            edgecolors=THEME["text"],
            zorder=5,
        )

    add_logic_labels(ax, candles, close_hit_idx, wick_hit_idx)

    all_prices = [price for candle in candles for price in candle.values()]
    if "prev_close" in scenario:
        all_prices.append(scenario["prev_close"])
    pad = 18
    ax.set_ylim(min(all_prices) - pad, max(all_prices) + pad)
    ax.set_xlim(-1.0 if "prev_close" in scenario else -0.6, len(candles) - 0.4)
    ax.set_title(scenario["title"], color=THEME["text"], fontsize=12, loc="left", pad=10)
    ax.set_xlabel("1-minute candle sequence", color=THEME["muted"], fontsize=9)
    ax.set_ylabel("Spot price", color=THEME["muted"], fontsize=9)
    ax.text(
        0.02,
        0.05,
        scenario["note"],
        transform=ax.transAxes,
        color=THEME["text"],
        fontsize=9,
        va="bottom",
    )


def add_summary_panel(ax):
    style_axis(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    lines = [
        "Short trade stop rule being tested",
        "Entry happens after a 5-minute close below or_low.",
        "While the trade is open, 1-minute spot candles are checked for a return to or_low.",
        "",
        "Current code in engine.py",
        "short stop on 1m close >= spot_sl",
        "This misses wick touches that never close above the level.",
        "",
        "Recommended live-trading-aligned rule",
        "short stop on 1m high >= spot_sl",
        "This catches both a clean reclaim and an intrabar touch.",
        "",
        "Most important case: Scenario 2",
        "The wick reaches the stop, so a real stop would trigger,",
        "but a close-based backtest keeps the trade alive incorrectly.",
    ]

    y = 0.94
    for line in lines:
        color = THEME["muted"] if not line or line.endswith("rule") else THEME["text"]
        if line.startswith("Current code"):
            color = THEME["close_logic"]
        elif line.startswith("Recommended"):
            color = THEME["wick_logic"]
        elif line.startswith("Most important"):
            color = THEME["highlight"]

        ax.text(
            0.03,
            y,
            line,
            color=color,
            fontsize=10,
            va="top",
            family="monospace" if line.startswith("if sig_dir") else None,
        )
        y -= 0.065 if line else 0.04


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), facecolor=THEME["bg"])
    axes = axes.flatten()

    for ax, scenario in zip(axes, SCENARIOS):
        plot_scenario(ax, scenario)

    add_summary_panel(axes[-1])

    fig.suptitle(
        "Why a wick-based spot stop catches the missed short trade stop",
        color=THEME["text"],
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.955,
        "Orange dashed line = or_low / stop level. Blue square = close-based hit. Orange dot = wick/high-based hit.",
        color=THEME["muted"],
        fontsize=10,
        ha="center",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_PATH, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
