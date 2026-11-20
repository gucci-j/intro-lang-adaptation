#!/usr/bin/env python3
"""
Generate a slide-ready two-panel chart from eval_results.json for Slide 9:
  Left:  Belebele reading comprehension accuracy (English, French ref., Amharic)
  Right: FLORES+ translation chrF++ (En<->Fr ref., En<->Cy, En<->Am)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    data = json.loads(Path("eval_results.json").read_text())
    model_name = data["model"].split("/")[-1]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Sized for the content area of a 16:9 widescreen slide (13.33" x 7.5")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.8), dpi=300)
    fig.patch.set_facecolor("white")

    # --- Panel 1: Belebele ---
    bel = data["belebele"]
    bel_spec = [
        ("eng_Latn", "English", "#2563eb"),
        ("fra_Latn", "French\n(ref.)", "#60a5fa"),
        ("amh_Ethi", "Amharic", "#f59e0b"),
    ]
    x1 = np.arange(len(bel_spec))
    ax1.axhline(25.0, color="#94a3b8", linestyle="--", linewidth=1.4)

    for idx, (code, label, color) in enumerate(bel_spec):
        val = bel[code]["accuracy"]
        ax1.bar(idx, val, color=color, width=0.52, edgecolor="none")
        ax1.text(
            idx,
            val + 2.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13,
        )

    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for _, lbl, _ in bel_spec], fontsize=13)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("Accuracy (%)", fontsize=13)
    ax1.set_title(f"Belebele Reading Comprehension (n={data['belebele_n']})", fontweight="bold", fontsize=14.5, pad=12)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.text(
        2.35,
        27.0,
        "Random chance (25%)",
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="#64748b",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0),
    )

    # --- Panel 2: FLORES+ ---
    tr = data["flores_plus"]["translations"]
    ref_langs = [("fra_Latn", "French\n(ref.)"), ("cym_Latn", "Welsh"), ("amh_Ethi", "Amharic")]
    en_to_x = [tr[f"eng_Latn->{code}"]["chrf++"] for code, _ in ref_langs]
    x_to_en = [tr[f"{code}->eng_Latn"]["chrf++"] for code, _ in ref_langs]
    labels2 = [lbl for _, lbl in ref_langs]

    x2 = np.arange(len(labels2))
    w = 0.35
    b_into = ax2.bar(x2 - w / 2, en_to_x, width=w, label="English → Target", color="#2563eb")
    b_from = ax2.bar(x2 + w / 2, x_to_en, width=w, label="Target → English", color="#93c5fd")

    ax2.set_ylim(0, 88)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=13)
    ax2.set_ylabel("chrF++", fontsize=13)
    ax2.set_title(f"FLORES+ Translation (n={data['flores_n']})", fontweight="bold", fontsize=14.5, pad=12)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", frameon=False, fontsize=11.5)

    for bars in (b_into, b_from):
        for bar in bars:
            val = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
            )

    fig.suptitle(f"{model_name} Performance Across Languages (Zero-Shot, Non-Thinking Mode)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=2.5)
    plt.savefig("slide9_performance_gap.png", dpi=300)
    plt.savefig("slide9_performance_gap.svg")
    print("Saved slide9_performance_gap.png and slide9_performance_gap.svg")


if __name__ == "__main__":
    main()
