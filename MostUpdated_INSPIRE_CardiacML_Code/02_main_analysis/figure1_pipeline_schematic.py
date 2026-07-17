#!/usr/bin/env python3
"""
Central illustration: INSPIRE → engineered features → phenotyping + supervised prediction.
Run from repo root or this directory:
  python central_illustration.py
Outputs: central_illustration.pdf and .png in the same folder as this script.
Requires: matplotlib
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent


def box(ax, xy, w, h, text, fontsize=8, facecolor="#f4f4f5", edgecolor="#333333"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transData,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.15,
        wrap=True,
    )


def arrow(ax, start, end, color="#444444"):
    a = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Row y positions (bottom of boxes)
    y_main = 1.35
    h = 1.35
    w_narrow = 1.35
    w_mid = 1.55
    w_wide = 1.75

    # --- Top row: data → features ---
    box(
        ax,
        (0.25, y_main + 1.15),
        w_wide,
        h,
        "INSPIRE cohort\nn = 2,747\nCardiac surgery\n2011–2020",
        facecolor="#e8eef5",
    )
    box(
        ax,
        (2.15, y_main + 1.15),
        w_wide + 0.35,
        h,
        "Minute-level\nintraoperative streams\n(vitals, meds, labs)",
        facecolor="#eef2f7",
    )
    box(
        ax,
        (4.35, y_main + 1.15),
        w_mid + 0.5,
        h,
        "Feature engineering\nOrthogonal set\n(no outcome leakage)",
        facecolor="#f0f4fa",
    )

    arrow(ax, (2.0, y_main + 1.15 + h / 2), (2.15, y_main + 1.15 + h / 2))
    arrow(ax, (3.95, y_main + 1.15 + h / 2), (4.35, y_main + 1.15 + h / 2))

    # Merge arrow down to "feature matrix"
    box(
        ax,
        (3.85, y_main - 0.05),
        2.3,
        0.85,
        "Patient × feature matrix\n(operative window only)",
        fontsize=7.5,
        facecolor="#ffffff",
    )
    ax.plot([5.0, 5.0], [y_main + 1.15, y_main + 0.95], color="#444444", lw=1.2)
    ax.add_patch(
        FancyArrowPatch(
            (5.0, y_main + 0.95),
            (5.0, y_main + 0.78),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color="#444444",
        )
    )

    # --- Bottom: two branches ---
    # Left branch: unsupervised
    box(
        ax,
        (0.35, y_main),
        w_mid,
        h,
        "PCA → K-means\n(k = 2 phenotypes)",
        facecolor="#e8f5e9",
    )
    box(
        ax,
        (0.25, y_main - 1.25),
        w_mid + 0.2,
        h,
        "Trajectory phenotypes\nLower- vs higher-risk\n(validation: bootstrap,\nfeature-split, temporal)",
        fontsize=7.5,
        facecolor="#c8e6c9",
    )
    arrow(ax, (1.22, y_main + h / 2), (1.22, y_main + 0.02))
    ax.add_patch(
        FancyArrowPatch(
            (1.22, y_main + 0.02),
            (1.22, y_main - 0.05),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color="#444444",
        )
    )

    # Right branch: supervised
    box(
        ax,
        (6.15, y_main),
        w_wide + 0.4,
        h,
        "Random forest\n(binary classifiers)",
        facecolor="#fff3e0",
    )
    box(
        ax,
        (6.05, y_main - 1.25),
        w_wide + 0.55,
        h,
        "Endpoints\nICU stay >3 d\nIn-hospital mortality\n(random + temporal test)",
        fontsize=7.5,
        facecolor="#ffe0b2",
    )
    arrow(ax, (6.85, y_main + h / 2), (6.85, y_main + 0.02))
    ax.add_patch(
        FancyArrowPatch(
            (6.85, y_main + 0.02),
            (6.85, y_main - 0.05),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color="#444444",
        )
    )

    # Split from center hub to branches
    hub = (5.0, y_main + 0.38)
    arrow(ax, hub, (1.22, y_main + h), color="#2e7d32")
    arrow(ax, hub, (6.85, y_main + h), color="#e65100")

    ax.text(
        0.15,
        3.75,
        "Central study design",
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.15,
        3.45,
        "Dynamic intraoperative physiology → unsupervised phenotypes and supervised outcome prediction",
        fontsize=8,
        style="italic",
        va="top",
        color="#333333",
    )

    plt.tight_layout()
    pdf = OUT_DIR / "central_illustration.pdf"
    png = OUT_DIR / "central_illustration.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close()
    print(f"Wrote {pdf}\nWrote {png}")


if __name__ == "__main__":
    main()
