#!/usr/bin/env python3
"""
Paper 9 — Figures 1-3.

Figure 1  Evidence sequence for deployment of a clinical AI intervention with
          heterogeneous effects. The paper's positioning, currently distributed
          across several paragraphs of Introduction and Methods.
Figure 2  The two independent constraints on local screening, as two panels,
          because conflating the axes is one of the paper's own findings.
Figure 3  Measured response across deployment conditions: the 52-cell section
          matrix with the whole-note comparison marked.

Output: 300 dpi PNG and vector PDF, greyscale-safe, no reliance on colour alone.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 300,
})
INK, MID, PALE = "#1a1a1a", "#6b6b6b", "#d9d9d9"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


# ------------------------------------------------------------------ Figure 1
def figure1():
    fig, ax = plt.subplots(figsize=(7.4, 7.9))
    ax.set_xlim(-1.2, 11.2); ax.set_ylim(4.9, 25.4); ax.axis("off")
    CX = 6.0                      # main column centre, leaves room at left

    def box(y, text, x=CX, w=7.0, h=1.5, style="solid", fs=8.5):
        ls = ":" if style == "dashed" else "-"
        ec = MID if style == "dashed" else INK
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.0, edgecolor=ec, facecolor="white",
            linestyle=ls, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=ec, zorder=3, linespacing=1.4)

    def arrow(y0, y1, x=CX):
        ax.add_patch(FancyArrowPatch(
            (x, y0), (x, y1), arrowstyle="-|>", mutation_scale=11,
            linewidth=1.0, color=INK, zorder=1))

    box(24.3, "Heterogeneous health-care setting\n"
              "biology · clinical practice · treatment effects · documentation",
        w=7.8, fs=8.0)
    arrow(23.55, 22.85)
    box(22.1, "AI intervention with potentially sign-changing effect\n"
              "beneficial in some deployments, harmful in others")
    arrow(21.35, 20.65)
    box(19.9, "Response-defining state of the deployed system\n"
              "not recoverable from published system descriptions\n"
              "in the empirical case examined", h=1.7, fs=8.2)
    arrow(19.0, 18.58)
    box(17.5, "Universal deployment has positive net benefit\n"
              "versus no action only where\n"
              "prevalence  p  >  p*  =  (|d_harm| + K/M) / (d_ben + |d_harm|)\n"
              "\u2009",
        h=2.45)
    ax.text(CX, 16.66, "≈  |d_harm| / (d_ben + |d_harm|)   when  K/M ≪ 1",
            ha="center", va="center", fontsize=7.1, style="italic", color=MID)

    # branch label sits beside the arrow, not on top of the next box
    ax.add_patch(FancyArrowPatch(
        (CX, 16.28), (CX, 15.68), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.0, color=INK, zorder=1))
    ax.text(CX + 0.35, 15.98,
            "state cannot be established from published evidence",
            ha="left", va="center", fontsize=7.3, style="italic", color=MID)

    box(15.0, "Local measurement of the deployed system", h=1.25)

    arrow(14.38, 13.5, x=3.4)
    arrow(14.38, 13.5, x=8.6)
    box(12.3, "A. Cross-condition\nevidence breadth\n\n"
              "corpus × query-format\nconditions scored",
        x=3.4, w=3.4, h=2.5, fs=8.0)
    box(12.3, "B. Within-site\ndocument sample\n\n"
              "documents scored\nper condition",
        x=8.6, w=3.4, h=2.5, fs=8.0)
    ax.text(CX, 12.3, "both\nrequired", ha="center", va="center",
            fontsize=7.6, style="italic", color=MID, linespacing=1.4)
    arrow(11.05, 10.2, x=3.4)
    arrow(11.05, 10.2, x=8.6)

    box(9.6, "Eligible deployment setting", h=1.2)
    arrow(9.0, 8.3)
    box(7.7, "Clinical-impact study", h=1.2)
    arrow(7.1, 6.4)
    box(5.8, "Setting-specific economic evaluation", h=1.2)

    # the insufficient route, clear of the main column
    box(19.9, "Model label\nTraining objective\nModel card",
        x=-0.05, w=2.5, h=2.4, style="dashed", fs=7.6)
    ax.plot([-1.05, 0.95], [18.85, 20.95], color=MID, linewidth=1.4,
            zorder=4, solid_capstyle="round", alpha=0.42)
    ax.plot([-1.05, 0.95], [20.95, 18.85], color=MID, linewidth=1.4,
            zorder=4, solid_capstyle="round", alpha=0.42)
    ax.add_patch(FancyArrowPatch(
        (1.30, 19.9), (2.45, 19.9), arrowstyle="-|>", mutation_scale=10,
        linewidth=1.0, color=MID, linestyle=":", zorder=1))
    ax.text(1.88, 20.05, "insufficient\nto infer", ha="center", va="bottom",
            fontsize=7.0, style="italic", color=MID, linespacing=1.3)

    save(fig, "figure1_evidence_sequence")


# ------------------------------------------------------------------ Figure 2
def figure2():
    heldout = pd.read_csv("expA_curve_heldout.csv")
    heldout = heldout[heldout.n_cond <= 6]
    ext = {1: 0.1244, 2: 0.0552, 3: 0.0261, 4: 0.0128, 5: 0.0065, 6: 0.0033}
    boot = pd.read_csv("appendices/appendix3_screen_resampling.csv")
    pooled = boot.groupby(["Nominal tier", "Documents sampled"])[
        "Agreement rate"].mean().unstack(0)
    req = {10: 0.4091, 20: 0.2631, 30: 0.1888, 50: 0.0915, 75: 0.0185}

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3))

    # -- Panel A
    a.axhline(0.5805, color=INK, linewidth=1.0, linestyle="--", zorder=1)
    a.text(6.0, 0.5805 * 1.13, "universal deployment  p* = 0.5805",
           ha="right", va="bottom", fontsize=7.2, color=INK)
    a.plot(heldout.n_cond, heldout.p_screen_star, "o-", color=INK,
           markersize=4.2, linewidth=1.3, label="derived on benchmark corpora",
           zorder=3)
    a.plot(list(ext), list(ext.values()), "s--", color=MID, markersize=4.0,
           linewidth=1.2, label="applied to external effect sizes", zorder=3)
    a.set_yscale("log")
    a.set_xlabel("corpus × query-format conditions scored")
    a.set_ylabel("minimum prevalence for positive net effect")
    a.set_title("A. Cross-condition evidence breadth", fontsize=9,
                loc="left", pad=8)
    a.set_xticks(range(1, 7))
    a.legend(frameon=False, fontsize=7.2, loc="lower left")
    a.grid(axis="y", linewidth=0.4, color=PALE, zorder=0)
    a.set_axisbelow(True)

    # -- Panel B
    n = list(pooled.index)
    b.plot(n, pooled[1], "o-", color=INK, markersize=4.2, linewidth=1.3,
           label="specificity (harmed subgroup)", zorder=3)
    b.plot(n, pooled[2], "^--", color=MID, markersize=4.0, linewidth=1.2,
           label="sensitivity (benefited subgroup)", zorder=3)
    b.axhline(0.5, color=MID, linewidth=0.8, linestyle=":", zorder=1)
    b.text(76, 0.505, "50% correct sign", ha="right", va="bottom",
           fontsize=7.0, color=MID)
    b.fill_between([8, 77], 0, 0.5, color=PALE, alpha=0.55, zorder=0)
    b.text(11, 0.38, "more often wrong than correct:\n"
                     "harm reported as benefit",
           fontsize=7.2, color=INK, va="center", linespacing=1.4)
    for x in (10, 75):
        b.annotate(f"{pooled.loc[x,1]:.3f}", (x, pooled.loc[x, 1]),
                   textcoords="offset points", xytext=(6, -10),
                   fontsize=7.0, color=INK)
    b.set_xlim(8, 79); b.set_ylim(0, 1.05)
    b.set_xlabel("documents scored within site")
    b.set_ylabel("probability of recovering the correct sign")
    b.set_title("B. Within-site document sample size", fontsize=9,
                loc="left", pad=8)
    b.set_xticks(n)
    lg = b.legend(fontsize=7.2, loc="lower right", frameon=True,
                  framealpha=1.0, edgecolor="none")
    lg.get_frame().set_facecolor("white")
    b.grid(axis="y", linewidth=0.4, color=PALE, zorder=0)
    b.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "figure2_screening_constraints")


# ------------------------------------------------------------------ Figure 3
def figure3():
    d = pd.read_csv("appendices/appendix2_transport_matrix.csv")
    lab = {("er_reason", "keyword"): "ER-Reason\nkeyword",
           ("er_reason", "nl"): "ER-Reason\nnatural lang.",
           ("mimic_discharge", "keyword"): "MIMIC\nkeyword",
           ("mimic_discharge", "nl"): "MIMIC\nnatural lang."}
    order = list(lab)
    hpi = d[d["Document variant"] == "hpi"]
    full = d[d["Document variant"] == "full512"]

    tiers = hpi[["Configuration", "Nominal tier"]].drop_duplicates()
    rows = (tiers[tiers["Nominal tier"] == 2].Configuration.sort_values().tolist()
            + tiers[tiers["Nominal tier"] == 1].Configuration.sort_values().tolist())
    n2 = (tiers["Nominal tier"] == 2).sum()

    M = np.full((len(rows), 4), np.nan)
    F = np.full((len(rows), 4), np.nan)
    for j, k in enumerate(order):
        for src, T in ((hpi, M), (full, F)):
            g = src[(src.Corpus == k[0]) & (src["Query format"] == k[1])]
            s = dict(zip(g.Configuration, g["dMRR@10"]))
            for i, r in enumerate(rows):
                if r in s:
                    T[i, j] = s[r]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    norm = TwoSlopeNorm(vmin=-0.28, vcenter=0, vmax=0.28)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(len(rows)):
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7.0,
                    color="white" if abs(v) > 0.15 else INK)
            # mark cells whose assignment changes under whole-note encoding
            tier = 2 if i < n2 else 1
            f = F[i, j]
            if not np.isnan(f) and ((tier == 2 and f < 0) or (tier == 1 and f > 0)):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="black", linewidth=2.0,
                                           linestyle=(0, (2, 1.4))))

    ax.axhline(n2 - 0.5, color=INK, linewidth=1.6)
    ax.set_xticks(range(4)); ax.set_xticklabels([lab[k] for k in order],
                                                fontsize=7.4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7.4)
    # subgroup brackets outside the tick labels, one per block
    for y0, y1, lab in ((-0.4, n2 - 0.6, "nominal\nbenefit tier"),
                        (n2 - 0.4, len(rows) - 0.6, "nominal\nharm tier")):
        ax.annotate("", xy=(-0.30, y0), xytext=(-0.30, y1),
                    xycoords=("axes fraction", "data"),
                    textcoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", linewidth=1.0,
                                    color=MID, shrinkA=0, shrinkB=0))
        ax.text(-0.335, (y0 + y1) / 2, lab, rotation=90, ha="center",
                va="center", fontsize=7.4, color=MID, linespacing=1.3,
                transform=ax.get_yaxis_transform())
    ax.set_title("Measured response (ΔMRR@10) by deployment condition",
                 fontsize=9.5, loc="left", pad=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("ΔMRR@10 after correction", fontsize=7.8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "figure3_response_heatmap")


if __name__ == "__main__":
    print("figures:")
    figure1(); figure2(); figure3()
