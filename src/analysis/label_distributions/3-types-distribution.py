"""
Analyze the distribution of type_1, type_2, and type_3 output scores
from the papers dataset and generate publication-quality visualizations.
Also prints values needed for the LaTeX paper tables.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import Counter

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
DATAFRAME_DIR = DATA_DIR / "train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet"

FIGURE_OUTPUT_DIR = OUTPUT_DIR / "analysis/all_papers_distributions"

# Headroom above the tallest bar, so value labels fit inside the axes and
# do not overlap with the title when tight_layout() runs.
Y_HEADROOM_SINGLE = 1.25   # single-panel figures
Y_HEADROOM_MULTI  = 1.2   # multi-panel combined figures (titles are closer)


def extract_all_scores(df: pd.DataFrame):
    """Extract all valid type_1, type_2, type_3 values from references across all papers."""
    all_type1_scores = []
    all_type2_scores = []
    all_type3_scores = []
    total_instances = 0
    total_papers_with_refs = 0

    for refs in df["references"]:
        if isinstance(refs, str):
            refs = json.loads(refs)
        if not refs:
            continue
        total_papers_with_refs += 1
        for ref in refs:
            total_instances += 1

            # Type 1
            val1 = ref.get("type_1_output", "")
            if val1 and str(val1).lower() not in ("nan", "", "none"):
                try:
                    score = float(val1)
                    if not np.isnan(score):
                        all_type1_scores.append(score)
                except (ValueError, TypeError):
                    pass

            # Type 2
            val2 = ref.get("type_2_output", "")
            if val2 and str(val2).lower() not in ("nan", "", "none"):
                try:
                    score = float(val2)
                    if not np.isnan(score):
                        all_type2_scores.append(score)
                except (ValueError, TypeError):
                    pass

            # Type 3
            val3 = ref.get("type_3_output", "")
            if val3 and str(val3).lower() not in ("nan", "", "none"):
                try:
                    score = float(val3)
                    if not np.isnan(score):
                        all_type3_scores.append(score)
                except (ValueError, TypeError):
                    pass

    return (
        np.array(all_type1_scores),
        np.array(all_type2_scores),
        np.array(all_type3_scores),
        total_instances,
        total_papers_with_refs,
    )


def print_latex_values(type1: np.ndarray, type2: np.ndarray, type3: np.ndarray, total_instances: int, total_papers: int):
    """Print all values needed to fill in the LaTeX paper."""

    print("\n" + "=" * 70)
    print("VALUES FOR LATEX PAPER")
    print("=" * 70)

    print(f"\n--- Section 2.2 Text ---")
    print(f"Total citation instances: {total_instances:,}")
    print(f"Total papers with references: {total_papers:,}")

    # ---- Table 1: Type I distribution ----
    print(f"\n--- Table 1: Type I Distribution (Binary) ---")
    t1_ones = int(np.sum(type1 == 1.0))
    t1_zeros = int(np.sum(type1 == 0.0))
    t1_total = len(type1)
    print(f"Type I = 1 (baseline):  {t1_ones:,}  ({t1_ones/t1_total*100:.1f}%)")
    print(f"Type I = 0:             {t1_zeros:,}  ({t1_zeros/t1_total*100:.1f}%)")
    print(f"Type I total valid:     {t1_total:,}")

    # ---- Table 1: Type II distribution ----
    print(f"\n--- Table 1: Type II Distribution (1-5 Scale) ---")
    t2_total = len(type2)
    for score in [5, 4, 3, 2, 1]:
        count = int(np.sum(type2 == float(score)))
        pct = count / t2_total * 100
        label = {5: "core", 4: "", 3: "", 2: "", 1: "background"}.get(score, "")
        suffix = f" ({label})" if label else ""
        print(f"Type II = {score}{suffix}:  {count:,}  ({pct:.1f}%)")
    print(f"Type II total valid:    {t2_total:,}")

    # ---- Type II must-cite (4 or 5) ----
    t2_mustcite = int(np.sum(type2 >= 4.0))
    print(f"Type II must-cite (>=4): {t2_mustcite:,}  ({t2_mustcite/t2_total*100:.1f}%)")

    # ---- Type III distribution ----
    print(f"\n--- Type III Distribution (Frequency) ---")
    t3_total = len(type3)
    print(f"Type III total valid:   {t3_total:,}")
    print(f"Min:    {type3.min():.0f}")
    print(f"Max:    {type3.max():.0f}")
    print(f"Mean:   {type3.mean():.2f}")
    print(f"Median: {np.median(type3):.0f}")
    print(f"Std:    {type3.std():.2f}")

    print(f"\n--- Type III Value Counts ---")
    counts = Counter(type3)
    for val in sorted(counts.keys()):
        pct = counts[val] / t3_total * 100
        print(f"  {val:5.0f}:  {counts[val]:>8,}  ({pct:5.1f}%)")

    print(f"\n--- Type III Percentiles ---")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(type3, p):.0f}")

    print(f"\n--- Type III Candidate Thresholds ---")
    for t in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        pos = int(np.sum(type3 >= t))
        neg = int(np.sum(type3 < t))
        ratio = pos / neg if neg > 0 else float('inf')
        print(f"  >= {t:2d}:  pos={pos:,} ({pos/t3_total*100:.1f}%)  neg={neg:,} ({neg/t3_total*100:.1f}%)  ratio={ratio:.3f}")


def setup_plot_style():
    """Set up publication-ready matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def plot_type1_distribution(type1: np.ndarray, output_dir: Path):
    """Bar chart for Type I binary distribution."""
    fig_dir = output_dir / "type_1_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))

    counts = [int(np.sum(type1 == 0.0)), int(np.sum(type1 == 1.0))]
    labels = ["0 (Non-baseline)", "1 (Baseline)"]
    colors = ['#A8DADC', '#2E86AB']

    bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=0.5)

    for bar, count in zip(bars, counts):
        pct = count / len(type1) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Type I Label")
    ax.set_ylabel("Number of References")
    ax.set_title("Distribution of Type I (Experimental Baseline)")
    ax.set_ylim(0, max(counts) * Y_HEADROOM_SINGLE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    fig.savefig(fig_dir / "type1_distribution.png")
    fig.savefig(fig_dir / "type1_distribution.pdf")
    print(f"\nSaved Type I plot to {fig_dir / 'type1_distribution.png'}")
    plt.close(fig)


def plot_type2_distribution(type2: np.ndarray, output_dir: Path):
    """Bar chart for Type II 1-5 scale distribution."""
    fig_dir = output_dir / "type_2_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    score_labels = ["1\n(Background)", "2", "3", "4", "5\n(Core)"]
    counts = [int(np.sum(type2 == float(s))) for s in range(1, 6)]
    colors = ['#F4A024', '#E8C547', '#A8DADC', '#457B9D', '#2E86AB']

    bars = ax.bar(score_labels, counts, color=colors, edgecolor='white', linewidth=0.5)

    for bar, count in zip(bars, counts):
        pct = count / len(type2) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)

    # Set y-limit BEFORE drawing threshold labels so headroom is reserved
    ax.set_ylim(0, max(counts) * Y_HEADROOM_SINGLE)

    # Draw the must-cite threshold line; put the label in the left gap (x<2.5)
    # so it does not sit on top of any bar.
    ax.axvline(x=2.5, color='#E8453C', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(2.4, max(counts) * 1.05, 'Must-cite threshold',
            fontsize=8, color='#E8453C', ha='right', va='center')

    ax.set_xlabel("Type II Score (Core Relevance)")
    ax.set_ylabel("Number of References")
    ax.set_title("Distribution of Type II (Core Relevance)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    fig.savefig(fig_dir / "type2_distribution.png")
    fig.savefig(fig_dir / "type2_distribution.pdf")
    print(f"Saved Type II plot to {fig_dir / 'type2_distribution.png'}")
    plt.close(fig)


def plot_type3_distribution(type3: np.ndarray, output_dir: Path):
    """Bar chart + CDF for Type III frequency distribution."""
    fig_dir = output_dir / "type_3_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    max_val = int(type3.max())
    counts = Counter(type3)
    values = sorted(counts.keys())

    # Cap x-axis at 20 for readability
    cap = min(max_val, 20)
    values_capped = [v for v in values if v <= cap]
    freqs_capped = [counts[v] for v in values_capped]
    if max_val > cap:
        overflow = sum(counts[v] for v in values if v > cap)
        values_capped.append(cap + 1)
        freqs_capped.append(overflow)

    labels = [f"{int(v)}" for v in values_capped[:-1]]
    if max_val > cap:
        labels.append(f">{int(cap)}")
    else:
        labels.append(f"{int(values_capped[-1])}")

    sorted_scores = np.sort(type3)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

    # ==============================
    # Figure: Bar chart
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(range(len(values_capped)), freqs_capped, color='#2E86AB', edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(values_capped)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Citation Frequency Score (Type III)")
    ax.set_ylabel("Number of References")
    ax.set_title("Distribution of Type III (Intra-Paper Mention Frequency)")
    ax.set_ylim(0, max(freqs_capped) * Y_HEADROOM_SINGLE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    fig.savefig(fig_dir / "type3_distribution_bar.png")
    fig.savefig(fig_dir / "type3_distribution_bar.pdf")
    print(f"Saved Type III bar chart to {fig_dir / 'type3_distribution_bar.png'}")
    plt.close(fig)

    # ==============================
    # Figure: CDF with thresholds
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(sorted_scores, cdf, color='#2E86AB', linewidth=1.5)
    ax.set_xlabel("Citation Frequency Score (Type III)")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title("CDF of Type III (Intra-Paper Mention Frequency)")
    ax.set_xlim(0, min(max_val, 20))
    ax.set_ylim(0, 1.05)

    for t, color, ls in [(2, '#E8453C', '-'), (3, '#F4A024', '--'), (5, '#2BA84A', ':')]:
        pct_below = np.sum(type3 < t) / len(type3)
        ax.axvline(x=t, color=color, linestyle=ls, alpha=0.7, linewidth=1.2)
        ax.axhline(y=pct_below, color=color, linestyle=ls, alpha=0.3, linewidth=0.8)
        ax.annotate(f'threshold={t}\n({pct_below:.1%} below)',
                    xy=(t, pct_below), xytext=(t + 1.5, pct_below - 0.08),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    plt.tight_layout()
    fig.savefig(fig_dir / "type3_cdf.png")
    fig.savefig(fig_dir / "type3_cdf.pdf")
    print(f"Saved Type III CDF to {fig_dir / 'type3_cdf.png'}")
    plt.close(fig)

    # ==============================
    # Figure: Combined (for paper)
    # ==============================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(len(values_capped)), freqs_capped, color='#2E86AB', edgecolor='white', linewidth=0.5)
    ax1.set_xticks(range(len(values_capped)))
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Citation Frequency Score")
    ax1.set_ylabel("Number of References")
    ax1.set_title("(a) Score Distribution")
    ax1.set_ylim(0, max(freqs_capped) * Y_HEADROOM_SINGLE)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    ax2.plot(sorted_scores, cdf, color='#2E86AB', linewidth=1.5)
    ax2.set_xlabel("Citation Frequency Score")
    ax2.set_ylabel("Cumulative Proportion")
    ax2.set_title("(b) Cumulative Distribution")
    ax2.set_xlim(0, min(max_val, 20))
    ax2.set_ylim(0, 1.05)

    for t, color, ls in [(2, '#E8453C', '-'), (3, '#F4A024', '--'), (5, '#2BA84A', ':')]:
        pct_below = np.sum(type3 < t) / len(type3)
        ax2.axvline(x=t, color=color, linestyle=ls, alpha=0.7, linewidth=1.2)
        ax2.annotate(f't={t} ({pct_below:.1%} below)',
                     xy=(t, pct_below), xytext=(t + 1.2, pct_below - 0.06),
                     fontsize=8, color=color,
                     arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    plt.tight_layout()
    fig.savefig(fig_dir / "type3_combined.png")
    fig.savefig(fig_dir / "type3_combined.pdf")
    print(f"Saved Type III combined plot to {fig_dir / 'type3_combined.png'}")
    plt.close(fig)


def plot_all_three_combined(type1: np.ndarray, type2: np.ndarray, type3: np.ndarray, output_dir: Path):
    """Single figure with all three distributions side by side for the paper."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    # --- Type I ---
    t1_counts = [int(np.sum(type1 == 0.0)), int(np.sum(type1 == 1.0))]
    t1_labels = ["0", "1"]
    bars1 = ax1.bar(t1_labels, t1_counts, color=['#A8DADC', '#2E86AB'], edgecolor='white', linewidth=0.5)
    for bar, count in zip(bars1, t1_counts):
        pct = count / len(type1) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(t1_counts) * 0.02,
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    ax1.set_xlabel("Label")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Type I: Experimental Baseline")
    ax1.set_ylim(0, max(t1_counts) * Y_HEADROOM_MULTI)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # --- Type II ---
    t2_labels = ["1", "2", "3", "4", "5"]
    t2_counts = [int(np.sum(type2 == float(s))) for s in range(1, 6)]
    colors2 = ['#F4A024', '#E8C547', '#A8DADC', '#457B9D', '#2E86AB']
    bars2 = ax2.bar(t2_labels, t2_counts, color=colors2, edgecolor='white', linewidth=0.5)
    for bar, count in zip(bars2, t2_counts):
        pct = count / len(type2) * 100
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(t2_counts) * 0.02,
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    ax2.axvline(x=2.5, color='#E8453C', linestyle='--', linewidth=1.2, alpha=0.7)
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Type II: Core Relevance")
    ax2.set_ylim(0, max(t2_counts) * Y_HEADROOM_MULTI)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # --- Type III ---
    max_val = int(type3.max())
    cap = min(max_val, 15)
    t3_counts_dict = Counter(type3)
    t3_values = [v for v in sorted(t3_counts_dict.keys()) if v <= cap]
    t3_freqs = [t3_counts_dict[v] for v in t3_values]
    if max_val > cap:
        overflow = sum(t3_counts_dict[v] for v in t3_counts_dict if v > cap)
        t3_values.append(cap + 1)
        t3_freqs.append(overflow)
    t3_labels = [f"{int(v)}" for v in t3_values[:-1]]
    if max_val > cap:
        t3_labels.append(f">{int(cap)}")
    else:
        t3_labels.append(f"{int(t3_values[-1])}")

    ax3.bar(range(len(t3_values)), t3_freqs, color='#2E86AB', edgecolor='white', linewidth=0.5)
    ax3.set_xticks(range(len(t3_values)))
    ax3.set_xticklabels(t3_labels, fontsize=8)
    ax3.set_xlabel("Frequency")
    ax3.set_ylabel("Count")
    ax3.set_title("(c) Type III: Mention Frequency")
    ax3.set_ylim(0, max(t3_freqs) * Y_HEADROOM_MULTI)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    fig.savefig(output_dir / "all_three_distributions.png")
    fig.savefig(output_dir / "all_three_distributions.pdf")
    print(f"\nSaved combined 3-type figure to {output_dir / 'all_three_distributions.png'}")
    plt.close(fig)


def suggest_type3_threshold(type3: np.ndarray):
    """Suggest a threshold for Type III based on distribution."""
    median = np.median(type3)
    mean = type3.mean()

    print("\n" + "=" * 70)
    print("TYPE III THRESHOLD RECOMMENDATION")
    print("=" * 70)
    print(f"\nMedian: {median:.0f}, Mean: {mean:.2f}")
    print(f"Distribution is right-skewed (most references cited few times).")
    print(f"\nCandidate thresholds with class balance:")

    for t in [2, 3, 4, 5]:
        pos = int(np.sum(type3 >= t))
        neg = int(np.sum(type3 < t))
        total = len(type3)
        ratio = pos / neg if neg > 0 else float('inf')
        print(f"\n  Threshold >= {t}:")
        print(f"    Positive (1): {pos:,} ({pos/total*100:.1f}%)")
        print(f"    Negative (0): {neg:,} ({neg/total*100:.1f}%)")
        print(f"    Pos/Neg ratio: {ratio:.3f}")


def main():
    print("Loading papers...")
    df = pd.read_parquet(DATAFRAME_DIR)
    print(f"Loaded {len(df)} papers.")

    type1, type2, type3, total_instances, total_papers = extract_all_scores(df)
    print(f"\nExtracted scores:")
    print(f"  Type 1: {len(type1):,} valid scores")
    print(f"  Type 2: {len(type2):,} valid scores")
    print(f"  Type 3: {len(type3):,} valid scores")
    print(f"  Total citation instances: {total_instances:,}")
    print(f"  Papers with references: {total_papers:,}")

    if len(type1) == 0 and len(type2) == 0 and len(type3) == 0:
        print("No valid scores found. Check your data.")
        return

    setup_plot_style()

    # Print all values needed for the LaTeX paper
    print_latex_values(type1, type2, type3, total_instances, total_papers)

    # Generate individual plots
    if len(type1) > 0:
        plot_type1_distribution(type1, FIGURE_OUTPUT_DIR)
    if len(type2) > 0:
        plot_type2_distribution(type2, FIGURE_OUTPUT_DIR)
    if len(type3) > 0:
        plot_type3_distribution(type3, FIGURE_OUTPUT_DIR)

    # Generate combined figure for the paper
    if len(type1) > 0 and len(type2) > 0 and len(type3) > 0:
        plot_all_three_combined(type1, type2, type3, FIGURE_OUTPUT_DIR)

    # Type III threshold suggestion
    if len(type3) > 0:
        suggest_type3_threshold(type3)


if __name__ == "__main__":
    main()