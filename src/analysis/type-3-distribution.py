"""
Analyze the distribution of type_3_output (citation frequency) scores
from the core papers dataset and generate publication-quality visualizations.
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

FIGURE_OUTPUT_DIR = OUTPUT_DIR / "analysis/all_papers_distributions/type_3_figures"


def extract_type3_scores(df: pd.DataFrame) -> np.ndarray:
    """Extract all valid type_3_output values from references across all papers."""
    all_scores = []

    for refs in df["references"]:
        if isinstance(refs, str):
            refs = json.loads(refs)
        if not refs:
            continue
        for ref in refs:
            val = ref.get("type_3_output", "")
            if val and str(val).lower() not in ("nan", "", "none"):
                try:
                    score = float(val)
                    if not np.isnan(score):
                        all_scores.append(score)
                except (ValueError, TypeError):
                    continue

    return np.array(all_scores)


def print_statistics(scores: np.ndarray):
    """Print descriptive statistics for the scores."""
    print("\n" + "=" * 60)
    print("TYPE 3 (CITATION FREQUENCY) — DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(f"Total valid scores:  {len(scores):,}")
    print(f"Min:                 {scores.min():.1f}")
    print(f"Max:                 {scores.max():.1f}")
    print(f"Mean:                {scores.mean():.2f}")
    print(f"Median:              {np.median(scores):.1f}")
    print(f"Std Dev:             {scores.std():.2f}")
    print(f"25th percentile:     {np.percentile(scores, 25):.1f}")
    print(f"50th percentile:     {np.percentile(scores, 50):.1f}")
    print(f"75th percentile:     {np.percentile(scores, 75):.1f}")
    print(f"90th percentile:     {np.percentile(scores, 90):.1f}")
    print(f"95th percentile:     {np.percentile(scores, 95):.1f}")
    print(f"99th percentile:     {np.percentile(scores, 99):.1f}")

    print("\n" + "-" * 60)
    print("VALUE DISTRIBUTION")
    print("-" * 60)
    counts = Counter(scores)
    for val in sorted(counts.keys()):
        pct = counts[val] / len(scores) * 100
        print(f"  {val:5.0f}:  {counts[val]:>8,}  ({pct:5.1f}%)")

    # Cumulative percentages at candidate thresholds
    print("\n" + "-" * 60)
    print("CUMULATIVE % AT CANDIDATE THRESHOLDS")
    print("-" * 60)
    for t in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        below = np.sum(scores < t) / len(scores) * 100
        at_or_above = 100 - below
        print(f"  >= {t:2d}:  {at_or_above:5.1f}% of references  ({np.sum(scores >= t):,} refs)")


def plot_distribution(scores: np.ndarray, output_dir: Path):
    """Generate publication-quality distribution plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot style setup (publication-ready) ---
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

    max_val = int(scores.max())

    # ==============================
    # Figure 1: Bar chart of value counts
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 4))

    counts = Counter(scores)
    values = sorted(counts.keys())
    freqs = [counts[v] for v in values]

    # Cap x-axis at a reasonable value for readability
    cap = min(max_val, 20)
    values_capped = [v for v in values if v <= cap]
    freqs_capped = [counts[v] for v in values_capped]
    if max_val > cap:
        overflow = sum(counts[v] for v in values if v > cap)
        values_capped.append(cap + 1)
        freqs_capped.append(overflow)

    bars = ax.bar(range(len(values_capped)), freqs_capped, color='#2E86AB', edgecolor='white', linewidth=0.5)

    # Labels
    labels = [f"{int(v)}" for v in values_capped[:-1]]
    if max_val > cap:
        labels.append(f">{int(cap)}")
    else:
        labels.append(f"{int(values_capped[-1])}")
    ax.set_xticks(range(len(values_capped)))
    ax.set_xticklabels(labels)

    ax.set_xlabel("Citation Frequency Score (Type 3)")
    ax.set_ylabel("Number of References")
    ax.set_title("Distribution of Citation Frequency Scores")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    fig.savefig(output_dir / "type3_distribution_bar.png")
    fig.savefig(output_dir / "type3_distribution_bar.pdf")
    print(f"\nSaved bar chart to {output_dir / 'type3_distribution_bar.png'}")
    plt.close(fig)

    # ==============================
    # Figure 2: CDF (Cumulative Distribution)
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 4))

    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

    ax.plot(sorted_scores, cdf, color='#2E86AB', linewidth=1.5)
    ax.set_xlabel("Citation Frequency Score (Type 3)")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title("CDF of Citation Frequency Scores")
    ax.set_xlim(0, min(max_val, 20))

    # Mark candidate thresholds
    for t, color, ls in [(2, '#E8453C', '-'), (3, '#F4A024', '--'), (5, '#2BA84A', ':')]:
        pct_below = np.sum(scores < t) / len(scores)
        ax.axvline(x=t, color=color, linestyle=ls, alpha=0.7, linewidth=1.2)
        ax.axhline(y=pct_below, color=color, linestyle=ls, alpha=0.3, linewidth=0.8)
        ax.annotate(f'threshold={t}\n({pct_below:.1%} below)',
                     xy=(t, pct_below), xytext=(t + 1.5, pct_below - 0.08),
                     fontsize=8, color=color,
                     arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "type3_cdf.png")
    fig.savefig(output_dir / "type3_cdf.pdf")
    print(f"Saved CDF plot to {output_dir / 'type3_cdf.png'}")
    plt.close(fig)

    # ==============================
    # Figure 3: Combined (for paper) — bar + CDF side by side
    # ==============================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: bar chart
    bars = ax1.bar(range(len(values_capped)), freqs_capped, color='#2E86AB', edgecolor='white', linewidth=0.5)
    ax1.set_xticks(range(len(values_capped)))
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Citation Frequency Score")
    ax1.set_ylabel("Number of References")
    ax1.set_title("(a) Score Distribution")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Right: CDF
    ax2.plot(sorted_scores, cdf, color='#2E86AB', linewidth=1.5)
    ax2.set_xlabel("Citation Frequency Score")
    ax2.set_ylabel("Cumulative Proportion")
    ax2.set_title("(b) Cumulative Distribution")
    ax2.set_xlim(0, min(max_val, 20))

    for t, color, ls in [(2, '#E8453C', '-'), (3, '#F4A024', '--'), (5, '#2BA84A', ':')]:
        pct_below = np.sum(scores < t) / len(scores)
        ax2.axvline(x=t, color=color, linestyle=ls, alpha=0.7, linewidth=1.2)
        ax2.annotate(f't={t} ({pct_below:.1%} below)',
                      xy=(t, pct_below), xytext=(t + 1.2, pct_below - 0.06),
                      fontsize=8, color=color,
                      arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "type3_combined.png")
    fig.savefig(output_dir / "type3_combined.pdf")
    print(f"Saved combined plot to {output_dir / 'type3_combined.png'}")
    plt.close(fig)


def suggest_threshold(scores: np.ndarray):
    """Suggest a threshold based on the distribution."""
    median = np.median(scores)
    mean = scores.mean()

    print("\n" + "=" * 60)
    print("THRESHOLD RECOMMENDATION")
    print("=" * 60)
    print(f"\nThe median score is {median:.0f} and the mean is {mean:.2f}.")
    print(f"This is a right-skewed distribution (most references are cited few times).")
    print(f"\nCommon approaches for binarization:")
    print(f"  1. Median split (>= {median:.0f}): balanced classes")
    print(f"  2. Mean split  (>= {mean:.0f}): slightly above center")
    print(f"  3. Top quartile (>= {np.percentile(scores, 75):.0f}): identifies high-frequency citations")

    # Suggest based on class balance
    for t in [2, 3, 4, 5]:
        pos = np.sum(scores >= t)
        neg = np.sum(scores < t)
        ratio = pos / neg if neg > 0 else float('inf')
        print(f"\n  Threshold >= {t}:")
        print(f"    Positive (1): {pos:,} ({pos/len(scores)*100:.1f}%)")
        print(f"    Negative (0): {neg:,} ({neg/len(scores)*100:.1f}%)")
        print(f"    Ratio pos/neg: {ratio:.2f}")


def main():
    print("Loading papers...")
    df = pd.read_parquet(DATAFRAME_DIR)
    print(f"Loaded {len(df)} papers.")

    scores = extract_type3_scores(df)
    print(f"Extracted {len(scores):,} valid type_3 scores.")

    if len(scores) == 0:
        print("No valid type_3_output scores found. Check your data.")
        return

    print_statistics(scores)
    plot_distribution(scores, FIGURE_OUTPUT_DIR)
    suggest_threshold(scores)


if __name__ == "__main__":
    main()