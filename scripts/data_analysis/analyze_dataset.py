#!/usr/bin/env python3
"""
Analyze dataset characteristics for data quality optimization.

This script analyzes the Europarl dataset to provide insights for data quality
optimization, including sentence length distribution, vocabulary coverage,
and other statistics.
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config
from src.data.tokenizer import BPETokenizer


def load_sample_lines(file_path, max_samples=10000):
    """Load a sample of lines from a file."""
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            lines.append(line.strip())
    return lines


def analyze_sentence_lengths(src_lines, tgt_lines, tokenizer=None):
    """Analyze sentence length distribution."""
    print("=" * 60)
    print("SENTENCE LENGTH ANALYSIS")
    print("=" * 60)

    src_lengths = []
    tgt_lengths = []
    length_ratios = []

    for src, tgt in zip(src_lines, tgt_lines):
        # Simple word count (split by whitespace)
        src_word_count = len(src.split())
        tgt_word_count = len(tgt.split())

        src_lengths.append(src_word_count)
        tgt_lengths.append(tgt_word_count)

        if tgt_word_count > 0:
            length_ratios.append(src_word_count / tgt_word_count)

    # Basic statistics
    print(f"Source sentences: {len(src_lengths)}")
    print(f"Target sentences: {len(tgt_lengths)}")
    print()

    print("Source length statistics (words):")
    print(f"  Min: {min(src_lengths):.1f}")
    print(f"  Max: {max(src_lengths):.1f}")
    print(f"  Mean: {np.mean(src_lengths):.1f}")
    print(f"  Median: {np.median(src_lengths):.1f}")
    print(f"  Std: {np.std(src_lengths):.1f}")
    print(f"  90th percentile: {np.percentile(src_lengths, 90):.1f}")
    print(f"  95th percentile: {np.percentile(src_lengths, 95):.1f}")
    print()

    print("Target length statistics (words):")
    print(f"  Min: {min(tgt_lengths):.1f}")
    print(f"  Max: {max(tgt_lengths):.1f}")
    print(f"  Mean: {np.mean(tgt_lengths):.1f}")
    print(f"  Median: {np.median(tgt_lengths):.1f}")
    print(f"  Std: {np.std(tgt_lengths):.1f}")
    print(f"  90th percentile: {np.percentile(tgt_lengths, 90):.1f}")
    print(f"  95th percentile: {np.percentile(tgt_lengths, 95):.1f}")
    print()

    print("Length ratio (src/tgt) statistics:")
    print(f"  Min: {min(length_ratios):.2f}")
    print(f"  Max: {max(length_ratios):.2f}")
    print(f"  Mean: {np.mean(length_ratios):.2f}")
    print(f"  Median: {np.median(length_ratios):.2f}")
    print(f"  Std: {np.std(length_ratios):.2f}")

    return src_lengths, tgt_lengths, length_ratios


def analyze_vocabulary(src_lines, tgt_lines, src_tokenizer, tgt_tokenizer):
    """Analyze vocabulary usage and coverage."""
    print("\n" + "=" * 60)
    print("VOCABULARY ANALYSIS")
    print("=" * 60)

    # Analyze source vocabulary
    src_token_counts = Counter()
    src_oov_counts = []
    src_total_tokens = 0

    for line in src_lines:
        tokens = src_tokenizer.encode(line)
        src_total_tokens += len(tokens)
        for token in tokens:
            src_token_counts[token] += 1

    # Analyze target vocabulary
    tgt_token_counts = Counter()
    tgt_oov_counts = []
    tgt_total_tokens = 0

    for line in tgt_lines:
        tokens = tgt_tokenizer.encode(line)
        tgt_total_tokens += len(tokens)
        for token in tokens:
            tgt_token_counts[token] += 1

    # Vocabulary statistics
    src_vocab_size = len(src_token_counts)
    tgt_vocab_size = len(tgt_token_counts)

    src_most_common = src_token_counts.most_common(20)
    tgt_most_common = tgt_token_counts.most_common(20)

    print(f"Source vocabulary size used: {src_vocab_size} / {src_tokenizer.sp.get_piece_size()}")
    print(f"Target vocabulary size used: {tgt_vocab_size} / {tgt_tokenizer.sp.get_piece_size()}")
    print()

    print(f"Source total tokens: {src_total_tokens}")
    print(f"Target total tokens: {tgt_total_tokens}")
    print()

    print("Source token frequency distribution:")
    src_freq = sorted(src_token_counts.values(), reverse=True)
    cum_coverage = np.cumsum(src_freq) / src_total_tokens
    for percentile in [0.5, 0.8, 0.9, 0.95, 0.99]:
        idx = next(i for i, cov in enumerate(cum_coverage) if cov >= percentile)
        print(f"  Top {idx+1} tokens cover {percentile*100:.0f}% of occurrences")

    print("\nTarget token frequency distribution:")
    tgt_freq = sorted(tgt_token_counts.values(), reverse=True)
    cum_coverage = np.cumsum(tgt_freq) / tgt_total_tokens
    for percentile in [0.5, 0.8, 0.9, 0.95, 0.99]:
        idx = next(i for i, cov in enumerate(cum_coverage) if cov >= percentile)
        print(f"  Top {idx+1} tokens cover {percentile*100:.0f}% of occurrences")

    return src_token_counts, tgt_token_counts


def analyze_data_quality(src_lines, tgt_lines):
    """Analyze basic data quality issues."""
    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)

    # Check for empty lines
    src_empty = sum(1 for line in src_lines if not line.strip())
    tgt_empty = sum(1 for line in tgt_lines if not line.strip())

    # Check for very short lines
    src_short = sum(1 for line in src_lines if len(line.split()) < 3)
    tgt_short = sum(1 for line in tgt_lines if len(line.split()) < 3)

    # Check for very long lines
    src_long = sum(1 for line in src_lines if len(line.split()) > 100)
    tgt_long = sum(1 for line in tgt_lines if len(line.split()) > 100)

    # Check for lines with unusual characters
    import string
    printable = set(string.printable)
    src_non_printable = sum(1 for line in src_lines if not all(c in printable for c in line))
    tgt_non_printable = sum(1 for line in tgt_lines if not all(c in printable for c in line))

    print(f"Empty source lines: {src_empty} ({src_empty/len(src_lines)*100:.1f}%)")
    print(f"Empty target lines: {tgt_empty} ({tgt_empty/len(tgt_lines)*100:.1f}%)")
    print()

    print(f"Very short source lines (<3 words): {src_short} ({src_short/len(src_lines)*100:.1f}%)")
    print(f"Very short target lines (<3 words): {tgt_short} ({tgt_short/len(tgt_lines)*100:.1f}%)")
    print()

    print(f"Very long source lines (>100 words): {src_long} ({src_long/len(src_lines)*100:.1f}%)")
    print(f"Very long target lines (>100 words): {tgt_long} ({tgt_long/len(tgt_lines)*100:.1f}%)")
    print()

    print(f"Source lines with non-printable chars: {src_non_printable} ({src_non_printable/len(src_lines)*100:.1f}%)")
    print(f"Target lines with non-printable chars: {tgt_non_printable} ({tgt_non_printable/len(tgt_lines)*100:.1f}%)")

    return {
        'src_empty': src_empty,
        'tgt_empty': tgt_empty,
        'src_short': src_short,
        'tgt_short': tgt_short,
        'src_long': src_long,
        'tgt_long': tgt_long,
        'src_non_printable': src_non_printable,
        'tgt_non_printable': tgt_non_printable,
    }


def plot_histograms(src_lengths, tgt_lengths, output_dir):
    """Create histogram plots of sentence lengths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Source length histogram
    axes[0, 0].hist(src_lengths, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Sentence Length (words)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Source Sentence Length Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # Target length histogram
    axes[0, 1].hist(tgt_lengths, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Sentence Length (words)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Target Sentence Length Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Combined histogram
    axes[1, 0].hist(src_lengths, bins=50, alpha=0.5, color='blue', label='Source')
    axes[1, 0].hist(tgt_lengths, bins=50, alpha=0.5, color='green', label='Target')
    axes[1, 0].set_xlabel('Sentence Length (words)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Combined Sentence Length Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Length ratio histogram
    length_ratios = [s/t if t > 0 else 0 for s, t in zip(src_lengths, tgt_lengths)]
    axes[1, 1].hist(length_ratios, bins=50, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Length Ratio (Source/Target)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Source/Target Length Ratio Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "sentence_length_distributions.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nHistogram saved to: {plot_path}")


def main():
    """Main analysis function."""
    print("=" * 60)
    print("DATASET ANALYSIS FOR DATA QUALITY OPTIMIZATION")
    print("=" * 60)

    # Setup paths
    data_dir = Path(__file__).parent.parent.parent
    src_file = data_dir / config.src_file
    tgt_file = data_dir / config.tgt_file

    # Load tokenizers - use enhanced tokenizers (consistent with training)
    print("Loading tokenizers...")
    src_tokenizer_path = data_dir / "models_enhanced" / "src_tokenizer_final.model"
    tgt_tokenizer_path = data_dir / "models_enhanced" / "tgt_tokenizer_final.model"

    if not src_tokenizer_path.exists() or not tgt_tokenizer_path.exists():
        print("Error: Tokenizer files not found. Run preprocess.py first.")
        return

    src_tokenizer = BPETokenizer(str(src_tokenizer_path))
    tgt_tokenizer = BPETokenizer(str(tgt_tokenizer_path))

    # Load sample data (first 10,000 lines for analysis)
    print("Loading dataset samples...")
    sample_size = 10000
    src_lines = load_sample_lines(src_file, sample_size)
    tgt_lines = load_sample_lines(tgt_file, sample_size)

    print(f"Loaded {len(src_lines)} source sentences")
    print(f"Loaded {len(tgt_lines)} target sentences")

    # Check alignment
    if len(src_lines) != len(tgt_lines):
        print(f"Warning: Source and target have different line counts ({len(src_lines)} vs {len(tgt_lines)})")
        min_len = min(len(src_lines), len(tgt_lines))
        src_lines = src_lines[:min_len]
        tgt_lines = tgt_lines[:min_len]
        print(f"Using first {min_len} aligned sentences")

    # Run analyses
    src_lengths, tgt_lengths, length_ratios = analyze_sentence_lengths(src_lines, tgt_lines)

    src_token_counts, tgt_token_counts = analyze_vocabulary(
        src_lines, tgt_lines, src_tokenizer, tgt_tokenizer
    )

    quality_issues = analyze_data_quality(src_lines, tgt_lines)

    # Create visualizations
    output_dir = data_dir / "data_analysis"
    plot_histograms(src_lengths, tgt_lengths, output_dir)

    # Generate recommendations
    print("\n" + "=" * 60)
    print("DATA QUALITY OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)

    # Based on analysis, generate recommendations
    src_mean_len = np.mean(src_lengths)
    tgt_mean_len = np.mean(tgt_lengths)

    print("1. Sentence Length Filtering:")
    print(f"   - Current max_len={config.max_len}")
    print(f"   - Mean source length: {src_mean_len:.1f} words")
    print(f"   - Mean target length: {tgt_mean_len:.1f} words")
    print(f"   - Recommendation: Keep current max_len={config.max_len} or increase to {max(32, int(max(src_mean_len, tgt_mean_len) * 1.5))}")

    print("\n2. Tokenizer Training:")
    print(f"   - Current vocab_size={config.vocab_size}")
    print(f"   - Source vocab used: {len(src_token_counts)}/{src_tokenizer.sp.get_piece_size()}")
    print(f"   - Target vocab used: {len(tgt_token_counts)}/{tgt_tokenizer.sp.get_piece_size()}")
    print(f"   - Recommendation: Consider reducing vocab_size if coverage is low")

    print("\n3. Data Cleaning:")
    total_samples = len(src_lines)
    if quality_issues['src_empty'] > 0 or quality_issues['tgt_empty'] > 0:
        print(f"   - Found {quality_issues['src_empty'] + quality_issues['tgt_empty']} empty lines")
        print(f"   - Recommendation: Remove empty lines")

    if quality_issues['src_short'] > 0 or quality_issues['tgt_short'] > 0:
        short_pct = (quality_issues['src_short'] + quality_issues['tgt_short']) / (2 * total_samples) * 100
        print(f"   - Found {quality_issues['src_short'] + quality_issues['tgt_short']} very short lines (<3 words)")
        print(f"   - Recommendation: Consider filtering very short sentences")

    if quality_issues['src_long'] > 0 or quality_issues['tgt_long'] > 0:
        long_pct = (quality_issues['src_long'] + quality_issues['tgt_long']) / (2 * total_samples) * 100
        print(f"   - Found {quality_issues['src_long'] + quality_issues['tgt_long']} very long lines (>100 words)")
        print(f"   - Recommendation: Filter very long sentences or split them")

    print("\n4. Training Data Size:")
    print(f"   - Current max_train_samples={config.max_train_samples}")
    print(f"   - Tokenizer trained on: 100,000 samples")
    print(f"   - Recommendation: Increase tokenizer training samples to match max_train_samples")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()