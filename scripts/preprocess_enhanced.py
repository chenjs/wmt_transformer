#!/usr/bin/env python3
"""
Enhanced preprocessing with data cleaning and quality filtering.

This script improves upon the basic preprocess.py by:
1. Cleaning data (removing empty lines, normalizing characters)
2. Filtering based on sentence length and quality
3. Training tokenizers on cleaned data
4. Providing detailed statistics
"""

import sys
import re
from pathlib import Path
import unicodedata

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.tokenizer import train_tokenizer


def is_valid_german_char(char):
    """Check if character is valid for German text."""
    # German-specific characters
    german_chars = {'ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü'}
    if char in german_chars:
        return True

    # Check Unicode category
    category = unicodedata.category(char)

    # Allow letters, marks, numbers, punctuation, symbols, spaces
    allowed_categories = {
        'L',  # Letter
        'M',  # Mark
        'N',  # Number
        'P',  # Punctuation
        'S',  # Symbol
        'Z',  # Separator
    }

    return category[0] in allowed_categories


def clean_text(text):
    """Clean and normalize text."""
    # Normalize Unicode
    text = unicodedata.normalize('NFKC', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text.strip()


def filter_sentence_pair(src_text, tgt_text, min_words=3, max_words=100, max_ratio=3.0):
    """Filter sentence pair based on quality criteria."""
    # Check for empty sentences
    if not src_text or not tgt_text:
        return False

    # Split into words
    src_words = src_text.split()
    tgt_words = tgt_text.split()

    # Check word count bounds
    if len(src_words) < min_words or len(tgt_words) < min_words:
        return False

    if len(src_words) > max_words or len(tgt_words) > max_words:
        return False

    # Check length ratio
    ratio = len(src_words) / max(1, len(tgt_words))
    if ratio > max_ratio or ratio < 1.0/max_ratio:
        return False

    return True


def prepare_cleaned_data(
    src_file: Path,
    tgt_file: Path,
    output_dir: Path,
    max_samples: int = 200000,
    min_words: int = 3,
    max_words: int = 100,
    max_ratio: float = 3.0,
):
    """Prepare cleaned text files for tokenizer training.

    Returns:
        Tuple of (src_text_file, tgt_text_file, stats_dict)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_text_file = output_dir / "src_text_cleaned.txt"
    tgt_text_file = output_dir / "tgt_text_cleaned.txt"

    stats = {
        'total_pairs': 0,
        'cleaned_pairs': 0,
        'removed_empty': 0,
        'removed_short': 0,
        'removed_long': 0,
        'removed_ratio': 0,
        'removed_other': 0,
    }

    print(f"Preparing cleaned tokenizer training data (max {max_samples} samples)...")
    print(f"Filters: min_words={min_words}, max_words={max_words}, max_ratio={max_ratio}")

    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt, \
         open(src_text_file, 'w', encoding='utf-8') as f_src_out, \
         open(tgt_text_file, 'w', encoding='utf-8') as f_tgt_out:

        for i, (src_line, tgt_line) in enumerate(zip(f_src, f_tgt)):
            if max_samples and i >= max_samples:
                break

            stats['total_pairs'] += 1

            # Clean text
            src_cleaned = clean_text(src_line)
            tgt_cleaned = clean_text(tgt_line)

            # Apply filters
            if not src_cleaned or not tgt_cleaned:
                stats['removed_empty'] += 1
                continue

            # Check for basic validity
            src_words = src_cleaned.split()
            tgt_words = tgt_cleaned.split()

            if len(src_words) < min_words or len(tgt_words) < min_words:
                stats['removed_short'] += 1
                continue

            if len(src_words) > max_words or len(tgt_words) > max_words:
                stats['removed_long'] += 1
                continue

            # Check length ratio
            ratio = len(src_words) / max(1, len(tgt_words))
            if ratio > max_ratio or ratio < 1.0/max_ratio:
                stats['removed_ratio'] += 1
                continue

            # Write cleaned data
            f_src_out.write(src_cleaned + "\n")
            f_tgt_out.write(tgt_cleaned + "\n")
            stats['cleaned_pairs'] += 1

            # Progress reporting
            if stats['total_pairs'] % 10000 == 0:
                print(f"  Processed {stats['total_pairs']} pairs, kept {stats['cleaned_pairs']}")

    # Calculate percentages
    stats['kept_percentage'] = (stats['cleaned_pairs'] / max(1, stats['total_pairs'])) * 100
    stats['removed_total'] = (
        stats['removed_empty'] +
        stats['removed_short'] +
        stats['removed_long'] +
        stats['removed_ratio']
    )

    print(f"\nCleaning completed:")
    print(f"  Total pairs processed: {stats['total_pairs']}")
    print(f"  Pairs kept: {stats['cleaned_pairs']} ({stats['kept_percentage']:.1f}%)")
    print(f"  Pairs removed: {stats['removed_total']}")
    print(f"    - Empty: {stats['removed_empty']}")
    print(f"    - Too short: {stats['removed_short']}")
    print(f"    - Too long: {stats['removed_long']}")
    print(f"    - Bad ratio: {stats['removed_ratio']}")

    return src_text_file, tgt_text_file, stats


def train_tokenizers_with_stats(
    src_text_file: Path,
    tgt_text_file: Path,
    output_dir: Path,
    vocab_size: int = 16000,  # Reduced from 32000 based on analysis
):
    """Train tokenizers with statistics."""
    output_dir = Path(output_dir)

    print(f"\nTraining tokenizers with vocab_size={vocab_size}")

    # Train source tokenizer (English)
    print("\n" + "=" * 50)
    print("Training source tokenizer (English)")
    print("=" * 50)
    src_model = str(output_dir / "src_tokenizer_enhanced")
    train_tokenizer(
        str(src_text_file),
        src_model,
        vocab_size=vocab_size,
    )

    # Train target tokenizer (German)
    print("\n" + "=" * 50)
    print("Training target tokenizer (German)")
    print("=" * 50)
    tgt_model = str(output_dir / "tgt_tokenizer_enhanced")
    train_tokenizer(
        str(tgt_text_file),
        tgt_model,
        vocab_size=vocab_size,
    )

    return f"{src_model}.model", f"{tgt_model}.model"


def analyze_tokenizer_coverage(text_file: Path, tokenizer_model: str, sample_size: int = 10000):
    """Analyze tokenizer coverage on sample data."""
    from src.data.tokenizer import BPETokenizer

    print(f"\nAnalyzing tokenizer coverage on {sample_size} samples...")

    # Load tokenizer
    tokenizer = BPETokenizer(tokenizer_model)
    vocab_size = tokenizer.sp.get_piece_size()

    # Load sample data
    sample_lines = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            sample_lines.append(line.strip())

    # Analyze token coverage
    token_counts = {}
    total_tokens = 0

    for line in sample_lines:
        tokens = tokenizer.encode(line)
        total_tokens += len(tokens)
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

    vocab_used = len(token_counts)
    coverage_percentage = (vocab_used / vocab_size) * 100

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Vocabulary used: {vocab_used} ({coverage_percentage:.1f}%)")
    print(f"  Total tokens: {total_tokens}")

    # Calculate coverage curve
    sorted_counts = sorted(token_counts.values(), reverse=True)
    cumulative = 0
    coverage_points = []

    for i, count in enumerate(sorted_counts):
        cumulative += count
        coverage_percent = (cumulative / total_tokens) * 100

        if coverage_percent >= 50 and not any(p[0] == 50 for p in coverage_points):
            coverage_points.append((50, i + 1))
        if coverage_percent >= 80 and not any(p[0] == 80 for p in coverage_points):
            coverage_points.append((80, i + 1))
        if coverage_percent >= 90 and not any(p[0] == 90 for p in coverage_points):
            coverage_points.append((90, i + 1))
        if coverage_percent >= 95 and not any(p[0] == 95 for p in coverage_points):
            coverage_points.append((95, i + 1))
        if coverage_percent >= 99 and not any(p[0] == 99 for p in coverage_points):
            coverage_points.append((99, i + 1))

    print(f"  Coverage curve:")
    for percent, tokens_needed in coverage_points:
        print(f"    Top {tokens_needed:5d} tokens cover {percent:2d}% of occurrences")

    return {
        'vocab_size': vocab_size,
        'vocab_used': vocab_used,
        'coverage_percentage': coverage_percentage,
        'total_tokens': total_tokens,
        'coverage_points': coverage_points,
    }


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("ENHANCED DATA PREPROCESSING WITH QUALITY FILTERING")
    print("=" * 60)

    # Setup paths
    data_dir = Path(__file__).parent.parent
    src_file = data_dir / config.src_file
    tgt_file = data_dir / config.tgt_file
    output_dir = data_dir / "models_enhanced"
    output_dir.mkdir(exist_ok=True)

    # Step 1: Prepare cleaned data
    print("\nStep 1: Cleaning and filtering data")
    print("-" * 40)

    src_text_file, tgt_text_file, cleaning_stats = prepare_cleaned_data(
        src_file=src_file,
        tgt_file=tgt_file,
        output_dir=output_dir,
        # max_samples=200000,  # Match max_train_samples
        max_samples=config.max_train_samples,
        min_words=3,
        max_words=100,
        max_ratio=3.0,
    )

    # Step 2: Train tokenizers with optimized vocab_size
    print("\nStep 2: Training optimized tokenizers")
    print("-" * 40)

    # Try different vocabulary sizes
    vocab_sizes = [16000, 20000, 24000]
    best_vocab_size = 16000  # Default

    for vocab_size in vocab_sizes:
        print(f"\nTrying vocab_size={vocab_size}")

        src_model, tgt_model = train_tokenizers_with_stats(
            src_text_file=src_text_file,
            tgt_text_file=tgt_text_file,
            output_dir=output_dir,
            vocab_size=vocab_size,
        )

        # Analyze coverage
        print(f"\nAnalyzing source tokenizer coverage...")
        src_coverage = analyze_tokenizer_coverage(src_text_file, src_model, 10000)

        print(f"\nAnalyzing target tokenizer coverage...")
        tgt_coverage = analyze_tokenizer_coverage(tgt_text_file, tgt_model, 10000)

        # Check if this vocab_size provides good coverage
        src_coverage_pct = src_coverage['coverage_percentage']
        tgt_coverage_pct = tgt_coverage['coverage_percentage']

        print(f"\nCoverage summary for vocab_size={vocab_size}:")
        print(f"  Source: {src_coverage_pct:.1f}% coverage")
        print(f"  Target: {tgt_coverage_pct:.1f}% coverage")

        # Good coverage is >70%
        if src_coverage_pct > 70 and tgt_coverage_pct > 70:
            print(f"  ✓ Good coverage achieved")
            best_vocab_size = vocab_size
            break
        else:
            print(f"  ⚠ Coverage below target (70%), trying larger vocab_size")

    print(f"\nSelected vocab_size: {best_vocab_size}")

    # Step 3: Create final tokenizers with best vocab_size
    print("\nStep 3: Creating final tokenizers")
    print("-" * 40)

    src_model_final = str(output_dir / "src_tokenizer_final")
    tgt_model_final = str(output_dir / "tgt_tokenizer_final")

    train_tokenizer(
        str(src_text_file),
        src_model_final,
        vocab_size=best_vocab_size,
    )

    train_tokenizer(
        str(tgt_text_file),
        tgt_model_final,
        vocab_size=best_vocab_size,
    )

    # Step 4: Generate configuration recommendations
    print("\n" + "=" * 60)
    print("CONFIGURATION RECOMMENDATIONS")
    print("=" * 60)

    # Analyze sentence lengths in cleaned data
    src_lengths = []
    tgt_lengths = []

    with open(src_text_file, 'r', encoding='utf-8') as f:
        for line in f:
            src_lengths.append(len(line.split()))

    with open(tgt_text_file, 'r', encoding='utf-8') as f:
        for line in f:
            tgt_lengths.append(len(line.split()))

    import numpy as np
    src_mean = np.mean(src_lengths)
    tgt_mean = np.mean(tgt_lengths)
    src_90th = np.percentile(src_lengths, 90)
    tgt_90th = np.percentile(tgt_lengths, 90)

    print(f"Cleaned dataset statistics:")
    print(f"  Samples: {len(src_lengths)}")
    print(f"  Source mean length: {src_mean:.1f} words")
    print(f"  Target mean length: {tgt_mean:.1f} words")
    print(f"  Source 90th percentile: {src_90th:.1f} words")
    print(f"  Target 90th percentile: {tgt_90th:.1f} words")

    print(f"\nRecommended configuration changes:")
    print(f"  1. Update config.vocab_size = {best_vocab_size}")
    print(f"  2. Update config.src_vocab_size = {best_vocab_size}")
    print(f"  3. Update config.tgt_vocab_size = {best_vocab_size}")
    print(f"  4. Update config.max_len = {int(max(src_90th, tgt_90th) * 1.2)}")
    print(f"  5. Update tokenizer paths to use enhanced tokenizers")

    print(f"\n" + "=" * 60)
    print("PREPROCESSING COMPLETED!")
    print("=" * 60)
    print(f"Enhanced tokenizers saved to:")
    print(f"  Source: {src_model_final}.model")
    print(f"  Target: {tgt_model_final}.model")
    print(f"\nTo use enhanced tokenizers, update config.py:")
    print(f"  src_tokenizer = \"models_enhanced/src_tokenizer_final.model\"")
    print(f"  tgt_tokenizer = \"models_enhanced/tgt_tokenizer_final.model\"")
    print(f"  vocab_size = {best_vocab_size}")
    print(f"  src_vocab_size = {best_vocab_size}")
    print(f"  tgt_vocab_size = {best_vocab_size}")


if __name__ == "__main__":
    main()