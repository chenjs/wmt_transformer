#!/usr/bin/env python3
"""
Test enhanced preprocessing on small sample.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preprocess_enhanced import prepare_cleaned_data, train_tokenizers_with_stats


def main():
    """Test preprocessing on small sample."""
    print("=" * 60)
    print("TESTING ENHANCED PREPROCESSING (10,000 samples)")
    print("=" * 60)

    # Setup paths
    data_dir = Path(__file__).parent.parent
    src_file = data_dir / "europarl-v7.de-en.en"
    tgt_file = data_dir / "europarl-v7.de-en.de"
    output_dir = data_dir / "models_test"
    output_dir.mkdir(exist_ok=True)

    # Step 1: Prepare cleaned data (small sample)
    print("\nStep 1: Cleaning and filtering data (10,000 samples)")
    print("-" * 40)

    src_text_file, tgt_text_file, cleaning_stats = prepare_cleaned_data(
        src_file=src_file,
        tgt_file=tgt_file,
        output_dir=output_dir,
        max_samples=10000,  # Small sample for testing
        min_words=3,
        max_words=100,
        max_ratio=3.0,
    )

    # Step 2: Train tokenizers with small vocab for testing
    print("\nStep 2: Training test tokenizers")
    print("-" * 40)

    src_model, tgt_model = train_tokenizers_with_stats(
        src_text_file=src_text_file,
        tgt_text_file=tgt_text_file,
        output_dir=output_dir,
        vocab_size=8000,  # Small for testing
    )

    print(f"\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("=" * 60)
    print(f"Test tokenizers saved to:")
    print(f"  Source: {src_model}")
    print(f"  Target: {tgt_model}")

    # Analyze cleaned data statistics
    print(f"\nCleaned data statistics:")
    print(f"  Original pairs: {cleaning_stats['total_pairs']}")
    print(f"  Cleaned pairs: {cleaning_stats['cleaned_pairs']} ({cleaning_stats['kept_percentage']:.1f}%)")
    print(f"  Removed: {cleaning_stats['removed_total']}")
    print(f"    - Empty: {cleaning_stats['removed_empty']}")
    print(f"    - Too short: {cleaning_stats['removed_short']}")
    print(f"    - Too long: {cleaning_stats['removed_long']}")
    print(f"    - Bad ratio: {cleaning_stats['removed_ratio']}")


if __name__ == "__main__":
    main()