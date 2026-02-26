"""
Preprocess data: train tokenizers.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.tokenizer import train_tokenizer, prepare_tokenizer_data


def main():
    """Main preprocessing function."""
    # Paths
    data_dir = Path(__file__).parent.parent
    src_file = data_dir / config.src_file
    tgt_file = data_dir / config.tgt_file
    output_dir = data_dir / "models"
    output_dir.mkdir(exist_ok=True)

    # Prepare tokenizer training data
    print("=" * 50)
    print("Step 1: Preparing tokenizer training data")
    print("=" * 50)
    src_text_file, tgt_text_file = prepare_tokenizer_data(
        src_file, tgt_file, output_dir, max_samples=100000
    )

    # Train source tokenizer (English)
    print("\n" + "=" * 50)
    print("Step 2: Training source tokenizer (English)")
    print("=" * 50)
    src_model = str(output_dir / "src_tokenizer")
    train_tokenizer(
        str(src_text_file),
        src_model,
        vocab_size=config.vocab_size,
    )

    # Train target tokenizer (German)
    print("\n" + "=" * 50)
    print("Step 3: Training target tokenizer (German)")
    print("=" * 50)
    tgt_model = str(output_dir / "tgt_tokenizer")
    train_tokenizer(
        str(tgt_text_file),
        tgt_model,
        vocab_size=config.vocab_size,
    )

    print("\n" + "=" * 50)
    print("Preprocessing completed!")
    print("=" * 50)
    print(f"Source tokenizer: {src_model}.model")
    print(f"Target tokenizer: {tgt_model}.model")


if __name__ == "__main__":
    main()
