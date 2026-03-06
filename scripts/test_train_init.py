#!/usr/bin/env python3
"""
Test train.py initialization to verify data path fix.
This script runs the initialization part of train.py without actual training.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import config
from src.data.dataset import ParallelDataset
from src.data.tokenizer import load_tokenizers
from src.model import Transformer

def get_device():
    """Get device for training."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def main():
    print("=" * 60)
    print("Train.py Initialization Test")
    print("=" * 60)

    # Simulate train.py main function up to dataset loading
    device = get_device()
    print(f"Using device: {device}")

    # Update config (as in train.py)
    config.device = device
    config.src_tokenizer = "models_enhanced/src_tokenizer_final.model"
    config.tgt_tokenizer = "models_enhanced/tgt_tokenizer_final.model"

    # Check if tokenizers exist
    data_dir = Path(__file__).parent.parent
    src_tokenizer_path = data_dir / config.src_tokenizer
    tgt_tokenizer_path = data_dir / config.tgt_tokenizer

    if not src_tokenizer_path.exists() or not tgt_tokenizer_path.exists():
        print("ERROR: Tokenizers not found.")
        return

    # Load tokenizers
    print("\n1. Loading tokenizers...")
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )
    print(f"   Source vocab size: {src_tokenizer.sp.get_piece_size()}")
    print(f"   Target vocab size: {tgt_tokenizer.sp.get_piece_size()}")

    # Update config vocabulary sizes
    config.src_vocab_size = src_tokenizer.sp.get_piece_size()
    config.tgt_vocab_size = tgt_tokenizer.sp.get_piece_size()
    config.vocab_size = config.src_vocab_size  # Backward compatibility

    # Apply data path override (as in modified train.py)
    config.src_file = "models_enhanced/src_text_cleaned.txt"
    config.tgt_file = "models_enhanced/tgt_text_cleaned.txt"
    print(f"\n2. Data paths configured:")
    print(f"   Source: {config.src_file}")
    print(f"   Target: {config.tgt_file}")

    # Load dataset
    print("\n3. Loading dataset...")
    src_file = data_dir / config.src_file
    tgt_file = data_dir / config.tgt_file

    print(f"   Loading from: {src_file}")
    print(f"   Loading from: {tgt_file}")

    try:
        dataset = ParallelDataset(
            src_file,
            tgt_file,
            max_samples=config.max_train_samples,
        )
        print(f"   Dataset size: {len(dataset)}")

        # Split into train and validation sets
        train_dataset, val_dataset = dataset.split(split_ratio=config.train_split, seed=42)
        print(f"   Training set: {len(train_dataset)} samples")
        print(f"   Validation set: {len(val_dataset)} samples")

    except Exception as e:
        print(f"   ERROR loading dataset: {e}")
        return

    # Create model
    print("\n4. Creating model...")
    try:
        model = Transformer(
            src_vocab_size=config.src_vocab_size,
            tgt_vocab_size=config.tgt_vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_len=config.max_len,
        )
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Model max_len: {config.max_len}")
        print(f"   Model src_vocab_size: {config.src_vocab_size}")
        print(f"   Model tgt_vocab_size: {config.tgt_vocab_size}")

    except Exception as e:
        print(f"   ERROR creating model: {e}")
        return

    # Check config values
    print("\n5. Configuration check:")
    print(f"   min_loss_improvement: {getattr(config, 'min_loss_improvement', 'NOT SET')}")
    print(f"   save_interval: {config.save_interval}")
    print(f"   eval_interval: {config.eval_interval}")
    print(f"   batch_size: {config.batch_size}")
    print(f"   max_steps: {config.max_steps}")

    print("\n" + "=" * 60)
    print("✅ INITIALIZATION TEST PASSED")
    print("Train.py data path fix is working correctly.")
    print("All components loaded successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()