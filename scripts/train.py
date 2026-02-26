"""
Train the Transformer model.
"""
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import config
from src.data.dataset import ParallelDataset
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.trainer import Trainer


def get_device():
    """Get device for training."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def main(resume_from: str = None):
    """Main training function."""
    print("=" * 50)
    print("Transformer Translation Training")
    print("=" * 50)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Update config
    config.device = device
    config.src_tokenizer = "models/src_tokenizer.model"
    config.tgt_tokenizer = "models/tgt_tokenizer.model"

    # Check if tokenizers exist
    data_dir = Path(__file__).parent.parent
    src_tokenizer_path = data_dir / config.src_tokenizer
    tgt_tokenizer_path = data_dir / config.tgt_tokenizer

    if not src_tokenizer_path.exists() or not tgt_tokenizer_path.exists():
        print("Tokenizers not found. Please run preprocess.py first.")
        print(f"Expected: {src_tokenizer_path} and {tgt_tokenizer_path}")
        return

    # Load tokenizers
    print("\nLoading tokenizers...")
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )
    print(f"Source vocab size: {src_tokenizer.sp.get_piece_size()}")
    print(f"Target vocab size: {tgt_tokenizer.sp.get_piece_size()}")

    # Update config vocab size
    config.vocab_size = src_tokenizer.sp.get_piece_size()

    # Load dataset
    print("\nLoading dataset...")
    src_file = data_dir / config.src_file
    tgt_file = data_dir / config.tgt_file
    dataset = ParallelDataset(
        src_file,
        tgt_file,
        max_samples=config.max_train_samples,
    )
    print(f"Dataset size: {len(dataset)}")

    # Create model
    print("\nCreating model...")
    model = Transformer(
        src_vocab_size=config.vocab_size,
        tgt_vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        config=config,
        device=device,
    )

    # Resume from checkpoint if specified
    start_step = 0
    if resume_from:
        checkpoint_path = data_dir / "models" / resume_from
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {resume_from}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Get saved step count
            if 'step' in checkpoint:
                start_step = checkpoint['step']
            print(f"Resuming from step {start_step}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    # Train
    print("\nStarting training...")
    trainer.train(
        dataset=dataset,
        batch_size=config.batch_size,
        max_steps=config.max_steps,
        max_len=config.max_len,
        start_step=start_step,
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint file")
    args = parser.parse_args()

    main(resume_from=args.resume)
