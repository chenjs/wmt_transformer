#!/usr/bin/env python3
"""
Fine-tune model on basic vocabulary to improve translation of common phrases.
"""
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import Config
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


def main(resume_from: str = None, max_steps: int = 5000):
    """Fine-tuning function for basic vocabulary."""
    print("=" * 50)
    print("Fine-tuning for Basic Vocabulary")
    print("=" * 50)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create fine-tuning config (override some defaults)
    config = Config()
    config.device = device

    # Use enhanced tokenizers (same as main training)
    config.src_tokenizer = "models_enhanced/src_tokenizer_final.model"
    config.tgt_tokenizer = "models_enhanced/tgt_tokenizer_final.model"

    # Use basic vocabulary data files
    config.src_file = "data_basic/basic_vocab.en"
    config.tgt_file = "data_basic/basic_vocab.de"

    # Fine-tuning hyperparameters
    config.batch_size = 8  # Smaller batch for fine-tuning
    config.learning_rate = 1e-4  # Lower learning rate for fine-tuning
    config.max_steps = max_steps
    config.max_train_samples = 100  # Only use basic vocabulary data
    config.train_split = 0.9  # Still use train/val split
    config.eval_interval = 500  # Evaluate more frequently
    config.save_interval = 1000  # Save checkpoints
    config.min_loss_improvement = 0.01  # Only save if loss improves by at least 1%

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

    # Update config vocabulary sizes
    config.src_vocab_size = src_tokenizer.sp.get_piece_size()
    config.tgt_vocab_size = tgt_tokenizer.sp.get_piece_size()
    config.vocab_size = config.src_vocab_size  # Backward compatibility

    # Load basic vocabulary dataset
    print("\nLoading basic vocabulary dataset...")
    src_file = data_dir / config.src_file
    tgt_file = data_dir / config.tgt_file

    if not src_file.exists() or not tgt_file.exists():
        print(f"Basic vocabulary data not found: {src_file}, {tgt_file}")
        print("Please create basic_vocab.en and basic_vocab.de in data_basic/ directory")
        return

    dataset = ParallelDataset(
        src_file,
        tgt_file,
        max_samples=config.max_train_samples,
    )
    print(f"Dataset size: {len(dataset)}")

    # Split into train and validation sets
    train_dataset, val_dataset = dataset.split(split_ratio=config.train_split, seed=42)
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")

    # Create model with separate vocabulary sizes
    print("\nCreating model...")
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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        config=config,
        device=device,
        val_dataset=val_dataset,
    )

    # Load model from checkpoint (required for fine-tuning)
    start_step = 0
    if resume_from:
        checkpoint_path = data_dir / "models" / resume_from
    else:
        # Default to best_model.pt
        checkpoint_path = data_dir / "models" / "best_model.pt"

    if checkpoint_path.exists():
        print(f"\nLoading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")

        # Restore scheduler step number (important for learning rate schedule)
        if hasattr(trainer.scheduler, 'step_num'):
            # First try to get scheduler_step_num from checkpoint
            if 'scheduler_step_num' in checkpoint:
                trainer.scheduler.step_num = checkpoint['scheduler_step_num']
                print(f"Restored scheduler step from checkpoint: {trainer.scheduler.step_num}")
            # Fallback to using step field
            elif 'step' in checkpoint:
                trainer.scheduler.step_num = checkpoint['step']
                print(f"Set scheduler step from step field: {trainer.scheduler.step_num}")
            else:
                print(f"Warning: No step information found in checkpoint, scheduler step may be incorrect")

        # Adjust learning rate for fine-tuning
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = config.learning_rate
        print(f"Learning rate set to {config.learning_rate} for fine-tuning")

        # For fine-tuning, reset step counter to 0 but keep scheduler state
        # The scheduler step_num is restored separately from checkpoint
        start_step = 0
        print(f"Starting fine-tuning from step 0 (original model at step {checkpoint.get('step', 'N/A')})")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Cannot fine-tune without a pretrained model. Exiting.")
        return

    # Train (fine-tune)
    print("\nStarting fine-tuning...")
    print(f"Fine-tuning steps: {config.max_steps}")
    print(f"Using basic vocabulary dataset with {len(train_dataset)} training samples")

    trainer.train(
        dataset=train_dataset,
        batch_size=config.batch_size,
        max_steps=config.max_steps,
        max_len=config.max_len,
        start_step=start_step,
    )

    # Save fine-tuned model
    fine_tuned_path = data_dir / "models" / "model_fine_tuned_basic.pt"
    trainer.save_checkpoint(fine_tuned_path, step=start_step + config.max_steps)
    print(f"\nFine-tuned model saved to: {fine_tuned_path}")
    print("Fine-tuning completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="best_model.pt",
                       help="Resume from checkpoint file (default: best_model.pt)")
    parser.add_argument("--max-steps", type=int, default=5000,
                       help="Fine-tuning steps (default: 5000)")
    args = parser.parse_args()

    main(resume_from=args.resume, max_steps=args.max_steps)