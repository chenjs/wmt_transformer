#!/usr/bin/env python3
"""
Test training for a few steps to see if loss decreases.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data.dataset import ParallelDataset
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.trainer import Trainer

def main():
    print("Testing training for a few steps...")

    # Modify config for testing
    config.label_smoothing = 0.0  # Disable label smoothing
    config.learning_rate = 1e-3   # Increase learning rate
    config.batch_size = 4         # Small batch
    config.max_train_samples = 100  # Small dataset
    config.max_steps = 10         # Train only 10 steps
    config.warmup_steps = 0       # No warmup for test
    config.save_interval = 1000   # Don't save

    device = "cpu"
    config.device = device

    # Load tokenizers
    data_dir = Path(__file__).parent
    src_tokenizer_path = data_dir / "models" / "src_tokenizer.model"
    tgt_tokenizer_path = data_dir / "models" / "tgt_tokenizer.model"

    if not src_tokenizer_path.exists() or not tgt_tokenizer_path.exists():
        print("Tokenizers not found. Please run preprocess.py first.")
        return

    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    # Update config vocabulary sizes
    config.src_vocab_size = src_tokenizer.sp.get_piece_size()
    config.tgt_vocab_size = tgt_tokenizer.sp.get_piece_size()

    # Load small dataset
    src_file = data_dir / config.src_file
    tgt_file = data_dir / config.tgt_file
    dataset = ParallelDataset(
        src_file,
        tgt_file,
        max_samples=config.max_train_samples,
    )
    print(f"Dataset size: {len(dataset)}")

    # Create fresh model
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
    )

    # Train for a few steps
    print("\nStarting training...")
    try:
        best_loss = trainer.train(
            dataset=dataset,
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            max_len=config.max_len,
            start_step=0,
        )
        print(f"\nFinal best loss: {best_loss:.4f}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

    # Test the trained model on a simple example
    print("\nTesting trained model...")
    model.eval()
    from src.evaluate import Evaluator
    evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device=device)

    test_sentences = ["Hello", "Good morning", "Thank you"]
    for text in test_sentences:
        translation = evaluator.translate(text, method="greedy")
        print(f"'{text}' → '{translation[:50]}...'")

if __name__ == "__main__":
    main()