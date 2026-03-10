#!/usr/bin/env python3
"""
Quick test of training with improved architecture.
Trains for only 5 steps to verify everything works.
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
    print("Quick training test with improved architecture")
    print("=" * 80)

    # Override config for quick test
    config.batch_size = 4
    config.max_steps = 5
    config.max_train_samples = 20
    config.warmup_steps = 10  # Small for test
    config.save_interval = 1000  # Don't save
    config.device = "cpu"

    print(f"Test config: batch_size={config.batch_size}, max_steps={config.max_steps}")
    print(f"  learning_rate={config.learning_rate}, warmup_steps={config.warmup_steps}")
    print(f"  label_smoothing={config.label_smoothing}, clip_grad={config.clip_grad}")

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

    # Create fresh model with improved architecture
    model = Transformer(
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        d_model=config.d_model,
        n_layers=2,  # Small for test
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Check initialization
    print("\nChecking initialization...")
    embedding_std = model.encoder.embedding.weight.std().item()
    print(f"  Encoder embedding std: {embedding_std:.6f} (expected ~0.02)")

    # Check if it's pre-norm by examining a layer
    encoder_layer = model.encoder.layers[0]
    print(f"  Encoder layer structure: {encoder_layer.__class__.__name__}")
    print(f"  SelfAttention has pre-norm: {'norm' in dir(encoder_layer.self_attn)}")

    # Create trainer
    trainer = Trainer(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        config=config,
        device="cpu",
    )

    # Train for a few steps
    print("\nTraining for 5 steps...")
    try:
        best_loss = trainer.train(
            dataset=dataset,
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            max_len=config.max_len,
            start_step=0,
        )
        print(f"Training completed. Best loss: {best_loss:.4f}")
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 80)
    if success:
        print("✅ Quick test passed! Improved architecture works correctly.")
        print("Recommendation: Delete old checkpoints and train from scratch.")
        print("  rm models/best_model.pt")
        print("  python scripts/train.py")
    else:
        print("❌ Quick test failed. Check error messages above.")
    print("=" * 80)