#!/usr/bin/env python3
"""
Test updated configuration after data optimization.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import config
from src.data.tokenizer import load_tokenizers
from src.data.dataset import ParallelDataset
from src.model import Transformer
from src.trainer import Trainer


def test_config():
    """Test configuration values."""
    print("=" * 60)
    print("TESTING UPDATED CONFIGURATION")
    print("=" * 60)

    # Check key configuration values
    print(f"vocab_size: {config.vocab_size} (expected: 16000)")
    print(f"src_vocab_size: {config.src_vocab_size} (expected: 16000)")
    print(f"tgt_vocab_size: {config.tgt_vocab_size} (expected: 16000)")
    print(f"max_len: {config.max_len} (expected: 54)")
    print(f"src_tokenizer: {config.src_tokenizer}")
    print(f"tgt_tokenizer: {config.tgt_tokenizer}")

    assert config.vocab_size == 16000, f"vocab_size should be 16000, got {config.vocab_size}"
    assert config.src_vocab_size == 16000, f"src_vocab_size should be 16000, got {config.src_vocab_size}"
    assert config.tgt_vocab_size == 16000, f"tgt_vocab_size should be 16000, got {config.tgt_vocab_size}"
    assert config.max_len == 54, f"max_len should be 54, got {config.max_len}"
    assert "models_enhanced" in config.src_tokenizer, f"src_tokenizer should point to models_enhanced"
    assert "models_enhanced" in config.tgt_tokenizer, f"tgt_tokenizer should point to models_enhanced"

    print("✅ Configuration values correct")


def test_tokenizers():
    """Test tokenizer loading and basic functionality."""
    print("\n" + "=" * 60)
    print("TESTING TOKENIZERS")
    print("=" * 60)

    # Build absolute paths
    data_dir = Path(__file__).parent.parent
    src_tokenizer_path = data_dir / config.src_tokenizer
    tgt_tokenizer_path = data_dir / config.tgt_tokenizer

    print(f"Source tokenizer path: {src_tokenizer_path}")
    print(f"Target tokenizer path: {tgt_tokenizer_path}")

    assert src_tokenizer_path.exists(), f"Source tokenizer not found: {src_tokenizer_path}"
    assert tgt_tokenizer_path.exists(), f"Target tokenizer not found: {tgt_tokenizer_path}"

    # Load tokenizers
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    print(f"Source vocab size: {src_tokenizer.sp.get_piece_size()}")
    print(f"Target vocab size: {tgt_tokenizer.sp.get_piece_size()}")

    assert src_tokenizer.sp.get_piece_size() == 16000, \
        f"Source tokenizer vocab size should be 16000, got {src_tokenizer.sp.get_piece_size()}"
    assert tgt_tokenizer.sp.get_piece_size() == 16000, \
        f"Target tokenizer vocab size should be 16000, got {tgt_tokenizer.sp.get_piece_size()}"

    # Test encoding/decoding
    test_sentence = "Hello, how are you?"
    src_ids = src_tokenizer.encode(test_sentence)
    src_decoded = src_tokenizer.decode(src_ids)

    print(f"Test sentence: '{test_sentence}'")
    print(f"Encoded IDs: {src_ids[:10]}...")
    print(f"Decoded: '{src_decoded}'")

    # Basic sanity check
    assert len(src_ids) > 0, "Encoding should produce tokens"
    assert isinstance(src_decoded, str), "Decoding should return string"

    print("✅ Tokenizers loaded and functional")


def test_dataset():
    """Test dataset loading with new configuration."""
    print("\n" + "=" * 60)
    print("TESTING DATASET")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent
    src_file = data_dir / config.src_file
    tgt_file = data_dir / config.tgt_file

    print(f"Source file: {src_file}")
    print(f"Target file: {tgt_file}")
    print(f"max_train_samples: {config.max_train_samples}")

    dataset = ParallelDataset(
        src_file,
        tgt_file,
        max_samples=config.max_train_samples,
    )

    print(f"Dataset size: {len(dataset)}")

    # Load tokenizers for dataset testing
    src_tokenizer_path = data_dir / config.src_tokenizer
    tgt_tokenizer_path = data_dir / config.tgt_tokenizer
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    # Test a few samples
    for i in range(min(3, len(dataset))):
        src_text, tgt_text = dataset[i]
        src_ids = src_tokenizer.encode(src_text)
        tgt_ids = tgt_tokenizer.encode(tgt_text)

        print(f"\nSample {i}:")
        print(f"  Source: '{src_text[:50]}...' (len={len(src_text.split())} words, {len(src_ids)} tokens)")
        print(f"  Target: '{tgt_text[:50]}...' (len={len(tgt_text.split())} words, {len(tgt_ids)} tokens)")

        # Check length constraints
        if len(src_ids) > config.max_len:
            print(f"  ⚠ Source exceeds max_len: {len(src_ids)} > {config.max_len}")
        if len(tgt_ids) > config.max_len:
            print(f"  ⚠ Target exceeds max_len: {len(tgt_ids)} > {config.max_len}")

    print(f"\n✅ Dataset loaded with {len(dataset)} samples")


def test_model():
    """Test model creation with new vocab sizes."""
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)

    # Get device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    print(f"Using device: {device}")

    # Create model with new vocab sizes
    model = Transformer(
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with dummy data
    batch_size = 2
    seq_len = 10

    src = torch.randint(0, config.src_vocab_size, (batch_size, seq_len)).to(device)
    tgt = torch.randint(0, config.tgt_vocab_size, (batch_size, seq_len)).to(device)

    # Create masks
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    seq_mask = torch.tril(torch.ones(seq_len, seq_len)).to(device).bool()
    tgt_mask = tgt_mask & seq_mask

    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)

    print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    assert output.shape == (batch_size, seq_len, config.tgt_vocab_size), \
        f"Expected output shape {(batch_size, seq_len, config.tgt_vocab_size)}, got {output.shape}"

    print("✅ Model created and forward pass successful")


def test_training_step():
    """Test a single training step with new configuration."""
    print("\n" + "=" * 60)
    print("TESTING TRAINING STEP")
    print("=" * 60)

    # Get device
    device = "cpu"  # Use CPU for testing to avoid MPS/CUDA issues
    print(f"Using device: {device}")

    # Load tokenizers
    data_dir = Path(__file__).parent.parent
    src_tokenizer_path = data_dir / config.src_tokenizer
    tgt_tokenizer_path = data_dir / config.tgt_tokenizer
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    # Create model
    model = Transformer(
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    ).to(device)

    # Create trainer
    trainer = Trainer(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        config=config,
        device=device,
        val_dataset=None,
    )

    print(f"Trainer created with:")
    print(f"  Source vocab: {src_tokenizer.sp.get_piece_size()}")
    print(f"  Target vocab: {tgt_tokenizer.sp.get_piece_size()}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Label smoothing: {config.label_smoothing}")

    # Create a dummy batch
    batch_size = 2
    seq_len = 8

    src = torch.randint(1, 100, (batch_size, seq_len)).to(device)
    tgt = torch.randint(1, 100, (batch_size, seq_len)).to(device)

    # Add BOS/EOS tokens (simplified)
    tgt[:, 0] = 1  # BOS
    tgt[:, -1] = 2  # EOS

    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    batch = {
        'src': src,
        'tgt': tgt,
        'src_mask': src_mask,
    }

    # Perform a training step
    try:
        loss = trainer.train_step(batch)
        print(f"Training step completed with loss: {loss:.4f}")
        print("✅ Training step successful")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all tests."""
    try:
        test_config()
        test_tokenizers()
        test_dataset()
        test_model()
        test_training_step()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60)
        print("Configuration updated successfully.")
        print("Ready for training with optimized data and tokenizers.")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()