"""
Translate text using trained model.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import Evaluator, greedy_decode, beam_search_decode


def get_device():
    """Get device for inference."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def main():
    """Main translation function."""
    print("=" * 50)
    print("Transformer Translation")
    print("=" * 50)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Paths
    data_dir = Path(__file__).parent.parent
    src_tokenizer_path = data_dir / "models" / "src_tokenizer.model"
    tgt_tokenizer_path = data_dir / "models" / "tgt_tokenizer.model"
    checkpoint_path = data_dir / "models" / "best_model.pt"

    # Check if tokenizers exist
    if not src_tokenizer_path.exists() or not tgt_tokenizer_path.exists():
        print("Tokenizers not found. Please run preprocess.py first.")
        return

    # Load tokenizers and compute vocabulary sizes
    # FIX 2026-02-26: Compute separate vocabulary sizes for backward compatibility
    print("\nLoading tokenizers...")
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )
    src_vocab_size = src_tokenizer.sp.get_piece_size()
    tgt_vocab_size = tgt_tokenizer.sp.get_piece_size()
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")

    # Load checkpoint first to get config
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Use config from checkpoint
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            config.max_len = saved_config.max_len
            config.d_model = saved_config.d_model
            config.n_layers = saved_config.n_layers
            config.n_heads = saved_config.n_heads
            config.d_ff = saved_config.d_ff
            config.dropout = saved_config.dropout
            # FIX 2026-02-26: Handle vocabulary sizes (backward compatibility)
            # Check if checkpoint has separate vocab sizes or uses old single vocab_size
            if hasattr(saved_config, 'src_vocab_size') and hasattr(saved_config, 'tgt_vocab_size'):
                config.src_vocab_size = saved_config.src_vocab_size
                config.tgt_vocab_size = saved_config.tgt_vocab_size
                print(f"Loaded config with separate vocab sizes: src={config.src_vocab_size}, tgt={config.tgt_vocab_size}")
            else:
                # Old checkpoint: use single vocab_size for both
                config.src_vocab_size = saved_config.vocab_size
                config.tgt_vocab_size = saved_config.vocab_size
                print(f"Loaded old config with single vocab size: {saved_config.vocab_size}")
            print(f"Loaded config: max_len={config.max_len}, d_model={config.d_model}")

        # Create model with saved config using separate vocabulary sizes
        # FIX 2026-02-26: Use src_vocab_size and tgt_vocab_size
        print("Creating model...")
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
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded!")
    else:
        print("No checkpoint found. Using randomly initialized model.")
        # FIX 2026-02-26: Use tokenizer vocab sizes for new model
        config.src_vocab_size = src_vocab_size
        config.tgt_vocab_size = tgt_vocab_size
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

    model = model.to(device)
    model.eval()

    # Create evaluator
    evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device)

    # Interactive translation
    print("\n" + "=" * 50)
    print("Translation Mode")
    print("Enter text to translate (Ctrl+C to quit)")
    print("=" * 50)

    while True:
        try:
            src_text = input("\nEnglish> ").strip()
            if not src_text:
                continue

            # Translate
            tgt_text = evaluator.translate(src_text, method="greedy")
            print(f"German> {tgt_text}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
