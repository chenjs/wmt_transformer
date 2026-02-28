"""
Diagnostic test for the translation model.
Check if model produces same output for different inputs.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer

def test_model():
    # Load tokenizers
    data_dir = Path(__file__).parent
    src_tokenizer_path = data_dir / "models" / "src_tokenizer.model"
    tgt_tokenizer_path = data_dir / "models" / "tgt_tokenizer.model"
    checkpoint_path = data_dir / "models" / "best_model.pt"

    if not checkpoint_path.exists():
        print("No checkpoint found")
        return

    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )
    # FIX 2026-02-26: Compute separate vocabulary sizes
    src_vocab_size = src_tokenizer.sp.get_piece_size()
    tgt_vocab_size = tgt_tokenizer.sp.get_piece_size()
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")
    # Set default vocab sizes in config (will be overridden by checkpoint config if present)
    config.src_vocab_size = src_vocab_size
    config.tgt_vocab_size = tgt_vocab_size

    # Load checkpoint
    device = "cpu"
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

    # Create model with separate vocabulary sizes
    # FIX 2026-02-26: Use src_vocab_size and tgt_vocab_size
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
    model.eval()

    # Test sentences
    test_sentences = [
        "Hello world",
        "How are you",
        "This is a test",
        "The quick brown fox",
        "I love machine learning"
    ]

    from src.evaluate import Evaluator
    evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device)

    print("Testing model translations:")
    for i, src_text in enumerate(test_sentences):
        tgt_text = evaluator.translate(src_text, method="greedy")
        print(f"{i+1}. Input: '{src_text}'")
        print(f"   Output: '{tgt_text}'")
        print()

    # Also check encoder outputs
    print("\nChecking encoder outputs...")
    for i, src_text in enumerate(test_sentences[:2]):
        src_tokens = src_tokenizer(src_text, add_bos=False, add_eos=True)
        src = torch.tensor([src_tokens], dtype=torch.long, device=device)
        src_mask = torch.ones(1, 1, 1, len(src_tokens), dtype=torch.bool, device=device)

        with torch.no_grad():
            encoder_output = model.encode(src, src_mask)
            print(f"Encoder output shape for '{src_text}': {encoder_output.shape}")
            print(f"Mean: {encoder_output.mean().item():.6f}, Std: {encoder_output.std().item():.6f}")
            print(f"Min: {encoder_output.min().item():.6f}, Max: {encoder_output.max().item():.6f}")
            print()

if __name__ == "__main__":
    test_model()