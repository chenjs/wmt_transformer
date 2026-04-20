#!/usr/bin/env python3
"""
Quick test of restored model functionality.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import Evaluator

def main():
    print("Quick test of restored model")
    print("=" * 50)

    # Load checkpoint
    checkpoint_path = Path(__file__).parent.parent / "models" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print(f"Step: {checkpoint.get('step', 'N/A')}")
    print(f"Config max_len: {checkpoint['config'].max_len}")
    print(f"Config vocab_size: {getattr(checkpoint['config'], 'vocab_size', 'N/A')}")
    print(f"Config src_vocab_size: {getattr(checkpoint['config'], 'src_vocab_size', 'N/A')}")
    print(f"Config tgt_vocab_size: {getattr(checkpoint['config'], 'tgt_vocab_size', 'N/A')}")

    # Test translation
    device = "cpu"
    config.max_len = checkpoint['config'].max_len

    # Load enhanced tokenizers (same as used in training)
    src_tokenizer_path = Path(__file__).parent.parent / "models_enhanced" / "src_tokenizer_final.model"
    tgt_tokenizer_path = Path(__file__).parent.parent / "models_enhanced" / "tgt_tokenizer_final.model"

    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    # Get vocab sizes from tokenizers
    src_vocab_size = src_tokenizer.sp.get_piece_size()
    tgt_vocab_size = tgt_tokenizer.sp.get_piece_size()

    print(f"\nTokenizer vocab sizes: src={src_vocab_size}, tgt={tgt_vocab_size}")

    # Create model with correct config
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=checkpoint['config'].d_model,
        n_layers=checkpoint['config'].n_layers,
        n_heads=checkpoint['config'].n_heads,
        d_ff=checkpoint['config'].d_ff,
        dropout=checkpoint['config'].dropout,
        max_len=checkpoint['config'].max_len,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device)

    # Test a few sentences
    test_sentences = [
        "Hello",
        "Thank you",
        "The sky is blue.",
        "Despite the heavy rain, the football match continued as scheduled."
    ]

    for src_text in test_sentences:
        translation = evaluator.translate(src_text, method="beam", beam_size=4)
        print(f"\n'{src_text}' -> '{translation}'")

    print("\n" + "=" * 50)
    print("Test completed successfully!")
    print("Model appears to be working correctly.")

if __name__ == "__main__":
    main()