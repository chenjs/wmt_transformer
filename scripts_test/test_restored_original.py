#!/usr/bin/env python3
"""
Test the restored original 200k-step model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import Config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import Evaluator

def test_original_model():
    print("=" * 70)
    print("Testing Restored Original 200k-step Model")
    print("=" * 70)

    checkpoint_path = Path(__file__).parent.parent / "models" / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = checkpoint['config']
    step = checkpoint.get('step', 'N/A')

    print(f"Model step: {step}")
    print(f"Config max_len: {saved_config.max_len}")
    print(f"Config src_vocab_size: {getattr(saved_config, 'src_vocab_size', getattr(saved_config, 'vocab_size', 'N/A'))}")
    print(f"Config tgt_vocab_size: {getattr(saved_config, 'tgt_vocab_size', getattr(saved_config, 'vocab_size', 'N/A'))}")

    # Create config from saved config
    test_config = Config()

    # Override with saved config values
    test_config.max_len = saved_config.max_len
    test_config.d_model = saved_config.d_model
    test_config.n_layers = saved_config.n_layers
    test_config.n_heads = saved_config.n_heads
    test_config.d_ff = saved_config.d_ff
    test_config.dropout = saved_config.dropout

    if hasattr(saved_config, 'src_vocab_size') and hasattr(saved_config, 'tgt_vocab_size'):
        test_config.src_vocab_size = saved_config.src_vocab_size
        test_config.tgt_vocab_size = saved_config.tgt_vocab_size
    else:
        test_config.src_vocab_size = saved_config.vocab_size
        test_config.tgt_vocab_size = saved_config.vocab_size

    # Get device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load tokenizers - use enhanced tokenizers that the model was trained with
    data_dir = Path(__file__).parent.parent

    # Enhanced model was trained with these tokenizers
    src_tokenizer_path = data_dir / "models_enhanced" / "src_tokenizer_final.model"
    tgt_tokenizer_path = data_dir / "models_enhanced" / "tgt_tokenizer_final.model"

    print(f"Using enhanced tokenizers (model training tokenizers):")
    print(f"  Source: {src_tokenizer_path}")
    print(f"  Target: {tgt_tokenizer_path}")

    if not src_tokenizer_path.exists() or not tgt_tokenizer_path.exists():
        print(f"Error: Tokenizers not found at {src_tokenizer_path} or {tgt_tokenizer_path}")
        return

    print(f"Loading tokenizers from:")
    print(f"  Source: {src_tokenizer_path}")
    print(f"  Target: {tgt_tokenizer_path}")

    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    # Create model with correct config
    model = Transformer(
        src_vocab_size=test_config.src_vocab_size,
        tgt_vocab_size=test_config.tgt_vocab_size,
        d_model=test_config.d_model,
        n_layers=test_config.n_layers,
        n_heads=test_config.n_heads,
        d_ff=test_config.d_ff,
        dropout=test_config.dropout,
        max_len=test_config.max_len,
    )

    # Load state dict
    print("Loading model state dict...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create evaluator
    evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device)

    # Test sentences
    test_sentences = [
        ("Basic vocabulary - should be poor in original", "Hello"),
        ("Basic vocabulary - should be poor in original", "Thank you"),
        ("Simple sentence - should work", "The sky is blue."),
        ("Complex sentence - should work well", "Despite the heavy rain, the football match continued as scheduled."),
        ("Abstract concept - should work", "The concept of democracy is fundamental to modern society."),
    ]

    print("\n" + "=" * 70)
    print("Translation Tests")
    print("=" * 70)

    for description, src_text in test_sentences:
        translation = evaluator.translate(src_text, method="beam", beam_size=4)
        print(f"\n[{description}]")
        print(f"English: {src_text}")
        print(f"German:  {translation}")

    # Quality assessment
    print("\n" + "=" * 70)
    print("Quality Assessment")
    print("=" * 70)

    # Check basic vocabulary quality (should be poor in original)
    basic_tests = ["Hello", "Thank you"]
    correct_basic = 0
    for src_text in basic_tests:
        translation = evaluator.translate(src_text, method="beam", beam_size=4).lower()
        src_lower = src_text.lower()

        if "hello" in src_lower:
            if "hallo" in translation or "guten" in translation:
                correct_basic += 1
                print(f"✅ '{src_text}' -> '{translation}' (correct basic vocabulary)")
            else:
                print(f"❌ '{src_text}' -> '{translation}' (expected German translation)")
        elif "thank you" in src_lower:
            if "danke" in translation:
                correct_basic += 1
                print(f"✅ '{src_text}' -> '{translation}' (correct basic vocabulary)")
            else:
                print(f"❌ '{src_text}' -> '{translation}' (expected German translation)")

    # Check complex sentence quality
    complex_text = "Despite the heavy rain, the football match continued as scheduled."
    complex_translation = evaluator.translate(complex_text, method="beam", beam_size=4)
    words = complex_translation.split()

    print(f"\nComplex sentence test:")
    print(f"English: {complex_text}")
    print(f"German:  {complex_translation}")

    if len(words) >= 5 and any(c.isalpha() for c in complex_translation):
        print(f"✅ Complex sentence: Reasonable output length ({len(words)} words)")
        complex_ok = True
    else:
        print(f"⚠️  Complex sentence: Short or odd output ({len(words)} words)")
        complex_ok = False

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Original Model Verification")
    print("=" * 70)

    if correct_basic == 0 and complex_ok:
        print("✅ **CONFIRMED: This is likely the original 200k-step model**")
        print("   - Basic vocabulary translation: Poor (as expected for original model)")
        print("   - Complex sentence translation: Good (as expected for original model)")
    elif correct_basic > 0 and complex_ok:
        print("⚠️  **MIXED: Model shows some fine-tuning effects**")
        print(f"   - Basic vocabulary correct: {correct_basic}/2")
        print("   - Complex sentence translation: Good")
    elif correct_basic > 0 and not complex_ok:
        print("❌ **FINE-TUNED: Model shows catastrophic forgetting**")
        print(f"   - Basic vocabulary correct: {correct_basic}/2")
        print("   - Complex sentence translation: Poor")
    else:
        print("❓ **UNKNOWN: Model behavior unexpected**")

    print(f"\nModel step: {step}")
    print(f"Config max_len: {saved_config.max_len}")
    print(f"Model successfully loaded and tested.")

if __name__ == "__main__":
    test_original_model()