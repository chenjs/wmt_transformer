#!/usr/bin/env python3
"""
Evaluate translation quality after training.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import Evaluator

def main():
    print("Translation Quality Evaluation")
    print("=" * 80)

    # Load tokenizers - use enhanced tokenizers (consistent with training)
    data_dir = Path(__file__).parent
    src_tokenizer_path = data_dir / "models_enhanced" / "src_tokenizer_final.model"
    tgt_tokenizer_path = data_dir / "models_enhanced" / "tgt_tokenizer_final.model"
    checkpoint_path = data_dir / "models" / "best_model.pt"

    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    # Load checkpoint
    device = "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        if hasattr(saved_config, 'src_vocab_size'):
            src_vocab_size = saved_config.src_vocab_size
            tgt_vocab_size = saved_config.tgt_vocab_size
        else:
            src_vocab_size = saved_config.vocab_size
            tgt_vocab_size = saved_config.vocab_size
    else:
        src_vocab_size = config.src_vocab_size
        tgt_vocab_size = config.tgt_vocab_size

    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create evaluator
    evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device=device)

    # Test sentences
    test_cases = [
        # Basic phrases
        ("Hello", "Hallo"),
        ("Good morning", "Guten Morgen"),
        ("Thank you", "Danke"),
        ("How are you?", "Wie geht es dir?"),
        ("I am fine", "Mir geht es gut"),

        # Simple sentences
        ("This is a test", "Das ist ein Test"),
        ("The sky is blue", "Der Himmel ist blau"),
        ("I like apples", "Ich mag Äpfel"),
        ("She reads a book", "Sie liest ein Buch"),
        ("We are learning", "Wir lernen"),

        # Slightly more complex
        ("The quick brown fox jumps over the lazy dog",
         "Der schnelle braune Fuchs springt über den faulen Hund"),
        ("I love machine learning and artificial intelligence",
         "Ich liebe maschinelles Lernen und künstliche Intelligenz"),
        ("The weather is beautiful today",
         "Das Wetter ist heute schön"),
        ("Can you help me please?",
         "Kannst du mir bitte helfen?"),
        ("What time is it?",
         "Wie spät ist es?"),
    ]

    print("Testing translation quality:")
    print("-" * 80)

    correct_count = 0
    partial_correct = 0
    total = len(test_cases)

    for i, (src, expected) in enumerate(test_cases):
        translation = evaluator.translate(src, method="beam", beam_size=4)

        # Clean up translation (remove special tokens)
        translation = translation.replace("[BOS]", "").replace("[EOS]", "").strip()

        # Simple evaluation
        src_lower = src.lower()
        trans_lower = translation.lower()
        expected_lower = expected.lower()

        # Check if translation contains key words
        key_words = expected_lower.split()
        matched_words = sum(1 for word in key_words if word in trans_lower)
        match_ratio = matched_words / len(key_words) if key_words else 0

        print(f"{i+1}. Input: '{src}'")
        print(f"   Expected: '{expected}'")
        print(f"   Translation: '{translation}'")

        if trans_lower == expected_lower:
            print(f"   ✅ Perfect match!")
            correct_count += 1
        elif match_ratio >= 0.5:
            print(f"   ⚠️  Partial match ({match_ratio:.0%} of key words)")
            partial_correct += 1
        else:
            print(f"   ❌ Poor match ({match_ratio:.0%} of key words)")

        print()

    print("=" * 80)
    print("Summary:")
    print(f"  Total test cases: {total}")
    print(f"  Perfect matches: {correct_count} ({correct_count/total:.0%})")
    print(f"  Partial matches: {partial_correct} ({partial_correct/total:.0%})")
    print(f"  Poor matches: {total - correct_count - partial_correct} ({(total - correct_count - partial_correct)/total:.0%})")

    # Test beam search vs greedy
    print("\n" + "=" * 80)
    print("Comparing decoding methods (on 3 examples):")

    comparison_cases = ["Hello", "Thank you", "This is a test"]
    for src in comparison_cases:
        greedy = evaluator.translate(src, method="greedy")  # Keep for comparison
        beam = evaluator.translate(src, method="beam", beam_size=4)

        greedy = greedy.replace("[BOS]", "").replace("[EOS]", "").strip()
        beam = beam.replace("[BOS]", "").replace("[EOS]", "").strip()

        print(f"\nInput: '{src}'")
        print(f"  Greedy: '{greedy}'")
        print(f"  Beam search (beam_size=4): '{beam}'")
        if greedy == beam:
            print(f"  ⚠️  Both methods produce same output")
        else:
            print(f"  ✅ Different outputs")

    # Test with different input lengths
    print("\n" + "=" * 80)
    print("Testing with varying input lengths:")

    length_test = [
        ("Hi", "短"),
        ("Hello world", "中"),
        ("This is a relatively longer sentence to test the model's capability", "长"),
    ]

    for src, desc in length_test:
        translation = evaluator.translate(src, method="beam", beam_size=4)
        translation = translation.replace("[BOS]", "").replace("[EOS]", "").strip()
        print(f"{desc} input ({len(src.split())} words): '{src}'")
        print(f"  → '{translation}' ({len(translation.split())} words)")
        print()

    print("=" * 80)
    print("Evaluation completed.")

if __name__ == "__main__":
    main()