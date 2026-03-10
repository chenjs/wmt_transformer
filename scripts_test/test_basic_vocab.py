#!/usr/bin/env python3
"""
Test basic vocabulary translation before and after fine-tuning.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import Evaluator


def get_device():
    """Get device for inference."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    device = get_device()
    data_dir = Path(__file__).parent.parent

    # Load tokenizers
    src_tokenizer_path = data_dir / "models_enhanced" / "src_tokenizer_final.model"
    tgt_tokenizer_path = data_dir / "models_enhanced" / "tgt_tokenizer_final.model"

    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Update config from checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        config.max_len = saved_config.max_len
        config.d_model = saved_config.d_model
        config.n_layers = saved_config.n_layers
        config.n_heads = saved_config.n_heads
        config.d_ff = saved_config.d_ff
        config.dropout = saved_config.dropout

        if hasattr(saved_config, 'src_vocab_size') and hasattr(saved_config, 'tgt_vocab_size'):
            config.src_vocab_size = saved_config.src_vocab_size
            config.tgt_vocab_size = saved_config.tgt_vocab_size
        else:
            config.src_vocab_size = saved_config.vocab_size
            config.tgt_vocab_size = saved_config.vocab_size

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
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create evaluator
    evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device)

    return evaluator


def test_basic_vocab():
    """Test basic vocabulary translation."""
    print("=" * 60)
    print("Basic Vocabulary Translation Test")
    print("=" * 60)

    # Test sentences (problematic basic vocabulary)
    test_sentences = [
        "Hello",
        "Thank you",
        "Good morning",
        "How are you?",
        "What time is it?",
        "Where is the station?",
        "I love you",
        "My name is John",
        "The sky is blue",
        "This is a test",
    ]

    # Load original model
    print("\n1. Loading original model (best_model.pt)...")
    original_evaluator = load_model(Path(__file__).parent.parent / "models" / "best_model.pt")

    # Load fine-tuned model
    print("2. Loading fine-tuned model (model_fine_tuned_basic.pt)...")
    finetuned_evaluator = load_model(Path(__file__).parent.parent / "models" / "model_fine_tuned_basic.pt")

    print("\n" + "=" * 60)
    print("Translation Results (Beam-4 decoding)")
    print("=" * 60)

    results = []
    for src_text in test_sentences:
        # Translate with both models
        original_translation = original_evaluator.translate(src_text, method="beam", beam_size=4)
        finetuned_translation = finetuned_evaluator.translate(src_text, method="beam", beam_size=4)

        results.append({
            'src': src_text,
            'original': original_translation,
            'finetuned': finetuned_translation,
        })

        print(f"\nEnglish: {src_text}")
        print(f"Original model: {original_translation}")
        print(f"Fine-tuned model: {finetuned_translation}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    improved = 0
    same = 0
    worse = 0

    for r in results:
        orig = r['original']
        fine = r['finetuned']

        # Simple heuristic: if translation changed
        if orig == fine:
            same += 1
        else:
            # Check if fine-tuned translation looks more correct
            # For basic vocabulary, we expect certain words
            src_lower = r['src'].lower()
            fine_lower = fine.lower()

            # Check for expected keywords
            expected_keywords = {
                'hello': ['hallo', 'guten'],
                'thank you': ['danke'],
                'good morning': ['guten morgen'],
                'how are you': ['wie geht'],
                'what time is it': ['wie spät', 'uhr'],
                'where is the station': ['bahnhof', 'station'],
                'i love you': ['liebe', 'lieb'],
                'my name is': ['name', 'heiße', 'heisse'],
                'the sky is blue': ['himmel', 'blau'],
                'this is a test': ['test', 'prüfung'],
            }

            # Determine if improvement
            is_improved = False
            for key, expected_list in expected_keywords.items():
                if key in src_lower:
                    for expected in expected_list:
                        if expected in fine_lower:
                            is_improved = True
                            break
                    break

            if is_improved:
                improved += 1
            else:
                worse += 1

    print(f"Total test sentences: {len(test_sentences)}")
    print(f"Improved translations: {improved}")
    print(f"Same translations: {same}")
    print(f"Worse translations: {worse}")

    if improved > 0:
        print("\n✅ Fine-tuning shows positive effect on basic vocabulary!")
    else:
        print("\n⚠️  Fine-tuning did not improve basic vocabulary translation.")

    # Save results to file
    output_file = Path(__file__).parent.parent / "evaluation_results" / "basic_vocab_test.json"
    output_file.parent.mkdir(exist_ok=True)

    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    test_basic_vocab()