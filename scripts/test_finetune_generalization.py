#!/usr/bin/env python3
"""
Test generalization of fine-tuned model on complex sentences.
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


def test_generalization():
    """Test model generalization on various sentence types."""
    print("=" * 70)
    print("Fine-tuned Model Generalization Test")
    print("=" * 70)

    # Test sentences of different complexity levels
    test_sentences = [
        # Basic vocabulary (should be improved)
        ("Basic", "Hello"),
        ("Basic", "Thank you"),
        ("Basic", "Good morning"),

        # Simple sentences
        ("Simple", "I am going to the store."),
        ("Simple", "She reads a book every day."),
        ("Simple", "The weather is nice today."),

        # Moderate complexity
        ("Moderate", "Can you help me with this problem please?"),
        ("Moderate", "I would like to order a cup of coffee."),
        ("Moderate", "What is the time of the next train to Berlin?"),

        # Complex sentences
        ("Complex", "Despite the heavy rain, the football match continued as scheduled."),
        ("Complex", "The scientist who discovered the new element was awarded the Nobel Prize."),
        ("Complex", "If I had known about the meeting, I would have prepared the presentation in advance."),

        # Abstract concepts
        ("Abstract", "The concept of democracy is fundamental to modern society."),
        ("Abstract", "Economic growth depends on innovation and investment."),
        ("Abstract", "Philosophy explores the nature of existence and knowledge."),
    ]

    # Load models
    print("\n1. Loading fine-tuned model...")
    finetuned_evaluator = load_model(Path(__file__).parent.parent / "models" / "model_fine_tuned_basic.pt")

    print("2. Loading original model (if available)...")
    original_path = Path(__file__).parent.parent / "models" / "best_model.pt"
    # Check if original model exists (might be overwritten by fine-tuning)
    if original_path.exists():
        original_evaluator = load_model(original_path)
        compare_mode = True
    else:
        compare_mode = False
        print("Original model not found (may have been overwritten)")

    print("\n" + "=" * 70)
    print("Translation Results (Beam-4 decoding)")
    print("=" * 70)

    results = []
    for category, src_text in test_sentences:
        # Translate with fine-tuned model
        finetuned_translation = finetuned_evaluator.translate(src_text, method="beam", beam_size=4)

        if compare_mode:
            original_translation = original_evaluator.translate(src_text, method="beam", beam_size=4)
            same = (original_translation == finetuned_translation)
        else:
            original_translation = "N/A"
            same = False

        results.append({
            'category': category,
            'src': src_text,
            'finetuned': finetuned_translation,
            'original': original_translation,
            'same': same,
        })

        print(f"\n[{category}] English: {src_text}")
        print(f"Fine-tuned: {finetuned_translation}")
        if compare_mode:
            print(f"Original:   {original_translation}")
            if same:
                print("Status: Same output")
            else:
                print("Status: Different output")

    # Summary by category
    print("\n" + "=" * 70)
    print("Summary by Category")
    print("=" * 70)

    from collections import defaultdict
    category_stats = defaultdict(lambda: {'count': 0, 'same': 0, 'different': 0})

    for r in results:
        cat = r['category']
        category_stats[cat]['count'] += 1
        if r['same']:
            category_stats[cat]['same'] += 1
        else:
            category_stats[cat]['different'] += 1

    for cat in ['Basic', 'Simple', 'Moderate', 'Complex', 'Abstract']:
        if cat in category_stats:
            stats = category_stats[cat]
            same_pct = (stats['same'] / stats['count']) * 100 if stats['count'] > 0 else 0
            print(f"\n{cat} sentences ({stats['count']}):")
            print(f"  Same as original: {stats['same']} ({same_pct:.1f}%)")
            print(f"  Different: {stats['different']}")

    # Save results
    output_file = Path(__file__).parent.parent / "evaluation_results" / "finetune_generalization_test.json"
    output_file.parent.mkdir(exist_ok=True)

    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Overall assessment
    print("\n" + "=" * 70)
    print("Generalization Assessment")
    print("=" * 70)

    if compare_mode:
        total_same = sum(stats['same'] for stats in category_stats.values())
        total_count = sum(stats['count'] for stats in category_stats.values())
        overall_same_pct = (total_same / total_count) * 100

        print(f"Overall similarity to original model: {overall_same_pct:.1f}%")

        if overall_same_pct > 80:
            print("✅ Model shows minimal catastrophic forgetting")
        elif overall_same_pct > 50:
            print("⚠️  Model shows moderate changes (possible selective forgetting)")
        else:
            print("❌ Model shows significant changes (possible catastrophic forgetting)")
    else:
        print("⚠️  Could not compare with original model (may have been overwritten)")
        print("Check if best_model.pt exists for comparison")

    # Qualitative assessment
    print("\n" + "=" * 70)
    print("Qualitative Analysis")
    print("=" * 70)

    # Check basic vocabulary quality
    basic_results = [r for r in results if r['category'] == 'Basic']
    correct_keywords = {
        'Hello': ['hallo', 'guten'],
        'Thank you': ['danke'],
        'Good morning': ['guten morgen'],
    }

    print("\nBasic vocabulary accuracy check:")
    for r in basic_results:
        src = r['src']
        translation = r['finetuned'].lower()

        correct = False
        for key, keywords in correct_keywords.items():
            if key.lower() in src.lower():
                for kw in keywords:
                    if kw in translation:
                        correct = True
                        break
                break

        if correct:
            print(f"  ✅ {src} → {r['finetuned']}")
        else:
            print(f"  ❌ {src} → {r['finetuned']} (expected German translation)")

    # Check complex sentence coherence
    complex_results = [r for r in results if r['category'] in ['Complex', 'Abstract']]
    print("\nComplex/Abstract sentence coherence:")
    for r in complex_results[:3]:  # First 3
        translation = r['finetuned']
        # Simple heuristic: reasonable length and German words
        if len(translation.split()) >= 3 and any(c.isalpha() for c in translation):
            print(f"  ✅ {r['src'][:40]}... → {translation[:40]}...")
        else:
            print(f"  ⚠️  {r['src'][:40]}... → {translation[:40]}... (short/odd output)")

    print("\nTest completed!")


if __name__ == "__main__":
    test_generalization()