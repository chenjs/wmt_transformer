#!/usr/bin/env python3
"""
Quick verification of restored original model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import Evaluator

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_model(checkpoint_path):
    device = get_device()
    data_dir = Path(__file__).parent.parent

    src_tokenizer_path = data_dir / "models_enhanced" / "src_tokenizer_final.model"
    tgt_tokenizer_path = data_dir / "models_enhanced" / "tgt_tokenizer_final.model"

    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device)
    return evaluator

def main():
    print("=" * 60)
    print("Original Model Verification")
    print("=" * 60)

    # Test different model files
    model_files = [
        ("original_150k_model.pt", "Restored original model"),
        ("best_model.pt", "Current best model"),
        ("model_fine_tuned_basic.pt", "Fine-tuned model"),
        ("checkpoint_step_3000.pt", "Checkpoint at step 3000"),
    ]

    test_sentences = [
        ("Hello", "Basic - Should be incorrect in original"),
        ("Thank you", "Basic - Should be incorrect in original"),
        ("The sky is blue", "Simple - Should work in original"),
        ("Despite the heavy rain, the football match continued as scheduled.", "Complex - Should work in original"),
    ]

    results = {}

    for model_file, description in model_files:
        model_path = Path(__file__).parent.parent / "models" / model_file
        if not model_path.exists():
            print(f"\n⚠️  {model_file} not found, skipping")
            continue

        print(f"\n{'='*40}")
        print(f"Testing: {description} ({model_file})")
        print(f"{'='*40}")

        try:
            evaluator = load_model(model_path)
            model_results = []

            for src_text, note in test_sentences:
                translation = evaluator.translate(src_text, method="beam", beam_size=4)
                model_results.append((src_text, translation, note))
                print(f"{src_text[:30]}... -> {translation[:40]}... ({note})")

            results[model_file] = model_results

        except Exception as e:
            print(f"Error loading {model_file}: {e}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)

    # Check which model is likely the original 150k model
    # Original model should have poor basic vocabulary but good complex sentences
    original_indicators = {}

    for model_file, model_results in results.items():
        if model_file not in original_indicators:
            original_indicators[model_file] = {"basic_correct": 0, "complex_works": 0}

        for src_text, translation, note in model_results:
            src_lower = src_text.lower()
            trans_lower = translation.lower()

            # Check basic vocabulary
            if "hello" in src_lower:
                if "hallo" in trans_lower or "guten" in trans_lower:
                    original_indicators[model_file]["basic_correct"] += 1
            elif "thank you" in src_lower:
                if "danke" in trans_lower:
                    original_indicators[model_file]["basic_correct"] += 1

            # Check complex sentence (should have reasonable length)
            if "despite" in src_lower.lower():
                if len(translation.split()) >= 5 and any(c.isalpha() for c in translation):
                    original_indicators[model_file]["complex_works"] += 1

    print("\nModel Type Identification:")
    for model_file, indicators in original_indicators.items():
        basic_score = indicators["basic_correct"]
        complex_score = indicators["complex_works"]

        if basic_score >= 1 and complex_score >= 1:
            model_type = "⚠️  Mixed (possibly fine-tuned intermediate)"
        elif basic_score >= 1 and complex_score == 0:
            model_type = "❌ Overfitted fine-tuned model"
        elif basic_score == 0 and complex_score >= 1:
            model_type = "✅ Likely original 150k model"
        else:
            model_type = "❓ Unknown"

        print(f"{model_file}: Basic correct={basic_score}, Complex works={complex_score} -> {model_type}")

    # Check checkpoint step information
    print("\n" + "=" * 60)
    print("Checkpoint Step Information")
    print("=" * 60)

    for model_file, description in model_files:
        model_path = Path(__file__).parent.parent / "models" / model_file
        if not model_path.exists():
            continue

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            step = checkpoint.get('step', 'N/A')
            print(f"{model_file}: step={step}")
        except:
            print(f"{model_file}: Could not read step info")

if __name__ == "__main__":
    main()