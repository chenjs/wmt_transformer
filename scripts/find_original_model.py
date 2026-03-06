#!/usr/bin/env python3
"""
Find the original 150,000-step model among backup files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

def check_model_step(model_path):
    """Check step information in model checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        step = checkpoint.get('step', 'N/A')

        # Also check config for model info
        config_info = {}
        if 'config' in checkpoint:
            config = checkpoint['config']
            config_info['d_model'] = getattr(config, 'd_model', 'N/A')
            config_info['n_layers'] = getattr(config, 'n_layers', 'N/A')

        return step, config_info
    except Exception as e:
        return f"Error: {e}", {}

def main():
    print("=" * 70)
    print("Searching for Original 150,000-step Model")
    print("=" * 70)

    # Find all .pt files
    model_files = []
    search_dirs = [
        Path(__file__).parent.parent / "models",
        Path(__file__).parent.parent / "models_backup",
        Path(__file__).parent.parent / "models_original",
        Path(__file__).parent.parent / "models_enhanced",
        Path(__file__).parent.parent / "models_test",
    ]

    for search_dir in search_dirs:
        if search_dir.exists():
            for pt_file in search_dir.glob("*.pt"):
                model_files.append(pt_file)

    print(f"Found {len(model_files)} model files")

    # Check each file
    results = []
    for model_path in model_files:
        step, config_info = check_model_step(model_path)

        # Get file size and modification time
        stat = model_path.stat()
        size_mb = stat.st_size / (1024 * 1024)

        results.append({
            'path': model_path,
            'step': step,
            'size_mb': size_mb,
            'mtime': stat.st_mtime,
            'config': config_info
        })

    # Sort by step (try to convert to int for sorting)
    def step_key(x):
        step = x['step']
        if isinstance(step, int):
            return step
        elif isinstance(step, str) and step.isdigit():
            return int(step)
        else:
            return float('inf')  # Put errors/unknown at end

    results.sort(key=step_key, reverse=True)  # Highest step first

    # Print results
    print("\n" + "=" * 70)
    print("Model Checkpoint Analysis")
    print("=" * 70)

    for r in results:
        path = r['path']
        step = r['step']
        size = r['size_mb']

        # Relative path for readability
        rel_path = path.relative_to(Path(__file__).parent.parent)

        print(f"\n{rel_path}:")
        print(f"  Step: {step}")
        print(f"  Size: {size:.1f} MB")

        # Interpret step value
        if isinstance(step, int):
            if step >= 150000 and step <= 200000:
                print(f"  ✅ Likely original model (150k-200k steps)")
            elif step >= 50000 and step < 150000:
                print(f"  ⚠️  Intermediate model ({step} steps)")
            elif step < 50000:
                print(f"  ⚠️  Early checkpoint or fine-tuned ({step} steps)")
            elif step > 200000:
                print(f"  ⚠️  Beyond 200k steps ({step} steps)")

        # Check if this might be the fine-tuned model
        if step == 5000 or step == 3564 or step == 3000 or step == 360:
            print(f"  ⚠️  Possibly fine-tuned model (step {step})")

    # Identify candidates for original 150k model
    print("\n" + "=" * 70)
    print("Original Model Candidates")
    print("=" * 70)

    candidates = []
    for r in results:
        step = r['step']
        if isinstance(step, int) and step >= 150000 and step <= 200000:
            candidates.append(r)

    if candidates:
        print(f"Found {len(candidates)} candidate(s) in 150k-200k step range:")
        for c in candidates:
            rel_path = c['path'].relative_to(Path(__file__).parent.parent)
            print(f"  ✅ {rel_path} (step {c['step']})")
    else:
        print("No candidates found in 150k-200k step range.")
        print("Checking for closest matches...")

        # Find closest to 150k
        int_steps = []
        for r in results:
            step = r['step']
            if isinstance(step, int):
                int_steps.append((abs(step - 150000), r))

        if int_steps:
            int_steps.sort(key=lambda x: x[0])  # Sort by distance to 150k
            for distance, r in int_steps[:3]:  # Top 3 closest
                rel_path = r['path'].relative_to(Path(__file__).parent.parent)
                print(f"  ⚠️  {rel_path} (step {r['step']}, {distance} steps from 150k)")

    # Quick test of top candidates on a complex sentence
    print("\n" + "=" * 70)
    print("Quick Translation Test on Complex Sentence")
    print("=" * 70)

    test_sentence = "Despite the heavy rain, the football match continued as scheduled."

    # Test top 3 candidates (highest step)
    test_candidates = []
    for r in results:
        step = r['step']
        if isinstance(step, int) and step >= 100000:
            test_candidates.append(r)

    # Limit to top 3
    test_candidates = sorted(test_candidates, key=lambda x: x['step'], reverse=True)[:3]

    if not test_candidates:
        # Fallback to any model
        test_candidates = results[:3]

    for r in test_candidates:
        path = r['path']
        step = r['step']
        rel_path = path.relative_to(Path(__file__).parent.parent)

        print(f"\nTesting: {rel_path} (step {step})")

        try:
            # Quick load and test
            import torch
            from src.config import config
            from src.data.tokenizer import load_tokenizers
            from src.model import Transformer
            from src.evaluate import Evaluator

            device = "cpu"  # Quick test on CPU
            checkpoint = torch.load(path, map_location=device, weights_only=False)

            # Minimal load just for translation
            src_tokenizer_path = Path(__file__).parent.parent / "models_enhanced" / "src_tokenizer_final.model"
            tgt_tokenizer_path = Path(__file__).parent.parent / "models_enhanced" / "tgt_tokenizer_final.model"

            src_tokenizer, tgt_tokenizer = load_tokenizers(
                str(src_tokenizer_path), str(tgt_tokenizer_path)
            )

            if 'config' in checkpoint:
                saved_config = checkpoint['config']
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
            model.eval()

            evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device)
            translation = evaluator.translate(test_sentence, method="beam", beam_size=4)

            print(f"Translation: {translation}")

            # Quality heuristic
            words = translation.split()
            if len(words) >= 5 and any(word.isalpha() for word in words):
                print("✅ Reasonable output length and content")
            else:
                print("⚠️  Short or odd output (possible fine-tuned model)")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("Recommendation")
    print("=" * 70)

    # Find best candidate
    best_candidate = None
    for r in results:
        step = r['step']
        if isinstance(step, int) and step >= 150000 and step <= 200000:
            best_candidate = r
            break

    if best_candidate:
        rel_path = best_candidate['path'].relative_to(Path(__file__).parent.parent)
        print(f"Recommended original model: {rel_path}")
        print(f"Steps: {best_candidate['step']}")
        print(f"\nTo restore as best_model.pt:")
        print(f"  cp '{best_candidate['path']}' models/best_model.pt")
    else:
        print("No clear 150k-step model found.")
        print("Consider using the highest-step model that produces reasonable complex translations.")

if __name__ == "__main__":
    main()