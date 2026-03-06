#!/usr/bin/env python3
"""
Check current training loss and state.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.data.batch import create_batch

def main():
    print("Checking Training Loss and Model State")
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

    # Get step from checkpoint
    step = checkpoint.get('step', 0)
    print(f"Current training step: {step:,}")

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

    # Test loss on a small batch
    print("\nTesting loss on sample data:")
    print("-" * 80)

    # Create a small test batch
    samples = [
        {'src': 'Hello world', 'tgt': 'Hallo Welt'},
        {'src': 'How are you', 'tgt': 'Wie geht es dir'},
        {'src': 'This is a test', 'tgt': 'Das ist ein Test'},
        {'src': 'Good morning', 'tgt': 'Guten Morgen'},
    ]

    batch = create_batch(samples, src_tokenizer, tgt_tokenizer, max_len=20, pad_id=0, device=device)

    # Compute loss
    src = batch['src'].to(device)
    tgt = batch['tgt'].to(device)
    src_mask = batch['src_mask'].to(device)

    # Create tgt_input and tgt_output (shifted)
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    # Create masks
    from src.data.batch import create_masks
    _, tgt_mask = create_masks(tgt_input, tgt_input, pad_id=0)

    # Forward pass
    with torch.no_grad():
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Compute cross-entropy loss
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(output, tgt_output)

        print(f"Cross-entropy loss on test batch: {loss.item():.4f}")

        # Check perplexity
        perplexity = torch.exp(loss).item()
        print(f"Perplexity: {perplexity:.2f}")

        # Analyze output distribution
        probs = torch.softmax(output, dim=-1)
        top_probs, top_ids = probs.topk(3, dim=-1)

        print(f"\nOutput statistics:")
        print(f"  Output mean: {output.mean().item():.6f}, std: {output.std().item():.6f}")
        print(f"  Probability mean: {probs.mean().item():.6e}")
        print(f"  Top-1 probability: {top_probs[:, 0].mean().item():.4f}")
        print(f"  Top-3 probability: {top_probs.mean().item():.4f}")

        # Check entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        uniform_entropy = torch.log(torch.tensor(tgt_vocab_size)).item()
        print(f"  Entropy: {entropy:.4f} (uniform: {uniform_entropy:.4f})")

    # Check model parameters statistics
    print("\n" + "=" * 80)
    print("Model parameter statistics:")

    param_stats = {
        'embedding': [],
        'attention': [],
        'ffn': [],
        'output': [],
        'norm': [],
    }

    for name, param in model.named_parameters():
        if param.requires_grad:
            mean_val = param.data.mean().item()
            std_val = param.data.std().item()

            if 'embedding' in name:
                param_stats['embedding'].append((name, mean_val, std_val))
            elif 'attention' in name:
                param_stats['attention'].append((name, mean_val, std_val))
            elif 'ff' in name or 'w_1' in name or 'w_2' in name:
                param_stats['ffn'].append((name, mean_val, std_val))
            elif 'output_proj' in name:
                param_stats['output'].append((name, mean_val, std_val))
            elif 'norm' in name:
                param_stats['norm'].append((name, mean_val, std_val))

    for category, params in param_stats.items():
        if params:
            means = [mean for _, mean, _ in params]
            stds = [std for _, _, std in params]
            avg_mean = sum(means) / len(means)
            avg_std = sum(stds) / len(stds)
            print(f"  {category}: {len(params)} parameters")
            print(f"    Average mean: {avg_mean:.6f}, std: {avg_std:.6f}")

    # Check if training should continue
    print("\n" + "=" * 80)
    print("Training Status Analysis:")

    # Based on typical Transformer training curves
    if step < 10000:
        print("  ⚠️  Early stage training (<10k steps)")
        print("  Recommendation: Continue training")
    elif step < 50000:
        print("  ⚠️  Mid-early stage training (10k-50k steps)")
        print("  Recommendation: Continue training, monitor loss")
    elif step < 150000:
        print("  ⚠️  Mid-stage training (50k-150k steps)")
        print("  Recommendation: Continue training, quality should improve")
    else:
        print("  ✅ Advanced training (>150k steps)")
        print("  Recommendation: Consider evaluation and fine-tuning")

    # Current max_steps
    print(f"\nCurrent max_steps in config: {config.max_steps:,}")
    print(f"Steps remaining: {config.max_steps - step:,}")
    print(f"Percentage complete: {step/config.max_steps*100:.1f}%")

    # Estimate additional epochs needed
    batch_size = config.batch_size
    max_train_samples = config.max_train_samples
    steps_per_epoch = max_train_samples // batch_size

    print(f"\nTraining statistics:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max train samples: {max_train_samples:,}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Current epochs: {step/steps_per_epoch:.2f}")
    print(f"  Additional epochs to max_steps: {(config.max_steps - step)/steps_per_epoch:.2f}")

    print("\n" + "=" * 80)
    print("Recommendation:")

    if loss.item() > 4.0:
        print("  Loss > 4.0 → Continue training until loss drops below 4.0")
    elif loss.item() > 2.0:
        print("  Loss 2.0-4.0 → Good progress, continue training")
    elif loss.item() > 1.0:
        print("  Loss 1.0-2.0 → Decent progress, consider extending training")
    else:
        print("  Loss < 1.0 → Good convergence, evaluate quality")

if __name__ == "__main__":
    main()