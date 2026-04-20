#!/usr/bin/env python3
"""
Compare weights between initialized and trained model.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.model import Transformer

def main():
    # Create fresh model
    fresh_model = Transformer(
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    )

    # Load trained model
    checkpoint_path = Path(__file__).parent / "models" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

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

    trained_model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    )
    trained_model.load_state_dict(checkpoint['model_state_dict'])

    print("Comparing weights between fresh and trained models:")
    print("=" * 80)

    total_params = 0
    changed_params = 0
    significant_changes = 0

    for (fresh_name, fresh_param), (trained_name, trained_param) in \
        zip(fresh_model.named_parameters(), trained_model.named_parameters()):

        assert fresh_name == trained_name, f"Parameter name mismatch: {fresh_name} != {trained_name}"

        diff = (fresh_param.data - trained_param.data).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        std_diff = diff.std().item()

        total_params += 1

        if max_diff > 1e-6:
            changed_params += 1

        if max_diff > 0.01:  # Significant change
            significant_changes += 1

            if significant_changes <= 5:  # Print first 5 significant changes
                print(f"\n{fresh_name}:")
                print(f"  Fresh: mean={fresh_param.data.mean().item():.6f}, std={fresh_param.data.std().item():.6f}")
                print(f"  Trained: mean={trained_param.data.mean().item():.6f}, std={trained_param.data.std().item():.6f}")
                print(f"  Diff: max={max_diff:.6f}, mean={mean_diff:.6f}, std={std_diff:.6f}")

        # Check if weights changed at all
        if total_params <= 10:  # Print first 10 params
            print(f"\n{fresh_name}: shape={fresh_param.shape}")
            print(f"  Fresh mean/std: {fresh_param.data.mean().item():.6f}/{fresh_param.data.std().item():.6f}")
            print(f"  Trained mean/std: {trained_param.data.mean().item():.6f}/{trained_param.data.std().item():.6f}")
            print(f"  Max diff: {max_diff:.6e}")

    print(f"\n\nSummary:")
    print(f"Total parameters: {total_params}")
    print(f"Parameters with any change (diff > 1e-6): {changed_params}")
    print(f"Parameters with significant change (diff > 0.01): {significant_changes}")

    # Check specific important layers
    print("\n\nSpecific layer comparison:")
    important_layers = [
        'encoder.embedding.weight',
        'decoder.embedding.weight',
        'decoder.output_proj.weight',
        'decoder.output_proj.bias',
    ]

    for layer_name in important_layers:
        fresh_param = dict(fresh_model.named_parameters())[layer_name]
        trained_param = dict(trained_model.named_parameters())[layer_name]

        diff = (fresh_param.data - trained_param.data).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n{layer_name}:")
        print(f"  Fresh: mean={fresh_param.data.mean().item():.6f}, std={fresh_param.data.std().item():.6f}")
        print(f"  Trained: mean={trained_param.data.mean().item():.6f}, std={trained_param.data.std().item():.6f}")
        print(f"  Diff: max={max_diff:.6e}, mean={mean_diff:.6e}")

        # Relative change
        fresh_std = fresh_param.data.std().item()
        if fresh_std > 1e-10:
            rel_change = mean_diff / fresh_std
            print(f"  Relative change (mean diff / fresh std): {rel_change:.6f}")

    # Check if trained model output is different
    print("\n\nForward pass comparison:")
    batch_size = 1
    src_len = 5
    tgt_len = 6

    src = torch.randint(0, 100, (batch_size, src_len))
    tgt = torch.randint(0, 100, (batch_size, tgt_len))

    src_mask = torch.ones(batch_size, 1, 1, src_len, dtype=torch.bool)
    tgt_mask = torch.tril(torch.ones(batch_size, 1, tgt_len, tgt_len, dtype=torch.bool))

    fresh_model.eval()
    trained_model.eval()

    with torch.no_grad():
        fresh_output = fresh_model(src, tgt, src_mask, tgt_mask)
        trained_output = trained_model(src, tgt, src_mask, tgt_mask)

    output_diff = (fresh_output - trained_output).abs()
    print(f"Fresh output: mean={fresh_output.mean().item():.6f}, std={fresh_output.std().item():.6f}")
    print(f"Trained output: mean={trained_output.mean().item():.6f}, std={trained_output.std().item():.6f}")
    print(f"Output diff: max={output_diff.max().item():.6e}, mean={output_diff.mean().item():.6e}")

    # Check output distribution
    print("\nOutput distribution comparison:")
    fresh_probs = torch.softmax(fresh_output.view(-1, 32000), dim=-1)
    trained_probs = torch.softmax(trained_output.view(-1, 32000), dim=-1)

    print(f"Fresh probs: mean={fresh_probs.mean().item():.6e}, entropy={-(fresh_probs * torch.log(fresh_probs + 1e-10)).sum(dim=-1).mean().item():.6f}")
    print(f"Trained probs: mean={trained_probs.mean().item():.6e}, entropy={-(trained_probs * torch.log(trained_probs + 1e-10)).sum(dim=-1).mean().item():.6f}")

if __name__ == "__main__":
    main()