#!/usr/bin/env python3
"""
Check model initialization.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.model import Transformer

def main():
    # Create a fresh model
    model = Transformer(
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    )

    print("Model initialization statistics:")

    # Check different types of layers
    layer_stats = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Classify layer type
            if 'embedding' in name:
                layer_type = 'embedding'
            elif 'pos_encoding' in name:
                layer_type = 'pos_encoding'
            elif 'norm.weight' in name:
                layer_type = 'norm.weight'
            elif 'norm.bias' in name:
                layer_type = 'norm.bias'
            elif '.weight' in name and ('w_q' in name or 'w_k' in name or 'w_v' in name or 'w_o' in name):
                layer_type = 'attention.weight'
            elif '.bias' in name and ('w_q' in name or 'w_k' in name or 'w_v' in name or 'w_o' in name):
                layer_type = 'attention.bias'
            elif 'w_1.weight' in name or 'w_2.weight' in name:
                layer_type = 'ff.weight'
            elif 'w_1.bias' in name or 'w_2.bias' in name:
                layer_type = 'ff.bias'
            elif 'output_proj.weight' in name:
                layer_type = 'output_proj.weight'
            elif 'output_proj.bias' in name:
                layer_type = 'output_proj.bias'
            else:
                layer_type = 'other'

            if layer_type not in layer_stats:
                layer_stats[layer_type] = []

            layer_stats[layer_type].append((name, param))

    # Print statistics for each layer type
    for layer_type, params in layer_stats.items():
        print(f"\n{layer_type}: {len(params)} parameters")

        # Calculate mean and std across all parameters of this type
        all_means = []
        all_stds = []

        for name, param in params:
            all_means.append(param.data.mean().item())
            all_stds.append(param.data.std().item())

        if params:
            mean_of_means = sum(all_means) / len(all_means)
            mean_of_stds = sum(all_stds) / len(all_stds)

            print(f"  Average mean: {mean_of_means:.6f}")
            print(f"  Average std: {mean_of_stds:.6f}")

            # Show first example
            name, param = params[0]
            print(f"  Example: {name}")
            print(f"    shape: {param.shape}")
            print(f"    mean: {param.data.mean().item():.6f}, std: {param.data.std().item():.6f}")

    # Check forward pass with small inputs
    print("\n\nForward pass test:")
    batch_size = 2
    src_len = 5
    tgt_len = 6

    src = torch.randint(0, 100, (batch_size, src_len))
    tgt = torch.randint(0, 100, (batch_size, tgt_len))

    src_mask = torch.ones(batch_size, 1, 1, src_len, dtype=torch.bool)
    tgt_mask = torch.tril(torch.ones(batch_size, 1, tgt_len, tgt_len, dtype=torch.bool))

    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)

    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}, std: {output.std().item():.6f}")

    # Check gradients with a simple forward/backward
    print("\n\nGradient flow test:")
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Simple forward
    output = model(src, tgt, src_mask, tgt_mask)
    loss = criterion(output.view(-1, 32000), tgt.view(-1))

    # Backward
    model.zero_grad()
    loss.backward()

    # Check gradient norms
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

    if grad_norms:
        print(f"Average gradient norm: {sum(grad_norms) / len(grad_norms):.6e}")
        print(f"Max gradient norm: {max(grad_norms):.6e}")
        print(f"Min gradient norm: {min(grad_norms):.6e}")
    else:
        print("No gradients")

    # Check activation statistics through layers (optional)
    print("\n\nChecking encoder activation statistics:")

    # Hook to capture activations
    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.detach())

    # Register hooks on encoder layers
    hooks = []
    for i, layer in enumerate(model.encoder.layers):
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Forward through encoder
    with torch.no_grad():
        encoder_output = model.encoder(src, src_mask)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print activation statistics
    for i, act in enumerate(activations):
        print(f"Encoder layer {i}: mean={act.mean().item():.6f}, std={act.std().item():.6f}")

    print(f"Final encoder output: mean={encoder_output.mean().item():.6f}, std={encoder_output.std().item():.6f}")

if __name__ == "__main__":
    main()