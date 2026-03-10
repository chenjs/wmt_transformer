#!/usr/bin/env python3
"""
Test the improved Transformer architecture with pre-norm and better initialization.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.model import Transformer

def test_architecture():
    print("Testing improved Transformer architecture...")
    print("=" * 80)

    # Create a model with improved architecture
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=config.d_model,
        n_layers=2,  # Small for testing
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    )

    print(f"Model created with pre-norm architecture")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 2
    src_len = 10
    tgt_len = 12

    src = torch.randint(0, 100, (batch_size, src_len))
    tgt = torch.randint(0, 100, (batch_size, tgt_len))

    src_mask = torch.ones(batch_size, 1, 1, src_len, dtype=torch.bool)
    tgt_mask = torch.tril(torch.ones(batch_size, 1, tgt_len, tgt_len, dtype=torch.bool))

    print("\n1. Forward pass test:")
    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)
        print(f"   Output shape: {output.shape}")
        print(f"   Output mean: {output.mean().item():.6f}, std: {output.std().item():.6f}")

    # Test encoder and decoder separately
    print("\n2. Encoder test:")
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        print(f"   Encoder output shape: {encoder_output.shape}")
        print(f"   Encoder output mean: {encoder_output.mean().item():.6f}, std: {encoder_output.std().item():.6f}")

    print("\n3. Decoder test:")
    with torch.no_grad():
        decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        print(f"   Decoder output shape: {decoder_output.shape}")
        print(f"   Decoder output mean: {decoder_output.mean().item():.6f}, std: {decoder_output.std().item():.6f}")

    # Check weight initialization statistics
    print("\n4. Weight initialization statistics:")

    layer_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Classify layer type
            if 'embedding' in name:
                layer_type = 'embedding'
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

    for layer_type, params in layer_stats.items():
        if params:
            means = [p.data.mean().item() for n, p in params]
            stds = [p.data.std().item() for n, p in params]
            avg_mean = sum(means) / len(means)
            avg_std = sum(stds) / len(stds)
            print(f"   {layer_type}: {len(params)} parameters")
            print(f"     Average mean: {avg_mean:.6f}, Average std: {avg_std:.6f}")

            # Check if initialization is appropriate
            if layer_type == 'embedding':
                # Embedding should have mean~0, std~0.02
                if abs(avg_mean) > 0.1 or abs(avg_std - 0.02) > 0.01:
                    print(f"     ⚠️  Unexpected: should be mean~0, std~0.02")
            elif layer_type == 'norm.weight':
                # LayerNorm weight should be ~1.0
                if abs(avg_mean - 1.0) > 0.1:
                    print(f"     ⚠️  Unexpected: should be ~1.0")
            elif layer_type == 'norm.bias':
                # LayerNorm bias should be ~0.0
                if abs(avg_mean) > 0.1:
                    print(f"     ⚠️  Unexpected: should be ~0.0")
            elif 'attention.weight' in layer_type or 'ff.weight' in layer_type:
                # Linear weights: std should be reasonable
                if avg_std < 0.01 or avg_std > 1.0:
                    print(f"     ⚠️  Unexpected std: {avg_std:.6f}")

    # Test gradient flow
    print("\n5. Gradient flow test:")
    model.train()
    criterion = nn.CrossEntropyLoss()

    output = model(src, tgt, src_mask, tgt_mask)
    loss = criterion(output.view(-1, 1000), tgt.view(-1))

    model.zero_grad()
    loss.backward()

    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

    if grad_norms:
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        max_grad_norm = max(grad_norms)
        min_grad_norm = min(grad_norms)
        print(f"   Average gradient norm: {avg_grad_norm:.6e}")
        print(f"   Max gradient norm: {max_grad_norm:.6e}")
        print(f"   Min gradient norm: {min_grad_norm:.6e}")

        # Check for vanishing/exploding gradients
        if avg_grad_norm < 1e-10:
            print(f"   ⚠️  Warning: Very small gradients (potential vanishing)")
        if max_grad_norm > 1000:
            print(f"   ⚠️  Warning: Very large gradients (potential exploding)")
    else:
        print(f"   No gradients computed")

    # Test with different inputs (check encoder output diversity)
    print("\n6. Encoder output diversity test:")
    test_sentences = [
        torch.randint(0, 100, (1, 5)),
        torch.randint(0, 100, (1, 7)),
        torch.randint(0, 100, (1, 6)),
    ]

    encoder_outputs = []
    with torch.no_grad():
        for i, src_input in enumerate(test_sentences):
            src_mask_i = torch.ones(1, 1, 1, src_input.size(1), dtype=torch.bool)
            encoder_output = model.encode(src_input, src_mask_i)
            encoder_outputs.append(encoder_output)

            std = encoder_output.std().item()
            print(f"   Input {i+1} (len={src_input.size(1)}): output std={std:.6f}")

        # Compare differences
        if len(encoder_outputs) >= 2:
            diff1 = torch.abs(encoder_outputs[0][0, 0, :] - encoder_outputs[1][0, 0, :]).mean().item()
            diff2 = torch.abs(encoder_outputs[0][0, 0, :] - encoder_outputs[2][0, 0, :]).mean().item()
            print(f"   Average difference between inputs:")
            print(f"     Input 1 vs 2: {diff1:.6f}")
            print(f"     Input 1 vs 3: {diff2:.6f}")

            if diff1 < 0.01 and diff2 < 0.01:
                print(f"   ⚠️  Warning: Encoder outputs too similar")
            else:
                print(f"   ✓ Encoder produces diverse outputs")

    print("\n" + "=" * 80)
    print("Architecture test completed.")

    # Save a small checkpoint for further testing
    test_checkpoint_path = Path(__file__).parent / "test_improved_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, test_checkpoint_path)
    print(f"Test model saved to {test_checkpoint_path}")

if __name__ == "__main__":
    test_architecture()