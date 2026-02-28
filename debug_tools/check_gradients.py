#!/usr/bin/env python3
"""
Check gradients in the model.
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
    # Load tokenizers
    data_dir = Path(__file__).parent.parent
    src_tokenizer_path = data_dir / "models" / "src_tokenizer.model"
    tgt_tokenizer_path = data_dir / "models" / "tgt_tokenizer.model"
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
    model.train()  # Set to training mode to enable gradients

    # Create a small batch
    samples = [
        {'src': 'Hello world', 'tgt': 'Hallo Welt'},
        {'src': 'How are you', 'tgt': 'Wie geht es dir'},
    ]

    batch = create_batch(samples, src_tokenizer, tgt_tokenizer, max_len=20, pad_id=0, device=device)

    # Forward pass
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
    output = model(src, tgt_input, src_mask, tgt_mask)

    # Compute loss
    output = output.reshape(-1, output.size(-1))
    tgt_output = tgt_output.reshape(-1)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(output, tgt_output)

    # Backward pass
    model.zero_grad()
    loss.backward()

    print(f"Loss: {loss.item():.6f}")

    # Check gradients
    print("\nGradient statistics:")
    total_params = 0
    zero_gradients = 0
    small_gradients = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_abs_mean = param.grad.abs().mean().item()

            total_params += 1

            if grad_norm < 1e-10:
                zero_gradients += 1
            elif grad_abs_mean < 1e-6:
                small_gradients += 1

            if total_params <= 5:  # Print first 5
                print(f"{name}:")
                print(f"  shape: {param.shape}")
                print(f"  grad norm: {grad_norm:.6e}")
                print(f"  grad mean: {grad_mean:.6e}, std: {grad_std:.6e}")
                print(f"  grad abs mean: {grad_abs_mean:.6e}")

    print(f"\nTotal parameters with gradients: {total_params}")
    print(f"Zero gradients (norm < 1e-10): {zero_gradients}")
    print(f"Small gradients (abs mean < 1e-6): {small_gradients}")

    # Check specific layers
    print("\n\nChecking specific layer gradients:")

    # Embedding layers
    print("\nEmbedding layers:")
    for name in ['encoder.embedding.weight', 'decoder.embedding.weight']:
        param = dict(model.named_parameters())[name]
        if param.grad is not None:
            grad_abs_mean = param.grad.abs().mean().item()
            print(f"  {name}: grad abs mean = {grad_abs_mean:.6e}")

    # Output projection
    print("\nOutput projection:")
    param = model.decoder.output_proj.weight
    if param.grad is not None:
        grad_abs_mean = param.grad.abs().mean().item()
        print(f"  decoder.output_proj.weight: grad abs mean = {grad_abs_mean:.6e}")

    param = model.decoder.output_proj.bias
    if param.grad is not None:
        grad_abs_mean = param.grad.abs().mean().item()
        print(f"  decoder.output_proj.bias: grad abs mean = {grad_abs_mean:.6e}")

    # Check loss value
    print(f"\nLoss value: {loss.item():.6f}")
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {tgt_output.shape}")

    # Check output statistics
    print(f"\nOutput statistics:")
    print(f"  Output mean: {output.mean().item():.6f}, std: {output.std().item():.6f}")
    print(f"  Output min: {output.min().item():.6f}, max: {output.max().item():.6f}")

    # Check softmax probabilities
    probs = torch.softmax(output, dim=-1)
    print(f"\nProbability statistics:")
    print(f"  Prob mean: {probs.mean().item():.6f}, std: {probs.std().item():.6f}")
    print(f"  Entropy: {-(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item():.6f}")

    # Expected entropy for uniform distribution over vocab size 32000
    uniform_entropy = torch.log(torch.tensor(32000.0)).item()
    print(f"  Uniform distribution entropy: {uniform_entropy:.6f}")

if __name__ == "__main__":
    main()