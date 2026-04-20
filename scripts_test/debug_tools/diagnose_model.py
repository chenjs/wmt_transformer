#!/usr/bin/env python3
"""
Diagnose model output issues.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer

def main():
    # Load tokenizers - use enhanced tokenizers (consistent with training)
    data_dir = Path(__file__).parent
    src_tokenizer_path = data_dir / "models_enhanced" / "src_tokenizer_final.model"
    tgt_tokenizer_path = data_dir / "models_enhanced" / "tgt_tokenizer_final.model"
    checkpoint_path = data_dir / "models" / "best_model.pt"

    if not checkpoint_path.exists():
        print("No checkpoint found")
        return

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

    print("Model loaded successfully")

    # Test with a simple input
    test_sentence = "Hello"
    src_tokens = src_tokenizer(test_sentence, add_bos=False, add_eos=True)
    src = torch.tensor([src_tokens], dtype=torch.long, device=device)
    src_mask = torch.ones(1, 1, 1, len(src_tokens), dtype=torch.bool, device=device)

    # Encode
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        print(f"Encoder output shape: {encoder_output.shape}")

        # Start with BOS
        tgt = torch.tensor([[tgt_tokenizer.bos_id]], device=device)

        # Decode step by step
        print("\nStep-by-step decoding:")
        for step in range(10):
            tgt_mask = torch.tril(torch.ones(1, 1, tgt.size(1), tgt.size(1), dtype=torch.bool, device=device))
            output = model.decode(tgt, encoder_output, src_mask, tgt_mask)

            # Get logits for last position
            logits = output[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            # Get top 5 predictions
            top_probs, top_ids = probs.topk(5)

            print(f"Step {step} (current sequence: {tgt[0].tolist()})")
            for i, (prob, token_id) in enumerate(zip(top_probs, top_ids)):
                token_text = tgt_tokenizer.decode([token_id.item()])
                print(f"  {i+1}. token_id={token_id.item()}, prob={prob.item():.4f}, text='{token_text}'")

            # Choose next token greedily
            next_token = probs.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tgt_tokenizer.eos_id:
                print("EOS reached")
                break

        # Check what token ID corresponds to "das" or "Das"
        print("\nChecking token 'das' and 'Das':")
        test_tokens = ["das", "Das", "the", "The", ".", ","]
        for token in test_tokens:
            ids = tgt_tokenizer.encode(token)
            print(f"  '{token}' -> ids: {ids}")

            if ids:
                # Check probability of this token at first step
                logits = output[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                prob = probs[ids[0]].item()
                print(f"    Probability at step 0: {prob:.6f}")

    # Check decoder output projection weights
    print("\nChecking decoder output projection layer:")
    output_proj = model.decoder.output_proj
    print(f"  Weight shape: {output_proj.weight.shape}")
    print(f"  Bias shape: {output_proj.bias.shape}")

    # Check if weights are zero or have small variance
    weight_mean = output_proj.weight.mean().item()
    weight_std = output_proj.weight.std().item()
    bias_mean = output_proj.bias.mean().item()
    bias_std = output_proj.bias.std().item()

    print(f"  Weight mean: {weight_mean:.6f}, std: {weight_std:.6f}")
    print(f"  Bias mean: {bias_mean:.6f}, std: {bias_std:.6f}")

    # Check embedding layers
    print("\nChecking embedding layers:")
    print(f"  Source embedding shape: {model.encoder.embedding.weight.shape}")
    print(f"  Target embedding shape: {model.decoder.embedding.weight.shape}")

    # Check if embeddings have gradients (were updated)
    print(f"  Source embedding requires_grad: {model.encoder.embedding.weight.requires_grad}")
    print(f"  Target embedding requires_grad: {model.decoder.embedding.weight.requires_grad}")

if __name__ == "__main__":
    main()