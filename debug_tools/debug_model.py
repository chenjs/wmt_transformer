"""
Debug model weights and activations.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer

def debug_model():
    # Load tokenizers
    data_dir = Path(__file__).parent
    src_tokenizer_path = data_dir / "models" / "src_tokenizer.model"
    tgt_tokenizer_path = data_dir / "models" / "tgt_tokenizer.model"
    checkpoint_path = data_dir / "models" / "best_model.pt"

    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )
    # FIX 2026-02-26: Compute separate vocabulary sizes
    src_vocab_size = src_tokenizer.sp.get_piece_size()
    tgt_vocab_size = tgt_tokenizer.sp.get_piece_size()
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")
    # Set default vocab sizes in config (will be overridden by checkpoint config if present)
    config.src_vocab_size = src_vocab_size
    config.tgt_vocab_size = tgt_vocab_size

    # Load checkpoint
    device = "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Use config from checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        config.max_len = saved_config.max_len
        config.d_model = saved_config.d_model
        config.n_layers = saved_config.n_layers
        config.n_heads = saved_config.n_heads
        config.d_ff = saved_config.d_ff
        config.dropout = saved_config.dropout
        # FIX 2026-02-26: Handle vocabulary sizes (backward compatibility)
        # Check if checkpoint has separate vocab sizes or uses old single vocab_size
        if hasattr(saved_config, 'src_vocab_size') and hasattr(saved_config, 'tgt_vocab_size'):
            config.src_vocab_size = saved_config.src_vocab_size
            config.tgt_vocab_size = saved_config.tgt_vocab_size
            print(f"Loaded config with separate vocab sizes: src={config.src_vocab_size}, tgt={config.tgt_vocab_size}")
        else:
            # Old checkpoint: use single vocab_size for both
            config.src_vocab_size = saved_config.vocab_size
            config.tgt_vocab_size = saved_config.vocab_size
            print(f"Loaded old config with single vocab size: {saved_config.vocab_size}")

    # Create model with separate vocabulary sizes
    # FIX 2026-02-26: Use src_vocab_size and tgt_vocab_size
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

    print("=" * 60)
    print("Model Architecture Debug")
    print("=" * 60)

    # Check encoder embedding
    encoder_embed = model.encoder.embedding
    print(f"\n1. Encoder embedding weight shape: {encoder_embed.weight.shape}")
    print(f"   Weight stats: mean={encoder_embed.weight.mean().item():.6f}, std={encoder_embed.weight.std().item():.6f}")
    print(f"   Min={encoder_embed.weight.min().item():.6f}, Max={encoder_embed.weight.max().item():.6f}")

    # Check positional encoding
    pos_enc = model.encoder.pos_encoding
    print(f"\n2. Positional encoding buffer shape: {pos_enc.pe.shape}")
    print(f"   PE stats: mean={pos_enc.pe.mean().item():.6f}, std={pos_enc.pe.std().item():.6f}")

    # Check first encoder layer
    if len(model.encoder.layers) > 0:
        layer = model.encoder.layers[0]
        print(f"\n3. First encoder layer:")
        print(f"   Self-attention weights:")
        attn = layer.self_attn.attention
        print(f"     w_q weight: {attn.w_q.weight.mean().item():.6f} ± {attn.w_q.weight.std().item():.6f}")
        print(f"     w_k weight: {attn.w_k.weight.mean().item():.6f} ± {attn.w_k.weight.std().item():.6f}")
        print(f"     w_v weight: {attn.w_v.weight.mean().item():.6f} ± {attn.w_v.weight.std().item():.6f}")
        print(f"     w_o weight: {attn.w_o.weight.mean().item():.6f} ± {attn.w_o.weight.std().item():.6f}")

    # Test forward pass with different inputs
    print(f"\n4. Testing forward pass with different inputs:")
    test_inputs = [
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [100, 200, 300, 400],
    ]

    for i, tokens in enumerate(test_inputs):
        src = torch.tensor([tokens], dtype=torch.long, device=device)
        src_mask = torch.ones(1, 1, 1, len(tokens), dtype=torch.bool, device=device)

        with torch.no_grad():
            # Get embedding
            emb = model.encoder.embedding(src) * torch.sqrt(torch.tensor(model.encoder.d_model, dtype=torch.float32))
            print(f"\n   Input {i+1}: tokens={tokens}")
            print(f"     Embedding shape: {emb.shape}")
            print(f"     Embedding mean: {emb.mean().item():.6f}, std: {emb.std().item():.6f}")

            # Add positional encoding
            emb_with_pos = model.encoder.pos_encoding(emb)
            print(f"     After positional encoding mean: {emb_with_pos.mean().item():.6f}")

            # Full encoder forward
            encoder_output = model.encoder(src, src_mask)
            print(f"     Encoder output mean: {encoder_output.mean().item():.6f}, std: {encoder_output.std().item():.6f}")

            # Check if outputs are identical
            if i > 0:
                prev_output = model.encoder(torch.tensor([test_inputs[i-1]], dtype=torch.long), src_mask)
                diff = (encoder_output - prev_output).abs().max().item()
                print(f"     Max diff with previous: {diff:.6f}")

    # Check attention masks
    print(f"\n5. Testing attention mask generation:")
    from src.data.batch import create_masks
    src = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)  # with padding
    tgt = torch.tensor([[10, 20, 30, 40, 0]], dtype=torch.long)
    src_mask, tgt_mask = create_masks(src, tgt, pad_id=0)
    print(f"   src shape: {src.shape}, tgt shape: {tgt.shape}")
    print(f"   src_mask shape: {src_mask.shape}")
    print(f"   tgt_mask shape: {tgt_mask.shape}")
    print(f"   src_mask:\n{src_mask}")
    print(f"   tgt_mask[0,0]:\n{tgt_mask[0,0]}")

    # Check if model produces same output for different inputs via full forward
    print(f"\n6. Full model forward test:")
    src1 = torch.tensor([[10, 20, 30, 2]], dtype=torch.long, device=device)  # last token is EOS?
    src2 = torch.tensor([[50, 60, 70, 2]], dtype=torch.long, device=device)
    tgt = torch.tensor([[1, 100, 200, 2, 0]], dtype=torch.long, device=device)  # BOS, tokens, EOS, pad

    src_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool, device=device)
    _, tgt_mask = create_masks(tgt, tgt, pad_id=0)

    with torch.no_grad():
        out1 = model(src1, tgt, src_mask, tgt_mask)
        out2 = model(src2, tgt, src_mask, tgt_mask)

        print(f"   Output shapes: {out1.shape}")
        print(f"   Output1 mean: {out1.mean().item():.6f}, Output2 mean: {out2.mean().item():.6f}")
        diff = (out1 - out2).abs().max().item()
        print(f"   Max difference between outputs: {diff:.6f}")

        # Check output distribution
        probs1 = torch.softmax(out1[:, -1, :], dim=-1)
        probs2 = torch.softmax(out2[:, -1, :], dim=-1)
        topk1 = probs1[0].topk(5)
        topk2 = probs2[0].topk(5)
        print(f"   Top-5 tokens for input1: {topk1.indices.tolist()} with probs {topk1.values.tolist()}")
        print(f"   Top-5 tokens for input2: {topk2.indices.tolist()} with probs {topk2.values.tolist()}")

if __name__ == "__main__":
    debug_model()