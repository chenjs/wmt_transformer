"""
Transformer model for machine translation.
"""
import torch
import torch.nn as nn
import torch.nn.init as init

from .positional import PositionalEncoding
from .attention import SelfAttention, CrossAttention
from .feedforward import FeedForwardBlock


def init_transformer_weights(module):
    """Initialize weights for Transformer modules.

    Args:
        module: nn.Module to initialize
    """
    if isinstance(module, nn.Linear):
        # Linear layers: use Xavier uniform initialization
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        # Embedding layers: use normal initialization with mean=0, std=0.02
        init.normal_(module.weight, mean=0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        # LayerNorm: weight=1, bias=0 (PyTorch default is already this)
        init.constant_(module.weight, 1.0)
        init.constant_(module.bias, 0.0)
    # Note: PositionalEncoding buffers are not parameters, so not initialized here


class EncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads, dropout)
        self.ff_block = FeedForwardBlock(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, src_len, d_model]
            mask: [batch, 1, 1, src_len]
        """
        x = self.self_attn(x, mask)
        x = self.ff_block(x)
        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads, dropout)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.ff_block = FeedForwardBlock(d_model, d_ff, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch, tgt_len, d_model]
            encoder_output: [batch, src_len, d_model]
            src_mask: [batch, 1, 1, src_len]
            tgt_mask: [batch, 1, tgt_len, tgt_len]
        """
        x = self.self_attn(x, tgt_mask)
        x = self.cross_attn(x, encoder_output, src_mask)
        x = self.ff_block(x)
        return x


class Encoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # Note: No final layer norm in pre-norm architecture

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, src_len]
            mask: [batch, 1, 1, src_len]
        Returns:
            [batch, src_len, d_model]
        """
        # Embedding and positional encoding
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=x.device))
        x = self.pos_encoding(x)

        # Pass through layers
        for layer in self.layers:
            x = layer(x, mask)

        return x  # No final layer norm in pre-norm architecture


class Decoder(nn.Module):
    """Transformer decoder."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # Note: No final layer norm in pre-norm architecture
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch, tgt_len]
            encoder_output: [batch, src_len, d_model]
            src_mask: [batch, 1, 1, src_len]
            tgt_mask: [batch, 1, tgt_len, tgt_len]
        Returns:
            [batch, tgt_len, vocab_size]
        """
        # Embedding and positional encoding
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=x.device))
        x = self.pos_encoding(x)

        # Pass through layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.output_proj(x)  # No final layer norm in pre-norm architecture


class Transformer(nn.Module):
    """Full Transformer model for translation."""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len
        )

        # Apply custom weight initialization
        self.apply(init_transformer_weights)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]
            src_mask: [batch, 1, 1, src_len]
            tgt_mask: [batch, 1, tgt_len, tgt_len]
        Returns:
            [batch, tgt_len, tgt_vocab_size]
        """
        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return output

    def encode(self, src, src_mask=None):
        """Encode source sequence."""
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence."""
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
