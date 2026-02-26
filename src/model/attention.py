"""
Multi-Head Attention for Transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, tgt_len, d_model]
            key: [batch, src_len, d_model]
            value: [batch, src_len, d_model]
            mask: [batch, 1, tgt_len, src_len] or [batch, 1, 1, src_len]

        Returns:
            output: [batch, tgt_len, d_model]
            attention_weights: [batch, n_heads, tgt_len, src_len]
        """
        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = key.size(1)

        # Linear projections and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Broadcast mask to match attention scores shape
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, 1, src_len] or [batch, 1, tgt_len, 1]
            elif mask.dim() == 4:
                pass  # Already [batch, 1, tgt_len, src_len]
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        # Concatenate heads and apply final linear
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)

        return output, attn_weights


class SelfAttention(nn.Module):
    """Self-attention layer (for encoder)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, 1, 1, seq_len]
        """
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm(x + self.dropout(attn_output))
        return x


class CrossAttention(nn.Module):
    """Cross-attention layer (for decoder)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask=None):
        """
        Args:
            x: [batch, tgt_len, d_model]
            encoder_output: [batch, src_len, d_model]
            mask: [batch, 1, tgt_len, src_len]
        """
        attn_output, _ = self.attention(x, encoder_output, encoder_output, mask)
        x = self.norm(x + self.dropout(attn_output))
        return x
