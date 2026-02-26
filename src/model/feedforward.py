"""
Feed-Forward Network for Transformer.
"""
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class FeedForwardBlock(nn.Module):
    """Feed-Forward block with residual connection and layer norm."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        ff_output = self.ff(x)
        x = self.norm(x + self.dropout(ff_output))
        return x
