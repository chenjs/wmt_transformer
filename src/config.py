"""
Configuration for Transformer translation model.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Data paths
    data_dir: Path = Path(__file__).parent.parent
    src_file: str = "europarl-v7.de-en.en"  # English
    tgt_file: str = "europarl-v7.de-en.de"  # German

    # Tokenizer
    vocab_size: int = 16000  # Deprecated: use src_vocab_size and tgt_vocab_size
    src_vocab_size: int = 16000  # FIX 2026-02-26: Added for separate source vocabulary
    tgt_vocab_size: int = 16000  # FIX 2026-02-26: Added for separate target vocabulary
    max_len: int = 54  # Based on 90th percentile of cleaned data (45 words * 1.2)

    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1

    # Training
    batch_size: int = 32    # 12  # Increased from 8 to reduce epoch time
    learning_rate: float = 1e-3  # Increased from 5e-4 for faster convergence with pre-norm
    warmup_steps: int = 8000     # Restored to standard warmup steps
    max_steps: int = 100000    # 200000       # Increased from 20000 for better convergence
    label_smoothing: float = 0.1  # Enable label smoothing for better generalization
    clip_grad: float = 10.0  # Increased from 5.0 to allow larger gradients

    # Data
    train_split: float = 0.99
    max_train_samples: int = 600000  # Increased from 100000 for better learning

    # Checkpoint
    checkpoint_dir: Path = Path(__file__).parent.parent / "models"
    save_interval: int = 10000
    eval_interval: int = 5000  # Evaluate on validation set every N steps
    min_loss_improvement: float = 0.01  # Only save best model if loss improves by at least 1%

    # Device
    device: str = "mps"  # Will fallback to cpu if mps unavailable

    # Tokenizer paths (will be created during preprocessing)
    src_tokenizer: str = "models_enhanced/src_tokenizer_final.model"
    tgt_tokenizer: str = "models_enhanced/tgt_tokenizer_final.model"


# Global config instance
config = Config()
