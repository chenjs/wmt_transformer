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
    vocab_size: int = 32000
    max_len: int = 64

    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    max_steps: int = 20000
    label_smoothing: float = 0.1
    clip_grad: float = 1.0

    # Data
    train_split: float = 0.99
    max_train_samples: int = 100000  # Limit for faster training on Mac

    # Checkpoint
    checkpoint_dir: Path = Path(__file__).parent.parent / "models"
    save_interval: int = 5000

    # Device
    device: str = "mps"  # Will fallback to cpu if mps unavailable

    # Tokenizer paths (will be created during preprocessing)
    src_tokenizer: str = ""
    tgt_tokenizer: str = ""


# Global config instance
config = Config()
