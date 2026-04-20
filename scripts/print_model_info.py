"""
Print Transformer model info.
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import math
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.dataset import ParallelDataset
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
# from src.trainer import Trainer

def create_model():
    print("\nCreating model...")
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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model    


def load_checkpoint(model, filename: str):
    """Load model checkpoint."""

    device = torch.device('cpu')

    data_dir = Path(__file__).parent.parent
    # src_tokenizer_path = data_dir / config.src_tokenizer
    # tgt_tokenizer_path = data_dir / config.tgt_tokenizer    

    checkpoint_path = data_dir / filename
    if not checkpoint_path.exists():
        print(f"checkpoint file: {checkpoint_path} not exists.")
        return
    
    print(f"\nLoading from checkpoint: {filename}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if 'optimizer_state_dict' in checkpoint:
        print("Model has optimizer state saved.")
        # trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_step_num' in checkpoint:
        print(f"scheduler_step_num: {checkpoint['scheduler_step_num']}")
    elif 'step' in checkpoint:
        print(f"step: {checkpoint['step']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="checkpoint file")
    args = parser.parse_args()

    model = create_model()

    filename = args.checkpoint

    load_checkpoint(model, filename)



