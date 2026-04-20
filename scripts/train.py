"""
Train the Transformer model.
"""
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import config
from src.data.dataset import ParallelDataset
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.trainer import Trainer


def get_device():
    """Get device for training."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# def get_last_step_from_log(log_dir: Path) -> int:
#     """
#     从训练日志文件获取最后一个步数。

#     Args:
#         log_dir: 日志目录路径

#     Returns:
#         最后一个步数，如果没有日志则返回0
#     """
#     step_log_path = log_dir / "step_log.csv"
#     if not step_log_path.exists():
#         return 0

#     last_step = 0
#     try:
#         import csv
#         with open(step_log_path, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 try:
#                     step = int(row['step'])
#                     if step > last_step:
#                         last_step = step
#                 except (ValueError, KeyError):
#                     continue
#     except Exception as e:
#         print(f"警告: 读取日志文件失败: {e}")

#     return last_step


def main(resume_from: str = None, max_steps: int = None):
    """Main training function."""
    print("=" * 50)
    print("Transformer Translation Training")
    print("=" * 50)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Update config
    config.device = device
    config.src_tokenizer = "models_enhanced/src_tokenizer_final.model"
    config.tgt_tokenizer = "models_enhanced/tgt_tokenizer_final.model"

    # Check if tokenizers exist
    data_dir = Path(__file__).parent.parent
    src_tokenizer_path = data_dir / config.src_tokenizer
    tgt_tokenizer_path = data_dir / config.tgt_tokenizer

    if not src_tokenizer_path.exists() or not tgt_tokenizer_path.exists():
        print("Tokenizers not found. Please run preprocess.py first.")
        print(f"Expected: {src_tokenizer_path} and {tgt_tokenizer_path}")
        return

    # Load tokenizers
    print("\nLoading tokenizers...")
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )
    print(f"Source vocab size: {src_tokenizer.sp.get_piece_size()}")
    print(f"Target vocab size: {tgt_tokenizer.sp.get_piece_size()}")

    # Update config vocabulary sizes
    # FIX 2026-02-26: Set separate vocabulary sizes for source and target
    config.src_vocab_size = src_tokenizer.sp.get_piece_size()
    config.tgt_vocab_size = tgt_tokenizer.sp.get_piece_size()
    config.vocab_size = config.src_vocab_size  # Backward compatibility

    # Ensure we use cleaned data (explicit override for safety)
    config.src_file = "models_enhanced/src_text_cleaned.txt"
    config.tgt_file = "models_enhanced/tgt_text_cleaned.txt"

    # Load dataset
    print("\nLoading dataset...")
    src_file = data_dir / config.src_file
    tgt_file = data_dir / config.tgt_file
    dataset = ParallelDataset(
        src_file,
        tgt_file,
        max_samples=config.max_train_samples,
    )
    print(f"Dataset size: {len(dataset)}")

    # Split into train and validation sets
    train_dataset, val_dataset = dataset.split(split_ratio=config.train_split, seed=42)
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")

    # Create model with separate vocabulary sizes
    # FIX 2026-02-26: Use src_vocab_size and tgt_vocab_size instead of single vocab_size
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

    # Create trainer
    trainer = Trainer(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        config=config,
        device=device,
        val_dataset=val_dataset,
    )

    # Resume from checkpoint if specified
    start_step = 0
    auto_resume_detected = False

    # # 如果没有指定检查点，但日志文件存在，尝试自动恢复步数
    # if not resume_from:
    #     log_dir = data_dir / "models" / "logs"
    #     last_step = get_last_step_from_log(log_dir)
    #     if last_step > 0:
    #         print(f"\n检测到上次训练日志，最后步数: {last_step}")
    #         print(f"注意: 自动从步数 {last_step + 1} 继续步数计数。")
    #         print("这只会恢复步数显示，不会恢复模型参数。")
    #         print("要恢复模型参数，请使用 --resume 参数指定检查点文件。")
    #         # 自动继续（从下一步开始）
    #         start_step = last_step + 1
    #         auto_resume_detected = True
    #         print(f"自动从步数 {start_step} 继续训练")

    if resume_from:
        checkpoint_path = data_dir / "models" / resume_from
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {resume_from}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore scheduler step number
            if hasattr(trainer.scheduler, 'step_num'):
                # First try to get scheduler_step_num from checkpoint
                if 'scheduler_step_num' in checkpoint:
                    trainer.scheduler.step_num = checkpoint['scheduler_step_num']
                    print(f"Restored scheduler step from checkpoint: {trainer.scheduler.step_num}")
                # Fallback to using step field
                elif 'step' in checkpoint:
                    trainer.scheduler.step_num = checkpoint['step']
                    print(f"Set scheduler step from step field: {trainer.scheduler.step_num}")
                else:
                    print(f"Warning: No step information found in checkpoint, scheduler step may be incorrect")

            # Get saved step count - prefer scheduler_step_num as it's more reliable
            if 'scheduler_step_num' in checkpoint:
                start_step = checkpoint['scheduler_step_num']
                print(f"Using scheduler_step_num as start step: {start_step}")
            elif 'step' in checkpoint:
                start_step = checkpoint['step']
                print(f"Using step field as start step: {start_step}")
            else:
                start_step = 0
                print("Warning: No step information found, starting from step 0")

            print(f"Resuming from step {start_step}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    # 如果自动检测到恢复，设置scheduler步数
    if auto_resume_detected and hasattr(trainer.scheduler, 'step_num'):
        trainer.scheduler.step_num = start_step
        print(f"设置scheduler步数为: {trainer.scheduler.step_num}")

    # Train
    print("\nStarting training...")
    # Use custom max_steps if provided, otherwise use config
    final_max_steps = max_steps if max_steps is not None else config.max_steps
    print(f"Training steps: {final_max_steps} (starting from step {start_step})")

    trainer.train(
        dataset=train_dataset,
        batch_size=config.batch_size,
        max_steps=final_max_steps,
        max_len=config.max_len,
        start_step=start_step,
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint file")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    args = parser.parse_args()

    main(resume_from=args.resume, max_steps=args.max_steps)
