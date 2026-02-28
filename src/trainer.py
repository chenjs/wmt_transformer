"""
Trainer for Transformer translation model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import math
from tqdm import tqdm

from .model import Transformer
from .data.batch import create_batch, create_masks


class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross entropy loss."""

    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_id = pad_id
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: [batch * seq_len, vocab_size]
            target: [batch * seq_len]
        """
        pred = pred.log_softmax(dim=-1)

        # Optimized path for smoothing = 0 (standard cross-entropy)
        if self.smoothing == 0:
            # Standard negative log likelihood loss
            loss = F.nll_loss(
                pred, target,
                ignore_index=self.pad_id,
                reduction='mean'
            )
            return loss

        # Original label smoothing implementation for smoothing > 0
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

            # Mask padding positions
            mask = (target != self.pad_id).unsqueeze(1).float()
            true_dist = true_dist * mask

        loss = torch.sum(-true_dist * pred, dim=-1)
        loss = loss.masked_fill(target == self.pad_id, 0)
        return loss.mean()


class WarmupScheduler:
    """Learning rate scheduler with warmup."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _get_lr(self):
        step = max(1, self.step_num)
        if self.warmup_steps <= 0:
            return self.d_model ** (-0.5) * step ** (-0.5)
        else:
            return self.d_model ** (-0.5) * min(
                step ** (-0.5),
                step * self.warmup_steps ** (-1.5)
            )


class Trainer:
    """Trainer for Transformer model."""

    def __init__(
        self,
        model: Transformer,
        src_tokenizer,
        tgt_tokenizer,
        config,
        device: str = "cpu",
        val_dataset = None,
    ):
        self.model = model.to(device)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.config = config
        self.device = device
        self.val_dataset = val_dataset

        # Compute target vocabulary size from tokenizer
        # FIX 2026-02-26: Use tgt_tokenizer's vocab size instead of config.vocab_size
        tgt_vocab_size = tgt_tokenizer.sp.get_piece_size()

        # Loss function
        self.criterion = LabelSmoothingLoss(
            tgt_vocab_size, config.label_smoothing, pad_id=0
        )

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # Learning rate scheduler
        self.scheduler = WarmupScheduler(
            self.optimizer, config.d_model, config.warmup_steps
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Log directory
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, batch):
        """Single training step."""
        src = batch['src'].to(self.device)
        tgt = batch['tgt'].to(self.device)
        src_mask = batch['src_mask'].to(self.device)

        # Create tgt_input and tgt_output (shifted)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Create masks for shifted targets (need to recreate for correct length)
        _, tgt_mask = create_masks(tgt_input, tgt_input, pad_id=0)

        # Forward pass
        output = self.model(src, tgt_input, src_mask, tgt_mask)

        # Compute loss
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = self.criterion(output, tgt_output)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def evaluate_loss(self, dataset, batch_size: int = None, max_len: int = 100):
        """Evaluate loss on dataset."""
        if dataset is None:
            return None

        self.model.eval()
        if batch_size is None:
            batch_size = self.config.batch_size

        total_loss = 0
        num_batches = 0

        indices = list(range(len(dataset)))

        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                samples = [dataset[j] for j in batch_indices]
                batch = create_batch(samples, self.src_tokenizer, self.tgt_tokenizer, max_len, pad_id=0, device=self.device)

                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)

                # Create tgt_input and tgt_output (shifted)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # Create masks
                _, tgt_mask = create_masks(tgt_input, tgt_input, pad_id=0)

                # Forward pass
                output = self.model(src, tgt_input, src_mask, tgt_mask)

                # Compute loss
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
                loss = self.criterion(output, tgt_output)

                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else None

    def train_epoch(self, dataset, batch_size: int, max_len: int = 100, global_step: int = 0, step_log_file=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Create batches
        indices = list(range(len(dataset)))
        import numpy as np
        np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            samples = [dataset[j] for j in batch_indices]
            batch = create_batch(samples, self.src_tokenizer, self.tgt_tokenizer, max_len, pad_id=0, device=self.device)  # FIX 2026-02-26: Pass both tokenizers

            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            global_step += 1

            # Print each step
            current_lr = self.scheduler._get_lr()
            print(f"Step {global_step}: loss={loss:.4f}, lr={current_lr:.6f}")

            # Log to file if provided
            if step_log_file is not None:
                step_log_file.write(f"{global_step},{loss:.6f},{current_lr:.6f}\n")
                step_log_file.flush()

        return total_loss / num_batches, global_step

    def train(self, dataset, batch_size: int, max_steps: int, max_len: int = 100, start_step: int = 0):
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Total training steps: {max_steps}")
        print(f"Dataset size: {len(dataset)}, batches per epoch: {len(dataset) // batch_size}")

        print(f"self.config.save_interval: {self.config.save_interval}")

        global_step = start_step
        best_loss = float('inf')
        dataset_size = len(dataset)
        batches_per_epoch = dataset_size // batch_size

        # Open log files
        step_log_path = self.log_dir / "step_log.csv"
        epoch_log_path = self.log_dir / "epoch_log.csv"
        val_log_path = self.log_dir / "val_log.csv"

        # Write headers if files don't exist
        if not step_log_path.exists():
            with open(step_log_path, 'w', encoding='utf-8') as f:
                f.write("step,loss,lr\n")
        if not epoch_log_path.exists():
            with open(epoch_log_path, 'w', encoding='utf-8') as f:
                f.write("step,epoch_loss,lr\n")
        if not val_log_path.exists():
            with open(val_log_path, 'w', encoding='utf-8') as f:
                f.write("step,val_loss\n")

        # Open files for appending
        step_log_file = open(step_log_path, 'a', encoding='utf-8')
        epoch_log_file = open(epoch_log_path, 'a', encoding='utf-8')
        val_log_file = open(val_log_path, 'a', encoding='utf-8')

        while global_step < max_steps:
            # Train for one epoch
            epoch_loss, global_step = self.train_epoch(dataset, batch_size, max_len, global_step, step_log_file)

            # Print epoch progress
            current_lr = self.scheduler._get_lr()
            print(f"Epoch complete (step {global_step}): avg_loss={epoch_loss:.4f}, lr={current_lr:.6f}")

            # Log epoch to file
            epoch_log_file.write(f"{global_step},{epoch_loss:.6f},{current_lr:.6f}\n")
            epoch_log_file.flush()

            # Evaluate on validation set periodically
            if (self.val_dataset is not None and
                self.config.eval_interval > 0 and
                global_step > 0 and
                global_step % self.config.eval_interval == 0):
                val_loss = self.evaluate_loss(self.val_dataset, batch_size, max_len)
                if val_loss is not None:
                    print(f"Validation (step {global_step}): loss={val_loss:.4f}")
                    val_log_file.write(f"{global_step},{val_loss:.6f}\n")
                    val_log_file.flush()

            # Save checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_checkpoint(f"best_model.pt", global_step)
                print(f"Saved best model (loss={best_loss:.4f})")

            # Periodic checkpoint (save every save_interval steps)
            if global_step > 0 and global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{global_step}.pt", global_step)

            # Stop if max steps reached
            if global_step >= max_steps:
                break

        print("Training completed!")

        # Close log files
        step_log_file.close()
        epoch_log_file.close()
        val_log_file.close()

        return best_loss

    def save_checkpoint(self, filename: str, step: int = None):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step': step,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
