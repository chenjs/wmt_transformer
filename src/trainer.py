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
from datetime import datetime

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


# class WarmupScheduler:
#     """Learning rate scheduler with warmup."""

#     def __init__(self, optimizer, d_model: int, warmup_steps: int):
#         self.optimizer = optimizer
#         self.d_model = d_model
#         self.warmup_steps = warmup_steps
#         self.step_num = 0

#     def step(self):
#         self.step_num += 1
#         lr = self._get_lr()
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = lr
#         return lr

#     def _get_lr(self):
#         step = max(1, self.step_num)
#         if self.warmup_steps <= 0:
#             return self.d_model ** (-0.5) * step ** (-0.5)
#         else:
#             return self.d_model ** (-0.5) * min(
#                 step ** (-0.5),
#                 step * self.warmup_steps ** (-1.5)
#             )

class WarmupScheduler:
    """Learning rate scheduler with warmup."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int, total_training_steps: int, peak_lr, initial_lr, min_lr):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.total_training_steps = total_training_steps
        self.step_num = 0
        self.initial_lr = initial_lr
        self.peak_lr  = peak_lr
        self.min_lr = min_lr
        self.lr_increment = (self.peak_lr - self.initial_lr) / self.warmup_steps

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _get_lr(self):
        # step = max(1, self.step_num)

        # Adjust the learning rate based on the current phase (warmup or cosine annealing)
        if self.step_num < self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr + self.step_num * self.lr_increment
        else:
            # Cosine annealing after warmup
            progress = ((self.step_num - self.warmup_steps) / (self.total_training_steps - self.warmup_steps))
            lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return lr     

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

        # # Optimizer
        # self.optimizer = optim.Adam(
        #     model.parameters(),
        #     lr=config.learning_rate,
        #     betas=(0.9, 0.98),
        #     eps=1e-9,
        # )

        # # Learning rate scheduler
        # self.scheduler = WarmupScheduler(
        #     self.optimizer, config.d_model, config.warmup_steps
        # )
        peak_lr = self.config.learning_rate
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)

        self.scheduler = WarmupScheduler(
            self.optimizer, config.d_model, config.warmup_steps, config.max_steps, 
            peak_lr=config.learning_rate, initial_lr=1e-6, min_lr=1e-6
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Log directory
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def calc_loss_batch(self, output, tgt_output):
        loss = torch.nn.functional.cross_entropy(output.flatten(0, 1), tgt_output.flatten())
        return loss

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

        # # Compute loss
        # output = output.reshape(-1, output.size(-1))
        # tgt_output = tgt_output.reshape(-1)
        # loss = self.criterion(output, tgt_output)

        # # Compute loss (Method2: without LabelSmoothing)
        loss = self.calc_loss_batch(output, tgt_output)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # # Compute gradient norm before clipping
        # total_norm = 0.0
        # for p in self.model.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.detach().data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # grad_norm_before = total_norm ** 0.5

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)

        # # Compute gradient norm after clipping
        # total_norm = 0.0
        # for p in self.model.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.detach().data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # grad_norm_after = total_norm ** 0.5

        # 梯度裁剪，并返回裁剪前的范数
        grad_norm_before = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.clip_grad
        ) 
        # 不裁剪，只计算当前范数
        grad_norm_after = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), float('inf')
        ) 

        self.optimizer.step()
        self.scheduler.step()

        # Print gradient info every 500 steps
        if self.scheduler.step_num % 500 == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Step {self.scheduler.step_num}: grad_norm before={grad_norm_before:.4f}, after={grad_norm_after:.4f}")

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

                # # Compute loss
                # output = output.reshape(-1, output.size(-1))
                # tgt_output = tgt_output.reshape(-1)
                # loss = self.criterion(output, tgt_output)

                # # Compute loss (Method2: without LabelSmoothing)
                loss = self.calc_loss_batch(output, tgt_output)                

                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        if num_batches == 0:
            print(f"Warning: evaluate_loss: num_batches=0, dataset size={len(dataset)}, batch_size={batch_size}")
        else:
            print(f"evaluate_loss: processed {num_batches} batches, avg_loss={total_loss/num_batches:.4f}")
        return total_loss / num_batches if num_batches > 0 else None

    # def _get_last_validation_step(self, val_log_path):
    #     """Get the last validation step from validation log file."""
    #     import csv
    #     last_step = 0
    #     try:
    #         if val_log_path.exists():
    #             with open(val_log_path, 'r', encoding='utf-8') as f:
    #                 reader = csv.reader(f)
    #                 # Skip header
    #                 next(reader, None)
    #                 for row in reader:
    #                     if row and len(row) >= 1:
    #                         try:
    #                             step = int(float(row[0]))
    #                             last_step = max(last_step, step)
    #                         except ValueError:
    #                             continue
    #     except Exception as e:
    #         print(f"Warning: Failed to read validation log: {e}")
    #     return last_step

    def _get_min_val_loss(self, val_log_path):
        """Get the minimum validation loss from validation log file."""
        import csv
        import os

        min_val_loss = float('inf')
        try:
            if val_log_path is None:
                return min_val_loss
            if os.path.exists(val_log_path):
                with open(val_log_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    # Skip header
                    next(reader, None)
                    for row in reader:
                        if row and len(row) >= 2:
                            try:
                                val_loss = float(row[1])
                                if val_loss < min_val_loss:
                                    min_val_loss = val_loss
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Warning: Failed to read validation log for min loss: {e}")
        return min_val_loss
    
    def _check_and_save_best_model(self, val_loss, best_val_loss, global_step):
        should_save = False
        if best_val_loss == float('inf'):
            should_save = True
        else:
            # Avoid division by zero
            if val_loss == 0:
                improvement = float('inf')  # Infinite improvement if loss goes to 0
            else:
                improvement = (best_val_loss - val_loss) / val_loss
            # Check if improvement meets threshold
            if (hasattr(self.config, 'min_loss_improvement') and
                self.config.min_loss_improvement is not None and
                self.config.min_loss_improvement > 0):

                if improvement >= self.config.min_loss_improvement:
                    should_save = True
                else:
                    print(f"Validation Loss improved by {improvement:.2%} but below threshold {self.config.min_loss_improvement:.2%}, skipping save")
            else:
                # Default behavior: any improvement
                should_save = True

        if should_save:
            self.save_checkpoint(f"best_model.pt", global_step)
            return val_loss  # Return new best validation loss
        else:
            return best_val_loss  # Return unchanged best validation loss

    def _prepare_loggers(self, start_step=0):
        # Open log files
        step_log_path = self.log_dir / "step_log.csv"
        epoch_log_path = self.log_dir / "epoch_log.csv"
        val_log_path = self.log_dir / "val_log.csv"

        # Write headers if files don't exist
        if not step_log_path.exists() or (start_step == 0):
            with open(step_log_path, 'w', encoding='utf-8') as f:
                f.write("step,loss,lr\n")
        if not epoch_log_path.exists() or (start_step == 0):
            with open(epoch_log_path, 'w', encoding='utf-8') as f:
                f.write("step,epoch_loss,lr\n")
        if not val_log_path.exists() or (start_step == 0):
            with open(val_log_path, 'w', encoding='utf-8') as f:
                f.write("step,val_loss\n")

        # Open files for appending
        step_log_file = open(step_log_path, 'a', encoding='utf-8')
        epoch_log_file = open(epoch_log_path, 'a', encoding='utf-8')
        val_log_file = open(val_log_path, 'a', encoding='utf-8')

        return (step_log_file, epoch_log_file, val_log_file, val_log_path)     
       

    def train_epoch(self, dataset, batch_size: int, max_len: int = 100, global_step: int = 0, step_log_file=None,
                   val_log_file=None, val_log_path=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        best_val_loss = self._get_min_val_loss(val_log_path)

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

            if global_step % 20 == 0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Step {global_step}: loss={loss:.4f}, lr={current_lr:.7f}")

                # Log to file if provided
                if step_log_file is not None:
                    step_log_file.write(f"{global_step},{loss:.6f},{current_lr:.6f}\n")
                    step_log_file.flush()

            # Check for validation at appropriate intervals
            if (val_log_file is not None and val_log_path is not None and
                self.val_dataset is not None and self.config.eval_interval > 0 and
                global_step > 0 and global_step % self.config.eval_interval == 0):

                print(f"\n{'='*60}")
                print(f"IN-EPOCH VALIDATION at step {global_step}")
                print(f"{'='*60}")

                val_loss = self.evaluate_loss(self.val_dataset, batch_size, max_len)
                if val_loss is not None:
                    print(f"In-epoch validation (step {global_step}): loss={val_loss:.4f}")
                    val_log_file.write(f"{global_step},{val_loss:.6f}\n")
                    val_log_file.flush()
                    print(f"In-epoch validation result saved")
                    # Check and save best model, update best_val_loss
                    best_val_loss = self._check_and_save_best_model(val_loss, best_val_loss, global_step)
                else:
                    print(f"ERROR: In-epoch validation loss is None at step {global_step}")

                print(f"{'='*60}\n")

            # Periodic checkpoint (save every save_interval steps)
            if global_step > 0 and global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{global_step}.pt", global_step)    

        # Calculate average loss for the epoch
        epoch_loss = total_loss / num_batches if num_batches > 0 else 0

        return epoch_loss, global_step

    def train(self, dataset, batch_size: int, max_steps: int, max_len: int = 100, start_step: int = 0):
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Total training steps: {max_steps}")
        print(f"Warmup steps: {self.config.warmup_steps}")
        print(f"Dataset size: {len(dataset)}, batches per epoch: {len(dataset) // batch_size}")

        print(f"self.config.save_interval: {self.config.save_interval}")
        print(f"self.config.eval_interval: {self.config.eval_interval}")
        print(f"self.config.clip_grad: {self.config.clip_grad}")
        print(f"Validation dataset size: {len(self.val_dataset) if self.val_dataset is not None else 0}")

        global_step = start_step
        best_loss = float('inf')
        dataset_size = len(dataset)
        batches_per_epoch = dataset_size // batch_size

        step_log_file, epoch_log_file, val_log_file, val_log_path = self._prepare_loggers(start_step)

        # i_epoch = 0
        i_epoch = global_step // (len(dataset) // batch_size)
        try:
            while global_step < max_steps:
                # Train for one epoch
                i_epoch += 1

                print(f"Epoch[{i_epoch}] ...")
                      
                epoch_loss, global_step = self.train_epoch(dataset, batch_size, max_len, global_step, step_log_file,
                                                          val_log_file, val_log_path)

                # Print epoch progress
                current_lr = self.scheduler._get_lr()
                print(f"Epoch[{i_epoch}] complete (step {global_step}): avg_loss={epoch_loss:.4f}, lr={current_lr:.6f}")

                # Log epoch to file
                epoch_log_file.write(f"{global_step},{epoch_loss:.6f},{current_lr:.6f}\n")
                epoch_log_file.flush()

                # TODO: Epoch-level validation check - now handled inside train_epoch
                # Keeping this commented for potential future use
                # Evaluate on validation set periodically
                # if (self.val_dataset is not None and
                #     self.config.eval_interval > 0 and
                #     global_step > 0 and
                #     global_step % self.config.eval_interval == 0):
                #
                #     print(f"\n{'='*60}")
                #     print(f"VALIDATION CHECK at step {global_step}")
                #     print(f"{'='*60}")
                #     print(f"Validation dataset size: {len(self.val_dataset)}")
                #     print(f"Eval interval: {self.config.eval_interval}")
                #     print(f"Last validation step: {self._get_last_validation_step(val_log_path)}")
                #
                #     val_loss = self.evaluate_loss(self.val_dataset, batch_size, max_len)
                #     if val_loss is not None:
                #         print(f"Validation (step {global_step}): loss={val_loss:.4f}")
                #         val_log_file.write(f"{global_step},{val_loss:.6f}\n")
                #         val_log_file.flush()
                #         print(f"Validation result saved to {val_log_path}")
                #     else:
                #         print(f"ERROR: Validation loss is None at step {global_step}")
                #         print(f"Check validation dataset and batch creation")
                #
                #     print(f"{'='*60}\n")

                # Save checkpoint with improvement threshold
                if epoch_loss < best_loss:
                    # Calculate improvement percentage
                    if best_loss == float('inf'):
                        # First save, always save
                        improvement = 1.0
                        should_save = True
                    else:
                        improvement = (best_loss - epoch_loss) / best_loss
                        should_save = False  # Default to not saving

                        # Check if improvement meets threshold
                        if (hasattr(self.config, 'min_loss_improvement') and
                            self.config.min_loss_improvement is not None and
                            self.config.min_loss_improvement > 0):

                            if improvement >= self.config.min_loss_improvement:
                                should_save = True
                            else:
                                print(f"Train Loss improved by {improvement:.2%} but below threshold {self.config.min_loss_improvement:.2%}, skipping save")
                        else:
                            # Default behavior: any improvement
                            should_save = True

                    if should_save:
                        best_loss = epoch_loss
                        self.save_checkpoint(f"best_model.pt", global_step)
                        print(f"Saved best model (loss={best_loss:.4f}, improvement={improvement:.2%})")

                # # Periodic checkpoint (save every save_interval steps)
                # if global_step > 0 and global_step % self.config.save_interval == 0:
                #     self.save_checkpoint(f"checkpoint_step_{global_step}.pt", global_step)

                # Stop if max steps reached
                if global_step >= max_steps:
                    break

            print("Training completed!")

        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("Training interrupted by user (Ctrl+C)")
            print(f"Saving interrupted checkpoint at step {global_step}...")
            self.save_checkpoint(f"checkpoint_interrupted.pt", global_step)
            print("Interrupted checkpoint saved. Exiting gracefully.")
            print("="*60)

        finally:
            # Close log files (always close, even on interrupt)
            try:
                step_log_file.close()
            except:
                pass
            try:
                epoch_log_file.close()
            except:
                pass
            try:
                val_log_file.close()
            except:
                pass

        return best_loss

    def save_checkpoint(self, filename: str, step: int = None):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step': step,
        }
        # Save scheduler state if available
        if hasattr(self.scheduler, 'step_num'):
            checkpoint['scheduler_step_num'] = self.scheduler.step_num
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Restore scheduler step number
        if hasattr(self.scheduler, 'step_num'):
            # First try to get scheduler_step_num from checkpoint
            if 'scheduler_step_num' in checkpoint:
                self.scheduler.step_num = checkpoint['scheduler_step_num']
                print(f"Restored scheduler step from checkpoint: {self.scheduler.step_num}")
            # Fallback to using step field
            elif 'step' in checkpoint:
                self.scheduler.step_num = checkpoint['step']
                print(f"Set scheduler step from step field: {self.scheduler.step_num}")
            else:
                print(f"Warning: No step information found in checkpoint, scheduler step may be incorrect")
        print(f"Checkpoint loaded from {checkpoint_path}")
