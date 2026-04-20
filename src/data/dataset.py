"""
Dataset classes for parallel corpus.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ParallelDataset(Dataset):
    """Dataset for parallel text (source-target pairs)."""

    def __init__(
        self,
        src_file: Path,
        tgt_file: Path,
        max_samples: int = None,
    ):
        self.src_lines = []
        self.tgt_lines = []

        print(f"Loading src data from {src_file} ...")
        print(f"Loading tgt data from {tgt_file} ...")

        with open(src_file, 'r', encoding='utf-8') as f_src, \
             open(tgt_file, 'r', encoding='utf-8') as f_tgt:

            for i, (src_line, tgt_line) in enumerate(zip(f_src, f_tgt)):
                if max_samples and i >= max_samples:
                    break
                src_text = src_line.strip()
                tgt_text = tgt_line.strip()
                if src_text and tgt_text:
                    self.src_lines.append(src_text)
                    self.tgt_lines.append(tgt_text)

        print(f"Loaded {len(self.src_lines)} parallel sentences")

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        return {
            'src': self.src_lines[idx],
            'tgt': self.tgt_lines[idx],
        }

    def split(self, split_ratio: float = 0.9, seed: int = 42):
        """Split dataset into two parts.

        Args:
            split_ratio: ratio for first part (e.g., 0.9 for 90% train, 10% val)
            seed: random seed for shuffling

        Returns:
            Two ParallelDataset instances (train, val)
        """
        import numpy as np
        np.random.seed(seed)

        indices = list(range(len(self)))
        np.random.shuffle(indices)

        split_point = int(len(self) * split_ratio)
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]

        # Create new datasets with selected lines
        train_src = [self.src_lines[i] for i in train_indices]
        train_tgt = [self.tgt_lines[i] for i in train_indices]
        val_src = [self.src_lines[i] for i in val_indices]
        val_tgt = [self.tgt_lines[i] for i in val_indices]

        # Create new dataset instances (reusing the same class)
        train_dataset = ParallelDataset.__new__(ParallelDataset)
        train_dataset.src_lines = train_src
        train_dataset.tgt_lines = train_tgt

        val_dataset = ParallelDataset.__new__(ParallelDataset)
        val_dataset.src_lines = val_src
        val_dataset.tgt_lines = val_tgt

        return train_dataset, val_dataset


# class TranslationDataset(Dataset):
#     """Dataset that returns tokenized sequences."""

#     def __init__(self, src_tokens, tgt_tokens):
#         """
#         Args:
#             src_tokens: list of source token lists
#             tgt_tokens: list of target token lists
#         """
#         self.src_tokens = src_tokens
#         self.tgt_tokens = tgt_tokens

#     def __len__(self):
#         return len(self.src_tokens)

#     def __getitem__(self, idx):
#         return {
#             'src': self.src_tokens[idx],
#             'tgt': self.tgt_tokens[idx],
#         }


# def collate_fn(batch, pad_id: int = 0, max_len: int = 100):
#     """Collate function for DataLoader.

#     Args:
#         batch: list of samples
#         pad_id: padding token id
#         max_len: maximum sequence length
#     """
#     src_batch = []
#     tgt_batch = []

#     for item in batch:
#         src_batch.append(item['src'][:max_len])
#         tgt_batch.append(item['tgt'][:max_len])

#     # Pad sequences
#     src_padded = []
#     tgt_padded = []
#     src_mask = []
#     tgt_mask = []

#     for src, tgt in zip(src_batch, tgt_batch):
#         src_len = len(src)
#         tgt_len = len(tgt)

#         # Pad to max length in batch
#         src_padded.append(
#             src + [pad_id] * (max_len - src_len)
#         )
#         tgt_padded.append(
#             tgt + [pad_id] * (max_len - tgt_len)
#         )

#         # Create masks (1 = valid, 0 = padding)
#         src_mask.append([1] * src_len + [0] * (max_len - src_len))
#         tgt_mask.append([1] * tgt_len + [0] * (max_len - tgt_len))

#     return {
#         'src': torch.tensor(src_padded, dtype=torch.long),
#         'tgt': torch.tensor(tgt_padded, dtype=torch.long),
#         'src_mask': torch.tensor(src_mask, dtype=torch.bool),
#         'tgt_mask': torch.tensor(tgt_mask, dtype=torch.bool),
#     }
