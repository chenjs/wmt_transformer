"""
Batch generation utilities.
"""
import torch
import numpy as np


def create_masks(src, tgt, pad_id: int = 0):
    """Create attention masks for Transformer.

    Args:
        src: source tokens [batch, src_len]
        tgt: target tokens [batch, tgt_len]
        pad_id: padding token id

    Returns:
        src_mask: [batch, 1, 1, src_len] - mask for encoder
        tgt_mask: [batch, 1, tgt_len, tgt_len] - mask for decoder
    """
    device = src.device
    batch_size = src.size(0)
    src_len = src.size(1)
    tgt_len = tgt.size(1)

    # Source mask: 1 for valid tokens, 0 for padding
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]

    # Target mask: causal mask + padding mask
    # Causal mask: prevent attending to future positions
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), diagonal=1
    )  # [tgt_len, tgt_len]

    # Padding mask for target
    tgt_pad_mask = (tgt != pad_id)  # [batch, tgt_len]

    # Combine: both causal and padding
    tgt_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2) & ~causal_mask.unsqueeze(0).unsqueeze(1)
    # [batch, 1, tgt_len, tgt_len]

    return src_mask, tgt_mask


def create_batch(samples, src_tokenizer, tgt_tokenizer=None, max_len: int = 100, pad_id: int = 0, device: str = "cpu"):
    """Create a batch from samples.

    FIX 2026-02-26: Added separate src_tokenizer and tgt_tokenizer parameters to fix
    tokenizer usage bug where German text was incorrectly tokenized with English tokenizer.

    Args:
        samples: list of {'src': str, 'tgt': str}
        src_tokenizer: source tokenizer instance
        tgt_tokenizer: target tokenizer instance (optional, defaults to src_tokenizer with warning)
        max_len: maximum sequence length
        pad_id: padding token id
        device: device for tensors

    Returns:
        dict with tensors
    """
    if tgt_tokenizer is None:
        import warnings
        warnings.warn("tgt_tokenizer not provided, using src_tokenizer for both source and target. "
                     "This may cause incorrect tokenization for translation tasks.", DeprecationWarning)
        tgt_tokenizer = src_tokenizer

    src_texts = [s['src'] for s in samples]
    tgt_texts = [s['tgt'] for s in samples]

    # Tokenize with BOS/EOS
    src_tokens = [
        src_tokenizer(s, add_bos=False, add_eos=True)[:max_len]
        for s in src_texts
    ]
    tgt_tokens = [
        tgt_tokenizer(s, add_bos=True, add_eos=True)[:max_len]
        for s in tgt_texts
    ]

    # Pad to max_len
    batch_size = len(samples)
    max_src_len = min(max_len, max(len(t) for t in src_tokens))
    max_tgt_len = min(max_len, max(len(t) for t in tgt_tokens))

    src_padded = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long, device=device)
    tgt_padded = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long, device=device)

    src_mask = torch.zeros(batch_size, 1, 1, max_src_len, dtype=torch.bool, device=device)
    tgt_mask = torch.zeros(batch_size, 1, max_tgt_len, max_tgt_len, dtype=torch.bool, device=device)

    # Causal mask for decoder
    causal_mask = torch.triu(
        torch.ones(max_tgt_len, max_tgt_len, dtype=torch.bool, device=device), diagonal=1
    )

    for i, (src, tgt) in enumerate(zip(src_tokens, tgt_tokens)):
        src_len = len(src)
        tgt_len = len(tgt)

        src_padded[i, :src_len] = torch.tensor(src, device=device)
        tgt_padded[i, :tgt_len] = torch.tensor(tgt, device=device)

        # Masks
        src_mask[i, 0, 0, :src_len] = True

        # Target mask: padding AND causal
        valid_tgt = torch.ones(max_tgt_len, dtype=torch.bool, device=device)
        valid_tgt[:tgt_len] = True

        # Causal mask
        valid_tgt = valid_tgt.unsqueeze(1) & ~causal_mask.unsqueeze(0)
        tgt_mask[i, 0] = valid_tgt

    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
    }


# class BatchIterator:
#     """Iterator for creating batches on-the-fly.

#     FIX 2026-02-26: Added separate src_tokenizer and tgt_tokenizer parameters to fix
#     tokenizer usage bug where German text was incorrectly tokenized with English tokenizer.
#     """

#     def __init__(self, dataset, src_tokenizer, tgt_tokenizer=None, batch_size: int = 64,
#                  max_len: int = 100, shuffle: bool = True):
#         if tgt_tokenizer is None:
#             import warnings
#             warnings.warn("tgt_tokenizer not provided, using src_tokenizer for both source and target. "
#                          "This may cause incorrect tokenization for translation tasks.", DeprecationWarning)
#         self.dataset = dataset
#         self.src_tokenizer = src_tokenizer
#         self.tgt_tokenizer = tgt_tokenizer if tgt_tokenizer is not None else src_tokenizer
#         self.batch_size = batch_size
#         self.max_len = max_len
#         self.shuffle = shuffle
#         self.indices = list(range(len(dataset)))

#         if self.shuffle:
#             np.random.shuffle(self.indices)

#     def __iter__(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)

#         for i in range(0, len(self.indices), self.batch_size):
#             batch_indices = self.indices[i:i + self.batch_size]
#             samples = [self.dataset[j] for j in batch_indices]
#             yield create_batch(samples, self.src_tokenizer, self.tgt_tokenizer, self.max_len)  # FIX 2026-02-26: Pass both tokenizers

#     def __len__(self):
#         return (len(self.indices) + self.batch_size - 1) // self.batch_size
