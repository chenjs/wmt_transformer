"""
Tokenizer using SentencePiece BPE.
"""
import sentencepiece as spm
from pathlib import Path
import os


class BPETokenizer:
    """BPE Tokenizer using SentencePiece."""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        """Encode text to token ids."""
        ids = self.sp.encode(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        """Decode token ids to text."""
        return self.sp.decode(ids)

    def __call__(self, text: str, add_bos: bool = False, add_eos: bool = False):
        return self.encode(text, add_bos, add_eos)


def train_tokenizer(
    text_file: str,
    model_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 1.0,
    model_type: str = "bpe",
):
    """Train a SentencePiece tokenizer.

    Args:
        text_file: input text file (one sentence per line)
        model_prefix: output model prefix
        vocab_size: vocabulary size
        character_coverage: character coverage
        model_type: model type (bpe, char, word)
    """
    model_prefix = str(model_prefix)

    # Train tokenizer
    train_cmd = (
        f"--input={text_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage={character_coverage} "
        f"--model_type={model_type} "
        f"--pad_id=0 "
        f"--bos_id=1 "
        f"--eos_id=2 "
        f"--unk_id=3 "
        f"--pad_piece=[PAD] "
        f"--bos_piece=[BOS] "
        f"--eos_piece=[EOS] "
        f"--unk_piece=[UNK]"
    )

    print(f"Training tokenizer: {model_prefix}")
    spm.SentencePieceTrainer.train(train_cmd)
    print(f"Tokenizer saved to {model_prefix}.model")

    return f"{model_prefix}.model"


def prepare_tokenizer_data(
    src_file: Path,
    tgt_file: Path,
    output_dir: Path,
    max_samples: int = 100000,
):
    """Prepare text files for tokenizer training.

    Args:
        src_file: source language file
        tgt_file: target language file
        output_dir: output directory
        max_samples: maximum samples to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_text_file = output_dir / "src_text.txt"
    tgt_text_file = output_dir / "tgt_text.txt"

    print(f"Preparing tokenizer training data (max {max_samples} samples)...")

    # Write source file
    with open(src_file, 'r', encoding='utf-8') as f_in, \
         open(src_text_file, 'w', encoding='utf-8') as f_out:

        for i, line in enumerate(f_in):
            if max_samples and i >= max_samples:
                break
            f_out.write(line.strip() + "\n")

    # Write target file
    with open(tgt_file, 'r', encoding='utf-8') as f_in, \
         open(tgt_text_file, 'w', encoding='utf-8') as f_out:

        for i, line in enumerate(f_in):
            if max_samples and i >= max_samples:
                break
            f_out.write(line.strip() + "\n")

    print(f"Tokenizer data saved to {output_dir}")
    return src_text_file, tgt_text_file


def load_tokenizers(src_model: str, tgt_model: str):
    """Load source and target tokenizers."""
    src_tokenizer = BPETokenizer(src_model)
    tgt_tokenizer = BPETokenizer(tgt_model)
    return src_tokenizer, tgt_tokenizer
