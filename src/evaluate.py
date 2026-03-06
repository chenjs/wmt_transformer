"""
Evaluation functions for translation model.
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import List

from .model import Transformer
from .data.batch import create_masks


def greedy_decode(model, src, src_mask, src_tokenizer, tgt_tokenizer,
                  max_len: int = 100, device: str = "cpu"):
    """Greedy decoding for translation.

    Args:
        model: Transformer model
        src: source tokens [1, src_len]
        src_mask: source mask [1, 1, 1, src_len]
        src_tokenizer: source tokenizer
        tgt_tokenizer: target tokenizer
        max_len: maximum decode length
        device: device

    Returns:
        translated text
    """
    model.eval()

    # Encode source
    encoder_output = model.encode(src, src_mask)

    # Start with BOS token
    tgt = torch.tensor([[tgt_tokenizer.bos_id]], device=device)

    for _ in range(max_len):
        # Create masks
        tgt_mask = create_masks(tgt, tgt, pad_id=0)[1]

        # Decode
        output = model.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Get next token
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

        # Append to target
        tgt = torch.cat([tgt, next_token], dim=1)

        # Stop if EOS
        if next_token.item() == tgt_tokenizer.eos_id:
            break

    # Decode to text
    tokens = tgt[0].tolist()
    text = tgt_tokenizer.decode(tokens)
    return text


def beam_search_decode(model, src, src_mask, src_tokenizer, tgt_tokenizer,
                        max_len: int = 100, beam_size: int = 5, device: str = "cpu"):
    """Beam search decoding for translation.

    Args:
        model: Transformer model
        src: source tokens [1, src_len]
        src_mask: source mask [1, 1, 1, src_len]
        src_tokenizer: source tokenizer
        tgt_tokenizer: target tokenizer
        max_len: maximum decode length
        beam_size: beam size
        device: device

    Returns:
        translated text
    """
    model.eval()

    # Encode source
    encoder_output = model.encode(src, src_mask)

    # Start with BOS token
    beams = [(torch.tensor([[tgt_tokenizer.bos_id]], device=device), 0.0)]

    completed = []

    for _ in range(max_len):
        all_candidates = []

        for beam in beams:
            tgt, score = beam

            # Stop if EOS
            if tgt[0, -1].item() == tgt_tokenizer.eos_id:
                completed.append(beam)
                continue

            # Create masks
            tgt_mask = create_masks(tgt, tgt, pad_id=0)[1]

            # Decode
            output = model.decode(tgt, encoder_output, src_mask, tgt_mask)

            # Get top-k next tokens
            log_probs = F.log_softmax(output[:, -1, :], dim=-1)
            topk_probs, topk_ids = log_probs.topk(beam_size)

            for i in range(beam_size):
                next_token = topk_ids[0, i].unsqueeze(0).unsqueeze(0)
                new_score = score + topk_probs[0, i].item()
                new_tgt = torch.cat([tgt, next_token], dim=1)
                all_candidates.append((new_tgt, new_score))

        # Select top beams
        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_size]

        # Stop if all beams completed
        if len(completed) >= beam_size:
            break

    # Add remaining beams to completed
    completed.extend(beams)

    # Select best
    completed = sorted(completed, key=lambda x: x[1], reverse=True)
    best_tgt = completed[0][0]

    # Decode to text
    tokens = best_tgt[0].tolist()
    text = tgt_tokenizer.decode(tokens)
    return text


def calculate_bleu(references: List[str], hypotheses: List[str]) -> float:
    """Calculate BLEU score.

    Args:
        references: list of reference texts
        hypotheses: list of predicted texts

    Returns:
        BLEU score
    """
    from collections import Counter

    def get_ngrams(tokens, n):
        """Get n-grams."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def count_ngrams(hypothesis, references, n):
        """Count matching n-grams."""
        hyp_ngrams = Counter(get_ngrams(hypothesis, n))

        # Find best match among references
        best_count = 0
        for ref in references:
            ref_ngrams = Counter(get_ngrams(ref, n))
            matches = sum((hyp_ngrams & ref_ngrams).values())
            best_count = max(best_count, matches)

        return best_count, sum(hyp_ngrams.values())

    # Tokenize
    references = [ref.split() for ref in references]
    hypotheses = [hyp.split() for hyp in hypotheses]

    # Calculate scores for each n-gram
    scores = {}
    for n in range(1, 5):
        total_matches = 0
        total_predicted = 0

        for ref, hyp in zip(references, hypotheses):
            matches, predicted = count_ngrams(hyp, [ref], n)
            total_matches += matches
            total_predicted += predicted

        if total_predicted > 0:
            scores[n] = total_matches / total_predicted
        else:
            scores[n] = 0

    # Calculate BLEU
    if all(scores[n] > 0 for n in range(1, 5)):
        bleu = 1.0
        for n in range(1, 5):
            bleu *= scores[n]
        bleu = bleu ** 0.25
    else:
        bleu = 0

    # Apply brevity penalty
    c = sum(len(h) for h in hypotheses)
    r = sum(len(r) for r in references) / len(references)

    if c > 0:
        bp = min(1, math.exp(1 - r / c))
    else:
        bp = 0

    return bp * bleu * 100


import math


class Evaluator:
    """Evaluator for translation model."""

    def __init__(self, model, src_tokenizer, tgt_tokenizer, device: str = "cpu"):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        # Get max_len from model's positional encoding
        self.max_len = model.decoder.pos_encoding.pe.size(1)

    def translate(self, src_text: str, method: str = "greedy", beam_size: int = 4) -> str:
        """Translate a single sentence.

        Args:
            src_text: source text
            method: decoding method ("greedy" or "beam")
            beam_size: beam size for beam search (default: 4)

        Returns:
            translated text
        """
        # Tokenize
        src_tokens = self.src_tokenizer(src_text, add_bos=False, add_eos=True)
        src = torch.tensor([src_tokens], dtype=torch.long, device=self.device)
        src_mask = torch.ones(1, 1, 1, len(src_tokens), dtype=torch.bool, device=self.device)

        # Translate
        if method == "greedy":
            return greedy_decode(self.model, src, src_mask,
                               self.src_tokenizer, self.tgt_tokenizer,
                               max_len=self.max_len, device=self.device)
        else:
            return beam_search_decode(self.model, src, src_mask,
                                     self.src_tokenizer, self.tgt_tokenizer,
                                     max_len=self.max_len, beam_size=beam_size, device=self.device)

    def evaluate(self, dataset, max_samples: int = 100, method: str = "greedy") -> dict:
        """Evaluate on a dataset.

        Args:
            dataset: ParallelDataset
            max_samples: maximum samples to evaluate
            method: decoding method

        Returns:
            dict with BLEU score and sample translations
        """
        self.model.eval()

        references = []
        hypotheses = []
        samples = []

        with torch.no_grad():
            for i in range(min(len(dataset), max_samples)):
                src_text = dataset[i]['src']
                tgt_text = dataset[i]['tgt']

                # Translate
                hyp_text = self.translate(src_text, method)

                references.append(tgt_text)
                hypotheses.append(hyp_text)

                if i < 5:  # Save first 5 samples
                    samples.append({
                        'src': src_text,
                        'ref': tgt_text,
                        'hyp': hyp_text,
                    })

        # Calculate BLEU
        bleu = calculate_bleu(references, hypotheses)

        return {
            'bleu': bleu,
            'samples': samples,
        }
