#!/usr/bin/env python3
"""
Calculate quantitative translation metrics (BLEU, TER, etc.) for Transformer model.
Version 2: Sample from training corpus instead of using fixed test cases.
"""
import sys
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Any
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import greedy_decode, beam_search_decode


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance (edit distance) between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_ter(reference: str, hypothesis: str) -> float:
    """Calculate Translation Edit Rate (TER).

    TER = (number of edits) / (length of reference)
    Lower is better (0.0 is perfect).
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    distance = levenshtein_distance(ref_tokens, hyp_tokens)

    ref_len = len(ref_tokens)
    if ref_len == 0:
        return 0.0 if distance == 0 else 1.0

    return distance / ref_len


def calculate_bleu(references: List[str], hypotheses: List[str]) -> float:
    """Calculate BLEU score (simplified implementation).

    Returns BLEU score (0-100 scale).
    """
    def get_ngrams(tokens, n):
        """Get n-grams."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def count_ngrams(hypothesis, references, n):
        """Count matching n-grams."""
        hyp_ngrams = Counter(get_ngrams(hypothesis, n))

        best_count = 0
        for ref in references:
            ref_ngrams = Counter(get_ngrams(ref, n))
            matches = sum((hyp_ngrams & ref_ngrams).values())
            best_count = max(best_count, matches)

        return best_count, sum(hyp_ngrams.values())

    references = [ref.split() for ref in references]
    hypotheses = [hyp.split() for hyp in hypotheses]

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

    if all(scores[n] > 0 for n in range(1, 5)):
        bleu = 1.0
        for n in range(1, 5):
            bleu *= scores[n]
        bleu = bleu ** 0.25
    else:
        bleu = 0

    c = sum(len(h) for h in hypotheses)
    r = sum(len(r) for r in references) / len(references)

    if c > 0:
        bp = min(1, math.exp(1 - r / c))
    else:
        bp = 0

    return bp * bleu * 100


def calculate_exact_match(reference: str, hypothesis: str) -> bool:
    """Calculate exact match (case-insensitive)."""
    return reference.lower().strip() == hypothesis.lower().strip()


def calculate_word_overlap(reference: str, hypothesis: str) -> float:
    """Calculate word overlap (Jaccard similarity)."""
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())

    if not ref_words or not hyp_words:
        return 0.0

    intersection = len(ref_words.intersection(hyp_words))
    union = len(ref_words.union(hyp_words))

    return intersection / union


def calculate_length_ratio(reference: str, hypothesis: str) -> float:
    """Calculate length ratio (hypothesis / reference)."""
    ref_len = len(reference.split())
    hyp_len = len(hypothesis.split())

    if ref_len == 0:
        return 1.0 if hyp_len == 0 else float('inf')

    return hyp_len / ref_len


def load_corpus(src_path: str, tgt_path: str) -> List[Tuple[str, str]]:
    """Load parallel corpus from files."""
    corpus = []

    with open(src_path, 'r', encoding='utf-8') as f_src, \
         open(tgt_path, 'r', encoding='utf-8') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            src = src_line.strip()
            tgt = tgt_line.strip()
            # Only keep pairs where both have content
            if src and tgt:
                corpus.append((src, tgt))

    return corpus


def sample_by_length(corpus: List[Tuple[str, str]],
                     src_tokenizer,
                     num_samples: int,
                     length_bins: List[Tuple[int, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """Sample sentences by length bins.

    Args:
        corpus: List of (src, tgt) pairs
        src_tokenizer: Source tokenizer for counting tokens
        num_samples: Number of samples per bin
        length_bins: List of (max_length, bin_name) tuples, e.g., [(10, "short"), (30, "medium"), (54, "long")]

    Returns:
        Dictionary mapping bin_name to list of samples
    """
    # Categorize by token length
    bins = {name: [] for _, name in length_bins}

    for src, tgt in corpus:
        src_tokens = src_tokenizer.encode(src, add_bos=False, add_eos=False)
        token_len = len(src_tokens)

        # Find appropriate bin
        for max_len, bin_name in length_bins:
            if token_len <= max_len:
                bins[bin_name].append((src, tgt))
                break

    # Sample from each bin
    result = {}
    for bin_name in bins:
        samples = bins[bin_name]
        if len(samples) == 0:
            result[bin_name] = []
            print(f"  Warning: No samples found for bin '{bin_name}'")
            continue

        # Sample without replacement if possible
        if len(samples) >= num_samples:
            selected = random.sample(samples, num_samples)
        else:
            selected = samples  # Use all available
            print(f"  Warning: Only {len(samples)} samples available for bin '{bin_name}', requested {num_samples}")

        result[bin_name] = selected
        print(f"  {bin_name}: {len(selected)} samples (token length <= {length_bins[[i for i, n in enumerate(length_bins) if n[1] == bin_name][0]][0]})")

    return result


class QuantitativeEvaluatorV2:
    """Quantitative metrics evaluator using sampled corpus."""

    def __init__(self, checkpoint_path: str = "models/best_model.pt", device: str = "cpu"):
        self.data_dir = Path(__file__).parent.parent
        self.checkpoint_path = self.data_dir / checkpoint_path
        self.device = device

        # Load tokenizers from config
        src_tokenizer_path = self.data_dir / config.src_tokenizer
        tgt_tokenizer_path = self.data_dir / config.tgt_tokenizer

        self.src_tokenizer, self.tgt_tokenizer = load_tokenizers(
            str(src_tokenizer_path), str(tgt_tokenizer_path)
        )

        # Load model
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)

        # Get vocabulary sizes
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            if hasattr(saved_config, 'src_vocab_size'):
                src_vocab_size = saved_config.src_vocab_size
                tgt_vocab_size = saved_config.tgt_vocab_size
            else:
                src_vocab_size = saved_config.vocab_size
                tgt_vocab_size = saved_config.vocab_size
        else:
            src_vocab_size = config.src_vocab_size
            tgt_vocab_size = config.tgt_vocab_size

        # Create model
        self.model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_len=config.max_len,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)

        print(f"Model loaded from: {checkpoint_path}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")

    def translate(self, src_text: str, method: str = "greedy") -> str:
        """Translate a single sentence."""
        if method == "beam":
            method = "beam4"

        # Check token length - skip if too long
        src_tokens = self.src_tokenizer.encode(src_text, add_bos=False, add_eos=True)
        if len(src_tokens) > config.max_len:
            # Truncate for demonstration (real evaluation should skip)
            src_tokens = src_tokens[:config.max_len]
            print(f"    Warning: Input truncated from {len(src_tokens)} to {config.max_len} tokens")

        src = torch.tensor([src_tokens], dtype=torch.long, device=self.device)
        src_mask = torch.ones(1, 1, 1, len(src_tokens), dtype=torch.bool, device=self.device)

        if method == "greedy":
            return greedy_decode(self.model, src, src_mask,
                               self.src_tokenizer, self.tgt_tokenizer,
                               max_len=config.max_len, device=self.device)
        elif method == "beam4":
            return beam_search_decode(self.model, src, src_mask,
                                     self.src_tokenizer, self.tgt_tokenizer,
                                     max_len=config.max_len, beam_size=4, device=self.device)
        elif method == "beam8":
            return beam_search_decode(self.model, src, src_mask,
                                     self.src_tokenizer, self.tgt_tokenizer,
                                     max_len=config.max_len, beam_size=8, device=self.device)
        else:
            raise ValueError(f"Unknown method: {method}. Supported: greedy, beam, beam4, beam8")

    def evaluate_sampled_corpus(self, sampled_data: Dict[str, List[Tuple[str, str]]],
                                 method: str = "greedy",
                                 max_samples_per_bin: int = 100) -> Dict[str, Any]:
        """Evaluate on sampled corpus and calculate metrics."""
        print(f"\nEvaluating with method: {method}")
        print("=" * 60)

        results_by_bin = {}
        all_references = []
        all_hypotheses = []

        for bin_name, samples in sampled_data.items():
            if not samples:
                continue

            print(f"\n{bin_name.upper()} ({len(samples)} samples):")
            print("-" * 40)

            bin_results = {
                "references": [],
                "hypotheses": [],
                "exact_matches": [],
                "ters": [],
                "word_overlaps": [],
                "length_ratios": [],
                "samples": []
            }

            for i, (src, ref) in enumerate(samples):
                # Translate
                hyp = self.translate(src, method=method)

                # Calculate metrics
                exact_match = calculate_exact_match(ref, hyp)
                ter = calculate_ter(ref, hyp)
                word_overlap = calculate_word_overlap(ref, hyp)
                length_ratio = calculate_length_ratio(ref, hyp)

                # Store results
                bin_results["references"].append(ref)
                bin_results["hypotheses"].append(hyp)
                bin_results["exact_matches"].append(exact_match)
                bin_results["ters"].append(ter)
                bin_results["word_overlaps"].append(word_overlap)
                bin_results["length_ratios"].append(length_ratio)

                # Store first few samples
                if i < 3:
                    bin_results["samples"].append({
                        "source": src[:100] + "..." if len(src) > 100 else src,
                        "reference": ref[:100] + "..." if len(ref) > 100 else ref,
                        "hypothesis": hyp[:100] + "..." if len(hyp) > 100 else hyp,
                        "exact_match": exact_match,
                        "ter": ter,
                        "word_overlap": word_overlap,
                        "length_ratio": length_ratio
                    })

                # Print first few translations
                if i < 3:
                    print(f"  Source: {src[:80]}...")
                    print(f"  Reference: {ref[:80]}...")
                    print(f"  Hypothesis: {hyp[:80]}...")
                    print(f"  Exact: {exact_match}, TER: {ter:.3f}, "
                          f"Overlap: {word_overlap:.3f}, Len ratio: {length_ratio:.2f}")
                    print()

            # Calculate aggregate metrics
            if bin_results["references"]:
                exact_match_rate = sum(bin_results["exact_matches"]) / len(bin_results["exact_matches"])
                avg_ter = sum(bin_results["ters"]) / len(bin_results["ters"])
                avg_overlap = sum(bin_results["word_overlaps"]) / len(bin_results["word_overlaps"])
                avg_length_ratio = sum(bin_results["length_ratios"]) / len(bin_results["length_ratios"])

                print(f"  Summary: Exact match: {exact_match_rate:.1%}, "
                      f"Avg TER: {avg_ter:.3f}, Avg overlap: {avg_overlap:.3f}, "
                      f"Avg length ratio: {avg_length_ratio:.2f}")

                bin_results["summary"] = {
                    "exact_match_rate": exact_match_rate,
                    "avg_ter": avg_ter,
                    "avg_word_overlap": avg_overlap,
                    "avg_length_ratio": avg_length_ratio
                }

            results_by_bin[bin_name] = bin_results

            # Add to overall lists
            all_references.extend(bin_results["references"])
            all_hypotheses.extend(bin_results["hypotheses"])

        # Calculate overall BLEU
        overall_bleu = 0.0
        if all_references and all_hypotheses:
            overall_bleu = calculate_bleu(all_references, all_hypotheses)

        # Calculate overall metrics
        all_exact_matches = []
        all_ters = []
        all_overlaps = []
        all_length_ratios = []

        for bin_results in results_by_bin.values():
            all_exact_matches.extend(bin_results["exact_matches"])
            all_ters.extend(bin_results["ters"])
            all_overlaps.extend(bin_results["word_overlaps"])
            all_length_ratios.extend(bin_results["length_ratios"])

        if all_exact_matches:
            overall_exact_match = sum(all_exact_matches) / len(all_exact_matches)
            overall_avg_ter = sum(all_ters) / len(all_ters)
            overall_avg_overlap = sum(all_overlaps) / len(all_overlaps)
            overall_avg_length_ratio = sum(all_length_ratios) / len(all_length_ratios)
        else:
            overall_exact_match = overall_avg_ter = overall_avg_overlap = overall_avg_length_ratio = 0.0

        overall_results = {
            "method": method,
            "bleu_score": overall_bleu,
            "exact_match_rate": overall_exact_match,
            "avg_ter": overall_avg_ter,
            "avg_word_overlap": overall_avg_overlap,
            "avg_length_ratio": overall_avg_length_ratio,
            "num_cases": len(all_references),
            "by_bin": results_by_bin
        }

        print("\n" + "=" * 60)
        print(f"OVERALL RESULTS ({method}):")
        print(f"  BLEU: {overall_bleu:.2f}")
        print(f"  Exact match rate: {overall_exact_match:.1%}")
        print(f"  Average TER: {overall_avg_ter:.3f} (lower is better)")
        print(f"  Average word overlap: {overall_avg_overlap:.3f}")
        print(f"  Average length ratio: {overall_avg_length_ratio:.2f}")
        print("=" * 60)

        return overall_results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate quantitative metrics by sampling from training corpus"
    )
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--use-cleaned", action="store_true", default=True,
                       help="Use cleaned data (same as training)")
    parser.add_argument("--src-corpus", type=str, default="europarl-v7.de-en.en",
                       help="Source (English) corpus file (used when --no-cleaned)")
    parser.add_argument("--tgt-corpus", type=str, default="europarl-v7.de-en.de",
                       help="Target (German) corpus file (used when --no-cleaned)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                       help="Which split to sample from (train or val)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples to load from corpus (same as training, default: all)")
    parser.add_argument("--samples", type=int, default=20,
                       help="Number of samples to evaluate")
    parser.add_argument("--max-bin", type=int, default=54,
                       help="Maximum token length bin")
    parser.add_argument("--methods", type=str, default="greedy,beam4,beam8",
                       help="Comma-separated list of decoding methods")
    parser.add_argument("--output", type=str, default="evaluation_results/quantitative_metrics_v2.json",
                       help="Output JSON file path")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Create evaluator
    evaluator = QuantitativeEvaluatorV2(checkpoint_path=args.checkpoint, device=args.device)

    # Load corpus
    data_dir = Path(__file__).parent.parent

    if args.use_cleaned:
        # Use cleaned data (same as training)
        src_corpus_path = data_dir / "models_enhanced/src_text_cleaned.txt"
        tgt_corpus_path = data_dir / "models_enhanced/tgt_text_cleaned.txt"
        print("\nUsing cleaned data (same as training)")
    else:
        src_corpus_path = data_dir / args.src_corpus
        tgt_corpus_path = data_dir / args.tgt_corpus
        print("\nUsing raw corpus")

    print(f"\nLoading corpus from:")
    print(f"  Source: {src_corpus_path}")
    print(f"  Target: {tgt_corpus_path}")

    # Load corpus with max_samples limit (same as training: 200000)
    from src.data.dataset import ParallelDataset
    import numpy as np

    max_samples = args.max_samples if args.max_samples else config.max_train_samples
    dataset = ParallelDataset(
        src_corpus_path,
        tgt_corpus_path,
        max_samples=max_samples,
    )
    print(f"Loaded {len(dataset)} sentence pairs (max_samples={max_samples})")

    # Split into train/val using same method as training
    train_split = config.train_split  # 0.99
    split_seed = 42  # same as training

    np.random.seed(split_seed)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    split_point = int(len(dataset) * train_split)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    if args.split == "val":
        selected_indices = val_indices
        split_name = "validation"
    else:
        selected_indices = train_indices
        split_name = "training"

    # Extract selected sentences
    corpus = [(dataset.src_lines[i], dataset.tgt_lines[i]) for i in selected_indices]
    print(f"Using {split_name} split: {len(corpus)} sentence pairs")

    # Define length bins
    length_bins = [
        (15, "short"),      # 1-15 tokens
        (30, "medium"),     # 16-30 tokens
        (args.max_bin, "long")  # 31-max tokens
    ]

    # Sample by length
    print(f"\nSampling {args.samples} sentences per length bin...")
    sampled_data = sample_by_length(
        corpus,
        evaluator.src_tokenizer,
        args.samples,
        length_bins
    )

    total_samples = sum(len(samples) for samples in sampled_data.values())
    print(f"Total samples: {total_samples}")

    # Evaluate each method
    methods = [m.strip() for m in args.methods.split(",")]
    all_results = {}

    for method in methods:
        print(f"\n{'='*80}")
        print(f"EVALUATING METHOD: {method}")
        print('='*80)

        results = evaluator.evaluate_sampled_corpus(sampled_data, method=method)
        all_results[method] = results

    # Save results
    output_path = data_dir / args.output
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "device": args.device,
            "seed": args.seed,
            "total_samples": total_samples,
            "samples_per_bin": args.samples,
            "max_bin": args.max_bin,
            "corpus_info": {
                "src": args.src_corpus,
                "tgt": args.tgt_corpus,
                "total_pairs": len(corpus)
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Print comparison table
    print("\n" + "="*80)
    print("METHOD COMPARISON")
    print("="*80)
    print(f"{'Method':<10} {'BLEU':<8} {'Exact Match':<12} {'Avg TER':<10} {'Avg Overlap':<12} {'Len Ratio':<10}")
    print("-"*80)

    for method, results in all_results.items():
        print(f"{method:<10} {results['bleu_score']:<8.2f} {results['exact_match_rate']:<12.1%} "
              f"{results['avg_ter']:<10.3f} {results['avg_word_overlap']:<12.3f} "
              f"{results['avg_length_ratio']:<10.2f}")

    print("="*80)

    # Print breakdown by length bin
    print("\nBREAKDOWN BY LENGTH BIN:")
    print("-"*80)
    for method, results in all_results.items():
        print(f"\n{method}:")
        for bin_name, bin_results in results.get("by_bin", {}).items():
            summary = bin_results.get("summary", {})
            print(f"  {bin_name}: BLEU=---, TER={summary.get('avg_ter', 0):.3f}, "
                  f"Overlap={summary.get('avg_word_overlap', 0):.3f}, "
                  f"Exact={summary.get('exact_match_rate', 0):.1%}")

    print("\nNote: TER (Translation Edit Rate) - lower is better (0.0 is perfect)")
    print("      BLEU - higher is better (0-100 scale)")
    print("      Word overlap - higher is better (Jaccard similarity)")


if __name__ == "__main__":
    main()
