#!/usr/bin/env python3
"""
Calculate quantitative translation metrics (BLEU, TER, etc.) for Transformer model.
Option B of training optimization plan.
"""
import sys
import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Any
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import Evaluator, greedy_decode, beam_search_decode


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
    # Tokenize into words
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # Calculate edit distance
    distance = levenshtein_distance(ref_tokens, hyp_tokens)

    # TER = edits / reference_length
    ref_len = len(ref_tokens)
    if ref_len == 0:
        return 0.0 if distance == 0 else 1.0

    return distance / ref_len


def calculate_bleu(references: List[str], hypotheses: List[str]) -> float:
    """Calculate BLEU score (simplified implementation).

    Returns BLEU score (0-100 scale).
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


class QuantitativeEvaluator:
    """Quantitative metrics evaluator for translation model."""

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
        # Map method names for compatibility
        if method == "beam":
            method = "beam4"  # default beam size 4

        # Tokenize (add EOS token as in the original evaluator)
        src_tokens = self.src_tokenizer.encode(src_text, add_bos=False, add_eos=True)
        src = torch.tensor([src_tokens], dtype=torch.long, device=self.device)
        src_mask = torch.ones(1, 1, 1, len(src_tokens), dtype=torch.bool, device=self.device)

        # Translate
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

    def evaluate_test_cases(self, test_cases: Dict[str, List[Tuple[str, str]]],
                           method: str = "greedy") -> Dict[str, Any]:
        """Evaluate on test cases and calculate metrics."""
        print(f"\nEvaluating with method: {method}")
        print("=" * 60)

        all_references = []
        all_hypotheses = []
        results_by_difficulty = {}

        for difficulty, cases in test_cases.items():
            print(f"\n{difficulty.upper()} level ({len(cases)} cases):")
            print("-" * 40)

            difficulty_results = {
                "references": [],
                "hypotheses": [],
                "exact_matches": [],
                "ters": [],
                "word_overlaps": [],
                "length_ratios": [],
                "samples": []
            }

            for i, (src, ref) in enumerate(cases):
                # Translate
                hyp = self.translate(src, method=method)

                # Calculate metrics
                exact_match = calculate_exact_match(ref, hyp)
                ter = calculate_ter(ref, hyp)
                word_overlap = calculate_word_overlap(ref, hyp)
                length_ratio = calculate_length_ratio(ref, hyp)

                # Store results
                difficulty_results["references"].append(ref)
                difficulty_results["hypotheses"].append(hyp)
                difficulty_results["exact_matches"].append(exact_match)
                difficulty_results["ters"].append(ter)
                difficulty_results["word_overlaps"].append(word_overlap)
                difficulty_results["length_ratios"].append(length_ratio)

                # Store first few samples for inspection
                if i < 3:
                    difficulty_results["samples"].append({
                        "source": src,
                        "reference": ref,
                        "hypothesis": hyp,
                        "exact_match": exact_match,
                        "ter": ter,
                        "word_overlap": word_overlap,
                        "length_ratio": length_ratio
                    })

                # Print first few translations
                if i < 3:
                    print(f"  Source: {src}")
                    print(f"  Reference: {ref}")
                    print(f"  Hypothesis: {hyp}")
                    print(f"  Exact match: {exact_match}, TER: {ter:.3f}, "
                          f"Overlap: {word_overlap:.3f}, Length ratio: {length_ratio:.2f}")
                    print()

            # Calculate aggregate metrics for this difficulty
            if difficulty_results["references"]:
                exact_match_rate = sum(difficulty_results["exact_matches"]) / len(difficulty_results["exact_matches"])
                avg_ter = sum(difficulty_results["ters"]) / len(difficulty_results["ters"])
                avg_overlap = sum(difficulty_results["word_overlaps"]) / len(difficulty_results["word_overlaps"])
                avg_length_ratio = sum(difficulty_results["length_ratios"]) / len(difficulty_results["length_ratios"])

                print(f"  Summary: Exact match: {exact_match_rate:.1%}, "
                      f"Avg TER: {avg_ter:.3f}, Avg overlap: {avg_overlap:.3f}, "
                      f"Avg length ratio: {avg_length_ratio:.2f}")

                difficulty_results["summary"] = {
                    "exact_match_rate": exact_match_rate,
                    "avg_ter": avg_ter,
                    "avg_word_overlap": avg_overlap,
                    "avg_length_ratio": avg_length_ratio
                }

            results_by_difficulty[difficulty] = difficulty_results

            # Add to overall lists for BLEU calculation
            all_references.extend(difficulty_results["references"])
            all_hypotheses.extend(difficulty_results["hypotheses"])

        # Calculate overall BLEU
        overall_bleu = 0.0
        if all_references and all_hypotheses:
            overall_bleu = calculate_bleu(all_references, all_hypotheses)

        # Calculate overall metrics
        all_exact_matches = []
        all_ters = []
        all_overlaps = []
        all_length_ratios = []

        for diff_results in results_by_difficulty.values():
            all_exact_matches.extend(diff_results["exact_matches"])
            all_ters.extend(diff_results["ters"])
            all_overlaps.extend(diff_results["word_overlaps"])
            all_length_ratios.extend(diff_results["length_ratios"])

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
            "by_difficulty": results_by_difficulty
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

    def get_test_cases(self) -> Dict[str, List[Tuple[str, str]]]:
        """Get test cases (same as in evaluate_translation_comprehensive.py)."""
        # Basic vocabulary (problematic cases from previous evaluation)
        basic_cases = [
            ("Hello", "Hallo"),
            ("Good morning", "Guten Morgen"),
            ("Thank you", "Danke"),
            ("How are you?", "Wie geht es dir?"),
            ("I am fine", "Mir geht es gut"),
            ("Goodbye", "Auf Wiedersehen"),
            ("Please", "Bitte"),
            ("Sorry", "Entschuldigung"),
            ("Yes", "Ja"),
            ("No", "Nein"),
        ]

        # Simple sentences (should work well)
        simple_cases = [
            ("This is a test", "Das ist ein Test"),
            ("The sky is blue", "Der Himmel ist blau"),
            ("I like apples", "Ich mag Äpfel"),
            ("She reads a book", "Sie liest ein Buch"),
            ("We are learning", "Wir lernen"),
            ("The cat is sleeping", "Die Katze schläft"),
            ("I have a car", "Ich habe ein Auto"),
            ("Water is important", "Wasser ist wichtig"),
            ("The sun is shining", "Die Sonne scheint"),
            ("Time flies quickly", "Die Zeit vergeht schnell"),
        ]

        # Moderate complexity
        moderate_cases = [
            ("Can you help me please?", "Können Sie mir bitte helfen?"),
            ("What time is it now?", "Wie spät ist es jetzt?"),
            ("Where is the nearest station?", "Wo ist die nächste Station?"),
            ("I would like to order coffee", "Ich möchte Kaffee bestellen"),
            ("The weather is beautiful today", "Das Wetter ist heute schön"),
            ("We need to buy some groceries", "Wir müssen Lebensmittel einkaufen"),
            ("She is studying at the university", "Sie studiert an der Universität"),
            ("He works in a big company", "Er arbeitet in einer großen Firma"),
            ("They are planning a trip to Berlin", "Sie planen eine Reise nach Berlin"),
            ("The meeting starts at three o'clock", "Das Meeting beginnt um drei Uhr"),
        ]

        # Complex sentences (challenging cases)
        complex_cases = [
            ("Despite the heavy rain, we decided to continue our journey",
             "Trotz des starken Regens haben wir beschlossen, unsere Reise fortzusetzen"),
            ("The development of artificial intelligence has accelerated significantly in recent years",
             "Die Entwicklung der künstlichen Intelligenz hat in den letzten Jahren erheblich beschleunigt"),
            ("If you have any questions, please don't hesitate to contact our customer service",
             "Wenn Sie Fragen haben, zögern Sie bitte nicht, unseren Kundendienst zu kontaktieren"),
            ("The conference will cover topics ranging from machine learning to natural language processing",
             "Die Konferenz wird Themen behandeln, die von maschinellem Lernen bis zur natürlichen Sprachverarbeitung reichen"),
            ("To achieve sustainable development, we must balance economic growth with environmental protection",
             "Um nachhaltige Entwicklung zu erreichen, müssen wir wirtschaftliches Wachstum mit Umweltschutz in Einklang bringen"),
        ]

        # Abstract/conceptual cases
        abstract_cases = [
            ("The concept of freedom varies across different cultures",
             "Das Konzept der Freiheit variiert in verschiedenen Kulturen"),
            ("Love and hate are two sides of the same coin",
             "Liebe und Hass sind zwei Seiten derselben Medaille"),
            ("The pursuit of happiness is a fundamental human right",
             "Das Streben nach Glück ist ein grundlegendes Menschenrecht"),
            ("Time is a relative concept in physics",
             "Zeit ist ein relatives Konzept in der Physik"),
            ("The meaning of life is a philosophical question",
             "Die Bedeutung des Lebens ist eine philosophische Frage"),
        ]

        return {
            "basic": basic_cases,
            "simple": simple_cases,
            "moderate": moderate_cases,
            "complex": complex_cases,
            "abstract": abstract_cases,
        }


def main():
    parser = argparse.ArgumentParser(description="Calculate quantitative translation metrics")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--methods", type=str, default="greedy,beam4,beam8",
                       help="Comma-separated list of decoding methods to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results/quantitative_metrics.json",
                       help="Output JSON file path")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu, cuda, mps)")

    args = parser.parse_args()

    # Create evaluator
    evaluator = QuantitativeEvaluator(checkpoint_path=args.checkpoint, device=args.device)

    # Get test cases
    test_cases = evaluator.get_test_cases()
    total_cases = sum(len(cases) for cases in test_cases.values())
    print(f"Total test cases: {total_cases}")

    # Evaluate each method
    methods = [m.strip() for m in args.methods.split(",")]
    all_results = {}

    for method in methods:
        print(f"\n{'='*80}")
        print(f"EVALUATING METHOD: {method}")
        print('='*80)

        results = evaluator.evaluate_test_cases(test_cases, method=method)
        all_results[method] = results

    # Save results
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "device": args.device,
            "total_cases": total_cases,
            "results": all_results,
            "test_cases": test_cases
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
    print("Note: TER (Translation Edit Rate) - lower is better (0.0 is perfect)")
    print("      BLEU - higher is better (0-100 scale)")
    print("      Exact match - higher is better")
    print("      Word overlap - higher is better (Jaccard similarity)")


if __name__ == "__main__":
    main()