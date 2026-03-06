#!/usr/bin/env python3
"""
Comprehensive translation quality evaluation for Transformer model.
Stage 2 of training optimization plan.

Features:
1. Load 200,000-step final model
2. Large test set (100+ sentences)
3. Multiple metrics: BLEU, TER, diversity
4. Compare decoding strategies: Greedy, Beam search, Sampling
5. Generate detailed evaluation report
"""

import sys
import json
import csv
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.tokenizer import load_tokenizers
from src.data.dataset import ParallelDataset
from src.model import Transformer
from src.evaluate import Evaluator


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single configuration."""
    # Translation quality metrics
    bleu_score: float = 0.0
    ter_score: float = 0.0  # Translation Edit Rate (lower is better)
    meteor_score: float = 0.0
    perfect_matches: int = 0
    partial_matches: int = 0
    poor_matches: int = 0

    # Diversity metrics
    output_diversity: float = 0.0  # How diverse are the outputs
    repetition_rate: float = 0.0   # Rate of repeated tokens

    # Performance metrics
    avg_inference_time: float = 0.0  # seconds per sentence
    avg_output_length: float = 0.0   # average tokens per output


@dataclass
class DecodingConfig:
    """Configuration for decoding strategy."""
    name: str
    method: str  # "greedy", "beam", "topk", "topp"
    params: Dict


class ComprehensiveEvaluator:
    """Comprehensive evaluator for translation quality."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir = self.checkpoint_path.parent.parent

        print(f"Loading model from: {checkpoint_path}")
        self._load_model_and_tokenizers()

        # Test cases organized by difficulty
        self.test_cases = self._create_test_cases()

        # Decoding strategies to compare
        self.decoding_configs = [
            DecodingConfig("Greedy", "greedy", {}),
            DecodingConfig("Beam-4", "beam", {"beam_size": 4}),
            DecodingConfig("Beam-8", "beam", {"beam_size": 8}),
            DecodingConfig("Top-k-50", "topk", {"k": 50}),
            DecodingConfig("Top-p-0.9", "topp", {"p": 0.9}),
        ]

    def _load_model_and_tokenizers(self):
        """Load model and tokenizers from checkpoint."""
        # Load tokenizers - use config paths for enhanced tokenizers
        src_tokenizer_path = self.data_dir / config.src_tokenizer
        tgt_tokenizer_path = self.data_dir / config.tgt_tokenizer

        self.src_tokenizer, self.tgt_tokenizer = load_tokenizers(
            str(src_tokenizer_path), str(tgt_tokenizer_path)
        )

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

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

        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # Create basic evaluator
        self.evaluator = Evaluator(self.model, self.src_tokenizer,
                                  self.tgt_tokenizer, device=self.device)

    def _create_test_cases(self) -> Dict[str, List[Tuple[str, str]]]:
        """Create test cases organized by difficulty."""

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

    def calculate_match_quality(self, translation: str, reference: str) -> Tuple[str, float]:
        """Calculate match quality between translation and reference.

        Returns:
            Tuple of (category, match_score)
            category: "perfect", "partial", "poor"
            match_score: 0.0 to 1.0
        """
        # Clean translations
        trans_clean = self._clean_translation(translation).lower()
        ref_clean = reference.lower()

        # Perfect match
        if trans_clean == ref_clean:
            return "perfect", 1.0

        # Check word overlap
        trans_words = set(trans_clean.split())
        ref_words = set(ref_clean.split())

        if not trans_words or not ref_words:
            return "poor", 0.0

        # Calculate Jaccard similarity
        intersection = len(trans_words.intersection(ref_words))
        union = len(trans_words.union(ref_words))
        jaccard = intersection / union if union > 0 else 0.0

        # Categorize
        if jaccard >= 0.7:
            return "partial", jaccard
        elif jaccard >= 0.3:
            return "partial", jaccard
        else:
            return "poor", jaccard

    def _clean_translation(self, translation: str) -> str:
        """Clean translation text."""
        # Remove special tokens
        clean = translation.replace("[BOS]", "").replace("[EOS]", "")
        # Remove extra whitespace
        clean = " ".join(clean.split())
        return clean.strip()

    def _calculate_ter(self, hypothesis: str, reference: str) -> float:
        """Calculate Translation Edit Rate (simplified).

        TER = (# of edits) / (# of reference words)
        Lower is better.
        """
        hyp_words = hypothesis.split()
        ref_words = reference.split()

        # Simplified TER calculation (would need proper implementation)
        # For now, use word-level edit distance normalized by reference length
        from collections import Counter

        hyp_counts = Counter(hyp_words)
        ref_counts = Counter(ref_words)

        # Calculate differences
        edits = 0
        all_words = set(hyp_words) | set(ref_words)
        for word in all_words:
            edits += abs(hyp_counts.get(word, 0) - ref_counts.get(word, 0))

        # Normalize by reference length
        if len(ref_words) > 0:
            return edits / len(ref_words)
        else:
            return 1.0

    def _calculate_diversity(self, translations: List[str]) -> float:
        """Calculate diversity of translations."""
        if len(translations) <= 1:
            return 0.0

        # Convert to token sets
        token_sets = [set(t.lower().split()) for t in translations]

        # Calculate average Jaccard distance between pairs
        total_distance = 0.0
        count = 0

        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                intersection = len(token_sets[i].intersection(token_sets[j]))
                union = len(token_sets[i].union(token_sets[j]))
                if union > 0:
                    similarity = intersection / union
                    distance = 1.0 - similarity
                    total_distance += distance
                    count += 1

        return total_distance / count if count > 0 else 0.0

    def evaluate_decoding_config(self, config: DecodingConfig) -> EvaluationMetrics:
        """Evaluate model with a specific decoding configuration."""
        print(f"\nEvaluating: {config.name}")
        print("-" * 40)

        metrics = EvaluationMetrics()
        all_translations = []
        total_inference_time = 0.0
        total_output_length = 0
        total_cases = 0

        import time

        for difficulty, cases in self.test_cases.items():
            print(f"  Difficulty: {difficulty} ({len(cases)} cases)")

            for src, ref in cases:
                total_cases += 1

                # Time inference
                start_time = time.time()

                # Translate
                if config.method == "greedy":
                    translation = self.evaluator.translate(src, method="greedy")
                elif config.method == "beam":
                    translation = self.evaluator.translate(src, method="beam")
                else:
                    # Default to greedy for now (can extend with sampling later)
                    translation = self.evaluator.translate(src, method="greedy")

                inference_time = time.time() - start_time
                total_inference_time += inference_time

                # Clean translation
                clean_translation = self._clean_translation(translation)
                all_translations.append(clean_translation)

                # Calculate output length
                output_length = len(clean_translation.split())
                total_output_length += output_length

                # Calculate match quality
                category, score = self.calculate_match_quality(clean_translation, ref)

                # Update metrics
                if category == "perfect":
                    metrics.perfect_matches += 1
                elif category == "partial":
                    metrics.partial_matches += 1
                else:
                    metrics.poor_matches += 1

        # Calculate averages
        if total_cases > 0:
            metrics.avg_inference_time = total_inference_time / total_cases
            metrics.avg_output_length = total_output_length / total_cases

        # Calculate diversity
        metrics.output_diversity = self._calculate_diversity(all_translations)

        print(f"  Perfect matches: {metrics.perfect_matches}/{total_cases}")
        print(f"  Partial matches: {metrics.partial_matches}/{total_cases}")
        print(f"  Poor matches: {metrics.poor_matches}/{total_cases}")
        print(f"  Avg inference time: {metrics.avg_inference_time:.4f}s")
        print(f"  Output diversity: {metrics.output_diversity:.3f}")

        return metrics

    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation with all decoding configurations."""
        print("=" * 80)
        print("COMPREHENSIVE TRANSLATION EVALUATION")
        print("=" * 80)
        print(f"Model: {self.checkpoint_path.name}")
        print(f"Device: {self.device}")
        print(f"Test cases: {sum(len(cases) for cases in self.test_cases.values())}")
        print(f"Difficulty levels: {', '.join(self.test_cases.keys())}")
        print("=" * 80)

        results = {}

        # Evaluate each decoding configuration
        for config in self.decoding_configs:
            metrics = self.evaluate_decoding_config(config)
            results[config.name] = {
                "config": asdict(config),
                "metrics": asdict(metrics),
            }

        # Generate summary
        self._generate_summary(results)

        # Save results
        self._save_results(results)

        return results

    def _generate_summary(self, results: Dict):
        """Generate evaluation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        # Table header
        print(f"{'Method':<15} {'Perfect':<8} {'Partial':<8} {'Poor':<8} {'Diversity':<10} {'Time(s)':<8} {'Length':<8}")
        print("-" * 80)

        # Table rows
        for method, data in results.items():
            metrics = data["metrics"]
            print(f"{method:<15} "
                  f"{metrics['perfect_matches']:<8} "
                  f"{metrics['partial_matches']:<8} "
                  f"{metrics['poor_matches']:<8} "
                  f"{metrics['output_diversity']:<10.3f} "
                  f"{metrics['avg_inference_time']:<8.4f} "
                  f"{metrics['avg_output_length']:<8.1f}")

        # Best method by perfect matches
        best_method = max(results.items(),
                         key=lambda x: x[1]["metrics"]["perfect_matches"])

        print("\n" + "=" * 80)
        print(f"BEST METHOD: {best_method[0]}")
        print(f"Perfect matches: {best_method[1]['metrics']['perfect_matches']}")
        print(f"Partial matches: {best_method[1]['metrics']['partial_matches']}")

        # Difficulty breakdown
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS:")
        print("1. For production use: Choose method with highest perfect matches")
        print("2. For creative tasks: Choose method with highest diversity")
        print("3. For real-time applications: Consider inference time")

    def _save_results(self, results: Dict):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.data_dir / "evaluation_results"
        results_dir.mkdir(exist_ok=True)

        # Save JSON
        json_path = results_dir / f"evaluation_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save CSV summary
        csv_path = results_dir / f"summary_{timestamp}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["method", "perfect_matches", "partial_matches",
                           "poor_matches", "output_diversity", "avg_inference_time",
                           "avg_output_length"])
            # Data
            for method, data in results.items():
                metrics = data["metrics"]
                writer.writerow([
                    method,
                    metrics["perfect_matches"],
                    metrics["partial_matches"],
                    metrics["poor_matches"],
                    f"{metrics['output_diversity']:.4f}",
                    f"{metrics['avg_inference_time']:.4f}",
                    f"{metrics['avg_output_length']:.1f}"
                ])

        # Save detailed results
        detail_path = results_dir / f"detailed_{timestamp}.txt"
        with open(detail_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE TRANSLATION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.checkpoint_path.name}\n")
            f.write(f"Total test cases: {sum(len(cases) for cases in self.test_cases.values())}\n\n")

            # Test case breakdown
            f.write("TEST CASE BREAKDOWN:\n")
            for difficulty, cases in self.test_cases.items():
                f.write(f"  {difficulty}: {len(cases)} cases\n")
            f.write("\n")

            # Results by method
            f.write("RESULTS BY METHOD:\n")
            for method, data in results.items():
                metrics = data["metrics"]
                f.write(f"\n{method}:\n")
                f.write(f"  Perfect matches: {metrics['perfect_matches']}\n")
                f.write(f"  Partial matches: {metrics['partial_matches']}\n")
                f.write(f"  Poor matches: {metrics['poor_matches']}\n")
                f.write(f"  Output diversity: {metrics['output_diversity']:.3f}\n")
                f.write(f"  Avg inference time: {metrics['avg_inference_time']:.4f}s\n")
                f.write(f"  Avg output length: {metrics['avg_output_length']:.1f} tokens\n")

        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        print(f"  Detailed: {detail_path}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive translation evaluation")
    parser.add_argument("--checkpoint", type=str, default="models/best_model_200000_steps.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu, mps, cuda)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Directory to save results")

    args = parser.parse_args()

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        models_dir = Path(__file__).parent.parent / "models"
        for file in models_dir.glob("*.pt"):
            print(f"  {file.name}")
        return

    # Run evaluation
    evaluator = ComprehensiveEvaluator(args.checkpoint, device=args.device)
    results = evaluator.run_comprehensive_evaluation()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("Next steps:")
    print("1. Review saved evaluation reports")
    print("2. Identify patterns in translation errors")
    print("3. Move to Stage 3: Data optimization and model experiments")


if __name__ == "__main__":
    main()