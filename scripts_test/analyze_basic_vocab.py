#!/usr/bin/env python3
"""
专项分析基础词汇翻译质量。
评估318,262步模型在基础词汇上的表现，识别问题词汇。
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import Evaluator


@dataclass
class BasicVocabResult:
    """单个基础词汇的翻译结果"""
    english: str
    german_reference: str
    greedy_translation: str
    beam4_translation: str
    beam8_translation: str
    greedy_exact_match: bool
    beam4_exact_match: bool
    beam8_exact_match: bool
    greedy_semantic_score: float  # 0-1, 语义相似度
    beam4_semantic_score: float
    beam8_semantic_score: float
    greedy_keyword_match: bool  # 是否包含关键词
    beam4_keyword_match: bool
    beam8_keyword_match: bool


@dataclass
class BasicVocabAnalysis:
    """基础词汇分析结果"""
    total_vocab: int
    greedy_exact_matches: int
    beam4_exact_matches: int
    beam8_exact_matches: int
    greedy_semantic_avg: float
    beam4_semantic_avg: float
    beam8_semantic_avg: float
    greedy_keyword_matches: int
    beam4_keyword_matches: int
    beam8_keyword_matches: int
    problematic_vocab: List[Dict]  # 问题词汇列表
    best_performing_method: str
    timestamp: str


class BasicVocabAnalyzer:
    """基础词汇分析器"""

    def __init__(self, checkpoint_path: str = "models/best_model.pt", device: str = "mps"):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir = Path(__file__).parent.parent

        print(f"基础词汇专项分析")
        print(f"模型: {self.checkpoint_path.name}")
        print(f"设备: {self.device}")
        print("=" * 60)

        self._load_model_and_tokenizers()
        self._load_basic_vocab()

    def _load_model_and_tokenizers(self):
        """加载模型和分词器"""
        # 使用增强分词器
        src_tokenizer_path = self.data_dir / "models_enhanced" / "src_tokenizer_final.model"
        tgt_tokenizer_path = self.data_dir / "models_enhanced" / "tgt_tokenizer_final.model"

        self.src_tokenizer, self.tgt_tokenizer = load_tokenizers(
            str(src_tokenizer_path), str(tgt_tokenizer_path)
        )

        # 加载检查点
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # 获取词汇表大小
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

        # 创建模型
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

        # 创建评估器
        self.evaluator = Evaluator(self.model, self.src_tokenizer, self.tgt_tokenizer, self.device)

        print(f"模型加载完成: {sum(p.numel() for p in self.model.parameters()):,} 参数")

    def _load_basic_vocab(self):
        """加载基础词汇表"""
        basic_dir = self.data_dir / "data_basic"
        english_file = basic_dir / "basic_vocab.en"
        german_file = basic_dir / "basic_vocab.de"

        with open(english_file, 'r', encoding='utf-8') as f:
            english_vocab = [line.strip() for line in f if line.strip()]

        with open(german_file, 'r', encoding='utf-8') as f:
            german_vocab = [line.strip() for line in f if line.strip()]

        if len(english_vocab) != len(german_vocab):
            print(f"警告: 词汇表长度不匹配 (英文: {len(english_vocab)}, 德文: {len(german_vocab)})")
            min_len = min(len(english_vocab), len(german_vocab))
            english_vocab = english_vocab[:min_len]
            german_vocab = german_vocab[:min_len]

        self.basic_vocab = list(zip(english_vocab, german_vocab))
        print(f"加载基础词汇: {len(self.basic_vocab)} 对")

    def _calculate_semantic_score(self, translation: str, reference: str) -> float:
        """计算语义相似度分数 (简化版)"""
        translation_lower = translation.lower()
        reference_lower = reference.lower()

        # 完全匹配
        if translation_lower == reference_lower:
            return 1.0

        # 单词重叠
        trans_words = set(translation_lower.split())
        ref_words = set(reference_lower.split())

        if not trans_words or not ref_words:
            return 0.0

        # Jaccard相似度
        intersection = len(trans_words.intersection(ref_words))
        union = len(trans_words.union(ref_words))

        return intersection / union if union > 0 else 0.0

    def _check_keyword_match(self, translation: str, reference: str) -> bool:
        """检查关键词匹配"""
        # 基础词汇的期望关键词映射
        keyword_mapping = {
            "hello": ["hallo"],
            "hi": ["hallo"],
            "good morning": ["guten", "morgen"],
            "good afternoon": ["guten", "tag"],
            "good evening": ["guten", "abend"],
            "good night": ["gute", "nacht"],
            "thank you": ["danke"],
            "thanks": ["danke"],
            "you're welcome": ["bitte"],
            "please": ["bitte"],
            "excuse me": ["entschuldigung"],
            "sorry": ["entschuldigung"],
            "yes": ["ja"],
            "no": ["nein"],
            "maybe": ["vielleicht"],
            "ok": ["ok"],
            "goodbye": ["auf", "wiedersehen"],
            "bye": ["tschüss"],
            "see you later": ["bis", "später"],
            "how are you": ["wie", "geht"],
        }

        translation_lower = translation.lower()

        for english, expected_keywords in keyword_mapping.items():
            if english in reference.lower():
                # 检查翻译中是否包含任一关键词
                for keyword in expected_keywords:
                    if keyword in translation_lower:
                        return True

        return False

    def analyze_vocab(self) -> BasicVocabAnalysis:
        """分析基础词汇翻译质量"""
        print("\n开始基础词汇分析...")
        print("-" * 60)

        results = []

        for i, (english, german) in enumerate(self.basic_vocab, 1):
            print(f"处理词汇 {i}/{len(self.basic_vocab)}: '{english}' -> '{german}'")

            try:
                # 使用不同解码策略翻译
                greedy_trans = self.evaluator.translate(english, method="greedy")
                beam4_trans = self.evaluator.translate(english, method="beam", beam_size=4)
                beam8_trans = self.evaluator.translate(english, method="beam", beam_size=8)

                # 计算匹配分数
                greedy_exact = greedy_trans.lower() == german.lower()
                beam4_exact = beam4_trans.lower() == german.lower()
                beam8_exact = beam8_trans.lower() == german.lower()

                greedy_semantic = self._calculate_semantic_score(greedy_trans, german)
                beam4_semantic = self._calculate_semantic_score(beam4_trans, german)
                beam8_semantic = self._calculate_semantic_score(beam8_trans, german)

                greedy_keyword = self._check_keyword_match(greedy_trans, german)
                beam4_keyword = self._check_keyword_match(beam4_trans, german)
                beam8_keyword = self._check_keyword_match(beam8_trans, german)

                result = BasicVocabResult(
                    english=english,
                    german_reference=german,
                    greedy_translation=greedy_trans,
                    beam4_translation=beam4_trans,
                    beam8_translation=beam8_trans,
                    greedy_exact_match=greedy_exact,
                    beam4_exact_match=beam4_exact,
                    beam8_exact_match=beam8_exact,
                    greedy_semantic_score=greedy_semantic,
                    beam4_semantic_score=beam4_semantic,
                    beam8_semantic_score=beam8_semantic,
                    greedy_keyword_match=greedy_keyword,
                    beam4_keyword_match=beam4_keyword,
                    beam8_keyword_match=beam8_keyword
                )

                results.append(result)

                if not (greedy_exact or beam4_exact or beam8_exact):
                    print(f"  ⚠️  无完全匹配")

            except Exception as e:
                print(f"  ❌ 翻译出错: {e}")
                # 添加默认结果
                result = BasicVocabResult(
                    english=english,
                    german_reference=german,
                    greedy_translation="[ERROR]",
                    beam4_translation="[ERROR]",
                    beam8_translation="[ERROR]",
                    greedy_exact_match=False,
                    beam4_exact_match=False,
                    beam8_exact_match=False,
                    greedy_semantic_score=0.0,
                    beam4_semantic_score=0.0,
                    beam8_semantic_score=0.0,
                    greedy_keyword_match=False,
                    beam4_keyword_match=False,
                    beam8_keyword_match=False
                )
                results.append(result)

        # 计算总体统计
        total = len(results)
        greedy_exact = sum(1 for r in results if r.greedy_exact_match)
        beam4_exact = sum(1 for r in results if r.beam4_exact_match)
        beam8_exact = sum(1 for r in results if r.beam8_exact_match)

        greedy_semantic_avg = np.mean([r.greedy_semantic_score for r in results])
        beam4_semantic_avg = np.mean([r.beam4_semantic_score for r in results])
        beam8_semantic_avg = np.mean([r.beam8_semantic_score for r in results])

        greedy_keyword = sum(1 for r in results if r.greedy_keyword_match)
        beam4_keyword = sum(1 for r in results if r.beam4_keyword_match)
        beam8_keyword = sum(1 for r in results if r.beam8_keyword_match)

        # 识别问题词汇 (语义分数 < 0.3)
        problematic = []
        for r in results:
            max_score = max(r.greedy_semantic_score, r.beam4_semantic_score, r.beam8_semantic_score)
            if max_score < 0.3:
                problematic.append({
                    'english': r.english,
                    'german_reference': r.german_reference,
                    'best_translation': r.beam4_translation if r.beam4_semantic_score >= r.greedy_semantic_score else r.greedy_translation,
                    'best_score': max_score,
                    'best_method': 'beam4' if r.beam4_semantic_score >= r.greedy_semantic_score else 'greedy'
                })

        # 确定最佳解码策略
        if beam4_exact > greedy_exact and beam4_exact > beam8_exact:
            best_method = "beam4"
        elif beam8_exact > greedy_exact and beam8_exact > beam4_exact:
            best_method = "beam8"
        else:
            best_method = "greedy"

        analysis = BasicVocabAnalysis(
            total_vocab=total,
            greedy_exact_matches=greedy_exact,
            beam4_exact_matches=beam4_exact,
            beam8_exact_matches=beam8_exact,
            greedy_semantic_avg=greedy_semantic_avg,
            beam4_semantic_avg=beam4_semantic_avg,
            beam8_semantic_avg=beam8_semantic_avg,
            greedy_keyword_matches=greedy_keyword,
            beam4_keyword_matches=beam4_keyword,
            beam8_keyword_matches=beam8_keyword,
            problematic_vocab=problematic,
            best_performing_method=best_method,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        return analysis, results

    def generate_report(self, analysis: BasicVocabAnalysis, results: List[BasicVocabResult]):
        """生成分析报告"""
        print("\n" + "=" * 60)
        print("基础词汇分析报告")
        print("=" * 60)

        # 总体统计
        print(f"\n📊 总体统计 (共 {analysis.total_vocab} 个基础词汇):")
        print("-" * 40)
        print(f"解码策略    完全匹配  语义相似度  关键词匹配")
        print(f"Greedy      {analysis.greedy_exact_matches:2d}/{analysis.total_vocab:2d}      {analysis.greedy_semantic_avg:.3f}       {analysis.greedy_keyword_matches:2d}/{analysis.total_vocab:2d}")
        print(f"Beam-4      {analysis.beam4_exact_matches:2d}/{analysis.total_vocab:2d}      {analysis.beam4_semantic_avg:.3f}       {analysis.beam4_keyword_matches:2d}/{analysis.total_vocab:2d}")
        print(f"Beam-8      {analysis.beam8_exact_matches:2d}/{analysis.total_vocab:2d}      {analysis.beam8_semantic_avg:.3f}       {analysis.beam8_keyword_matches:2d}/{analysis.total_vocab:2d}")

        # 最佳策略
        print(f"\n🏆 最佳解码策略: {analysis.best_performing_method}")

        # 问题词汇
        if analysis.problematic_vocab:
            print(f"\n⚠️  问题词汇 ({len(analysis.problematic_vocab)} 个，语义相似度 < 0.3):")
            print("-" * 60)
            for i, problem in enumerate(analysis.problematic_vocab, 1):
                print(f"{i:2d}. '{problem['english']}' -> 参考: '{problem['german_reference']}'")
                print(f"    最佳翻译 ({problem['best_method']}): '{problem['best_translation']}'")
                print(f"    相似度: {problem['best_score']:.3f}")

        # 详细结果示例
        print(f"\n📝 详细结果示例 (前10个词汇):")
        print("-" * 60)
        for i, result in enumerate(results[:10], 1):
            print(f"{i:2d}. '{result.english}' -> 参考: '{result.german_reference}'")
            print(f"    Greedy: '{result.greedy_translation}' (准确: {result.greedy_exact_match}, 语义: {result.greedy_semantic_score:.3f})")
            print(f"    Beam-4: '{result.beam4_translation}' (准确: {result.beam4_exact_match}, 语义: {result.beam4_semantic_score:.3f})")
            print(f"    Beam-8: '{result.beam8_translation}' (准确: {result.beam8_exact_match}, 语义: {result.beam8_semantic_score:.3f})")

        # 建议
        print(f"\n💡 改进建议:")
        print("-" * 40)
        exact_match_rate = max(analysis.greedy_exact_matches, analysis.beam4_exact_matches, analysis.beam8_exact_matches) / analysis.total_vocab

        if exact_match_rate >= 0.8:
            print("✅ 基础词汇翻译准确率良好 (>80%)，保持当前训练策略")
        elif exact_match_rate >= 0.5:
            print("⚠️  基础词汇翻译准确率中等 (50-80%)，建议:")
            print("    1. 增加基础词汇在训练数据中的出现频率")
            print("    2. 微调模型专门针对基础词汇")
            print("    3. 检查分词器对基础词汇的处理")
        else:
            print("❌ 基础词汇翻译准确率较低 (<50%)，急需改进:")
            print("    1. 创建基础词汇专项训练数据集")
            print("    2. 调整模型架构或训练策略")
            print("    3. 验证分词器词汇表是否包含基础词汇")
            print("    4. 考虑从头训练新的分词器")

        if analysis.problematic_vocab:
            print(f"\n🎯 针对问题词汇的建议:")
            for problem in analysis.problematic_vocab[:5]:  # 显示前5个最严重的问题
                print(f"    - '{problem['english']}': 确保训练数据中包含此词汇的平行语料")

        # 保存结果到文件
        output_dir = Path(__file__).parent.parent / "evaluation_results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_dir / f"basic_vocab_analysis_{timestamp}.json"

        report_data = {
            "analysis": asdict(analysis),
            "results": [asdict(r) for r in results],
            "summary": {
                "total_vocab": analysis.total_vocab,
                "best_method": analysis.best_performing_method,
                "exact_match_rate": exact_match_rate,
                "problematic_count": len(analysis.problematic_vocab),
                "timestamp": analysis.timestamp
            }
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"\n📁 详细报告已保存: {json_file}")
        print("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="基础词汇翻译质量分析")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt",
                       help="模型检查点路径")
    parser.add_argument("--device", type=str, default="mps",
                       help="设备 (cpu, mps, cuda)")

    args = parser.parse_args()

    # 检查模型文件
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误: 模型文件不存在: {checkpoint_path}")
        return

    # 运行分析
    analyzer = BasicVocabAnalyzer(args.checkpoint, args.device)
    analysis, results = analyzer.analyze_vocab()
    analyzer.generate_report(analysis, results)


if __name__ == "__main__":
    main()