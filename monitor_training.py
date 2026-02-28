#!/usr/bin/env python3
"""
训练过程监控脚本。
在训练过程中定期运行此脚本来检查模型进展。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import Evaluator

def check_training_progress():
    """检查训练进展"""
    print("=" * 60)
    print("训练进展检查")
    print("=" * 60)

    # 检查checkpoint是否存在
    checkpoint_path = Path(__file__).parent / "models" / "best_model.pt"
    if not checkpoint_path.exists():
        print("尚未生成checkpoint，请继续训练...")
        print("建议至少训练1000步后再进行检查")
        return

    # 加载tokenizer
    data_dir = Path(__file__).parent
    src_tokenizer_path = data_dir / "models" / "src_tokenizer.model"
    tgt_tokenizer_path = data_dir / "models" / "tgt_tokenizer.model"

    if not src_tokenizer_path.exists() or not tgt_tokenizer_path.exists():
        print("tokenizer文件不存在，请先运行preprocess.py")
        return

    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    # 加载checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✓ 找到checkpoint: {checkpoint_path.name}")

        if 'step' in checkpoint:
            print(f"训练步数: {checkpoint['step']}")

        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            print(f"配置信息:")
            print(f"  d_model: {saved_config.d_model}")
            print(f"  max_len: {saved_config.max_len}")
            if hasattr(saved_config, 'learning_rate'):
                print(f"  原始learning_rate: {saved_config.learning_rate}")

        # 创建模型
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            model = Transformer(
                src_vocab_size=saved_config.src_vocab_size if hasattr(saved_config, 'src_vocab_size') else saved_config.vocab_size,
                tgt_vocab_size=saved_config.tgt_vocab_size if hasattr(saved_config, 'tgt_vocab_size') else saved_config.vocab_size,
                d_model=saved_config.d_model,
                n_layers=saved_config.n_layers,
                n_heads=saved_config.n_heads,
                d_ff=saved_config.d_ff,
                dropout=saved_config.dropout,
                max_len=saved_config.max_len,
            )
        else:
            # 使用默认配置
            model = Transformer(
                src_vocab_size=config.src_vocab_size,
                tgt_vocab_size=config.tgt_vocab_size,
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                max_len=config.max_len,
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ 模型加载成功")

        # 测试encoder输出多样性
        print("\n" + "=" * 60)
        print("Encoder输出测试")
        print("=" * 60)

        test_sentences = ["Hello world", "How are you", "This is a test"]
        encoder_outputs = []

        with torch.no_grad():
            for i, text in enumerate(test_sentences):
                src_tokens = src_tokenizer(text, add_bos=False, add_eos=True)
                src = torch.tensor([src_tokens], dtype=torch.long)
                src_mask = torch.ones(1, 1, 1, len(src_tokens), dtype=torch.bool)

                encoder_output = model.encode(src, src_mask)
                encoder_outputs.append(encoder_output)

                std = encoder_output.std().item()
                print(f"输入 '{text}':")
                print(f"  Token长度: {len(src_tokens)}")
                print(f"  Encoder输出标准差: {std:.6f}")

                if std < 0.1:
                    print(f"  ⚠️ 警告: 输出变化较小 (应大于0.5)")
                else:
                    print(f"  ✓ 输出变化正常")

            # 比较不同输入的输出差异
            if len(encoder_outputs) >= 2:
                diff1 = torch.abs(encoder_outputs[0][0, 0, :] - encoder_outputs[1][0, 0, :]).mean().item()
                diff2 = torch.abs(encoder_outputs[0][0, 0, :] - encoder_outputs[2][0, 0, :]).mean().item()
                print(f"\n不同输入间第一个token输出的平均差异:")
                print(f"  'Hello world' vs 'How are you': {diff1:.6f}")
                print(f"  'Hello world' vs 'This is a test': {diff2:.6f}")

                if diff1 < 0.01 or diff2 < 0.01:
                    print("  ⚠️ 警告: 不同输入产生相似输出")
                else:
                    print("  ✓ 不同输入产生不同输出")

        # 测试简单翻译
        print("\n" + "=" * 60)
        print("简单翻译测试")
        print("=" * 60)

        evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device='cpu')
        test_inputs = ["Hello", "Good morning", "Thank you"]

        outputs = []
        for text in test_inputs:
            translation = evaluator.translate(text, method="greedy")
            outputs.append(translation)
            print(f"'{text}' → '{translation[:50]}...'")

        # 检查输出是否相同
        if len(set(outputs)) == 1:
            print("\n⚠️ 警告: 所有输入产生相同的输出")
        elif len(set(outputs)) < len(outputs):
            print(f"\n⚠️ 警告: 部分输入产生相同的输出 ({len(set(outputs))}/{len(outputs)} 唯一)")
        else:
            print(f"\n✓ 所有输入产生不同的输出 ({len(set(outputs))}/{len(outputs)} 唯一)")

        # 检查权重统计
        print("\n" + "=" * 60)
        print("权重统计")
        print("=" * 60)

        layers_to_check = [
            'encoder.norm.weight',
            'encoder.norm.bias',
        ]

        for name in layers_to_check:
            if name in dict(model.named_parameters()):
                param = dict(model.named_parameters())[name]
                mean_val = param.data.mean().item()
                std_val = param.data.std().item()

                print(f"{name}:")
                print(f"  均值: {mean_val:.6f}")

                # LayerNorm特殊检查
                if 'norm.weight' in name:
                    if abs(mean_val - 1.0) > 0.5:  # 允许一定偏差
                        print(f"  ⚠️ 警告: 均值偏离1.0较大 (应为~1.0)")
                    else:
                        print(f"  ✓ 均值接近1.0 (正常)")
                elif 'norm.bias' in name:
                    if abs(mean_val) > 0.1:  # 允许一定偏差
                        print(f"  ⚠️ 警告: 均值偏离0.0较大 (应为~0.0)")
                    else:
                        print(f"  ✓ 均值接近0.0 (正常)")

    except Exception as e:
        print(f"检查过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_training_progress()
    print("\n" + "=" * 60)
    print("监控完成")
    print("=" * 60)
    print("\n建议:")
    print("1. 如果encoder输出标准差>0.5且不同输入输出不同 → 训练正常")
    print("2. 如果输出相同或标准差<0.1 → 可能需要调整训练参数")
    print("3. 继续训练直到loss稳定下降")