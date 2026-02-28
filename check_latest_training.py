#!/usr/bin/env python3
"""
检查最新训练成果的脚本。
"""
import sys
import os
from pathlib import Path
import torch
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluate import Evaluator

def analyze_training_logs():
    """分析训练日志"""
    print("=" * 80)
    print("训练日志分析")
    print("=" * 80)

    step_log_path = project_root / "models" / "logs" / "step_log.csv"
    epoch_log_path = project_root / "models" / "logs" / "epoch_log.csv"
    val_log_path = project_root / "models" / "logs" / "val_log.csv"

    # 分析 step_log.csv
    if step_log_path.exists():
        with open(step_log_path, 'r') as f:
            lines = f.readlines()

        if len(lines) > 1:
            # 获取最新记录
            last_line = lines[-1].strip()
            last_parts = last_line.split(',')
            if len(last_parts) >= 3:
                last_step = int(last_parts[0])
                last_loss = float(last_parts[1])
                last_lr = float(last_parts[2])
                print(f"最新训练步数: {last_step:,}")
                print(f"最新损失值: {last_loss:.4f}")
                print(f"最新学习率: {last_lr:.6f}")

            # 计算统计数据
            losses = []
            for line in lines[1:]:  # 跳过标题行
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    losses.append(float(parts[1]))

            if losses:
                print(f"\n损失统计 ({len(losses):,} 步记录):")
                print(f"  平均损失: {np.mean(losses):.4f}")
                print(f"  最小损失: {np.min(losses):.4f}")
                print(f"  最大损失: {np.max(losses):.4f}")
                print(f"  损失标准差: {np.std(losses):.4f}")

                # 分析最近1000步
                recent_losses = losses[-1000:] if len(losses) > 1000 else losses
                print(f"\n最近 {len(recent_losses)} 步统计:")
                print(f"  平均损失: {np.mean(recent_losses):.4f}")
                print(f"  最小损失: {np.min(recent_losses):.4f}")
                print(f"  最大损失: {np.max(recent_losses):.4f}")

                # 趋势分析
                if len(losses) > 100:
                    first_100 = np.mean(losses[:100])
                    last_100 = np.mean(losses[-100:])
                    trend = last_100 - first_100
                    print(f"\n趋势分析 (前100步 vs 后100步):")
                    print(f"  前100步平均: {first_100:.4f}")
                    print(f"  后100步平均: {last_100:.4f}")
                    print(f"  变化: {trend:+.4f} ({'下降' if trend < 0 else '上升'})")

    # 分析 epoch_log.csv
    if epoch_log_path.exists():
        with open(epoch_log_path, 'r') as f:
            lines = f.readlines()

        if len(lines) > 1:
            print(f"\nEpoch 记录 ({len(lines)-1} 条):")
            for i, line in enumerate(lines[1:], 1):
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    step = int(parts[0])
                    epoch_loss = float(parts[1])
                    lr = float(parts[2])
                    print(f"  Epoch {i}: 步数={step:,}, 损失={epoch_loss:.4f}, 学习率={lr:.6f}")

    # 检查验证日志
    if val_log_path.exists():
        with open(val_log_path, 'r') as f:
            val_lines = f.readlines()
        if len(val_lines) > 1:
            print(f"\n验证记录: {len(val_lines)-1} 条")
        else:
            print(f"\n验证记录: 无 (val_log.csv 为空)")
    else:
        print(f"\n验证记录: 文件不存在")

def evaluate_model():
    """评估模型翻译质量"""
    print("\n" + "=" * 80)
    print("模型翻译质量评估")
    print("=" * 80)

    # 加载 tokenizers
    src_tokenizer_path = project_root / "models" / "src_tokenizer.model"
    tgt_tokenizer_path = project_root / "models" / "tgt_tokenizer.model"
    checkpoint_path = project_root / "models" / "best_model.pt"

    if not checkpoint_path.exists():
        print("模型检查点不存在!")
        return

    print("加载 tokenizers...")
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )

    # 加载 checkpoint
    print("加载检查点...")
    device = "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    step = checkpoint.get('step', 0)
    print(f"检查点步数: {step:,}")

    # 获取配置
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
    print("创建模型...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建评估器
    evaluator = Evaluator(model, src_tokenizer, tgt_tokenizer, device=device)

    # 测试句子
    test_cases = [
        # 基础短语
        ("Hello", "Hallo"),
        ("Good morning", "Guten Morgen"),
        ("Thank you", "Danke"),
        ("How are you?", "Wie geht es dir?"),
        ("I am fine", "Mir geht es gut"),

        # 简单句子
        ("This is a test", "Das ist ein Test"),
        ("The sky is blue", "Der Himmel ist blau"),
        ("I like apples", "Ich mag Äpfel"),
        ("She reads a book", "Sie liest ein Buch"),
        ("We are learning", "Wir lernen"),

        # 日常用语
        ("What time is it?", "Wie spät ist es?"),
        ("Can you help me?", "Kannst du mir helfen?"),
        ("Where is the station?", "Wo ist der Bahnhof?"),
        ("My name is John", "Mein Name ist John"),
        ("I love you", "Ich liebe dich"),
    ]

    print(f"\n翻译测试 ({len(test_cases)} 个例子):")
    print("-" * 80)

    results = []
    for src, expected in test_cases:
        translation = evaluator.translate(src, method="greedy")

        # 清理翻译结果
        translation = translation.replace("[BOS]", "").replace("[EOS]", "").strip()

        # 简单评估
        src_lower = src.lower()
        trans_lower = translation.lower()
        expected_lower = expected.lower()

        # 检查关键词匹配
        key_words = expected_lower.split()
        matched_words = sum(1 for word in key_words if word in trans_lower)
        match_ratio = matched_words / len(key_words) if key_words else 0

        result = {
            'src': src,
            'expected': expected,
            'translation': translation,
            'match_ratio': match_ratio
        }
        results.append(result)

        print(f"输入: '{src}'")
        print(f"  预期: '{expected}'")
        print(f"  翻译: '{translation}'")

        if match_ratio >= 0.8:
            print(f"  ✅ 良好匹配 ({match_ratio:.0%})")
        elif match_ratio >= 0.4:
            print(f"  ⚠️  部分匹配 ({match_ratio:.0%})")
        else:
            print(f"  ❌ 匹配差 ({match_ratio:.0%})")
        print()

    # 统计结果
    good = sum(1 for r in results if r['match_ratio'] >= 0.8)
    partial = sum(1 for r in results if 0.4 <= r['match_ratio'] < 0.8)
    poor = sum(1 for r in results if r['match_ratio'] < 0.4)

    print("=" * 80)
    print("翻译质量统计:")
    print(f"  良好匹配: {good}/{len(results)} ({good/len(results):.1%})")
    print(f"  部分匹配: {partial}/{len(results)} ({partial/len(results):.1%})")
    print(f"  匹配差: {poor}/{len(results)} ({poor/len(results):.1%})")

    # 显示一些有代表性的例子
    print(f"\n代表性例子:")
    best = max(results, key=lambda r: r['match_ratio'])
    worst = min(results, key=lambda r: r['match_ratio'])

    print(f"  最佳: '{best['src']}' → '{best['translation']}' (匹配度: {best['match_ratio']:.0%})")
    print(f"  最差: '{worst['src']}' → '{worst['translation']}' (匹配度: {worst['match_ratio']:.0%})")

def check_training_progress():
    """检查训练进度"""
    print("\n" + "=" * 80)
    print("训练进度分析")
    print("=" * 80)

    # 从日志获取最新步数
    step_log_path = project_root / "models" / "logs" / "step_log.csv"
    if step_log_path.exists():
        with open(step_log_path, 'r') as f:
            lines = f.readlines()
        if len(lines) > 1:
            last_line = lines[-1].strip()
            last_step = int(last_line.split(',')[0])

            # 计算进度
            max_steps = config.max_steps
            progress = last_step / max_steps * 100
            remaining = max_steps - last_step

            print(f"当前步数: {last_step:,}")
            print(f"目标步数: {max_steps:,}")
            print(f"完成进度: {progress:.1f}%")
            print(f"剩余步数: {remaining:,}")

            # 估算剩余时间 (基于历史速度)
            # 假设每步时间大致相同
            step_log_len = len(lines) - 1  # 跳过标题行
            if step_log_len > 1000:
                # 简单估算: 如果日志记录了较长时间的训练，可以估算速度
                print(f"\n基于 {step_log_len:,} 步记录的统计:")
                print(f"  平均每步损失: {config.batch_size} 样本/步")
                print(f"  已处理样本: {last_step * config.batch_size:,}")
                print(f"  目标样本: {max_steps * config.batch_size:,}")

                # 计算epoch
                steps_per_epoch = config.max_train_samples // config.batch_size
                epochs_completed = last_step / steps_per_epoch
                total_epochs = max_steps / steps_per_epoch

                print(f"\nEpoch 统计:")
                print(f"  每epoch步数: {steps_per_epoch:,}")
                print(f"  已完成epoch: {epochs_completed:.2f}")
                print(f"  总epoch: {total_epochs:.2f}")
                print(f"  Epoch进度: {epochs_completed/total_epochs*100:.1f}%")

def main():
    """主函数"""
    print("最新训练成果检查")
    print("=" * 80)

    analyze_training_logs()
    evaluate_model()
    check_training_progress()

    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)

if __name__ == "__main__":
    main()