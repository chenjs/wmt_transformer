#!/usr/bin/env python3
"""
实验性训练脚本 - 支持不同模型配置的Transformer训练。
用于模型容量实验。
"""
import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import Config
from src.data.dataset import ParallelDataset
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.trainer import Trainer


@dataclass
class ExperimentConfig:
    """实验配置 - 可覆盖默认配置"""
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 54

    # 训练参数
    batch_size: int = 12
    learning_rate: float = 1e-3
    warmup_steps: int = 8000
    max_steps: int = 100000  # 实验步数，较短的初始目标
    label_smoothing: float = 0.1
    clip_grad: float = 10.0

    # 数据参数
    train_split: float = 0.99
    max_train_samples: int = 200000

    # 检查点参数
    save_interval: int = 10000
    eval_interval: int = 5000
    min_loss_improvement: float = 0.01


def get_device():
    """Get device for training."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_last_step_from_log(log_dir: Path) -> int:
    """
    从训练日志文件获取最后一个步数。

    Args:
        log_dir: 日志目录路径

    Returns:
        最后一个步数，如果没有日志则返回0
    """
    step_log_path = log_dir / "step_log.csv"
    if not step_log_path.exists():
        return 0

    last_step = 0
    try:
        import csv
        with open(step_log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(row['step'])
                    if step > last_step:
                        last_step = step
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"警告: 读取日志文件失败: {e}")

    return last_step


def create_model_with_config(config: Config, experiment_config: ExperimentConfig,
                           src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    """根据实验配置创建模型"""
    print(f"\n创建实验模型:")
    print(f"  d_model: {experiment_config.d_model} (默认: {config.d_model})")
    print(f"  n_layers: {experiment_config.n_layers} (默认: {config.n_layers})")
    print(f"  n_heads: {experiment_config.n_heads} (默认: {config.n_heads})")
    print(f"  d_ff: {experiment_config.d_ff} (默认: {config.d_ff})")
    print(f"  dropout: {experiment_config.dropout} (默认: {config.dropout})")

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=experiment_config.d_model,
        n_layers=experiment_config.n_layers,
        n_heads=experiment_config.n_heads,
        d_ff=experiment_config.d_ff,
        dropout=experiment_config.dropout,
        max_len=experiment_config.max_len,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")
    print(f"  参数增加: {((total_params / (68_730_496)) - 1) * 100:.1f}% (相对于68.7M基准)")

    return model


def transfer_parameters(old_state_dict: dict, new_model: Transformer,
                       old_config: Config, new_config: ExperimentConfig) -> Transformer:
    """
    将参数从旧模型迁移到新模型架构。
    处理维度不匹配的情况。
    """
    new_state_dict = new_model.state_dict()

    print("\n参数迁移:")
    transferred = 0
    skipped = 0
    expanded = 0

    for name, param in old_state_dict.items():
        if name in new_state_dict:
            old_shape = param.shape
            new_shape = new_state_dict[name].shape

            if old_shape == new_shape:
                # 直接复制
                new_state_dict[name].copy_(param)
                transferred += 1
            elif len(old_shape) == 2 and len(new_shape) == 2:
                # 处理二维参数 (权重矩阵)
                if old_shape[0] <= new_shape[0] and old_shape[1] <= new_shape[1]:
                    # 维度扩展 - 将旧参数放在左上角，其余部分随机初始化
                    new_state_dict[name][:old_shape[0], :old_shape[1]] = param
                    expanded += 1
                elif old_shape[0] >= new_shape[0] and old_shape[1] >= new_shape[1]:
                    # 维度缩小 - 只复制部分参数
                    new_state_dict[name].copy_(param[:new_shape[0], :new_shape[1]])
                    expanded += 1
                else:
                    # 维度不匹配，跳过
                    print(f"  ⚠️  跳过 {name}: 形状不匹配 {old_shape} -> {new_shape}")
                    skipped += 1
            elif len(old_shape) == 1 and len(new_shape) == 1:
                # 处理一维参数 (偏置)
                if old_shape[0] <= new_shape[0]:
                    new_state_dict[name][:old_shape[0]] = param
                    expanded += 1
                else:
                    new_state_dict[name].copy_(param[:new_shape[0]])
                    expanded += 1
            else:
                print(f"  ⚠️  跳过 {name}: 不支持的形状转换 {old_shape} -> {new_shape}")
                skipped += 1

    new_model.load_state_dict(new_state_dict)
    print(f"  迁移完成: {transferred}个直接复制, {expanded}个维度扩展, {skipped}个跳过")

    return new_model


def load_checkpoint_for_experiment(checkpoint_path: Path, experiment_config: ExperimentConfig,
                                 src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    """加载检查点并适配到实验配置"""
    print(f"加载检查点: {checkpoint_path.name}")

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 创建新模型
    model = create_model_with_config(Config(), experiment_config, src_vocab_size, tgt_vocab_size)

    if 'model_state_dict' in checkpoint:
        # 迁移参数
        model = transfer_parameters(checkpoint['model_state_dict'], model, Config(), experiment_config)

        # 如果检查点中有优化器状态，可能需要调整
        if 'optimizer_state_dict' in checkpoint:
            print("  ⚠️  优化器状态需要重新初始化 (架构变化)")
    else:
        print("  ⚠️  检查点中没有模型状态，从头开始训练")

    return model


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="实验性Transformer训练")

    # 实验配置参数
    parser.add_argument("--d-model", type=int, default=512, help="模型维度")
    parser.add_argument("--n-layers", type=int, default=6, help="Transformer层数")
    parser.add_argument("--n-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d-ff", type=int, default=2048, help="前馈网络维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="丢弃率")
    parser.add_argument("--max-len", type=int, default=54, help="最大序列长度")

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=12, help="批量大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--warmup-steps", type=int, default=8000, help="预热步数")
    parser.add_argument("--max-steps", type=int, default=100000, help="最大训练步数")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="标签平滑")
    parser.add_argument("--clip-grad", type=float, default=10.0, help="梯度裁剪")

    # 实验选项
    parser.add_argument("--resume", type=str, help="从检查点恢复训练")
    parser.add_argument("--experiment-name", type=str, default="exp_a",
                       help="实验名称 (用于保存结果)")
    parser.add_argument("--no-transfer", action="store_true",
                       help="不迁移参数，从头开始训练")

    args = parser.parse_args()

    # 创建实验配置
    experiment_config = ExperimentConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        label_smoothing=args.label_smoothing,
        clip_grad=args.clip_grad,
    )

    print("=" * 60)
    print("实验性Transformer训练")
    print("=" * 60)
    print(f"实验名称: {args.experiment_name}")
    print(f"配置: d_model={experiment_config.d_model}, "
          f"n_layers={experiment_config.n_layers}, "
          f"n_heads={experiment_config.n_heads}")

    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")

    # 设置数据路径和分词器
    data_dir = Path(__file__).parent.parent
    src_tokenizer_path = data_dir / "models_enhanced" / "src_tokenizer_final.model"
    tgt_tokenizer_path = data_dir / "models_enhanced" / "tgt_tokenizer_final.model"

    if not src_tokenizer_path.exists() or not tgt_tokenizer_path.exists():
        print("错误: 增强分词器未找到")
        print(f"请确保以下文件存在:")
        print(f"  {src_tokenizer_path}")
        print(f"  {tgt_tokenizer_path}")
        return

    # 加载分词器
    print("\n加载分词器...")
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        str(src_tokenizer_path), str(tgt_tokenizer_path)
    )
    src_vocab_size = src_tokenizer.sp.get_piece_size()
    tgt_vocab_size = tgt_tokenizer.sp.get_piece_size()
    print(f"源词汇表大小: {src_vocab_size}")
    print(f"目标词汇表大小: {tgt_vocab_size}")

    # 创建或加载模型
    if args.resume and not args.no_transfer:
        # 从检查点加载并迁移参数
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            print(f"错误: 检查点不存在: {checkpoint_path}")
            return

        model = load_checkpoint_for_experiment(
            checkpoint_path, experiment_config, src_vocab_size, tgt_vocab_size
        )
        start_step = get_last_step_from_log(data_dir / "models" / "logs")
    else:
        # 创建新模型
        model = create_model_with_config(
            Config(), experiment_config, src_vocab_size, tgt_vocab_size
        )
        start_step = 0

    model = model.to(device)
    model.train()

    # 创建数据集
    print("\n创建数据集...")
    src_file = data_dir / "models_enhanced" / "src_text_cleaned.txt"
    tgt_file = data_dir / "models_enhanced" / "tgt_text_cleaned.txt"

    if not src_file.exists() or not tgt_file.exists():
        print("错误: 清洗后的数据文件未找到")
        return

    dataset = ParallelDataset(
        src_file=str(src_file),
        tgt_file=str(tgt_file),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=experiment_config.max_len,
        max_samples=experiment_config.max_train_samples,
    )

    # 创建实验输出目录
    experiment_dir = data_dir / "experiments" / args.experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建训练器
    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=experiment_config.batch_size,
        learning_rate=experiment_config.learning_rate,
        warmup_steps=experiment_config.warmup_steps,
        max_steps=experiment_config.max_steps,
        label_smoothing=experiment_config.label_smoothing,
        clip_grad=experiment_config.clip_grad,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        save_interval=experiment_config.save_interval,
        eval_interval=experiment_config.eval_interval,
        min_loss_improvement=experiment_config.min_loss_improvement,
    )

    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    try:
        trainer.train(start_step=start_step)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print(f"检查点已保存到: {checkpoint_dir}")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("实验训练完成")
    print("=" * 60)
    print(f"实验结果保存在: {experiment_dir}")
    print(f"后续步骤:")
    print(f"  1. 评估模型: python scripts/evaluate_translation_comprehensive.py --checkpoint {checkpoint_dir}/best_model.pt")
    print(f"  2. 分析基础词汇: python scripts/analyze_basic_vocab.py --checkpoint {checkpoint_dir}/best_model.pt")
    print(f"  3. 比较实验结果")


if __name__ == "__main__":
    main()