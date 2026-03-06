#!/usr/bin/env python3
"""
分析优化后的训练结果。
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_training_results():
    # 路径
    log_dir = Path(__file__).parent.parent / "models" / "logs"
    step_log_path = log_dir / "step_log.csv"
    val_log_path = log_dir / "val_log.csv"
    epoch_log_path = log_dir / "epoch_log.csv"

    print("=" * 60)
    print("训练结果分析 - 数据优化后")
    print("=" * 60)

    # 读取验证日志
    if val_log_path.exists():
        val_df = pd.read_csv(val_log_path)
        print(f"\n验证损失数据 ({len(val_df)} 个检查点):")
        print(val_df)

        if len(val_df) > 0:
            initial_val_loss = val_df['val_loss'].iloc[0]
            final_val_loss = val_df['val_loss'].iloc[-1]
            improvement = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100

            print(f"\n验证损失分析:")
            print(f"  初始验证损失 (步 {val_df['step'].iloc[0]}): {initial_val_loss:.4f}")
            print(f"  最终验证损失 (步 {val_df['step'].iloc[-1]}): {final_val_loss:.4f}")
            print(f"  相对改进: {improvement:.1f}%")

            # 计算验证损失下降率
            steps = val_df['step'].values
            losses = val_df['val_loss'].values
            if len(steps) > 1:
                total_step_increase = steps[-1] - steps[0]
                total_loss_decrease = losses[0] - losses[-1]
                loss_per_step = total_loss_decrease / total_step_increase if total_step_increase > 0 else 0
                print(f"  验证损失下降率: {loss_per_step:.6f} 损失/步")

    # 读取步骤日志（采样分析，避免内存问题）
    if step_log_path.exists():
        print(f"\n步骤日志分析 (文件大小: {step_log_path.stat().st_size:,} 字节)")

        # 读取前1000行和后1000行进行分析
        try:
            # 获取总行数
            with open(step_log_path, 'r') as f:
                total_lines = sum(1 for _ in f)

            print(f"  总行数: {total_lines:,}")

            # 读取头部和尾部
            step_df_head = pd.read_csv(step_log_path, nrows=1000)
            step_df_tail = pd.read_csv(step_log_path, skiprows=total_lines-1000 if total_lines > 1000 else 0)

            print(f"\n  早期训练 (前1000步样本):")
            print(f"    平均损失: {step_df_head['loss'].mean():.4f}")
            print(f"    损失标准差: {step_df_head['loss'].std():.4f}")
            print(f"    学习率范围: [{step_df_head['lr'].min():.6f}, {step_df_head['lr'].max():.6f}]")

            print(f"\n  后期训练 (后1000步样本):")
            print(f"    平均损失: {step_df_tail['loss'].mean():.4f}")
            print(f"    损失标准差: {step_df_tail['loss'].std():.4f}")
            print(f"    学习率范围: [{step_df_tail['lr'].min():.6f}, {step_df_tail['lr'].max():.6f}]")

        except Exception as e:
            print(f"  步骤日志分析出错: {e}")

    # 读取epoch日志
    if epoch_log_path.exists():
        epoch_df = pd.read_csv(epoch_log_path)
        print(f"\nEpoch 日志分析 ({len(epoch_df)} 个epoch):")
        if len(epoch_df) > 0:
            print(epoch_df.tail(10))  # 显示最后10个epoch

    print("\n" + "=" * 60)
    print("分析与建议")
    print("=" * 60)

    # 基于验证损失给出初步评估
    if 'initial_val_loss' in locals() and 'final_val_loss' in locals():
        print(f"\n1. 验证损失从 {initial_val_loss:.4f} 下降到 {final_val_loss:.4f}，")
        print(f"   改善 {improvement:.1f}%，表明数据优化有效。")

        if final_val_loss < 2.5:
            print("2. 验证损失 < 2.5，表明模型学习良好。")
        elif final_val_loss < 3.0:
            print("2. 验证损失 < 3.0，模型学习中等，仍有改进空间。")
        else:
            print("2. 验证损失 ≥ 3.0，模型学习可能不足，需进一步优化。")

        # 与之前训练对比（215,000步时3.0364）
        previous_val_loss = 3.0364  # 从之前报告中获取
        comparison = ((previous_val_loss - final_val_loss) / previous_val_loss) * 100
        print(f"3. 与优化前训练对比 (215,000步时验证损失={previous_val_loss:.4f}):")
        print(f"   新训练在 {val_df['step'].iloc[-1]:,} 步达到 {final_val_loss:.4f}，")
        print(f"   相对改进: {comparison:.1f}%")

    print("\n4. 建议下一步:")
    print("   a. 运行翻译质量评估: python scripts/evaluate_translation_comprehensive.py")
    print("   b. 如果验证损失持续下降，可继续训练到100,000步")
    print("   c. 评估是否需要模型容量实验 (d_model=768, n_layers=8)")

if __name__ == "__main__":
    analyze_training_results()