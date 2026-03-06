#!/usr/bin/env python3
"""
训练进度报告更新脚本。
自动从训练日志中提取信息，生成/更新进度跟踪报告。
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import csv
import math

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config


def get_training_process_info():
    """获取训练进程信息"""
    try:
        # 查找训练进程
        result = subprocess.run(
            ["pgrep", "-f", "python.*train.py"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        pid = result.stdout.strip().split()[0]

        # 获取进程详细信息
        ps_result = subprocess.run(
            ["ps", "-o", "pid,%cpu,%mem,start,etime,time", "-p", pid],
            capture_output=True,
            text=True,
            check=True
        )

        lines = ps_result.stdout.strip().split('\n')
        if len(lines) < 2:
            return None

        headers = lines[0].split()
        values = lines[1].split()

        info = dict(zip(headers, values))
        info['PID'] = pid

        # 解析运行时间
        elapsed = info.get('ELAPSED', '00:00')
        if ':' in elapsed:
            if elapsed.count(':') == 1:  # MM:SS
                m, s = map(int, elapsed.split(':'))
                info['elapsed_seconds'] = m * 60 + s
            else:  # HH:MM:SS
                h, m, s = map(int, elapsed.split(':'))
                info['elapsed_seconds'] = h * 3600 + m * 60 + s
        else:
            info['elapsed_seconds'] = int(elapsed) if elapsed.isdigit() else 0

        return info

    except Exception as e:
        print(f"获取进程信息失败: {e}")
        return None


def read_step_log(log_path):
    """读取步数日志文件"""
    if not log_path.exists():
        return None

    steps = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['step'] = int(row['step'])
                row['loss'] = float(row['loss'])
                row['lr'] = float(row['lr'])
                steps.append(row)
    except Exception as e:
        print(f"读取日志文件失败: {e}")
        return None

    return steps


def read_val_log(log_path):
    """读取验证日志文件"""
    if not log_path.exists():
        return None

    vals = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['step'] = int(row['step'])
                row['val_loss'] = float(row['val_loss'])
                vals.append(row)
    except Exception as e:
        print(f"读取验证日志失败: {e}")
        return None

    return vals


def calculate_statistics(steps, vals=None):
    """计算统计信息"""
    if not steps:
        return {}

    stats = {}

    # 基础统计
    stats['total_steps'] = len(steps)
    stats['current_step'] = steps[-1]['step']
    stats['current_loss'] = steps[-1]['loss']
    stats['current_lr'] = steps[-1]['lr']

    # 初始损失
    stats['initial_loss'] = steps[0]['loss'] if steps else 0

    # 最近N步平均损失
    recent_n = min(20, len(steps))
    recent_losses = [s['loss'] for s in steps[-recent_n:]]
    stats['recent_avg_loss'] = sum(recent_losses) / len(recent_losses)

    # 损失改善百分比
    if len(steps) >= 10:
        early_avg = sum(s['loss'] for s in steps[:10]) / 10
        recent_avg = stats['recent_avg_loss']
        if early_avg > 0:
            stats['improvement_pct'] = ((early_avg - recent_avg) / early_avg) * 100
        else:
            stats['improvement_pct'] = 0
    else:
        stats['improvement_pct'] = 0

    # 验证统计
    if vals and len(vals) > 0:
        stats['val_steps'] = len(vals)
        stats['latest_val_step'] = vals[-1]['step']
        stats['latest_val_loss'] = vals[-1]['val_loss']
        stats['val_improvement'] = vals[0]['val_loss'] - vals[-1]['val_loss'] if len(vals) > 1 else 0
    else:
        stats['val_steps'] = 0
        stats['latest_val_step'] = 0
        stats['latest_val_loss'] = None

    return stats


def generate_progress_report(stats, process_info, output_path):
    """生成进度报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if process_info:
        start_time = "13:50:00"  # 从配置或日志推断
        elapsed_seconds = process_info.get('elapsed_seconds', 0)
        elapsed_str = str(timedelta(seconds=elapsed_seconds))

        # 计算训练速度
        if elapsed_seconds > 0 and stats['current_step'] > 0:
            speed = stats['current_step'] / elapsed_seconds
            eta_seconds = (200000 - stats['current_step']) / speed if speed > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            eta_time = (datetime.now() + timedelta(seconds=eta_seconds)).strftime("%Y-%m-%d %H:%M")
        else:
            speed = 0
            eta_str = "N/A"
            eta_time = "N/A"
    else:
        start_time = "N/A"
        elapsed_str = "N/A"
        speed = 0
        eta_str = "N/A"
        eta_time = "N/A"

    # 进度百分比
    progress_pct = (stats['current_step'] / 200000) * 100

    # 生成报告
    report = f"""# 训练进度跟踪报告
**报告生成时间:** {now}
**训练开始时间:** 2026-03-03 {start_time}
**当前分支:** training-optimization
**训练状态:** {'进行中 ✅' if process_info else '未运行 ⚠️'}

## 📊 训练概述
- **模型:** Transformer翻译模型 (英→德)
- **总训练步数:** 200,000步
- **当前步数:** {stats['current_step']:,}步 ({progress_pct:.2f}%)
- **训练设备:** MPS (Metal Performance Shaders)
- **数据:** 196,984个清洗后句子对
  - 训练集: 195,014样本
  - 验证集: 1,970样本

## 🎯 当前状态
### 训练进程
- **进程ID:** {process_info.get('PID', 'N/A') if process_info else 'N/A'}
- **运行时间:** {elapsed_str}
- **资源使用:** CPU {process_info.get('%CPU', 'N/A') if process_info else 'N/A'}%, 内存 {process_info.get('%MEM', 'N/A') if process_info else 'N/A'}%
- **训练速度:** {speed:.2f}步/秒
- **预计完成时间:** {eta_str} ({eta_time})

### 性能指标
| 指标 | 当前值 | 趋势 |
|------|--------|------|
| **当前损失** | {stats['current_loss']:.4f} | {'📉' if stats.get('improvement_pct', 0) > 0 else '➡️'} 改善{stats.get('improvement_pct', 0):.1f}% |
| **当前学习率** | {stats['current_lr']:.6f} | {'📈' if stats['current_lr'] > 0 else '➡️'} |
| **最近20步平均损失** | {stats.get('recent_avg_loss', 0):.4f} | {'📉' if stats['current_loss'] < stats.get('recent_avg_loss', stats['current_loss']) else '➡️'} |
| **初始损失** | {stats.get('initial_loss', 0):.4f} | - |

### 验证状态
- **验证间隔:** 每{config.eval_interval}步
- **验证集大小:** 1,970样本
- **验证次数:** {stats['val_steps']}次
- **最新验证损失:** {stats['latest_val_loss'] if stats['latest_val_loss'] is not None else '尚未验证'}
- **下次验证:** 步数{((stats['current_step'] // config.eval_interval) + 1) * config.eval_interval}

## 📈 训练进度
### 进度可视化
```
[{'█' * int(progress_pct / 2)}{'░' * (50 - int(progress_pct / 2))}] {progress_pct:.2f}%
{stats['current_step']:,} / 200,000步
```

### 里程碑追踪
- {'✅' if stats['current_step'] >= 100 else '🔄'} **前100步**: 完成初始波动期
- {'✅' if stats['current_step'] >= 500 else '🔄'} **第500步**: 损失降至~5.0以下
- {'✅' if stats['current_step'] >= 1000 else '🔄'} **第1,000步**: 预计损失~4.0
- {'✅' if stats['current_step'] >= 5000 else '🔄'} **第5,000步**: 首次验证评估
- {'✅' if stats['current_step'] >= 10000 else '🔄'} **第10,000步**: 首次模型保存
- {'🔄' if stats['current_step'] >= 50000 else '⏳'} **第50,000步**: 损失目标~2.5
- {'🔄' if stats['current_step'] >= 100000 else '⏳'} **第100,000步**: 损失目标~1.5

## 💾 检查点状态
### 自动保存
- **保存间隔:** 每{config.save_interval}步
- **最佳模型保存:** 仅当验证损失改善≥{config.min_loss_improvement*100}%
- **当前检查点:** {'暂无' if stats['current_step'] < config.save_interval else f'步数{((stats["current_step"] // config.save_interval) * config.save_interval)}'}

## 🔄 更新记录
- **{now}:** 自动更新进度报告
- **训练开始:** 2026-03-03 13:50:00
- **下次更新建议:** 每1,000步或重大里程碑更新

---
**报告生成脚本:** scripts/update_training_progress.py
**数据来源:** `models/logs/step_log.csv`, `models/logs/val_log.csv`, 训练进程信息
**跟踪频率:** 建议每1,000步或每小时自动更新
"""

    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"进度报告已生成: {output_path}")
    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="更新训练进度报告")
    parser.add_argument("--output", "-o", type=str, default="doc/training_progress_tracking.md",
                       help="输出报告文件路径")
    parser.add_argument("--checkpoint-dir", type=str, default="models",
                       help="检查点目录路径")
    args = parser.parse_args()

    print("=" * 60)
    print("训练进度报告生成器")
    print("=" * 60)

    # 获取训练进程信息
    print("\n1. 检查训练进程...")
    process_info = get_training_process_info()
    if process_info:
        print(f"   ✅ 训练进程运行中 (PID: {process_info['PID']})")
        print(f"       运行时间: {process_info.get('ELAPSED', 'N/A')}")
        print(f"       CPU使用: {process_info.get('%CPU', 'N/A')}%, 内存使用: {process_info.get('%MEM', 'N/A')}%")
    else:
        print("   ⚠️  未检测到训练进程")

    # 读取日志文件
    print("\n2. 读取训练日志...")
    base_dir = Path(__file__).parent.parent
    step_log_path = base_dir / args.checkpoint_dir / "logs" / "step_log.csv"
    val_log_path = base_dir / args.checkpoint_dir / "logs" / "val_log.csv"

    steps = read_step_log(step_log_path)
    if steps:
        print(f"   ✅ 读取{len(steps)}条训练记录")
        print(f"       当前步数: {steps[-1]['step']:,}")
        print(f"       当前损失: {steps[-1]['loss']:.4f}")
    else:
        print("   ❌ 训练日志文件不存在或为空")
        steps = []

    vals = read_val_log(val_log_path)
    if vals:
        print(f"   ✅ 读取{len(vals)}条验证记录")
        if vals:
            print(f"       最新验证步数: {vals[-1]['step']:,}, 损失: {vals[-1]['val_loss']:.4f}")
    else:
        print("   ℹ️  验证日志文件不存在或为空")
        vals = []

    # 计算统计信息
    print("\n3. 计算统计信息...")
    stats = calculate_statistics(steps, vals)

    if stats:
        print(f"   当前步数: {stats['current_step']:,}")
        print(f"   当前损失: {stats['current_loss']:.4f}")
        print(f"   学习率: {stats['current_lr']:.6f}")
        print(f"   最近20步平均损失: {stats.get('recent_avg_loss', 0):.4f}")
        if 'improvement_pct' in stats:
            print(f"   损失改善: {stats['improvement_pct']:.1f}%")
    else:
        print("   ⚠️  无法计算统计信息")
        stats = {}

    # 生成报告
    print("\n4. 生成进度报告...")
    output_path = base_dir / args.output
    report = generate_progress_report(stats, process_info, output_path)

    print("\n" + "=" * 60)
    print("✅ 进度报告更新完成")
    print("=" * 60)


if __name__ == "__main__":
    main()