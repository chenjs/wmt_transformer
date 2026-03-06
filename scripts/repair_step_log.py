#!/usr/bin/env python3
"""
修复步数日志文件，移除重复和无效记录。
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

def analyze_log_file(log_path):
    """分析日志文件问题"""
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"日志文件不存在: {log_path}")
        return None

    print(f"分析日志文件: {log_path}")
    print(f"文件大小: {log_path.stat().st_size:,} 字节")

    # 读取所有行
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"总行数: {len(lines)}")

    # 解析所有记录
    records = []
    errors = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if i == 0 and line.startswith('step'):
            continue  # 跳过标题行

        parts = line.split(',')
        if len(parts) < 3:
            errors += 1
            continue

        try:
            step = int(parts[0])
            loss = float(parts[1])
            lr = float(parts[2])
            records.append({
                'line_num': i + 1,
                'step': step,
                'loss': loss,
                'lr': lr,
                'raw': line
            })
        except ValueError:
            errors += 1
            continue

    print(f"有效记录: {len(records)}")
    print(f"解析错误: {errors}")

    if not records:
        return None

    # 分析步数
    steps = [r['step'] for r in records]
    min_step = min(steps)
    max_step = max(steps)
    print(f"步数范围: {min_step} - {max_step}")

    # 检查重复步数
    step_counts = defaultdict(int)
    for step in steps:
        step_counts[step] += 1

    duplicate_steps = {step: count for step, count in step_counts.items() if count > 1}
    if duplicate_steps:
        print(f"重复步数: {len(duplicate_steps)} 个")
        print(f"最多重复的步数: {max(duplicate_steps.items(), key=lambda x: x[1])}")

    # 检查连续性
    sorted_steps = sorted(set(steps))
    gaps = []
    for i in range(1, len(sorted_steps)):
        if sorted_steps[i] != sorted_steps[i-1] + 1:
            gaps.append((sorted_steps[i-1], sorted_steps[i]))

    if gaps:
        print(f"步数间隙: {len(gaps)} 个")
        for i, gap in enumerate(gaps[:5]):
            print(f"  间隙 {i+1}: {gap[0]} -> {gap[1]} (跳过了 {gap[1]-gap[0]-1} 步)")
        if len(gaps) > 5:
            print(f"  还有 {len(gaps)-5} 个间隙未显示")

    return {
        'records': records,
        'min_step': min_step,
        'max_step': max_step,
        'duplicate_steps': duplicate_steps,
        'gaps': gaps
    }

def repair_log_file(log_path, backup=True):
    """修复日志文件"""
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"日志文件不存在: {log_path}")
        return False

    print(f"\n修复日志文件: {log_path}")
    print("-" * 50)

    # 备份原文件
    if backup:
        backup_path = log_path.with_suffix('.csv.backup')
        import shutil
        shutil.copy2(log_path, backup_path)
        print(f"已备份原文件到: {backup_path}")

    # 分析问题
    analysis = analyze_log_file(log_path)
    if not analysis or not analysis['records']:
        print("无有效记录可修复")
        return False

    records = analysis['records']
    max_step = analysis['max_step']

    # 方案1: 只保留每个步数的第一次出现
    print("\n方案1: 保留每个步数的第一次出现")
    unique_records = {}
    for record in records:
        step = record['step']
        if step not in unique_records:
            unique_records[step] = record

    print(f"唯一步数: {len(unique_records)}")
    print(f"移除重复: {len(records) - len(unique_records)} 条记录")

    # 按步数排序
    sorted_records = sorted(unique_records.values(), key=lambda x: x['step'])

    # 检查排序后的连续性
    sorted_steps = [r['step'] for r in sorted_records]
    continuous = all(sorted_steps[i] == sorted_steps[i-1] + 1 for i in range(1, len(sorted_steps)))

    if not continuous:
        print("警告: 排序后步数仍不连续")
        # 方案2: 重新编号步数
        print("\n方案2: 重新编号步数 (从1开始连续)")
        for i, record in enumerate(sorted_records):
            record['step'] = i + 1

    # 写入修复后的文件
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("step,loss,lr\n")
        for record in sorted_records:
            f.write(f"{record['step']},{record['loss']:.6f},{record['lr']:.6f}\n")

    print(f"\n已写入修复后文件: {log_path}")
    print(f"总记录数: {len(sorted_records)}")
    print(f"步数范围: {sorted_records[0]['step']} - {sorted_records[-1]['step']}")

    # 验证修复
    print("\n验证修复结果:")
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"行数: {len(lines)} (包含标题行)")

    return True

def main():
    """主函数"""
    log_path = Path(__file__).parent.parent / "models" / "logs" / "step_log.csv"

    print("=" * 60)
    print("步数日志修复工具")
    print("=" * 60)

    if not log_path.exists():
        print(f"错误: 日志文件不存在: {log_path}")
        sys.exit(1)

    # 分析当前状态
    analysis = analyze_log_file(log_path)
    if not analysis:
        print("无法分析日志文件")
        sys.exit(1)

    # 询问用户是否修复
    print("\n是否修复日志文件？")
    print("这将会:")
    print("  1. 备份原文件为 .csv.backup")
    print("  2. 移除重复的步数记录")
    print("  3. 确保步数连续")

    response = input("\n继续修复？(y/N): ").strip().lower()
    if response != 'y':
        print("取消修复")
        sys.exit(0)

    # 执行修复
    success = repair_log_file(log_path, backup=True)

    if success:
        print("\n" + "=" * 60)
        print("✅ 修复完成!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 修复失败!")
        print("=" * 60)

if __name__ == "__main__":
    main()