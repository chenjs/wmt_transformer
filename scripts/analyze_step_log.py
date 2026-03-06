#!/usr/bin/env python3
"""
分析步数日志文件，检查步数计数问题。
"""
import csv
import sys
from pathlib import Path

def analyze_step_log(log_path):
    """分析步数日志文件"""
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"日志文件不存在: {log_path}")
        return None

    print(f"分析日志文件: {log_path}")
    print(f"文件大小: {log_path.stat().st_size} 字节")

    # 读取所有行
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    print(f"总行数: {len(lines)}")

    # 解析CSV
    steps = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    step = int(row['step'])
                    loss = float(row['loss'])
                    lr = float(row['lr'])
                    steps.append((step, loss, lr))
                except (ValueError, KeyError) as e:
                    print(f"第{i+2}行解析错误: {row}, 错误: {e}")
                    continue
    except Exception as e:
        print(f"CSV解析错误: {e}")
        # 尝试原始解析
        steps = []
        for i, line in enumerate(lines[1:], start=2):  # 跳过标题行
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        step = int(parts[0])
                        loss = float(parts[1])
                        lr = float(parts[2])
                        steps.append((step, loss, lr))
                    except ValueError:
                        print(f"第{i}行格式错误: {line}")

    print(f"有效步数记录: {len(steps)}")

    if steps:
        print(f"步数范围: {steps[0][0]} - {steps[-1][0]}")
        print(f"最后一步: 步数={steps[-1][0]}, 损失={steps[-1][1]:.4f}, 学习率={steps[-1][2]:.6f}")

        # 检查步数连续性
        expected_step = steps[0][0]
        gaps = []
        for i, (step, loss, lr) in enumerate(steps):
            if step != expected_step:
                gaps.append((i, expected_step, step))
            expected_step = step + 1

        if gaps:
            print(f"发现 {len(gaps)} 个步数间隙:")
            for gap in gaps[:5]:  # 只显示前5个
                print(f"  第{gap[0]+1}行: 期望步数{gap[1]}, 实际步数{gap[2]}")
            if len(gaps) > 5:
                print(f"  还有 {len(gaps)-5} 个间隙未显示")
        else:
            print("步数连续正常")

        # 检查是否有重复步数
        step_set = set()
        duplicates = []
        for step, loss, lr in steps:
            if step in step_set:
                duplicates.append(step)
            step_set.add(step)

        if duplicates:
            print(f"发现 {len(duplicates)} 个重复步数: {sorted(set(duplicates))[:10]}")
        else:
            print("无重复步数")

        return steps[-1][0]  # 返回最后一个步数

    return 0

def check_for_corruption(log_path):
    """检查文件是否损坏"""
    log_path = Path(log_path)
    with open(log_path, 'rb') as f:
        data = f.read()

    # 检查空行和异常字符
    lines = data.split(b'\n')
    empty_lines = 0
    corrupted_lines = 0

    for i, line in enumerate(lines):
        if not line.strip():
            empty_lines += 1
        # 检查是否有非ASCII字符（在CSV中可能是问题）
        try:
            line.decode('utf-8')
        except UnicodeDecodeError:
            corrupted_lines += 1
            if corrupted_lines <= 3:
                print(f"第{i+1}行编码错误: {line[:50]}")

    if empty_lines:
        print(f"发现 {empty_lines} 个空行")
    if corrupted_lines:
        print(f"发现 {corrupted_lines} 行编码错误")

    return empty_lines > 0 or corrupted_lines > 0

def main():
    """主函数"""
    log_path = Path(__file__).parent.parent / "models" / "logs" / "step_log.csv"

    print("=" * 60)
    print("步数日志分析工具")
    print("=" * 60)

    if not log_path.exists():
        print(f"错误: 日志文件不存在: {log_path}")
        sys.exit(1)

    # 检查文件损坏
    print("\n1. 检查文件完整性...")
    if check_for_corruption(log_path):
        print("⚠️  文件可能损坏")
    else:
        print("✅ 文件完整性正常")

    # 分析步数
    print("\n2. 分析步数记录...")
    last_step = analyze_step_log(log_path)

    if last_step is not None:
        print(f"\n3. 建议:")
        print(f"   最后训练步数: {last_step}")
        print(f"   下次训练应从步数: {last_step + 1} 开始")

        # 检查是否有检查点文件
        models_dir = Path(__file__).parent.parent / "models"
        checkpoint_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("checkpoint*"))
        if checkpoint_files:
            print(f"\n   找到 {len(checkpoint_files)} 个检查点文件:")
            for cf in sorted(checkpoint_files)[:5]:
                print(f"     - {cf.name}")
            if len(checkpoint_files) > 5:
                print(f"     还有 {len(checkpoint_files)-5} 个未显示")
        else:
            print("\n   未找到检查点文件")

    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)

if __name__ == "__main__":
    main()