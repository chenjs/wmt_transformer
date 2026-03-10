#!/usr/bin/env python3
"""
测试步数恢复逻辑。
"""
import sys
from pathlib import Path
import tempfile
import csv

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入修改后的函数
from scripts.train import get_last_step_from_log

def test_get_last_step_from_log():
    """测试日志读取函数"""
    print("测试 get_last_step_from_log 函数")
    print("=" * 50)

    # 测试1: 空目录
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        result = get_last_step_from_log(log_dir)
        print(f"测试1 - 空目录: {result} (期望: 0)")
        assert result == 0, f"期望0, 得到{result}"

    # 测试2: 空日志文件
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        step_log = log_dir / "step_log.csv"
        step_log.write_text("step,loss,lr\n")
        result = get_last_step_from_log(log_dir)
        print(f"测试2 - 空日志文件: {result} (期望: 0)")
        assert result == 0, f"期望0, 得到{result}"

    # 测试3: 正常日志文件
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        step_log = log_dir / "step_log.csv"

        # 创建测试日志
        with open(step_log, 'w', encoding='utf-8') as f:
            f.write("step,loss,lr\n")
            for i in range(1, 11):
                f.write(f"{i},{i*0.1:.4f},0.001\n")

        result = get_last_step_from_log(log_dir)
        print(f"测试3 - 正常日志(1-10步): {result} (期望: 10)")
        assert result == 10, f"期望10, 得到{result}"

    # 测试4: 不连续步数
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        step_log = log_dir / "step_log.csv"

        with open(step_log, 'w', encoding='utf-8') as f:
            f.write("step,loss,lr\n")
            f.write("1,0.1000,0.001\n")
            f.write("5,0.5000,0.001\n")
            f.write("10,1.0000,0.001\n")
            f.write("8,0.8000,0.001\n")  # 非顺序

        result = get_last_step_from_log(log_dir)
        print(f"测试4 - 不连续步数: {result} (期望: 10)")
        assert result == 10, f"期望10, 得到{result}"

    # 测试5: 有错误行的日志
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        step_log = log_dir / "step_log.csv"

        with open(step_log, 'w', encoding='utf-8') as f:
            f.write("step,loss,lr\n")
            f.write("1,0.1000,0.001\n")
            f.write("invalid,line,here\n")
            f.write("2,0.2000,0.001\n")
            f.write("3,0.3000,not_a_number\n")
            f.write("5,0.5000,0.001\n")

        result = get_last_step_from_log(log_dir)
        print(f"测试5 - 有错误行的日志: {result} (期望: 5)")
        assert result == 5, f"期望5, 得到{result}"

    print("\n" + "=" * 50)
    print("✅ 所有测试通过!")
    print("=" * 50)

def test_actual_log_file():
    """测试实际日志文件"""
    print("\n测试实际日志文件")
    print("=" * 50)

    log_dir = Path(__file__).parent.parent / "models" / "logs"
    result = get_last_step_from_log(log_dir)

    print(f"实际日志目录: {log_dir}")
    print(f"检测到的最后步数: {result}")

    # 手动验证
    step_log = log_dir / "step_log.csv"
    if step_log.exists():
        with open(step_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 1:
                # 尝试解析最后一行
                last_line = lines[-1].strip()
                if last_line:
                    parts = last_line.split(',')
                    if len(parts) >= 1 and parts[0].isdigit():
                        manual_last = int(parts[0])
                        print(f"手动解析最后一行步数: {manual_last}")

    print("=" * 50)

if __name__ == "__main__":
    test_get_last_step_from_log()
    test_actual_log_file()