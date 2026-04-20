#!/usr/bin/env python3
"""
检查检查点文件内容。
"""
import sys
from pathlib import Path

# 添加父目录到路径，以便导入src模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

def main():
    checkpoint_path = Path(__file__).parent.parent / "models" / "checkpoint_interrupted.pt"

    if not checkpoint_path.exists():
        print(f"检查点文件不存在: {checkpoint_path}")
        sys.exit(1)

    print(f"检查检查点文件: {checkpoint_path}")
    print(f"文件大小: {checkpoint_path.stat().st_size:,} 字节")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"加载检查点失败: {e}")
        sys.exit(1)

    print(f"\n检查点键: {list(checkpoint.keys())}")

    # 检查重要字段
    important_keys = ['step', 'scheduler_step_num', 'model_state_dict', 'optimizer_state_dict', 'config']
    for key in important_keys:
        if key in checkpoint:
            value = checkpoint[key]
            if key == 'step':
                print(f"  {key}: {value} (类型: {type(value).__name__})")
            elif key == 'scheduler_step_num':
                print(f"  {key}: {value} (类型: {type(value).__name__})")
            elif key == 'config':
                print(f"  {key}: 存在 (类型: {type(value).__name__})")
                if hasattr(value, 'save_interval'):
                    print(f"    配置: save_interval={value.save_interval}")
            else:
                print(f"  {key}: 存在 (类型: {type(value).__name__})")
        else:
            print(f"  {key}: 不存在")

    # 检查模型状态
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        print(f"\n模型状态键数量: {len(model_state)}")
        # 显示一些参数形状
        for key in list(model_state.keys())[:3]:
            tensor = model_state[key]
            print(f"  {key}: {tuple(tensor.shape)}")

    # 检查优化器状态
    if 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        print(f"\n优化器状态键: {list(opt_state.keys())}")
        if 'state' in opt_state:
            print(f"  状态条目数: {len(opt_state['state'])}")

    print("\n" + "=" * 50)
    print("检查完成")

if __name__ == "__main__":
    main()