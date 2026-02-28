# MPS内存优化指南

## 问题分析

训练在1028步后出现MPS内存不足错误：
```
MPS backend out of memory (MPS allocated: 7.30 GiB, other allocations: 1.55 GiB, max allowed: 9.07 GiB)
```

## 已实施的解决方案

### 1. 减少批次大小和序列长度

**修改文件**: `src/config.py`

| 参数 | 原值 | 新值 | 内存减少倍数 |
|------|------|------|-------------|
| batch_size | 32 | 8 | 4× |
| max_len | 64 | 32 | 2× |

**总内存减少**: 约8×（理论上）

### 2. 禁用标签平滑

标签平滑已设置为0.0，避免创建大型true_dist张量。

## 重新开始训练

### 步骤1：清理旧检查点
```bash
rm models/best_model.pt
rm models/checkpoint_*.pt
```

### 步骤2：开始训练
```bash
python scripts/train.py
```

## 备选方案（如果仍然OOM）

### 方案A：进一步减小配置

编辑 `src/config.py`:

```python
# 进一步减小
batch_size: int = 4
max_len: int = 24

# 或减小模型大小
d_model: int = 256  # 从512减小
n_layers: int = 4   # 从6减小
```

### 方案B：使用CPU训练（更慢但更稳定）

编辑 `src/config.py`:
```python
device: str = "cpu"  # 从"mps"改为"cpu"
```

### 方案C：梯度累积（保持有效batch_size）

修改 `src/trainer.py` 添加梯度累积（需要代码修改）。

### 方案D：启用混合精度训练（AMP）

需要修改训练循环使用 `torch.cuda.amp`（MPS支持有限）。

## 内存使用分析

### 主要内存消耗
1. **输出投影层**: `[batch_size * seq_len, vocab_size]`
   - batch_size=8, seq_len=32, vocab_size=32000 → [256, 32000] ≈ 32MB
   - 之前：batch_size=32, seq_len=64 → [2048, 32000] ≈ 262MB

2. **标签平滑张量**: 相同大小（当smoothing>0时）

3. **中间激活**: 12层（6 encoder + 6 decoder）的attention和FFN中间结果

### 估计内存使用
- 模型参数: ~250MB
- 梯度: ~250MB
- 优化器状态: ~500MB
- 激活值: ~1-2GB（取决于序列长度和批次大小）
- 总计: ~2-3GB（应适合MPS的9GB限制）

## 监控内存使用

### 方法1：使用MPS内存监控
```python
import torch
print(torch.mps.current_allocated_memory() / 1024**3)  # GB
print(torch.mps.driver_allocated_memory() / 1024**3)   # GB
```

### 方法2：定期检查点
每1000步保存检查点，如果OOM可以从最近检查点恢复。

## 恢复训练建议

### 如果希望从step 1028继续
虽然配置已改变，但可以尝试：
```bash
# 用旧检查点恢复，但新配置
python scripts/train.py --resume best_model.pt
```

**注意**: 由于batch_size和max_len改变，优化器状态可能不完全兼容。

### 推荐：重新开始训练
由于早期训练（1028步）相对容易恢复，建议重新开始。

## 高级优化选项

### 1. 梯度检查点（Activation Checkpointing）
```python
from torch.utils.checkpoint import checkpoint

# 在模型forward中包装计算密集型部分
output = checkpoint(self.attention, x_norm, x_norm, x_norm, mask)
```

### 2. 更高效的自定义损失
当label_smoothing=0.0时，使用`nn.CrossEntropyLoss`替代自定义实现。

### 3. 稀疏注意力（未来考虑）
使用局部注意力或稀疏模式减少计算和内存。

## 故障排除

### 错误：仍然OOM
1. 检查实际批次大小：确保data loader正确限制序列长度
2. 监控每一步的内存使用
3. 尝试CPU训练验证是否代码问题

### 错误：训练缓慢
1. MPS可能在某些操作上不如CUDA高效
2. 考虑使用CPU或减小模型复杂度
3. 确保使用最新PyTorch版本

### 错误：精度下降
1. MPS可能有数值精度差异
2. 考虑使用CPU进行最终训练
3. 验证损失曲线是否正常下降

## 推荐配置

对于16GB M1/M2 Mac:
```python
batch_size: int = 8
max_len: int = 32
d_model: int = 512  # 保持原质量
n_layers: int = 6   # 保持原深度
device: str = "mps"  # 优先使用GPU
```

如果仍然有问题，尝试：
```python
batch_size: int = 4
max_len: int = 24
d_model: int = 256  # 降低质量换取内存
device: str = "cpu"  # 回退到CPU
```

## 结论

已实施最直接的内存优化（减小batch_size和max_len）。应能解决MPS内存不足问题。如果问题持续，建议按上述备选方案逐步优化。

关键：监控训练早期（前1000步）的内存使用和损失下降。