# 模型容量实验计划

**创建时间:** 2026-03-08
**目标:** 通过增加模型容量提升翻译质量，特别是基础词汇准确率和整体BLEU分数
**当前模型:** 318,262步训练完成，验证损失2.08，基础词汇准确率待分析
**问题识别:** 基础词汇翻译不准确，BLEU分数仅21.06，TER较高(0.73)

## 📊 当前模型配置 (基准)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| **d_model** | 512 | 模型维度 |
| **n_layers** | 6 | Transformer层数 |
| **n_heads** | 8 | 注意力头数 |
| **d_ff** | 2048 | 前馈网络维度 |
| **dropout** | 0.1 | 丢弃率 |
| **总参数** | 68.7M | 模型参数量 |

## 🎯 实验目标

1. **基础词汇准确率**: 从当前<50%提升至>80%
2. **BLEU分数**: 从21.06提升至>25.0
3. **TER指标**: 从0.73降低至<0.5
4. **验证损失**: 从2.08降低至<2.0

## 🔬 实验方案设计

### 方案A: 温和增加容量 (推荐优先尝试)
| 参数 | 新值 | 增加幅度 | 预期效果 |
|------|------|----------|----------|
| **d_model** | 640 | +25% | 提升语义表示能力 |
| **n_layers** | 7 | +17% | 增加模型深度 |
| **d_ff** | 2560 | +25% | 增强前馈网络 |
| **总参数** | ~95M | +38% | 中等复杂度增加 |

**优势**: 平衡计算成本和性能提升，适合当前数据集规模

### 方案B: 显著增加容量
| 参数 | 新值 | 增加幅度 | 预期效果 |
|------|------|----------|----------|
| **d_model** | 768 | +50% | 显著提升表示能力 |
| **n_layers** | 8 | +33% | 深度显著增加 |
| **n_heads** | 12 | +50% | 提升注意力多样性 |
| **d_ff** | 3072 | +50% | 增强前馈网络 |
| **总参数** | ~135M | +96% | 接近翻倍 |

**优势**: 最大性能潜力，适合追求最佳翻译质量

### 方案C: 专项优化 (针对基础词汇)
| 参数 | 新值 | 增加幅度 | 预期效果 |
|------|------|----------|----------|
| **d_model** | 512 | 0% | 保持相同 |
| **n_layers** | 8 | +33% | 增加深度 |
| **n_heads** | 8 | 0% | 保持相同 |
| **d_ff** | 2048 | 0% | 保持相同 |
| **训练数据** | 基础词汇增强 | - | 针对性改进 |

**优势**: 针对基础词汇问题，计算成本最低

## 🛠️ 训练策略

### 1. 训练起点
- **选项1**: 从头开始训练
  - 优点: 避免当前模型的偏差
  - 缺点: 训练时间最长
- **选项2**: 从当前模型迁移 (推荐)
  - 优点: 利用已有知识，加速收敛
  - 缺点: 需要处理架构不匹配

### 2. 迁移学习方案
```python
# 处理架构不匹配的策略
1. 扩展维度: 新维度参数随机初始化
2. 复制参数: 重复现有参数到新维度
3. 平均参数: 对新增维度使用平均值
4. 零初始化: 新增参数从零开始学习
```

### 3. 训练步数
- **初步目标**: 100,000步 (验证收敛性)
- **完整训练**: 200,000-300,000步 (达到最佳性能)
- **微调阶段**: 额外50,000步针对基础词汇

## 📈 预期性能提升

| 指标 | 方案A | 方案B | 方案C |
|------|-------|-------|-------|
| **基础词汇准确率** | +20-30% | +30-40% | +15-25% |
| **BLEU分数** | +3-4点 | +4-6点 | +1-2点 |
| **TER改进** | -0.15 | -0.20 | -0.10 |
| **训练时间增加** | +40% | +80% | +20% |
| **推理延迟增加** | +30% | +60% | +10% |

## 🧪 实验执行步骤

### 阶段1: 准备 (1天)
1. **创建实验配置**
   ```python
   # config_experiment_a.py
   d_model = 640
   n_layers = 7
   d_ff = 2560
   n_heads = 8
   ```

2. **修改训练脚本支持配置参数**
   ```bash
   python scripts/train_experiment.py --config config_experiment_a.py
   ```

3. **准备迁移学习工具**
   - 创建模型参数映射工具
   - 准备架构扩展脚本

### 阶段2: 实验A执行 (3-5天)
1. **启动方案A训练**
   ```bash
   python scripts/train_experiment.py --config config_experiment_a.py \
     --resume models/best_model.pt --steps 100000
   ```

2. **监控训练过程**
   - 每10,000步评估基础词汇
   - 每20,000步全面评估

3. **达到100,000步后分析**
   - 与基准模型对比
   - 决定是否继续训练

### 阶段3: 实验B执行 (可选)
如果方案A效果显著但未达到目标:
1. **启动方案B训练**
2. **使用方案A最佳检查点作为起点**
3. **训练100,000-150,000步**

### 阶段4: 实验C执行 (并行)
针对基础词汇问题:
1. **创建基础词汇增强数据集**
2. **微调最佳模型**
3. **评估针对性改进效果**

## 🏗️ 技术实现

### 1. 配置系统修改
```python
# 当前: src/config.py 中的硬编码配置
# 修改为: 支持外部配置文件
import argparse
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    # ... 其他参数
```

### 2. 模型加载器修改
```python
def load_model_with_config(checkpoint_path, experiment_config):
    """根据实验配置加载或创建模型"""
    # 1. 加载现有检查点
    checkpoint = torch.load(checkpoint_path)

    # 2. 创建新架构模型
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

    # 3. 迁移参数 (处理维度不匹配)
    model = transfer_parameters(checkpoint['model_state_dict'], model)

    return model
```

### 3. 参数迁移策略
```python
def transfer_parameters(old_state_dict, new_model):
    """将参数从旧模型迁移到新模型架构"""
    new_state_dict = new_model.state_dict()

    for name, param in old_state_dict.items():
        if name in new_state_dict:
            old_shape = param.shape
            new_shape = new_state_dict[name].shape

            if old_shape == new_shape:
                # 直接复制
                new_state_dict[name].copy_(param)
            elif len(old_shape) == 2 and len(new_shape) == 2:
                # 矩阵维度扩展 (如d_model增加)
                if old_shape[0] < new_shape[0]:
                    # 扩展行维度
                    new_state_dict[name][:old_shape[0], :old_shape[1]] = param
                elif old_shape[1] < new_shape[1]:
                    # 扩展列维度
                    new_state_dict[name][:old_shape[0], :old_shape[1]] = param
            # ... 其他维度处理

    new_model.load_state_dict(new_state_dict)
    return new_model
```

## 📋 成功标准

### 主要成功指标
1. **基础词汇准确率 ≥ 80%** (关键目标)
2. **BLEU分数 ≥ 25.0**
3. **TER ≤ 0.5**
4. **验证损失 < 2.0**

### 次要成功指标
1. 训练收敛速度不慢于基准50%
2. 推理延迟增加不超过100%
3. 模型稳定性良好 (无训练崩溃)

## 🚨 风险与缓解

### 风险1: 训练不收敛
- **原因**: 架构变化过大，学习率不合适
- **缓解**:
  1. 使用更小的学习率 (5e-4)
  2. 增加warmup步数 (16,000)
  3. 从方案A开始，逐步增加复杂度

### 风险2: 过拟合
- **原因**: 模型容量增加，数据量不变
- **缓解**:
  1. 增加dropout (0.15-0.2)
  2. 使用更强的标签平滑 (0.15)
  3. 早停策略

### 风险3: 计算资源不足
- **原因**: 参数增加导致内存和计算需求增加
- **缓解**:
  1. 使用梯度累积
  2. 减小batch_size (从12降到8)
  3. 使用混合精度训练

## 📅 时间计划

| 阶段 | 任务 | 预计时间 | 负责人 |
|------|------|----------|--------|
| 准备 | 创建实验配置和脚本 | 1天 | Claude Code |
| 实验A | 运行方案A (100k步) | 3-5天 | 系统 |
| 评估A | 分析方案A结果 | 0.5天 | Claude Code |
| 实验B | 运行方案B (可选) | 4-6天 | 系统 |
| 评估B | 分析方案B结果 | 0.5天 | Claude Code |
| 实验C | 基础词汇微调 | 2-3天 | 系统 |
| 总结 | 撰写实验报告 | 1天 | Claude Code |

**总时间**: 7-11天

## 🎯 优先级建议

1. **立即执行方案A**: 平衡风险和收益，最有可能成功
2. **并行准备方案C**: 针对基础词汇问题，可与方案A并行
3. **方案B作为后备**: 如果方案A效果不足，再执行方案B

## 📁 文件结构

```
scripts/
  train_experiment.py           # 实验训练脚本
  analyze_experiment_results.py # 实验分析工具
  transfer_parameters.py        # 参数迁移工具

configs/
  experiment_a.py              # 方案A配置
  experiment_b.py              # 方案B配置
  experiment_c.py              # 方案C配置

experiments/
  exp_a/                       # 方案A结果
    logs/
    checkpoints/
    evaluation/
  exp_b/                       # 方案B结果
  exp_c/                       # 方案C结果
```

## 📝 下一步行动

### 立即行动 (今天)
1. ✅ 创建本实验计划文档
2. ✅ 创建基础词汇分析脚本并运行
3. 🔄 创建实验训练脚本 (`train_experiment.py`)

### 短期行动 (1-2天)
1. 运行基础词汇分析，获取准确基准
2. 创建并测试实验配置系统
3. 启动方案A实验 (100,000步目标)

### 中期行动 (3-7天)
1. 监控实验A进展
2. 每20,000步评估性能
3. 根据结果调整策略

---
**计划创建**: Claude Code
**创建时间**: 2026-03-08
**参考模型**: `models/best_model.pt` (318,262步)
**数据来源**: `models_enhanced/` 清洗后数据