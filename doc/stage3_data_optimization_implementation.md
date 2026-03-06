# 阶段3数据优化实施方案
**日期:** 2026-03-01
**分支:** training-optimization

## 概述
基于数据质量分析，实施全面数据优化（选项A）。包括数据清洗、tokenizer优化和配置更新。

## 当前状态
- ✅ 验证监控修复完成（阶段3-1）
- 🔄 增强预处理脚本运行中（处理200,000样本）
- ⏳ 等待预处理完成，获取优化配置建议

## 实施步骤

### 步骤1：完成增强预处理
**脚本:** `scripts/preprocess_enhanced.py`
**状态:** 运行中
**输出目录:** `models_enhanced/`

**预期输出:**
1. 清洁数据文件：`src_text_cleaned.txt`, `tgt_text_cleaned.txt`
2. 优化tokenizer：`src_tokenizer_final.model`, `tgt_tokenizer_final.model`
3. 配置建议：最佳vocab_size、max_len等参数

### 步骤2：更新配置文件
**需要更新的文件:** `src/config.py`

**需要更新的参数:**
```python
# 当前值
vocab_size: int = 32000
src_vocab_size: int = 32000
tgt_vocab_size: int = 32000
max_len: int = 32
src_tokenizer: str = ""  # 实际使用 "models/src_tokenizer.model"
tgt_tokenizer: str = ""  # 实际使用 "models/tgt_tokenizer.model"

# 更新为（示例，基于预处理输出）
vocab_size: int = 16000  # 或20000、24000
src_vocab_size: int = 16000
tgt_vocab_size: int = 16000
max_len: int = 48  # 基于清洁数据90百分位长度
src_tokenizer: str = "models_enhanced/src_tokenizer_final.model"
tgt_tokenizer: str = "models_enhanced/tgt_tokenizer_final.model"
```

### 步骤3：更新训练脚本
**文件:** `scripts/train.py`
**修改:** 更新tokenizer路径引用

**当前代码 (train.py 第43-44行):**
```python
config.src_tokenizer = "models/src_tokenizer.model"
config.tgt_tokenizer = "models/tgt_tokenizer.model"
```

**更新为:**
```python
config.src_tokenizer = "models_enhanced/src_tokenizer_final.model"
config.tgt_tokenizer = "models_enhanced/tgt_tokenizer_final.model"
```

### 步骤4：备份现有模型和tokenizer
**建议操作:**
```bash
# 备份现有模型目录
mv models models_original

# 使用增强tokenizer和模型目录
cp -r models_enhanced models
```

**或保持分离:**
- 保留`models_original/` - 原始tokenizer和检查点
- 使用`models_enhanced/` - 新tokenizer
- 新训练检查点可保存到`models/`或新目录

### 步骤5：验证优化效果
**测试步骤:**
1. 加载新tokenizer测试编码/解码功能
2. 验证配置参数是否正确加载
3. 运行小规模训练测试（如100步）
4. 检查训练日志和验证监控

### 步骤6：重新训练模型
**选项A:** 从头开始训练
- 使用清洁数据和优化tokenizer
- 监控验证损失改进

**选项B:** 微调现有模型
- 加载现有检查点
- 适配新vocab_size（可能需要调整embedding层）
- 继续训练

**推荐:** 选项A（从头训练），因为：
1. vocab_size改变需要调整embedding层
2. 清洁数据质量显著提升
3. 可建立新的性能基准

## 预期改进指标

| 指标 | 优化前 | 优化目标 | 测量方法 |
|------|--------|----------|----------|
| **词汇表覆盖率** | 40-58% | >70% | tokenizer使用统计 |
| **数据清洁度** | 86.9%德语行有"非打印字符" | <1% | 字符编码分析 |
| **句子截断率** | ~10% (max_len=32) | <5% | 长度分布分析 |
| **训练效率** | 当前收敛曲线 | 更快收敛 | 损失下降速度 |
| **验证损失** | 3.0364 (step 215,000) | 降低10-20% | 验证集评估 |

## 风险缓解

### 风险1：新tokenizer与现有模型不兼容
**缓解:** 采用选项A（从头训练），避免适配问题

### 风险2：清洁数据样本不足
**缓解:** 分析显示保留~98%样本（196,984/200,000），足够训练

### 风险3：配置错误导致训练失败
**缓解:** 先运行小规模测试（100步），验证配置正确性

### 风险4：内存增加（max_len增加）
**缓解:** 监控内存使用，必要时调整batch_size

## 时间估计

| 任务 | 时间 | 状态 |
|------|------|------|
| 增强预处理 | 30-60分钟 | 🔄 进行中 |
| 配置更新 | 10分钟 | ⏳ 待完成 |
| 小规模测试 | 15分钟 | ⏳ 待完成 |
| 完整训练 | 2-3天 | ⏳ 待完成 |
| 评估优化效果 | 1小时 | ⏳ 待完成 |

## 验证检查点

**预处理完成后检查:**
1. ✅ `models_enhanced/`目录包含所有文件
2. ✅ tokenizer文件可正常加载
3. ✅ 清洁数据统计符合预期
4. ✅ 配置建议合理

**配置更新后检查:**
1. ✅ `src/config.py`参数正确
2. ✅ `scripts/train.py`路径正确
3. ✅ 可正常导入配置

**训练验证检查:**
1. ✅ 模型可正常创建（新vocab_size）
2. ✅ 数据加载正常
3. ✅ 前向传播正常
4. ✅ 损失计算正常

## 备份策略

**重要文件备份:**
1. `models/` → `models_original/` (现有tokenizer和检查点)
2. `src/config.py` → `src/config_original.py`
3. `scripts/train.py` → `scripts/train_original.py`

**Git提交:** 创建提交点记录优化前状态

## 后续步骤

### 阶段3-3：模型容量实验（待开始）
基于清洁数据，实验不同模型配置：
1. `d_model=768`, `n_layers=6`
2. `d_model=512`, `n_layers=8`
3. `d_model=768`, `n_layers=8` (内存允许时)

### 阶段3-4：训练策略优化（待开始）
实验不同训练策略：
1. 课程学习（curriculum learning）
2. 动态批次大小
3. 自适应学习率调度

## 文档更新
- `doc/stage3_progress_update.md` - 记录实施进度
- `doc/training_optimization_plan.md` - 更新完成状态
- `doc/data_optimization_report.md` - 创建详细优化报告

---

**等待:** 增强预处理脚本完成，获取具体配置建议
**下一步:** 根据预处理输出更新配置文件，开始小规模测试