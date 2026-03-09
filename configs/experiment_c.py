"""
实验方案C配置: 针对基础词汇优化的专项实验
目标: 显著提升基础词汇翻译准确率
紧急程度: 高 (基础词汇准确率仅1.8-2.7%)
方法: 保持模型架构不变，专注于数据增强和训练策略优化
"""

# 模型架构参数 (保持不变)
d_model = 512      # 保持不变
n_layers = 6       # 保持不变
n_heads = 8        # 保持不变
d_ff = 2048        # 保持不变
dropout = 0.1      # 保持不变
max_len = 54       # 保持不变

# 训练参数 (优化针对基础词汇)
batch_size = 16         # 增加以加速训练
learning_rate = 1e-3    # 保持不变
warmup_steps = 4000     # 减少，基础词汇学习较快
max_steps = 50000       # 较少的步数，专注于基础词汇
label_smoothing = 0.05  # 减少，基础词汇需要更精确的输出
clip_grad = 5.0         # 减小梯度裁剪

# 数据增强参数
basic_vocab_repeat = 10  # 基础词汇在训练数据中重复次数
augment_basic_data = True  # 启用基础词汇数据增强
use_basic_vocab_dataset = True  # 使用专门的基础词汇数据集

# 检查点参数
save_interval = 5000     # 更频繁保存
eval_interval = 2500     # 更频繁评估基础词汇
min_loss_improvement = 0.005  # 更敏感的最佳模型保存

# 基础词汇数据路径
basic_vocab_src = "data_basic/basic_vocab.en"
basic_vocab_tgt = "data_basic/basic_vocab.de"

# 实验元数据
experiment_name = "exp_c"
description = "基础词汇专项优化: 数据增强和训练策略调整"
current_performance = {
    "exact_match_rate": "2.7% (3/110)",
    "semantic_similarity": "0.20 (平均)",
    "keyword_match_rate": "0.9% (1/110)",
    "problematic_vocab": "107个词汇语义相似度<0.3"
}
target_performance = {
    "exact_match_rate": ">80% (>88/110)",
    "semantic_similarity": ">0.70",
    "keyword_match_rate": ">90% (>99/110)",
    "improvement_needed": "40倍提升"
}

# 启动命令示例
start_command = """
# 步骤1: 创建基础词汇增强数据集
python scripts/create_basic_vocab_dataset.py \
  --repeat 10 \
  --output models_enhanced/basic_augmented/

# 步骤2: 微调现有模型 (推荐)
python scripts/train_experiment.py \
  --max-steps 50000 \
  --batch-size 16 \
  --warmup-steps 4000 \
  --label-smoothing 0.05 \
  --clip-grad 5.0 \
  --save-interval 5000 \
  --eval-interval 2500 \
  --resume models/best_model.pt \
  --experiment-name exp_c

# 步骤3: 评估改进
python scripts/analyze_basic_vocab.py \
  --checkpoint experiments/exp_c/checkpoints/best_model.pt
"""

# 数据增强策略
data_augmentation_strategies = [
    "重复基础词汇: 在训练数据中重复基础词汇10-20次",
    "同义词替换: 创建基础词汇的变体 (如 'Hello' -> 'Hi', 'Hey')",
    "上下文扩展: 将基础词汇放入简单句子中",
    "混合训练: 将基础词汇数据与原始数据混合",
    "课程学习: 先训练基础词汇，再逐渐增加复杂度"
]

# 训练策略调整
training_strategies = [
    "专注损失: 为基础词汇添加额外的损失权重",
    "早停机制: 当基础词汇准确率达到目标时停止",
    "渐进学习率: 为基础词汇使用更高的学习率",
    "定期评估: 每1000步评估基础词汇准确率",
    "检查点选择: 基于基础词汇准确率选择最佳模型"
]

if __name__ == "__main__":
    print(f"实验方案C配置")
    print(f"描述: {description}")
    print(f"\n⚠️ 紧急: 当前基础词汇准确率仅{current_performance['exact_match_rate']}")
    print(f"目标: {target_performance['exact_match_rate']}")
    print(f"需要改进: {target_performance['improvement_needed']}")
    print(f"\n数据增强策略:")
    for i, strategy in enumerate(data_augmentation_strategies[:3], 1):
        print(f"  {i}. {strategy}")
    print(f"\n训练策略:")
    for i, strategy in enumerate(training_strategies[:3], 1):
        print(f"  {i}. {strategy}")