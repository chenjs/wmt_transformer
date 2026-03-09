"""
实验方案B配置: 显著增加模型容量
目标: 最大化翻译质量提升
预期: 基础词汇准确率 +30-40%, BLEU +4-6点
警告: 计算需求显著增加，可能需要梯度累积或减小batch_size
"""

# 模型架构参数
d_model = 768      # 增加50% (原512)
n_layers = 8       # 增加2层 (原6)
n_heads = 12       # 增加50% (原8)
d_ff = 3072        # 增加50% (原2048)
dropout = 0.15     # 增加以防止过拟合 (原0.1)
max_len = 54       # 保持不变

# 训练参数 (可能需要调整以适应更大模型)
batch_size = 8          # 减小以适应更大内存需求 (原12)
learning_rate = 8e-4    # 略微减小学习率 (原1e-3)
warmup_steps = 16000    # 增加以适应更大模型 (原8000)
max_steps = 200000      # 完整训练步数
label_smoothing = 0.15  # 增加以改进泛化 (原0.1)
clip_grad = 5.0         # 减小梯度裁剪 (原10.0)

# 数据参数
train_split = 0.99      # 保持不变
max_train_samples = 200000  # 保持不变

# 检查点参数
save_interval = 10000   # 保持不变
eval_interval = 5000    # 保持不变
min_loss_improvement = 0.01  # 保持不变

# 梯度累积 (如果内存不足)
gradient_accumulation_steps = 2  # 如果需要可启用

# 实验元数据
experiment_name = "exp_b"
description = "显著增加模型容量: d_model=768, n_layers=8, n_heads=12, d_ff=3072"
expected_improvement = {
    "basic_vocab_accuracy": "+30-40%",
    "bleu_score": "+4-6 points",
    "ter_improvement": "-0.20",
    "training_time_increase": "+80%",
    "inference_latency_increase": "+60%"
}

# 启动命令示例 (包含梯度累积选项)
start_command = """
python scripts/train_experiment.py \
  --d-model 768 \
  --n-layers 8 \
  --n-heads 12 \
  --d-ff 3072 \
  --dropout 0.15 \
  --batch-size 8 \
  --learning-rate 8e-4 \
  --warmup-steps 16000 \
  --max-steps 100000 \
  --label-smoothing 0.15 \
  --clip-grad 5.0 \
  --resume models/best_model.pt \
  --experiment-name exp_b
"""

# 内存优化选项
memory_optimization = {
    "gradient_checkpointing": True,  # 如果启用需要修改模型
    "mixed_precision": True,         # 混合精度训练
    "gradient_accumulation": 2,      # 梯度累积步数
}

if __name__ == "__main__":
    print(f"实验方案B配置")
    print(f"描述: {description}")
    print(f"警告: 此配置显著增加计算需求，确保有足够GPU内存")
    print(f"启动命令: {start_command.strip()}")