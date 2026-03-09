"""
实验方案A配置: 温和增加模型容量
目标: 提升翻译质量，特别是基础词汇准确率
预期: 基础词汇准确率 +20-30%, BLEU +3-4点
"""

# 模型架构参数
d_model = 640      # 增加25% (原512)
n_layers = 7       # 增加1层 (原6)
n_heads = 8        # 保持不变
d_ff = 2560        # 增加25% (原2048)
dropout = 0.1      # 保持不变
max_len = 54       # 保持不变

# 训练参数
batch_size = 12         # 保持不变 (如果内存不足可减少到8)
learning_rate = 1e-3    # 保持不变
warmup_steps = 10000    # 略微增加 (原8000)
max_steps = 200000      # 完整训练步数 (实验目标100000步)
label_smoothing = 0.1   # 保持不变
clip_grad = 10.0        # 保持不变

# 数据参数
train_split = 0.99      # 保持不变
max_train_samples = 200000  # 保持不变

# 检查点参数
save_interval = 10000   # 保持不变
eval_interval = 5000    # 保持不变
min_loss_improvement = 0.01  # 保持不变

# 实验元数据
experiment_name = "exp_a"
description = "温和增加模型容量: d_model=640, n_layers=7, d_ff=2560"
expected_improvement = {
    "basic_vocab_accuracy": "+20-30%",
    "bleu_score": "+3-4 points",
    "ter_improvement": "-0.15",
    "training_time_increase": "+40%",
    "inference_latency_increase": "+30%"
}

# 启动命令示例
start_command = """
python scripts/train_experiment.py \
  --d-model 640 \
  --n-layers 7 \
  --d-ff 2560 \
  --max-steps 100000 \
  --resume models/best_model.pt \
  --experiment-name exp_a
"""

if __name__ == "__main__":
    print(f"实验方案A配置")
    print(f"描述: {description}")
    print(f"启动命令: {start_command.strip()}")