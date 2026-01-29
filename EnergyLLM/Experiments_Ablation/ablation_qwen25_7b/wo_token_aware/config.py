"""
消融实验: w/o Token-aware (Qwen2.5-7B版本)
去掉 token 长度特征，验证 token 感知的价值
状态空间从 9 维减少到 7 维（去掉 avg_input, avg_output）
"""

import os

# 路径配置 - 指向 Qwen2.5-7B 的结果目录
BASE_DIR = "/home/data/Fjw/test/proposed/"
RESULTS_DIR = os.path.join(BASE_DIR, "results_qwen25_7b")  # Qwen2.5-7B 模型结果
RL_DIR = os.path.join(BASE_DIR, "ablation_qwen25_7b/wo_token_aware")
DATASET_PATH = os.path.join(BASE_DIR, "prompt_datasets_jsonl/sharegpt_en_mixed_all_buckets.jsonl")

# 模型路径 - 使用 Qwen2.5-7B 的预测模型
LATENCY_MODEL_PATH = os.path.join(RESULTS_DIR, "latency_model.joblib")
POWER_MODEL_PATH = os.path.join(RESULTS_DIR, "power_model.joblib")
MODEL_META_PATH = os.path.join(RESULTS_DIR, "model_meta.joblib")
PROFILE_DATA_PATH = os.path.join(RESULTS_DIR, "performance_profile.csv")

# DVFS 配置空间（与 Llama 版本一致）
FREQ_OPTIONS = [705, 810, 900, 960, 1005, 1050, 1110, 1155, 1200, 1245, 1290, 1320, 1350, 1380, 1410]
BATCH_OPTIONS = [1, 2, 4, 8, 16, 32, 64]

# ============================================
# 消融: 去掉 token 长度特征
# 状态空间从 9 维减少到 7 维
# ============================================
STATE_DIM = 7  # 消融: 去掉 avg_input, avg_output

# 动作空间
NUM_FREQS = len(FREQ_OPTIONS)  # 15
NUM_BATCHES = len(BATCH_OPTIONS)  # 7
ACTION_DIM = NUM_FREQS * NUM_BATCHES  # 105

# 环境配置
DEFAULT_SLO = 12.0
MAX_QUEUE_SIZE = 100
MAX_EPISODE_STEPS = 200
FREQ_SWITCH_OVERHEAD = 0.05

# 12 种配置
EXPERIMENT_CONFIGS = [
    (10.0, 0.6), (10.0, 0.8),
    (11.0, 0.6), (11.0, 0.8), (11.0, 1.0), (11.0, 1.2),
    (12.0, 0.6), (12.0, 0.8), (12.0, 1.0), (12.0, 1.2),
    (13.0, 0.8), (13.0, 1.2),
]

SLO_OPTIONS = [10.0, 11.0, 12.0, 13.0]
RATE_OPTIONS = [0.6, 0.8, 1.0, 1.2]

# PPO 超参数（针对 Qwen2.5-7B 调整，降低学习率提高稳定性）
PPO_CONFIG = {
    'learning_rate': 1e-4,  # 降低学习率: 3e-4 -> 1e-4
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'max_grad_norm': 0.5,
    'batch_size': 64,
    'n_epochs': 10,
    'n_steps': 1024,
}

# 奖励函数（与 Llama 版本一致）
REWARD_CONFIG = {
    'slo_satisfied_reward': 0.9,
    'slo_violated_penalty': -1.8,
    'energy_weight': 0.005,
    'low_energy_bonus': 0.4,
    'energy_threshold': 180,
    'idle_penalty': -0.05,
}

# 训练配置（增加训练步数提高收敛性）
TRAIN_CONFIG = {
    'total_timesteps': 1000000,  # 增加: 500000 -> 1000000
    'eval_freq': 10000,
    'save_freq': 100000,
    'n_eval_episodes': 5,
}

# 模仿学习配置
IMITATION_CONFIG = {
    'pretrain_epochs': 50,
    'pretrain_batch_size': 64,
    'pretrain_lr': 1e-3,
    'episodes_per_config': 10,
    'target_accuracy': 0.95,
}

# 微调配置
FINETUNE_CONFIG = {
    'slo_target': 0.85,
}
