"""
消融实验: Fixed Freq
只用 RL 调 batch，频率固定在 1200MHz
验证 DVFS 联合优化的价值
"""

import os

# 路径配置
BASE_DIR = "/home/data/Fjw/test/proposed/"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RL_DIR = os.path.join(BASE_DIR, "ablation/fixed_freq")
DATASET_PATH = os.path.join(BASE_DIR, "prompt_datasets_jsonl/sharegpt_en_mixed_all_buckets.jsonl")

# 模型路径
LATENCY_MODEL_PATH = os.path.join(RESULTS_DIR, "latency_model.joblib")
POWER_MODEL_PATH = os.path.join(RESULTS_DIR, "power_model.joblib")
MODEL_META_PATH = os.path.join(RESULTS_DIR, "model_meta.joblib")
PROFILE_DATA_PATH = os.path.join(RESULTS_DIR, "performance_profile.csv")

# DVFS 配置空间
FREQ_OPTIONS = [705, 810, 900, 960, 1005, 1050, 1110, 1155, 1200, 1245, 1290, 1320, 1350, 1380, 1410]
BATCH_OPTIONS = [1, 2, 4, 8, 16, 32, 64]

# ============================================
# 消融: 固定频率
# 动作空间只选 batch，频率固定 1200MHz
# ============================================
FIXED_FREQ = 1200  # 固定频率
FIXED_FREQ_IDX = FREQ_OPTIONS.index(1200)  # 索引 8

# 状态空间维度（9 维，与 PPO-v4 相同）
STATE_DIM = 9

# 动作空间（只选 batch）
NUM_BATCHES = len(BATCH_OPTIONS)  # 7
ACTION_DIM = NUM_BATCHES  # 消融: 只有 7 个动作

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

# PPO 超参数
PPO_CONFIG = {
    'learning_rate': 3e-4,
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

# 奖励函数（与 PPO-v4 相同）
REWARD_CONFIG = {
    'slo_satisfied_reward': 0.9,
    'slo_violated_penalty': -1.8,
    'energy_weight': 0.005,
    'low_energy_bonus': 0.4,
    'energy_threshold': 180,
    'idle_penalty': -0.05,
}

# 训练配置
TRAIN_CONFIG = {
    'total_timesteps': 500000,
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
