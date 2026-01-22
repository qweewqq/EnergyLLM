"""
消融实验: w/o Token-aware
去掉 token 长度特征，状态空间从 9 维减少到 7 维
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
import json
import random
from collections import deque

from config import (
    FREQ_OPTIONS, BATCH_OPTIONS, STATE_DIM, ACTION_DIM,
    DEFAULT_SLO, MAX_QUEUE_SIZE, MAX_EPISODE_STEPS, FREQ_SWITCH_OVERHEAD,
    PROFILE_DATA_PATH, DATASET_PATH, REWARD_CONFIG,
    EXPERIMENT_CONFIGS
)


class LLMSchedulingEnv(gym.Env):
    """消融: w/o Token-aware（7 维状态）"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, slo_target=DEFAULT_SLO, arrival_rate=0.8, randomize_config=False):
        super().__init__()
        
        self.slo_target = slo_target
        self.arrival_rate = arrival_rate
        self.randomize_config = randomize_config
        
        self.action_space = spaces.Discrete(ACTION_DIM)
        
        # ============================================
        # 消融: 7 维状态空间（去掉 avg_input, avg_output）
        # ============================================
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self._load_lut()
        self._load_dataset()
        self.reset()
    
    def _load_lut(self):
        self.lut = {}
        try:
            df = pd.read_csv(PROFILE_DATA_PATH)
            for _, row in df.iterrows():
                key = (int(row['gpu_frequency_mhz']), int(row['batch_size']))
                self.lut[key] = {
                    'latency': row['total_latency_sec'],
                    'power': row['avg_gpu_power_watts']
                }
            print(f"LUT: {len(self.lut)} 条")
        except Exception as e:
            print(f"LUT 加载失败: {e}")
    
    def _load_dataset(self):
        self.dataset_items = []
        try:
            with open(DATASET_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.dataset_items.append(json.loads(line))
                        if len(self.dataset_items) >= 1000:
                            break
            print(f"数据集: {len(self.dataset_items)} 条")
        except:
            self.dataset_items = [{'input_len': 150, 'output_len': 100}] * 100
    
    def _predict(self, freq, batch_size):
        key = (freq, batch_size)
        if key in self.lut:
            return self.lut[key]['latency'], self.lut[key]['power']
        return 2.0 + batch_size * 0.1, 180 + freq * 0.05
    
    def _generate_arrivals(self, time_delta):
        n = np.random.poisson(self.arrival_rate * time_delta)
        for _ in range(n):
            if len(self.queue) >= MAX_QUEUE_SIZE:
                break
            item = random.choice(self.dataset_items)
            self.queue.append({
                'arrival_time': self.current_time,
                'input_len': item.get('input_len', 150),
                'output_len': item.get('output_len', 100)
            })
    
    def _action_to_config(self, action):
        freq_idx = action // len(BATCH_OPTIONS)
        batch_idx = action % len(BATCH_OPTIONS)
        return FREQ_OPTIONS[freq_idx], BATCH_OPTIONS[batch_idx], freq_idx
    
    def _get_state(self):
        """
        消融: 7 维状态（去掉 avg_input, avg_output）
        原 PPO-v4 状态:
        [0] queue_size, [1] avg_input, [2] avg_output, [3] freq_idx,
        [4] recent_slo, [5] recent_latency, [6] arrival_rate, [7] avg_wait, [8] slo_target
        
        消融后状态:
        [0] queue_size, [1] freq_idx, [2] recent_slo, [3] recent_latency,
        [4] arrival_rate, [5] avg_wait, [6] slo_target
        """
        queue_size = len(self.queue)
        
        if queue_size > 0:
            avg_wait = np.mean([self.current_time - r['arrival_time'] for r in self.queue])
        else:
            avg_wait = 0
        
        recent_slo_rate = np.mean(self.recent_slo) if self.recent_slo else 1.0
        recent_latency = np.mean(self.recent_latency) if self.recent_latency else 0
        
        state = np.array([
            min(queue_size / MAX_QUEUE_SIZE, 1.0),
            # 消融: 去掉 avg_input, avg_output
            self.current_freq_idx / (len(FREQ_OPTIONS) - 1),
            recent_slo_rate,
            min(recent_latency / 30, 1.0),
            min(self.arrival_rate / 2.0, 1.0),
            min(avg_wait / 30, 1.0),
            (self.slo_target - 8) / 8
        ], dtype=np.float32)
        
        return state
    
    def reset(self):
        if self.randomize_config:
            slo, rate = random.choice(EXPERIMENT_CONFIGS)
            self.slo_target = slo
            self.arrival_rate = rate
        
        self.queue = deque()
        self.current_time = 0.0
        self.current_freq_idx = len(FREQ_OPTIONS) // 2
        self.current_freq = FREQ_OPTIONS[self.current_freq_idx]
        
        self.step_count = 0
        self.total_requests = 0
        self.slo_satisfied = 0
        self.total_energy = 0.0
        
        self.recent_slo = deque(maxlen=20)
        self.recent_latency = deque(maxlen=20)
        
        self._generate_arrivals(3.0 + self.arrival_rate * 2)
        
        return self._get_state()
    
    def step(self, action):
        self.step_count += 1
        
        freq, batch_size, freq_idx = self._action_to_config(action)
        
        if freq_idx != self.current_freq_idx:
            self.current_time += FREQ_SWITCH_OVERHEAD
            self.current_freq_idx = freq_idx
            self.current_freq = freq
        
        if len(self.queue) == 0:
            self._generate_arrivals(1.0)
            self.current_time += 1.0
            reward = REWARD_CONFIG['idle_penalty']
            done = self.step_count >= MAX_EPISODE_STEPS
            return self._get_state(), reward, done, {'idle': True}
        
        actual_batch = min(batch_size, len(self.queue))
        tasks = [self.queue.popleft() for _ in range(actual_batch)]
        
        latency, power = self._predict(freq, actual_batch)
        self.current_time += latency
        
        self._generate_arrivals(latency)
        
        # 奖励计算（与 PPO-v4 相同）
        reward = 0.0
        for task in tasks:
            e2e = self.current_time - task['arrival_time']
            satisfied = e2e <= self.slo_target
            
            self.total_requests += 1
            if satisfied:
                self.slo_satisfied += 1
                reward += REWARD_CONFIG['slo_satisfied_reward']
            else:
                reward += REWARD_CONFIG['slo_violated_penalty']
            
            self.recent_slo.append(1 if satisfied else 0)
            self.recent_latency.append(e2e)
        
        # 能耗
        energy = power * latency
        self.total_energy += energy
        energy_per_req = energy / actual_batch
        reward -= REWARD_CONFIG['energy_weight'] * energy_per_req
        
        if energy_per_req < REWARD_CONFIG.get('energy_threshold', 180):
            reward += REWARD_CONFIG.get('low_energy_bonus', 0.4)
        
        done = self.step_count >= MAX_EPISODE_STEPS
        
        info = {
            'freq': freq,
            'batch_size': actual_batch,
            'slo_rate': self.slo_satisfied / max(1, self.total_requests),
            'energy_per_req': self.total_energy / max(1, self.total_requests),
            'slo_target': self.slo_target,
            'arrival_rate': self.arrival_rate
        }
        
        return self._get_state(), reward, done, info
    
    def render(self, mode='human'):
        slo = self.slo_satisfied / max(1, self.total_requests) * 100
        energy = self.total_energy / max(1, self.total_requests)
        print(f"Step {self.step_count}: SLO={self.slo_target}s, Rate={self.arrival_rate}, "
              f"Queue={len(self.queue)}, SLO%={slo:.1f}%, Energy={energy:.1f}")


if __name__ == '__main__':
    env = LLMSchedulingEnv(randomize_config=True)
    
    for ep in range(3):
        state = env.reset()
        print(f"\nEpisode {ep+1}: SLO={env.slo_target}, Rate={env.arrival_rate}")
        print(f"State dim: {len(state)}")
        
        for _ in range(50):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                break
        
        env.render()
