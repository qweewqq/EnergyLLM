"""
消融实验训练脚本: Fixed Freq (1200MHz) - Qwen2.5-7B版本
固定频率，只调整批次大小
"""

import os
import sys
import numpy as np
import torch
import json
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import LLMSchedulingEnv
from agent import PPOAgent
from config import (
    FREQ_OPTIONS, BATCH_OPTIONS, RL_DIR,
    TRAIN_CONFIG, PPO_CONFIG, EXPERIMENT_CONFIGS, FIXED_FREQ
)


def evaluate(agent, n_episodes=3):
    """12 种配置评估"""
    results = {}
    
    for slo, rate in EXPERIMENT_CONFIGS:
        env = LLMSchedulingEnv(slo_target=slo, arrival_rate=rate)
        
        total_slo = 0
        total_energy = 0
        
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                action, _, _ = agent.select_action(state, deterministic=True)
                state, _, done, _ = env.step(action)
            
            total_slo += env.slo_satisfied / max(1, env.total_requests) * 100
            total_energy += env.total_energy / max(1, env.total_requests)
        
        key = f"slo{int(slo)}_rate{rate}"
        results[key] = {
            'slo_rate': total_slo / n_episodes,
            'energy': total_energy / n_episodes
        }
    
    return results


def train(agent, total_timesteps, eval_freq=10000, save_freq=100000):
    """PPO 强化学习训练"""
    print("\n" + "=" * 50)
    print(f"消融实验: Fixed Freq ({FIXED_FREQ}MHz) (Qwen2.5-7B)")
    print("固定频率，只调整批次大小")
    print("=" * 50)
    
    save_dir = os.path.join(RL_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    log = {'timesteps': [], 'rewards': [], 'slo_rates': [], 'energies': [], 'losses': []}
    
    env = LLMSchedulingEnv(randomize_config=True)
    
    state = env.reset()
    episode_reward = 0
    episode_rewards = deque(maxlen=100)
    
    n_steps = PPO_CONFIG['n_steps']
    timestep = 0
    episode = 0
    last_eval = 0
    
    while timestep < total_timesteps:
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            timestep += 1
            
            if done:
                episode_rewards.append(episode_reward)
                episode += 1
                
                if episode % 50 == 0:
                    slo = env.slo_satisfied / max(1, env.total_requests) * 100
                    energy = env.total_energy / max(1, env.total_requests)
                    print(f"Ep {episode}, Step {timestep}: SLO_target={env.slo_target}, "
                          f"Rate={env.arrival_rate}, SLO%={slo:.1f}%, Energy={energy:.1f}")
                
                episode_reward = 0
                state = env.reset()
            
            if timestep >= total_timesteps:
                break
        
        with torch.no_grad():
            _, _, next_value = agent.select_action(state)
        loss = agent.update(next_value)
        
        if timestep - last_eval >= eval_freq:
            results = evaluate(agent, n_episodes=3)
            avg_slo = np.mean([r['slo_rate'] for r in results.values()])
            avg_energy = np.mean([r['energy'] for r in results.values()])
            
            log['timesteps'].append(timestep)
            log['rewards'].append(np.mean(episode_rewards) if episode_rewards else 0)
            log['slo_rates'].append(avg_slo)
            log['energies'].append(avg_energy)
            log['losses'].append(loss)
            
            print(f"\n[Eval] Step {timestep}: Avg SLO={avg_slo:.1f}%, Avg Energy={avg_energy:.1f}\n")
            
            last_eval = timestep
        
        if timestep % save_freq == 0:
            agent.save(os.path.join(save_dir, f"ppo_step_{timestep}.pt"))
    
    agent.save(os.path.join(save_dir, "ppo_final.pt"))
    
    log_path = os.path.join(RL_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"日志: {log_path}")
    
    return log


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=500000)
    args = parser.parse_args()
    
    print("=" * 50)
    print(f"消融实验: Fixed Freq ({FIXED_FREQ}MHz) (Qwen2.5-7B)")
    print(f"总步数: {args.timesteps}")
    print("=" * 50)
    
    agent = PPOAgent()
    
    log = train(agent, args.timesteps,
                eval_freq=TRAIN_CONFIG['eval_freq'],
                save_freq=TRAIN_CONFIG['save_freq'])
    
    print("\n" + "=" * 50)
    print("最终评估")
    print("=" * 50)
    results = evaluate(agent, n_episodes=10)
    for k, v in results.items():
        print(f"  {k}: SLO={v['slo_rate']:.1f}%, Energy={v['energy']:.1f}")


if __name__ == '__main__':
    main()
