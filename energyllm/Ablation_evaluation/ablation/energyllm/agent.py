
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from config import STATE_DIM, ACTION_DIM, PPO_CONFIG


class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        return self.actor(features), self.critic(features)
    
    def get_action(self, state, deterministic=False):
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()
    
    def evaluate(self, states, actions):
        probs, values = self.forward(states)
        dist = Categorical(probs)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy()


class PPOAgent:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, device='cpu'):
        self.device = torch.device(device)
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=PPO_CONFIG['learning_rate'])
        
        self.gamma = PPO_CONFIG['gamma']
        self.gae_lambda = PPO_CONFIG['gae_lambda']
        self.clip_epsilon = PPO_CONFIG['clip_epsilon']
        self.entropy_coef = PPO_CONFIG['entropy_coef']
        self.value_coef = PPO_CONFIG['value_coef']
        self.max_grad_norm = PPO_CONFIG['max_grad_norm']
        self.batch_size = PPO_CONFIG['batch_size']
        self.n_epochs = PPO_CONFIG['n_epochs']
        
        self.buffer = {'states': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': [], 'dones': []}
    
    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value, _ = self.policy.get_action(state_t, deterministic)
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def compute_gae(self, next_value):
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'] + [next_value])
        dones = np.array(self.buffer['dones'])
        
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1-dones[t]) * gae
            advantages[t] = gae
        
        return advantages, advantages + values[:-1]
    
    def update(self, next_value):
        if not self.buffer['states']:
            return 0.0
        
        advantages, returns = self.compute_gae(next_value)
        
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.LongTensor(self.buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        n_samples = len(self.buffer['states'])
        
        for _ in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                idx = indices[start:start+self.batch_size]
                
                new_log_probs, values, entropy = self.policy.evaluate(states[idx], actions[idx])
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages[idx]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, returns[idx])
                entropy_loss = -entropy.mean()
                
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        self.buffer = {'states': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': [], 'dones': []}
        return total_loss / max(1, self.n_epochs * (n_samples // self.batch_size + 1))
    
    def save(self, path):
        torch.save({'policy': self.policy.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)
        print(f"model save: {path}")
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        # 兼容两种保存格式
        if 'policy_state_dict' in ckpt:
            self.policy.load_state_dict(ckpt['policy_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            self.policy.load_state_dict(ckpt['policy'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f"model start: {path}")
