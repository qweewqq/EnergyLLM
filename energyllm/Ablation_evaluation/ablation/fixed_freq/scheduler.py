
import time
import random
import threading
import subprocess
from collections import deque
import pandas as pd
import numpy as np
import os
import json
import torch
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import PPOAgent
from config import (FREQ_OPTIONS, BATCH_OPTIONS, DATASET_PATH, RL_DIR, 
                    LATENCY_MODEL_PATH, POWER_MODEL_PATH, MODEL_META_PATH, MAX_QUEUE_SIZE,
                    FIXED_FREQ, FIXED_FREQ_IDX)


class RLScheduler:

    
    def __init__(self, model_path, slo_latency_sec, dataset_path=None, arrival_rate=0.8):
        self.slo_latency_sec = slo_latency_sec
        self.arrival_rate = arrival_rate
        
        self.agent = PPOAgent()
        self.agent.load(model_path)
        self.agent.policy.eval()
        
        self.request_queue = deque()
        self.stop_event = threading.Event()
        self.history = []
        

        self.current_freq_idx = FIXED_FREQ_IDX
        self.current_freq = FIXED_FREQ
        
        self.recent_slo = deque(maxlen=20)
        self.recent_latency = deque(maxlen=20)
        
        self.dataset_items = []
        if dataset_path:
            self._load_dataset(dataset_path)
        
        self._load_ml_models()
        
        print(f" (Fixed Freq {FIXED_FREQ}MHz): SLO={slo_latency_sec}s")
    
    def _load_dataset(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.dataset_items.append(json.loads(line))
                        if len(self.dataset_items) >= 1000:
                            break
        except:
            pass
    
    def _load_ml_models(self):
        try:
            self.latency_model = joblib.load(LATENCY_MODEL_PATH)
            self.power_model = joblib.load(POWER_MODEL_PATH)
            self.model_meta = joblib.load(MODEL_META_PATH)
            self.feature_names = self.model_meta.get('feature_names', 
                ['gpu_frequency_mhz', 'batch_size', 'avg_input_len', 'avg_output_len'])
        except Exception as e:
            print(f"unable load model: {e}")
            self.latency_model = None
            self.power_model = None

    def _predict(self, freq, batch_size, avg_input_len=150, avg_output_len=100):
        if self.latency_model is None:
            return 2.0 + batch_size * 0.1, 180 + freq * 0.05
        
        features = {
            'gpu_frequency_mhz': freq,
            'batch_size': batch_size,
            'avg_input_len': avg_input_len,
            'avg_output_len': avg_output_len
        }
        X = pd.DataFrame([features])[self.feature_names]
        
        latency = self.latency_model.predict(X)[0]
        power = self.power_model.predict(X)[0]
        
        return max(0.1, latency), max(50, power)
    
    def _get_state(self, current_time):
        queue_size = len(self.request_queue)
        
        if queue_size > 0:
            queue_list = list(self.request_queue)
            avg_wait = np.mean([current_time - r['arrival_time'] for r in queue_list])
        else:
            avg_wait = 0
        
        recent_slo_rate = np.mean(self.recent_slo) if self.recent_slo else 1.0
        recent_lat = np.mean(self.recent_latency) if self.recent_latency else 0
        
        state = np.array([
            min(queue_size / MAX_QUEUE_SIZE, 1.0),
            FIXED_FREQ_IDX / (len(FREQ_OPTIONS) - 1),  # 固定频率索引
            recent_slo_rate,
            min(recent_lat / 30, 1.0),
            min(self.arrival_rate / 2.0, 1.0),
            min(avg_wait / 30, 1.0),
            (self.slo_latency_sec - 8) / 8
        ], dtype=np.float32)
        
        return state
    
    def _action_to_config(self, action):
        batch_size = BATCH_OPTIONS[action]
        return FIXED_FREQ, batch_size, FIXED_FREQ_IDX
    
    def _set_gpu_frequency(self, freq_mhz):
        if self.current_freq == freq_mhz:
            return 0.0
        try:
            subprocess.run(["sudo", "nvidia-smi", "-lgc", str(freq_mhz)], capture_output=True)
            self.current_freq = freq_mhz
            return 0.05
        except:
            return 0.0
    
    def _reset_gpu_frequency(self):
        try:
            subprocess.run(["sudo", "nvidia-smi", "-rgc"], capture_output=True)
        except:
            pass
    
    def _request_generator(self, arrival_rate):
        request_id = 0
        idx = 0
        while not self.stop_event.is_set():
            time.sleep(random.expovariate(arrival_rate))
            
            if self.dataset_items:
                item = self.dataset_items[idx % len(self.dataset_items)]
                idx += 1
            else:
                item = {'input_len': 150, 'output_len': 100}
            
            self.request_queue.append({
                'id': request_id,
                'arrival_time': time.time(),
                'input_len': item.get('input_len', 150),
                'output_len': item.get('output_len', 100)
            })
            request_id += 1

    def run_simulation(self, duration_sec, arrival_rate):
        self.arrival_rate = arrival_rate
        print(f"\n {duration_sec}s, {arrival_rate} req/s, SLO={self.slo_latency_sec}s")
        
        self._set_gpu_frequency(FIXED_FREQ)
        
        gen_thread = threading.Thread(target=self._request_generator, args=(arrival_rate,))
        gen_thread.start()
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration_sec:
                if len(self.request_queue) == 0:
                    time.sleep(0.01)
                    continue
                
                current_time = time.time()
                state = self._get_state(current_time)
                
                with torch.no_grad():
                    action, _, _ = self.agent.select_action(state, deterministic=True)
                
                freq, batch_size, freq_idx = self._action_to_config(action)
                batch_to_process = min(len(self.request_queue), batch_size)
                
                tasks = [self.request_queue.popleft() for _ in range(batch_to_process)]
                
                avg_input = sum(t['input_len'] for t in tasks) / len(tasks)
                avg_output = sum(t['output_len'] for t in tasks) / len(tasks)
                
                latency, power = self._predict(freq, batch_to_process, avg_input, avg_output)
                
                time.sleep(latency)
                end_time = time.time()
                
                for task in tasks:
                    e2e = end_time - task['arrival_time']
                    satisfied = e2e <= self.slo_latency_sec
                    
                    self.recent_slo.append(1 if satisfied else 0)
                    self.recent_latency.append(e2e)
                    
                    self.history.append({
                        'timestamp': end_time,
                        'freq': freq,
                        'batch_size': batch_to_process,
                        'latency': latency,
                        'power': power,
                        'e2e_latency': e2e,
                        'slo_satisfied': satisfied
                    })
        finally:
            self._reset_gpu_frequency()
        
        self.stop_event.set()
        gen_thread.join()
        
        self._print_stats()
    
    def _print_stats(self):
        if not self.history:
            return
        
        df = pd.DataFrame(self.history)
        total = len(df)
        satisfied = df['slo_satisfied'].sum()
        energy = (df['power'] * df['latency']).sum()
        
        print("\n" + "=" * 50)
        print(f"SLO attainment: {satisfied/total*100:.2f}%")
        print(f"avg latency: {df['e2e_latency'].mean():.2f}s")
        print(f"avg energy: {energy/total:.2f} J/req")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--slo', type=float, default=12.0)
    parser.add_argument('--rate', type=float, default=0.8)
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()
    
    scheduler = RLScheduler(args.model, args.slo, DATASET_PATH)
    scheduler.run_simulation(args.duration, args.rate)
