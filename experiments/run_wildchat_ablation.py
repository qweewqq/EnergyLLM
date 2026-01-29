#!/usr/bin/env python3

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = "/home/data/EnergyLLM/test"
PROPOSED_DIR = os.path.join(BASE_DIR, "proposed")
ABLATION_V2_DIR = os.path.join(PROPOSED_DIR, "ablation_v2")

WILDCHAT_DATASET = "/home/data/Fjw/datasets/WildChat/processed/wildchat_en_mixed_1000.jsonl"

OUTPUT_DIR = os.path.join(BASE_DIR, "generalization/wildchat_ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABLATION_VARIANTS = {
    'proposed_v3': {
        'name': 'Proposed-v3',
        'type': 'heuristic',
        'scheduler': os.path.join(PROPOSED_DIR, 'scheduler_v3.py'),
    },
    'wo_energy_reward': {
        'name': 'w/o Energy Reward',
        'type': 'rl',
        'dir': os.path.join(ABLATION_V2_DIR, 'wo_energy_reward'),
        'model': 'checkpoints/ppo_final.pt'
    },
    'fixed_freq': {
        'name': 'Fixed Freq (1200MHz)',
        'type': 'rl',
        'dir': os.path.join(ABLATION_V2_DIR, 'fixed_freq'),
        'model': 'checkpoints/ppo_final.pt'
    },
    'fixed_batch': {
        'name': 'Fixed Batch (16)',
        'type': 'rl',
        'dir': os.path.join(ABLATION_V2_DIR, 'fixed_batch_16'),
        'model': 'checkpoints/ppo_final.pt'
    },
    'full': {
        'name': '(Ours)',
        'type': 'rl',
        'dir': os.path.join(PROPOSED_DIR, 'ablation/wo_token_aware'),
        'model': 'checkpoints/ppo_final.pt'
    }
}

TEST_CONFIGS = [
    {'slo': 10.0, 'rate': 0.6, 'name': 'slo10_rate0.6'},
    {'slo': 10.0, 'rate': 0.8, 'name': 'slo10_rate0.8'},
    {'slo': 11.0, 'rate': 0.6, 'name': 'slo11_rate0.6'},
    {'slo': 11.0, 'rate': 0.8, 'name': 'slo11_rate0.8'},
    {'slo': 11.0, 'rate': 1.0, 'name': 'slo11_rate1.0'},
    {'slo': 11.0, 'rate': 1.2, 'name': 'slo11_rate1.2'},
    {'slo': 12.0, 'rate': 0.6, 'name': 'slo12_rate0.6'},
    {'slo': 12.0, 'rate': 0.8, 'name': 'slo12_rate0.8'},
    {'slo': 12.0, 'rate': 1.0, 'name': 'slo12_rate1.0'},
    {'slo': 12.0, 'rate': 1.2, 'name': 'slo12_rate1.2'},
    {'slo': 13.0, 'rate': 0.8, 'name': 'slo13_rate0.8'},
    {'slo': 13.0, 'rate': 1.2, 'name': 'slo13_rate1.2'},
]

DURATION = 60
NUM_RUNS = 10


def check_models():
    print("Checking model files...")
    all_exist = True
    for key, variant in ABLATION_VARIANTS.items():
        if variant['type'] == 'heuristic':
            exists = os.path.exists(variant['scheduler'])
            status = "✓" if exists else "✗"
            print(f"  {status} {variant['name']}: {variant['scheduler']}")
        else:
            model_path = os.path.join(variant['dir'], variant['model'])
            exists = os.path.exists(model_path)
            status = "✓" if exists else "✗"
            print(f"  {status} {variant['name']}: {model_path}")
        if not exists:
            all_exist = False
    return all_exist


def extract_metrics(history):
    if not history or len(history) == 0:
        return None
    
    df = pd.DataFrame(history)
    total = len(df)
    satisfied = df['slo_satisfied'].sum()
    total_energy = (df['power'] * df['latency']).sum()
    
    return {
        'total_requests': total,
        'slo_satisfied': int(satisfied),
        'slo_rate': satisfied / total * 100 if total > 0 else 0,
        'avg_e2e_latency': df['e2e_latency'].mean(),
        'p99_latency': np.percentile(df['e2e_latency'], 99) if total > 0 else 0,
        'avg_power': df['power'].mean(),
        'total_energy': total_energy,
        'energy_per_req': total_energy / total if total > 0 else 0
    }


def run_ablation_variant(variant_key, slo, rate, duration):
    variant = ABLATION_VARIANTS[variant_key]
    
    if variant['type'] == 'heuristic':
        return run_proposed_v3(slo, rate, duration)
    else:
        return run_rl_variant(variant, slo, rate, duration)


def run_proposed_v3(slo, rate, duration):
    original_path = sys.path.copy()
    
    new_path = [p for p in sys.path if 'ablation' not in p]
    new_path.insert(0, PROPOSED_DIR)
    sys.path = new_path
    
    for mod_name in ['scheduler_v3', 'optimizer_v3']:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    
    try:
        from scheduler_v3 import HierarchicalScheduler
        from optimizer_v3 import HierarchicalOptimizer
        
        profile_data = pd.read_csv(os.path.join(PROPOSED_DIR, 'results/performance_profile.csv'))
        config_options = {
            'gpu_frequency_mhz': profile_data['gpu_frequency_mhz'].unique().tolist(),
            'batch_size': profile_data['batch_size'].unique().tolist()
        }
        
        optimizer = HierarchicalOptimizer(
            os.path.join(PROPOSED_DIR, 'results/latency_model.joblib'),
            os.path.join(PROPOSED_DIR, 'results/power_model.joblib'),
            config_options,
            os.path.join(PROPOSED_DIR, 'results/model_meta.joblib'),
            os.path.join(PROPOSED_DIR, 'results/performance_profile.csv')
        )
        
        scheduler = HierarchicalScheduler(
            optimizer=optimizer,
            slo_latency_sec=slo,
            dataset_path=WILDCHAT_DATASET
        )
        scheduler.run_simulation(duration_sec=duration, arrival_rate_per_sec=rate)
        
        if scheduler.history:
            df = pd.DataFrame(scheduler.history)
            total = len(df)
            satisfied = df['slo_satisfied'].sum()
            total_energy = (df['simulated_power_watts'] * df['simulated_latency_sec']).sum()
            return {
                'total_requests': total,
                'slo_satisfied': int(satisfied),
                'slo_rate': satisfied / total * 100 if total > 0 else 0,
                'avg_e2e_latency': df['end_to_end_latency_sec'].mean(),
                'p99_latency': np.percentile(df['end_to_end_latency_sec'], 99) if total > 0 else 0,
                'avg_power': df['simulated_power_watts'].mean(),
                'total_energy': total_energy,
                'energy_per_req': total_energy / total if total > 0 else 0
            }
        return None
    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()
        return None
    finally:
        sys.path = original_path


def run_rl_variant(variant, slo, rate, duration):
    variant_dir = variant['dir']
    model_path = os.path.join(variant_dir, variant['model'])
    
    original_path = sys.path.copy()
    original_modules = set(sys.modules.keys())
    
    new_path = [p for p in sys.path if 'ablation' not in p and 'rl_ppo_version' not in p]
    new_path.insert(0, variant_dir)
    sys.path = new_path
    
    for mod_name in list(sys.modules.keys()):
        if mod_name in ['scheduler', 'agent', 'config', 'env']:
            del sys.modules[mod_name]
    
    try:
        from scheduler import RLScheduler
        
        scheduler = RLScheduler(
            model_path=model_path,
            slo_latency_sec=slo,
            dataset_path=WILDCHAT_DATASET,
            arrival_rate=rate
        )
        scheduler.run_simulation(duration_sec=duration, arrival_rate=rate)
        
        return extract_metrics(scheduler.history)
    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()
        return None
    finally:
        sys.path = original_path
        for mod_name in list(sys.modules.keys()):
            if mod_name not in original_modules:
                if 'ablation' in str(sys.modules.get(mod_name, '')):
                    del sys.modules[mod_name]


def aggregate_results(results_list):
    valid_results = [r for r in results_list if r is not None]
    if not valid_results:
        return None
    
    aggregated = {}
    for key in valid_results[0].keys():
        values = [r[key] for r in valid_results]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1) if len(values) > 1 else 0,
        }
    
    return aggregated


def run_all_experiments():
    print("=" * 70)
    print("WildChat Ablation Experiments")
    print("=" * 70)
    print(f"Variants: {list(ABLATION_VARIANTS.keys())}")
    print(f"Configs: {len(TEST_CONFIGS)}")
    print(f"Runs per config: {NUM_RUNS} × {DURATION}s")
    print("=" * 70)
    
    if not check_models():
        print("\n[ERROR] Model files incomplete, please train first")
        return
    
    all_results = {}
    
    for config in TEST_CONFIGS:
        slo = config['slo']
        rate = config['rate']
        config_name = config['name']
        
        print(f"\n{'='*70}")
        print(f"Config: SLO={slo}s, Rate={rate} req/s")
        print(f"{'='*70}")
        
        all_results[config_name] = {}
        
        for variant_key, variant_info in ABLATION_VARIANTS.items():
            print(f"\n--- {variant_info['name']} ---")
            
            variant_results = []
            for run_idx in range(NUM_RUNS):
                print(f"  Run {run_idx + 1}/{NUM_RUNS}...", end=" ")
                
                result = run_ablation_variant(variant_key, slo, rate, DURATION)
                
                if result:
                    print(f"SLO={result['slo_rate']:.1f}%, Energy={result['energy_per_req']:.1f}J/req")
                    variant_results.append(result)
                else:
                    print("Failed")
                
                time.sleep(1)
            
            aggregated = aggregate_results(variant_results)
            all_results[config_name][variant_key] = {
                'name': variant_info['name'],
                'aggregated': aggregated
            }
            
            if aggregated:
                print(f"  Average: SLO={aggregated['slo_rate']['mean']:.2f}%, Energy={aggregated['energy_per_req']['mean']:.2f}J/req")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(all_results, timestamp)
    
    return all_results


def save_results(all_results, timestamp):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    json_path = os.path.join(OUTPUT_DIR, f"wildchat_ablation_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved: {json_path}")
    
    rows = []
    for config_name, variants in all_results.items():
        for variant_key, data in variants.items():
            if data['aggregated']:
                agg = data['aggregated']
                rows.append({
                    'Config': config_name,
                    'Variant': data['name'],
                    'SLO_Rate_Mean': agg['slo_rate']['mean'],
                    'SLO_Rate_Std': agg['slo_rate']['std'],
                    'Energy_Mean': agg['energy_per_req']['mean'],
                    'Energy_Std': agg['energy_per_req']['std'],
                    'Latency_Mean': agg['avg_e2e_latency']['mean'],
                    'P99_Latency': agg['p99_latency']['mean'],
                    'Requests': agg['total_requests']['mean']
                })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, f"wildchat_ablation_summary_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary table: {csv_path}")
    
    print("\n" + "=" * 80)
    print("Average Metrics by Variant")
    print("=" * 80)
    summary = df.groupby('Variant').agg({
        'SLO_Rate_Mean': 'mean',
        'Energy_Mean': 'mean'
    }).round(2)
    print(summary.to_string())


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()
    
    NUM_RUNS = args.runs
    DURATION = args.duration
    
    run_all_experiments()
