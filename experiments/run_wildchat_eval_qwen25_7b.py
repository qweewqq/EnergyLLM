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
BASELINES_DIR = os.path.join(BASE_DIR, "baselines")
PPO_V5_DIR = os.path.join(PROPOSED_DIR, "ablation_qwen25_7b/wo_token_aware")
DYNAMOLLM_DIR = os.path.join(BASE_DIR, "dynamollm")

WILDCHAT_DATASET = "/home/data/EnergyLLM/datasets/WildChat/processed/wildchat_en_mixed_1000.jsonl"

VLLM_MODEL_PATH = "/home/data/models/Qwen2.5-7B-Instruct"
PPO_MODEL_PATH = os.path.join(PPO_V5_DIR, "checkpoints/ppo_final.pt")

RESULTS_DIR = os.path.join(PROPOSED_DIR, "results_qwen25_7b")
LATENCY_MODEL_PATH = os.path.join(RESULTS_DIR, "latency_model.joblib")
POWER_MODEL_PATH = os.path.join(RESULTS_DIR, "power_model.joblib")
MODEL_META_PATH = os.path.join(RESULTS_DIR, "model_meta.joblib")
PROFILE_DATA_PATH = os.path.join(RESULTS_DIR, "performance_profile.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "generalization/wildchat_results_qwen25_7b")

CONDA_PYTHON = "/home/vipuser/anaconda3/envs/vllm_env/bin/python"

TEST_CONFIGS = [
    (10.0, 0.6), (10.0, 0.8),
    (11.0, 0.6), (11.0, 0.8), (11.0, 1.0), (11.0, 1.2),
    (12.0, 0.6), (12.0, 0.8), (12.0, 1.0), (12.0, 1.2),
    (13.0, 0.8), (13.0, 1.2),
]

DURATION = 60
NUM_RUNS = 10

METHODS = [
    'Static-High',
    'DVFS-Only',
    'Batch-Only',
    'Reactive-DVFS',
    'Token-Aware',
    'DynamoLLM-DVFS',
    'Ours',
]

if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)


def check_dataset():
    if not os.path.exists(WILDCHAT_DATASET):
        print(f"Dataset not found: {WILDCHAT_DATASET}")
        return False
    
    with open(WILDCHAT_DATASET, 'r') as f:
        count = sum(1 for _ in f)
    print(f"WildChat dataset: {count} entries")
    return True


def check_models():
    files = [
        (LATENCY_MODEL_PATH, "Latency Model"),
        (POWER_MODEL_PATH, "Power Model"),
        (MODEL_META_PATH, "Model Meta"),
        (PPO_MODEL_PATH, "PPO Model"),
    ]
    
    all_exist = True
    for path, name in files:
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name} not found: {path}")
            all_exist = False
    
    return all_exist


def run_single_test(method, slo, rate, run_id, results_dir):
    config_name = f"slo{int(slo)}_rate{rate}"
    
    print(f"  [{method}] {config_name} Run {run_id+1}/{NUM_RUNS}...", end=" ", flush=True)
    
    import numpy as np
    import pandas as pd
    
    if method == 'Static-High':
        from static_high import StaticHighScheduler
        import joblib
        latency_model = joblib.load(LATENCY_MODEL_PATH)
        power_model = joblib.load(POWER_MODEL_PATH)
        scheduler = StaticHighScheduler(
            slo_latency_sec=slo,
            dataset_path=WILDCHAT_DATASET,
            latency_model=latency_model,
            power_model=power_model,
            fixed_freq=1410,
            fixed_batch=8
        )
        stats = scheduler.run_simulation(DURATION, rate)
        if scheduler.history:
            df = pd.DataFrame(scheduler.history)
            stats['p99_latency'] = np.percentile(df['end_to_end_latency_sec'], 99)
        
    elif method == 'DVFS-Only':
        from dvfs_only import DVFSOnlyScheduler
        import joblib
        latency_model = joblib.load(LATENCY_MODEL_PATH)
        power_model = joblib.load(POWER_MODEL_PATH)
        scheduler = DVFSOnlyScheduler(
            slo_latency_sec=slo,
            dataset_path=WILDCHAT_DATASET,
            latency_model=latency_model,
            power_model=power_model,
            fixed_batch=8,
            freq_options=[810, 960, 1050, 1110, 1155, 1200]
        )
        stats = scheduler.run_simulation(DURATION, rate)
        if scheduler.history:
            df = pd.DataFrame(scheduler.history)
            stats['p99_latency'] = np.percentile(df['end_to_end_latency_sec'], 99)
        
    elif method == 'Batch-Only':
        from batch_only import BatchOnlyScheduler
        import joblib
        latency_model = joblib.load(LATENCY_MODEL_PATH)
        power_model = joblib.load(POWER_MODEL_PATH)
        scheduler = BatchOnlyScheduler(
            slo_latency_sec=slo,
            dataset_path=WILDCHAT_DATASET,
            latency_model=latency_model,
            power_model=power_model,
            fixed_freq=1410,
            batch_options=[1, 2, 4, 8, 16]
        )
        stats = scheduler.run_simulation(DURATION, rate)
        if scheduler.history:
            df = pd.DataFrame(scheduler.history)
            stats['p99_latency'] = np.percentile(df['end_to_end_latency_sec'], 99)

    elif method == 'DynamoLLM-DVFS':
        original_path = sys.path.copy()
        new_path = [p for p in sys.path if p != DYNAMOLLM_DIR]
        new_path.insert(0, DYNAMOLLM_DIR)
        sys.path = new_path
        
        for mod_name in ['scheduler_dvfs_v2', 'optimizer_dvfs_v2']:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        
        try:
            from scheduler_dvfs_v2 import RealTimeSchedulerDVFSv2
            from optimizer_dvfs_v2 import DynamoLLMOptimizerDVFSv2
            import joblib
            
            profile_data = pd.read_csv(PROFILE_DATA_PATH)
            config_options = {
                'gpu_frequency_mhz': profile_data['gpu_frequency_mhz'].unique().tolist(),
                'batch_size': profile_data['batch_size'].unique().tolist()
            }
            
            optimizer = DynamoLLMOptimizerDVFSv2(
                LATENCY_MODEL_PATH, POWER_MODEL_PATH, config_options, MODEL_META_PATH,
                slo_margin_ratio=0.61, min_freq=1050, throughput_weight=0.72
            )
            scheduler = RealTimeSchedulerDVFSv2(
                optimizer=optimizer,
                slo_latency_sec=slo,
                dataset_path=WILDCHAT_DATASET
            )
            scheduler.run_simulation(DURATION, rate)
            
            if scheduler.history:
                df = pd.DataFrame(scheduler.history)
                total = len(df)
                satisfied = df['slo_satisfied'].sum()
                total_energy = (df['simulated_power_watts'] * df['simulated_latency_sec']).sum()
                stats = {
                    'slo_rate': satisfied / total * 100,
                    'energy_per_req': total_energy / total,
                    'avg_e2e_latency': df['end_to_end_latency_sec'].mean(),
                    'p99_latency': np.percentile(df['end_to_end_latency_sec'], 99) if total > 0 else 0,
                    'total_requests': total
                }
            else:
                stats = None
        finally:
            sys.path = original_path
    
    elif method == 'Reactive-DVFS':
        from reactive_dvfs import ReactiveDVFSScheduler
        import joblib
        latency_model = joblib.load(LATENCY_MODEL_PATH)
        power_model = joblib.load(POWER_MODEL_PATH)
        scheduler = ReactiveDVFSScheduler(
            slo_latency_sec=slo,
            dataset_path=WILDCHAT_DATASET,
            latency_model=latency_model,
            power_model=power_model
        )
        stats = scheduler.run_simulation(DURATION, rate)
        if scheduler.history:
            df = pd.DataFrame(scheduler.history)
            stats['p99_latency'] = np.percentile(df['end_to_end_latency_sec'], 99)
    
    elif method == 'Token-Aware':
        from token_aware_heuristic import TokenAwareHeuristicScheduler
        import joblib
        latency_model = joblib.load(LATENCY_MODEL_PATH)
        power_model = joblib.load(POWER_MODEL_PATH)
        scheduler = TokenAwareHeuristicScheduler(
            slo_latency_sec=slo,
            dataset_path=WILDCHAT_DATASET,
            latency_model=latency_model,
            power_model=power_model
        )
        stats = scheduler.run_simulation(DURATION, rate)
        if scheduler.history:
            df = pd.DataFrame(scheduler.history)
            stats['p99_latency'] = np.percentile(df['end_to_end_latency_sec'], 99)

    elif method == 'PPO-v5':
        original_path = sys.path.copy()
        new_path = [p for p in sys.path if p != PROPOSED_DIR and p != PPO_V5_DIR]
        new_path.insert(0, PPO_V5_DIR)
        sys.path = new_path
        
        for mod_name in ['scheduler', 'agent', 'config', 'env']:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        
        try:
            from scheduler import RLScheduler
            scheduler = RLScheduler(
                model_path=PPO_MODEL_PATH,
                slo_latency_sec=slo,
                dataset_path=WILDCHAT_DATASET,
                arrival_rate=rate
            )
            scheduler.run_simulation(duration_sec=DURATION, arrival_rate=rate)
            
            if scheduler.history:
                df = pd.DataFrame(scheduler.history)
                total = len(df)
                satisfied = df['slo_satisfied'].sum()
                total_energy = (df['power'] * df['latency']).sum()
                stats = {
                    'slo_rate': satisfied / total * 100,
                    'energy_per_req': total_energy / total,
                    'avg_e2e_latency': df['e2e_latency'].mean(),
                    'p99_latency': np.percentile(df['e2e_latency'], 99) if total > 0 else 0,
                    'total_requests': total
                }
            else:
                stats = None
        finally:
            sys.path = original_path
    
    else:
        print(f"Unknown method: {method}")
        return None
    
    print(f"SLO={stats.get('slo_rate', 0):.1f}%, Energy={stats.get('energy_per_req', 0):.1f}J")
    return stats


def run_all_tests():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}
    
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Testing method: {method}")
        print(f"{'='*60}")
        
        method_results = {}
        
        for slo, rate in TEST_CONFIGS:
            config_name = f"slo{int(slo)}_rate{rate}"
            runs = []
            
            for run_id in range(NUM_RUNS):
                try:
                    stats = run_single_test(method, slo, rate, run_id, OUTPUT_DIR)
                    if stats:
                        runs.append(stats)
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if runs:
                import numpy as np
                method_results[config_name] = {
                    'slo_rate_mean': np.mean([r.get('slo_rate', 0) for r in runs]),
                    'slo_rate_std': np.std([r.get('slo_rate', 0) for r in runs]),
                    'energy_mean': np.mean([r.get('energy_per_req', 0) for r in runs]),
                    'energy_std': np.std([r.get('energy_per_req', 0) for r in runs]),
                    'latency_mean': np.mean([r.get('avg_e2e_latency', 0) for r in runs]),
                    'p99_latency_mean': np.mean([r.get('p99_latency', 0) for r in runs]),
                    'requests_mean': np.mean([r.get('total_requests', 0) for r in runs]),
                    'num_runs': len(runs)
                }
        
        all_results[method] = method_results
    
    results_file = os.path.join(OUTPUT_DIR, f"wildchat_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    generate_summary(all_results, OUTPUT_DIR, timestamp)
    
    return all_results


def generate_summary(results, output_dir, timestamp):
    import pandas as pd
    
    rows = []
    for method, configs in results.items():
        for config, stats in configs.items():
            rows.append({
                'Method': method,
                'Config': config,
                'SLO_Rate_Mean': stats['slo_rate_mean'],
                'SLO_Rate_Std': stats['slo_rate_std'],
                'Energy_Mean': stats['energy_mean'],
                'Energy_Std': stats['energy_std'],
                'Latency_Mean': stats['latency_mean'],
                'P99_Latency_Mean': stats.get('p99_latency_mean', 0),
                'Requests_Mean': stats['requests_mean'],
            })
    
    df = pd.DataFrame(rows)
    csv_file = os.path.join(output_dir, f"wildchat_summary_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Summary table saved to: {csv_file}")
    
    print("\n" + "="*80)
    print("WildChat Dataset Baseline Comparison Results (Qwen2.5-7B)")
    print("="*80)
    print(f"{'Method':<18} {'SLO Rate':<12} {'Energy (J/req)':<15} {'P99 (s)':<10}")
    print("-"*55)
    
    for method in METHODS:
        if method in results:
            configs = results[method]
            if configs:
                avg_slo = sum(c['slo_rate_mean'] for c in configs.values()) / len(configs)
                avg_energy = sum(c['energy_mean'] for c in configs.values()) / len(configs)
                avg_p99 = sum(c.get('p99_latency_mean', 0) for c in configs.values()) / len(configs)
                print(f"{method:<18} {avg_slo:>10.2f}% {avg_energy:>13.2f} {avg_p99:>8.2f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='WildChat Baseline Comparison - Qwen2.5-7B')
    parser.add_argument('--method', type=str, default=None, 
                        help='Specify test method')
    parser.add_argument('--slo', type=float, default=None, help='Specify SLO')
    parser.add_argument('--rate', type=float, default=None, help='Specify arrival rate')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per config')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds per run')
    args = parser.parse_args()
    
    NUM_RUNS = args.runs
    DURATION = args.duration
    
    if args.method:
        METHODS = [args.method]
    
    if args.slo is not None and args.rate is not None:
        TEST_CONFIGS = [(args.slo, args.rate)]
    
    print("="*60)
    print("WildChat Baseline Comparison - Qwen2.5-7B")
    print("="*60)
    print(f"Methods: {METHODS}")
    print(f"Configs: {TEST_CONFIGS}")
    print(f"Runs: {NUM_RUNS}, Duration: {DURATION}s")
    print("="*60)
    
    if not check_dataset():
        sys.exit(1)
    
    if not check_models():
        print("\nWarning: Some model files not found, please check paths")
    
    run_all_tests()
