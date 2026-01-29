import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

def plot_obs3_penalty(profile_path, output_path):
    df = pd.read_csv(profile_path)
    df_curve = df[df['batch_size']==4].groupby('gpu_frequency_mhz').mean().reset_index()
    freqs = df_curve['gpu_frequency_mhz'].values
    power = df_curve['avg_gpu_power_watts'].values
    
    df['per_token_ms'] = df['total_latency_sec'] * 1000 / (df['avg_input_len'] + df['avg_output_len'])
    
    lat_min_ms = df['per_token_ms'].min()
    lat_mean_ms = df['per_token_ms'].mean()
    lat_max_ms = df['per_token_ms'].max()
    
    overhead_ms = 50
    n_requests = np.array([1, 2, 3, 5, 10, 20, 50, 100])
    
    def calc_penalty(n, lat):
        ideal_time = n * lat
        real_time = n * lat + overhead_ms
        return (ideal_time / real_time) * 100
        
    pen_short = [calc_penalty(n, lat_min_ms) for n in n_requests]
    pen_med = [calc_penalty(n, lat_mean_ms) for n in n_requests]
    pen_long = [calc_penalty(n, lat_max_ms) for n in n_requests]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.25)

    color_p = '#d62728'
    
    ax1.plot(freqs, power, color=color_p, marker='o', linewidth=3, label='Power (W)')
    ax1.fill_between(freqs, power, alpha=0.1, color=color_p)
    ax1.set_xlabel('GPU Frequency (MHz)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Power Consumption (W)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Incentive: Power Scaling (Batch=4)', fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(n_requests, pen_short, marker='o', linewidth=3, label=f'Short Reqs ({lat_min_ms:.1f}ms)', color='#2ca02c')
    ax2.plot(n_requests, pen_med, marker='s', linewidth=3, label=f'Avg Reqs ({lat_mean_ms:.1f}ms)', color='#ff7f0e')
    ax2.plot(n_requests, pen_long, marker='^', linewidth=3, label=f'Long Reqs ({lat_max_ms:.1f}ms)', color='#1f77b4')
    
    ax2.set_xlabel('Consecutive Steps before Switching', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Effective Throughput (% of Ideal)', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Overhead: Cost Amortization', fontsize=16, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.set_xscale('log')
    ax2.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_obs3_penalty("test/proposed/results/performance_profile.csv", "paper/figures/observation3_penalty.pdf")
