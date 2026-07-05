import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[df['batch_size'] == 4]
    df['energy_per_req'] = (df['avg_gpu_power_watts'] * df['total_latency_sec']) / df['batch_size']
    return df

def plot_combined_observations(profile_path, output_path):
    df = pd.read_csv(profile_path)
    
    # Create figure with 3 subplots with different widths
    # Give more space to the heatmap (subplot a)
    fig = plt.figure(figsize=(18, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1], wspace=0.3, bottom=0.15)
    
    # ========== Subplot (a): Heatmap - Observation 2 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    df_heat = load_data(profile_path)
    
    def get_label(row):
        length = row['avg_input_len']
        if length < 100:
            return "Short (~50 tok)"
        elif length < 300:
            return "Medium (~250 tok)"
        else:
            return "Long (~500 tok)"
    
    df_heat['label'] = df_heat.apply(get_label, axis=1)
    df_heat = df_heat.sort_values('avg_input_len')
    
    # Select frequencies
    all_freqs = sorted(df_heat['gpu_frequency_mhz'].unique())
    selected_freqs = all_freqs[::2]
    if 1410 in all_freqs and 1410 not in selected_freqs:
        selected_freqs.append(1410)
    
    df_heat = df_heat[df_heat['gpu_frequency_mhz'].isin(selected_freqs)]
    pivot_table = df_heat.pivot(index='label', columns='gpu_frequency_mhz', values='energy_per_req')
    
    ordered_index = ["Short (~50 tok)", "Medium (~250 tok)", "Long (~500 tok)"]
    existing_labels = [l for l in ordered_index if l in pivot_table.index]
    pivot_table = pivot_table.reindex(existing_labels)
    
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={'label': 'Energy (J)'},
                linewidths=1, linecolor='white',
                annot_kws={"size": 10, "weight": "bold"},
                ax=ax1)
    
    # Get colorbar and set label properties
    cbar = ax1.collections[0].colorbar
    cbar.ax.set_ylabel('Energy (J)', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    ax1.set_xlabel('GPU Frequency (MHz)', fontsize=11, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Request Type', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=0, labelsize=9)
    ax1.tick_params(axis='y', rotation=0, labelsize=10)
    
    # Add title below using text
    ax1.text(0.5, -0.22, '(a) Request Heterogeneity', 
             fontsize=13, fontweight='bold', ha='center',
             transform=ax1.transAxes)
    
    # ========== Subplot (b): Power Scaling - Observation 3 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    df_curve = df[df['batch_size']==4].groupby('gpu_frequency_mhz').mean().reset_index()
    freqs = df_curve['gpu_frequency_mhz'].values
    power = df_curve['avg_gpu_power_watts'].values
    
    color_p = '#d62728'
    ax2.plot(freqs, power, color=color_p, marker='o', linewidth=2.5, markersize=6)
    ax2.fill_between(freqs, power, alpha=0.15, color=color_p)
    
    ax2.set_xlabel('GPU Frequency (MHz)', fontsize=11, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Power (W)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.tick_params(labelsize=9)
    
    # Add title below using text
    ax2.text(0.5, -0.22, '(b) Power Scaling Incentive', 
             fontsize=13, fontweight='bold', ha='center',
             transform=ax2.transAxes)
    
    # ========== Subplot (c): Switching Overhead - Observation 3 ==========
    ax3 = fig.add_subplot(gs[0, 2])
    
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
    
    ax3.plot(n_requests, pen_short, marker='o', linewidth=2.5, markersize=6,
             label=f'Short ({lat_min_ms:.1f}ms)', color='#2ca02c')
    ax3.plot(n_requests, pen_med, marker='s', linewidth=2.5, markersize=6,
             label=f'Avg ({lat_mean_ms:.1f}ms)', color='#ff7f0e')
    ax3.plot(n_requests, pen_long, marker='^', linewidth=2.5, markersize=6,
             label=f'Long ({lat_max_ms:.1f}ms)', color='#1f77b4')
    
    ax3.set_xlabel('Steps Before Switching', fontsize=11, fontweight='bold', labelpad=10)
    ax3.set_ylabel('Throughput (% of Ideal)', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.set_xscale('log')
    ax3.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax3.grid(True, linestyle='--', alpha=0.4)
    
    # Set legend with consistent font properties
    legend = ax3.legend(fontsize=9, loc='lower right', framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontweight('normal')
    
    ax3.tick_params(labelsize=9)
    
    # Add title below using text
    ax3.text(0.5, -0.22, '(c) Switching Overhead Penalty', 
             fontsize=13, fontweight='bold', ha='center',
             transform=ax3.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {output_path}")

if __name__ == "__main__":
    plot_combined_observations(
        "test/proposed/results/performance_profile.csv",
        "paper/figures/observation2_3_combined.pdf"
    )
