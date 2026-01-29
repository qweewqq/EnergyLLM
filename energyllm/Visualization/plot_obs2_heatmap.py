import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="white", palette="muted")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[df['batch_size'] == 4]
    df['energy_per_req'] = (df['avg_gpu_power_watts'] * df['total_latency_sec']) / df['batch_size']
    return df

def plot_heatmap(file_path, output_path):
    df = load_data(file_path)
    
    def get_label(row):
        length = row['avg_input_len']
        if length < 100:
            return "Short (~50 tok)"
        elif length < 300:
            return "Medium (~250 tok)"
        else:
            return "Long (~500 tok)"

    df['label'] = df.apply(get_label, axis=1)
    df = df.sort_values('avg_input_len')
    
    all_freqs = sorted(df['gpu_frequency_mhz'].unique())
    selected_freqs = all_freqs[::2]

    if 1410 in all_freqs and 1410 not in selected_freqs:
        selected_freqs.append(1410)
        
    df = df[df['gpu_frequency_mhz'].isin(selected_freqs)]
    pivot_table = df.pivot(index='label', columns='gpu_frequency_mhz', values='energy_per_req')
    ordered_index = ["Short (~50 tok)", "Medium (~250 tok)", "Long (~500 tok)"]
    existing_labels = [l for l in ordered_index if l in pivot_table.index]
    pivot_table = pivot_table.reindex(existing_labels)
    
    plt.figure(figsize=(8, 4.5))
    ax = sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", 
                     cbar_kws={'label': 'Energy per Request (J)'},
                     linewidths=1, linecolor='white',
                     annot_kws={"size": 12, "weight": "bold"})
    
    ax.set_title('(c) Types vs. Frequency Energy Map', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('GPU Frequency (MHz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Request Type', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    plot_heatmap("test/proposed/results/performance_profile.csv", "paper/figures/observation2_heatmap.pdf")
