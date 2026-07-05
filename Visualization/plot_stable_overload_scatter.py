"""
可视化稳定/过载边界下的系统性能

展示两个配置的Energy vs SLO scatter plot:
- Overload boundary: SLO=10s, rate=0.6 req/s (左图)
- Stable boundary: SLO=13s, rate=0.8 req/s (右图)
"""

import matplotlib.pyplot as plt
import numpy as np

# 数据：Overload boundary (SLO=10s, rate=0.6)
overload_data = {
    'Static-High': {'slo': 89.19, 'energy': 877.78, 'marker': 's', 'color': '#1f77b4'},
    'DVFS-Only': {'slo': 65.85, 'energy': 588.81, 'marker': '^', 'color': '#ff7f0e'},
    'Batch-Only': {'slo': 69.99, 'energy': 605.09, 'marker': 'v', 'color': '#2ca02c'},
    'Reactive-DVFS': {'slo': 56.61, 'energy': 574.52, 'marker': 'D', 'color': '#d62728'},
    'Token-Aware': {'slo': 73.72, 'energy': 847.24, 'marker': 'p', 'color': '#9467bd'},
    'DynamoLLM': {'slo': 58.93, 'energy': 528.80, 'marker': 'X', 'color': '#8c564b'},
    'EnergyLLM': {'slo': 76.19, 'energy': 528.78, 'marker': '*', 'color': '#e377c2'},
}

# 数据：Stable boundary (SLO=13s, rate=0.8)
stable_data = {
    'Static-High': {'slo': 100.0, 'energy': 917.70, 'marker': 's', 'color': '#1f77b4'},
    'DVFS-Only': {'slo': 99.58, 'energy': 630.17, 'marker': '^', 'color': '#ff7f0e'},
    'Batch-Only': {'slo': 93.82, 'energy': 675.80, 'marker': 'v', 'color': '#2ca02c'},
    'Reactive-DVFS': {'slo': 90.59, 'energy': 597.18, 'marker': 'D', 'color': '#d62728'},
    'Token-Aware': {'slo': 83.65, 'energy': 910.10, 'marker': 'p', 'color': '#9467bd'},
    'DynamoLLM': {'slo': 92.98, 'energy': 574.06, 'marker': 'X', 'color': '#8c564b'},
    'EnergyLLM': {'slo': 100.0, 'energy': 561.55, 'marker': '*', 'color': '#e377c2'},
}

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# 设置字体大小（与observation图一致）
LABEL_SIZE = 11
TICK_SIZE = 9
TITLE_SIZE = 13
LEGEND_SIZE = 9
MARKER_SIZE = 150

# ========== 左图：Overload Boundary ==========
for method, data in overload_data.items():
    label = 'EnergyLLM (Ours)' if method == 'EnergyLLM' else method
    zorder = 10 if method == 'EnergyLLM' else 5
    ax1.scatter(data['energy'], data['slo'], 
                s=MARKER_SIZE, marker=data['marker'], 
                color=data['color'], label=label,
                edgecolors='black', linewidths=1.5, zorder=zorder, alpha=0.9)

ax1.set_ylabel('SLO Attainment Rate (%)', fontsize=LABEL_SIZE, fontweight='bold')
ax1.set_xlabel('Energy per Request (J)\n\n(a) Overload Boundary (SLO=10s, Rate=0.6 req/s)', 
              fontsize=LABEL_SIZE, fontweight='bold', labelpad=10)
ax1.tick_params(axis='both', labelsize=TICK_SIZE)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_xlim(500, 900)
ax1.set_ylim(50, 95)

# ========== 右图：Stable Boundary ==========
for method, data in stable_data.items():
    label = 'EnergyLLM (Ours)' if method == 'EnergyLLM' else method
    zorder = 10 if method == 'EnergyLLM' else 5
    ax2.scatter(data['energy'], data['slo'], 
                s=MARKER_SIZE, marker=data['marker'], 
                color=data['color'], label=label,
                edgecolors='black', linewidths=1.5, zorder=zorder, alpha=0.9)

ax2.set_ylabel('SLO Attainment Rate (%)', fontsize=LABEL_SIZE, fontweight='bold')
ax2.set_xlabel('Energy per Request (J)\n\n(b) Stable Boundary (SLO=13s, Rate=0.8 req/s)', 
              fontsize=LABEL_SIZE, fontweight='bold', labelpad=10)
ax2.tick_params(axis='both', labelsize=TICK_SIZE)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_xlim(550, 950)
ax2.set_ylim(80, 101)

# 共享图例（放在右图右侧）
handles, labels = ax2.get_legend_handles_labels()
# 重新排序：把EnergyLLM放在最后
energyllm_idx = labels.index('EnergyLLM (Ours)')
handles = handles[:energyllm_idx] + handles[energyllm_idx+1:] + [handles[energyllm_idx]]
labels = labels[:energyllm_idx] + labels[energyllm_idx+1:] + [labels[energyllm_idx]]

ax2.legend(handles, labels, loc='lower left', fontsize=LEGEND_SIZE, 
           framealpha=0.95, edgecolor='black')

plt.tight_layout()

# 保存
output_pdf = 'test/baselines/batch_experiment_results/stable_overload_comparison.pdf'
output_png = 'test/baselines/batch_experiment_results/stable_overload_comparison.png'
plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"✓ 已保存: {output_pdf}")
print(f"✓ 已保存: {output_png}")

# 复制到paper/figures
import shutil
paper_pdf = 'paper/figures/stable_overload_comparison.pdf'
shutil.copy(output_pdf, paper_pdf)
print(f"✓ 已复制到: {paper_pdf}")

plt.show()

