"""
生成等SLO对比柱状图 - SLO=12s配置
横轴为arrival rate，每个rate下对比4个方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# 读取数据
df = pd.read_excel('llama-sharegpt_results -bak.xlsx')
df_config = df[df['table_type'] == 'config_result'].copy()

# 筛选SLO=12s的数据
slo_12_data = df_config[df_config['slo_target_s'] == 12.0]

# 定义方法和颜色
methods = ['PPO-v5 (Ours)', 'Batch-Only', 'DVFS-Only', 'Static-High']
method_labels = ['EnergyLLM', 'Batch-Only', 'DVFS-Only', 'Static-High']
colors = ['#2E86AB', '#06A77D', '#F77F00', '#E63946']

# 获取arrival rates并排序
arrival_rates = sorted(slo_12_data['arrival_rate_req_s'].unique())

# 构建数据矩阵
slo_matrix = []
energy_matrix = []

for rate in arrival_rates:
    rate_data = slo_12_data[slo_12_data['arrival_rate_req_s'] == rate]
    slo_row = []
    energy_row = []
    for method in methods:
        method_data = rate_data[rate_data['method'] == method]
        if len(method_data) > 0:
            slo_row.append(method_data['SLO_pct'].values[0])
            energy_row.append(method_data['Energy'].values[0])
        else:
            slo_row.append(0)
            energy_row.append(0)
    slo_matrix.append(slo_row)
    energy_matrix.append(energy_row)

slo_matrix = np.array(slo_matrix)
energy_matrix = np.array(energy_matrix)

# 创建图表 - 1行2列
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 设置柱状图参数
x = np.arange(len(arrival_rates))
width = 0.2  # 每个柱子的宽度
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

# ========================
# 左图: SLO Attainment
# ========================
ax1 = axes[0]

for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
    slo_values = slo_matrix[:, i]
    bars = ax1.bar(x + offsets[i], slo_values, width, label=label,
                   color=color, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # 在柱子上方添加数值
    for j, (bar, val) in enumerate(zip(bars, slo_values)):
        if val > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                    color=color)

ax1.set_ylabel('SLO Attainment (%)', fontsize=11, fontweight='bold')
ax1.set_xlabel('')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{r:.1f}' for r in arrival_rates], fontsize=9)
ax1.set_ylim([0, 115])
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_axisbelow(True)
ax1.tick_params(labelsize=9)

ax1.text(0.5, -0.24, 'Arrival Rate (req/s)', transform=ax1.transAxes,
         fontsize=11, fontweight='bold', ha='center')
ax1.text(0.5, -0.38, '(a) SLO Performance', transform=ax1.transAxes,
         fontsize=13, fontweight='bold', ha='center')

# 添加100%参考线
ax1.axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.6, zorder=2)

# ========================
# 右图: Energy Consumption
# ========================
ax2 = axes[1]

for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
    energy_values = energy_matrix[:, i]
    bars = ax2.bar(x + offsets[i], energy_values, width, label=label,
                   color=color, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # 在柱子上方添加数值
    for j, (bar, val) in enumerate(zip(bars, energy_values)):
        if val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 15,
                    f'{val:.0f}',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                    color=color)

ax2.set_ylabel('Energy per Request (J)', fontsize=11, fontweight='bold')
ax2.set_xlabel('')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{r:.1f}' for r in arrival_rates], fontsize=9)
ax2.set_ylim([0, 1000])
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_axisbelow(True)
ax2.tick_params(labelsize=9)

ax2.text(0.5, -0.24, 'Arrival Rate (req/s)', transform=ax2.transAxes,
         fontsize=11, fontweight='bold', ha='center')
ax2.text(0.5, -0.38, '(b) Energy Efficiency', transform=ax2.transAxes,
         fontsize=13, fontweight='bold', ha='center')

plt.tight_layout(rect=[0, 0.11, 1, 0.96])  # 调整布局，为下方文字和上方图例留出空间


handles = []
labels_list = []
for i, (label, color) in enumerate(zip(method_labels, colors)):
    handle = plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='black', linewidth=1.5, alpha=0.85)
    handles.append(handle)
    labels_list.append(label)

# 将图例放在图的上方
fig.legend(handles, labels_list, loc='upper center', bbox_to_anchor=(0.5, 0.98),
          ncol=4, fontsize=10, frameon=False, columnspacing=1.8)

# 保存图片
output_path = 'equal_slo_comparison_bar.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存: {output_path}")

output_path_png = 'equal_slo_comparison_bar.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"图片已保存: {output_path_png}")

plt.show()

# 打印统计总结
print("\n" + "=" * 80)
print("柱状图总结 (SLO Target = 12s)")
print("=" * 80)

print("\n各方法在不同arrival rate下的性能：")
print("-" * 80)
print(f"{'Rate':<8} {'指标':<10} {'EnergyLLM':<12} {'Batch-Only':<12} {'DVFS-Only':<12} {'Static-High':<12}")
print("-" * 80)

for i, rate in enumerate(arrival_rates):
    print(f"{rate:.1f}     SLO%       {slo_matrix[i,0]:>10.1f}% {slo_matrix[i,1]:>10.1f}% {slo_matrix[i,2]:>10.1f}% {slo_matrix[i,3]:>10.1f}%")
    print(f"         Energy     {energy_matrix[i,0]:>10.1f}J {energy_matrix[i,1]:>10.1f}J {energy_matrix[i,2]:>10.1f}J {energy_matrix[i,3]:>10.1f}J")
    print("-" * 80)

# 计算平均值
avg_slo = slo_matrix.mean(axis=0)
avg_energy = energy_matrix.mean(axis=0)

print(f"\n平均值    SLO%       {avg_slo[0]:>10.1f}% {avg_slo[1]:>10.1f}% {avg_slo[2]:>10.1f}% {avg_slo[3]:>10.1f}%")
print(f"         Energy     {avg_energy[0]:>10.1f}J {avg_energy[1]:>10.1f}J {avg_energy[2]:>10.1f}J {avg_energy[3]:>10.1f}J")

print("\n\n能耗节省（相对于Static-High）：")
print("-" * 80)
for i, label in enumerate(method_labels[:-1]):
    savings = (avg_energy[3] - avg_energy[i]) / avg_energy[3] * 100
    print(f"  {label:<20}: {savings:>6.1f}%")