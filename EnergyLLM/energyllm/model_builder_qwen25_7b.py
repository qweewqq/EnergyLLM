"""
Proposed Method - 模型构建器 (Qwen2.5-7B 版本)

构建延迟和功耗预测模型，支持 DVFS 频率控制。
特征：(gpu_frequency_mhz, batch_size, avg_input_len, avg_output_len)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import os


def build_and_evaluate_models(data_path, output_dir):
    """
    加载性能数据，构建并评估延迟和功耗模型。
    
    Proposed Method (DVFS) 版本：
    - 特征：(gpu_frequency_mhz, batch_size, avg_input_len, avg_output_len)
    - 目标：total_latency_sec, avg_gpu_power_watts
    """
    print("--- 1. 加载和预处理数据 ---")
    df = pd.read_csv(data_path)
    
    # 检测数据格式
    if 'gpu_frequency_mhz' in df.columns:
        power_column = 'gpu_frequency_mhz'
        control_type = 'dvfs'
        print("  - 检测到 DVFS 频率数据格式 (Proposed Method)")
    elif 'gpu_power_limit_watts' in df.columns:
        power_column = 'gpu_power_limit_watts'
        control_type = 'power_limit'
        print("  - 检测到 Power Limit 数据格式")
    else:
        raise ValueError("数据文件中未找到 gpu_frequency_mhz 或 gpu_power_limit_watts 列")
    
    # 检查是否包含 token 长度列
    has_token_length = 'avg_input_len' in df.columns and 'avg_output_len' in df.columns
    
    if has_token_length:
        print("  - 检测到 Token 长度特征，使用 4 维特征空间。")
        feature_columns = [power_column, 'batch_size', 'avg_input_len', 'avg_output_len']
    else:
        print("  - 未检测到 Token 长度特征，使用 2 维特征空间。")
        feature_columns = [power_column, 'batch_size']
    
    X = df[feature_columns]
    y_latency = df['total_latency_sec']
    y_power = df['avg_gpu_power_watts']
    
    print(f"  - 数据点数量: {len(df)}")
    print(f"  - 特征维度: {feature_columns}")
    print(f"  - 频率/功率范围: {df[power_column].min()} - {df[power_column].max()}")
    print(f"  - Batch 范围: {df['batch_size'].min()} - {df['batch_size'].max()}")
    
    # --- 2. 构建和训练模型 ---
    print("\n--- 2. 开始构建和训练模型 ---")
    
    print("  - 正在训练延迟模型 (Random Forest)...")
    latency_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    latency_model.fit(X, y_latency)
    print("  - 延迟模型训练完成。")

    print("  - 正在训练功耗模型 (Random Forest)...")
    power_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    power_model.fit(X, y_power)
    print("  - 功耗模型训练完成。")

    # --- 3. 评估模型准确度 ---
    print("\n--- 3. 评估模型准确度 ---")
    y_pred_latency = latency_model.predict(X)
    y_pred_power = power_model.predict(X)
    
    def calculate_metrics(y_true, y_pred, name):
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"  [{name} 模型评估]")
        print(f"    - R² (决定系数): {r2:.4f}")
        print(f"    - MAPE: {mape:.2f}%")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - MAE:  {mae:.4f}")
        return {'mape': mape, 'rmse': rmse, 'mae': mae, 'r2': r2}

    metrics_latency = calculate_metrics(y_latency, y_pred_latency, "延迟")
    metrics_power = calculate_metrics(y_power, y_pred_power, "功耗")
    
    # --- 4. 特征重要性分析 ---
    print("\n--- 4. 特征重要性分析 ---")
    print("  [延迟模型特征重要性]")
    for feat, imp in zip(feature_columns, latency_model.feature_importances_):
        print(f"    - {feat}: {imp:.4f}")
    
    print("  [功耗模型特征重要性]")
    for feat, imp in zip(feature_columns, power_model.feature_importances_):
        print(f"    - {feat}: {imp:.4f}")
    
    # --- 5. 保存模型到文件 ---
    print("\n--- 5. 保存模型到文件 ---")
    os.makedirs(output_dir, exist_ok=True)
    
    latency_model_path = os.path.join(output_dir, 'latency_model.joblib')
    power_model_path = os.path.join(output_dir, 'power_model.joblib')
    
    model_info = {
        'latency_model': latency_model,
        'power_model': power_model,
        'feature_columns': feature_columns,
        'has_token_length': has_token_length,
        'power_column': power_column,
        'control_type': control_type
    }
    
    joblib.dump(latency_model, latency_model_path)
    joblib.dump(power_model, power_model_path)
    
    model_meta_path = os.path.join(output_dir, 'model_meta.joblib')
    joblib.dump(model_info, model_meta_path)
    
    print(f"  - 延迟模型已保存到: {latency_model_path}")
    print(f"  - 功耗模型已保存到: {power_model_path}")
    print(f"  - 模型元信息已保存到: {model_meta_path}")
    print(f"  - 控制类型: {control_type}")

    # --- 6. 可视化 ---
    print("\n--- 6. 生成可视化图 ---")
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_latency, y_pred_latency, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_latency.min(), y_latency.max()], [y_latency.min(), y_latency.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Latency (sec)')
    axes[0].set_ylabel('Predicted Latency (sec)')
    axes[0].set_title(f'Latency Model (R²={metrics_latency["r2"]:.4f}, MAPE={metrics_latency["mape"]:.2f}%)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_power, y_pred_power, alpha=0.5, edgecolors='k', linewidth=0.5, color='orange')
    axes[1].plot([y_power.min(), y_power.max()], [y_power.min(), y_power.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Power (Watts)')
    axes[1].set_ylabel('Predicted Power (Watts)')
    axes[1].set_title(f'Power Model (R²={metrics_power["r2"]:.4f}, MAPE={metrics_power["mape"]:.2f}%)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    pred_vs_actual_path = os.path.join(plots_dir, 'model_prediction_accuracy.png')
    plt.savefig(pred_vs_actual_path, dpi=150)
    plt.close()
    print(f"  - 预测准确度图已保存到: {pred_vs_actual_path}")

    # 特征重要性图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].barh(feature_columns, latency_model.feature_importances_, color='steelblue')
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Latency Model Feature Importance (Qwen2.5-7B)')
    
    axes[1].barh(feature_columns, power_model.feature_importances_, color='darkorange')
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Power Model Feature Importance (Qwen2.5-7B)')
    
    plt.tight_layout()
    importance_path = os.path.join(plots_dir, 'feature_importance.png')
    plt.savefig(importance_path, dpi=150)
    plt.close()
    print(f"  - 特征重要性图已保存到: {importance_path}")

    print("\n--- Qwen2.5-7B 模型构建完成！ ---")
    
    return model_info


if __name__ == "__main__":
    BASE_DIR = "/home/data/Fjw/test/proposed/"
    
    # Qwen2.5-7B 的数据和输出路径
    data_file = os.path.join(BASE_DIR, "results_qwen25_7b", "performance_profile.csv")
    output_dir = os.path.join(BASE_DIR, "results_qwen25_7b")
    
    if not os.path.exists(data_file):
        print(f"错误：找不到数据文件 {data_file}")
        print("请先运行 profiler_qwen25_7b.py 采集性能数据。")
    else:
        build_and_evaluate_models(data_file, output_dir)
