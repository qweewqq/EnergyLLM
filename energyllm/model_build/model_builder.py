import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import os


def build_and_evaluate_models(data_path, output_dir):
    print("--- 1. Loading and Preprocessing Data ---")
    df = pd.read_csv(data_path)
    
    if 'gpu_frequency_mhz' in df.columns:
        power_column = 'gpu_frequency_mhz'
        control_type = 'dvfs'
        print("  - Detected DVFS frequency data format (Proposed Method)")
    elif 'gpu_power_limit_watts' in df.columns:
        power_column = 'gpu_power_limit_watts'
        control_type = 'power_limit'
        print("  - Detected Power Limit data format")
    else:
        raise ValueError("Neither gpu_frequency_mhz nor gpu_power_limit_watts column found in data file")
    
    has_token_length = 'avg_input_len' in df.columns and 'avg_output_len' in df.columns
    
    if has_token_length:
        print("  - Detected token length features, using 4-dimensional feature space.")
        feature_columns = [power_column, 'batch_size', 'avg_input_len', 'avg_output_len']
    else:
        print("  - Token length features not detected, using 2-dimensional feature space.")
        feature_columns = [power_column, 'batch_size']
    
    X = df[feature_columns]
    y_latency = df['total_latency_sec']
    y_power = df['avg_gpu_power_watts']
    
    print(f"  - Number of data points: {len(df)}")
    print(f"  - Feature dimensions: {feature_columns}")
    print(f"  - Frequency/Power range: {df[power_column].min()} - {df[power_column].max()}")
    print(f"  - Batch size range: {df['batch_size'].min()} - {df['batch_size'].max()}")
    
    print("\n--- 2. Building and Training Models ---")
    
    print("  - Training latency model (Random Forest)...")
    latency_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    latency_model.fit(X, y_latency)
    print("  - Latency model training completed.")

    print("  - Training power model (Random Forest)...")
    power_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    power_model.fit(X, y_power)
    print("  - Power model training completed.")

    print("\n--- 3. Evaluating Model Accuracy ---")
    y_pred_latency = latency_model.predict(X)
    y_pred_power = power_model.predict(X)
    
    def calculate_metrics(y_true, y_pred, name):
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"  [{name} Model Evaluation]")
        print(f"    - R² (Coefficient of Determination): {r2:.4f}")
        print(f"    - MAPE: {mape:.2f}%")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - MAE:  {mae:.4f}")
        return {'mape': mape, 'rmse': rmse, 'mae': mae, 'r2': r2}

    metrics_latency = calculate_metrics(y_latency, y_pred_latency, "Latency")
    metrics_power = calculate_metrics(y_power, y_pred_power, "Power")
    
    print("\n--- 4. Feature Importance Analysis ---")
    print("  [Latency Model Feature Importance]")
    for feat, imp in zip(feature_columns, latency_model.feature_importances_):
        print(f"    - {feat}: {imp:.4f}")
    
    print("  [Power Model Feature Importance]")
    for feat, imp in zip(feature_columns, power_model.feature_importances_):
        print(f"    - {feat}: {imp:.4f}")
    
    print("\n--- 5. Saving Models to Files ---")
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    latency_model_path = os.path.join(results_dir, 'latency_model.joblib')
    power_model_path = os.path.join(results_dir, 'power_model.joblib')
    
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
    
    model_meta_path = os.path.join(results_dir, 'model_meta.joblib')
    joblib.dump(model_info, model_meta_path)
    
    print(f"  - Latency model saved to: {latency_model_path}")
    print(f"  - Power model saved to: {power_model_path}")
    print(f"  - Model metadata saved to: {model_meta_path}")
    print(f"  - Control type: {control_type}")

    print("\n--- 6. Generating Visualization Plots ---")
    plots_dir = os.path.join(results_dir, 'plots')
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
    print(f"  - Prediction accuracy plot saved to: {pred_vs_actual_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].barh(feature_columns, latency_model.feature_importances_, color='steelblue')
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Latency Model Feature Importance')
    
    axes[1].barh(feature_columns, power_model.feature_importances_, color='darkorange')
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Power Model Feature Importance')
    
    plt.tight_layout()
    importance_path = os.path.join(plots_dir, 'feature_importance.png')
    plt.savefig(importance_path, dpi=150)
    plt.close()
    print(f"  - Feature importance plot saved to: {importance_path}")

    print("\n--- Model Building Completed! ---")
    
    return model_info


if __name__ == "__main__":
    BASE_DIR = "/home/data/EnergyLLM/test/proposed/"
    data_file = os.path.join(BASE_DIR, "results", "performance_profile.csv")
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        print("Please run profiler.py first to collect performance data.")
    else:
        build_and_evaluate_models(data_file, BASE_DIR)
