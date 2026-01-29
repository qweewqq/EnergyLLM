"""
Proposed Method Profiler - Qwen2.5-7B-Instruct 版本

使用 DVFS (频率调节) 作为控制方式
Profile 维度：(gpu_frequency_mhz, batch_size, avg_input_len, avg_output_len)

与 Llama-3.1-8B 版本的区别：
- 模型路径改为 Qwen2.5-7B-Instruct
- 结果保存到 results_qwen25_7b 目录，不覆盖其他模型的结果
"""

import os
import subprocess
import time
import re
import csv
import numpy as np
import pandas as pd
import psutil
import signal
import sys

# --- 全局配置 ---

# 定义要测试的 GPU 频率列表 (MHz) - DVFS 控制
GPU_FREQUENCIES = [705, 810, 900, 960, 1005, 1050, 1110, 1155, 1200, 1245, 1290, 1320, 1350, 1380, 1410]

# 定义要测试的批处理大小列表
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]  # 7 个档位

# 定义要测试的 Token 长度组合 (input_len, output_len, name)
TOKEN_LENGTH_CONFIGS = [
    (50, 50, "short"),      # 短输入短输出
    (250, 150, "medium"),   # 中等长度
    (500, 300, "long"),     # 长输入长输出
]

# vLLM 服务器启动的超时时间 (秒)
VLLM_TIMEOUT = 180

# 定义了各项工具和文件的路径
VLLM_ENV_PYTHON = "/home/vipuser/anaconda3/envs/vllm_env/bin/python"
VLLM_SERVER_SCRIPT = "-m vllm.entrypoints.openai.api_server"

# ========== 关键修改：模型路径 ==========
MODEL_PATH = "/home/data/models/Qwen2.5-7B-Instruct"
# ========================================

BATCH_SENDER_SCRIPT = "/home/data/Fjw/test/proposed/batch_sender.py"
MONITOR_SCRIPT = "/home/data/Fjw/test/proposed/monitor_full_system.py"
PROMPT_FILE = "prompt_datasets_jsonl/sharegpt_en_mixed_all_buckets.jsonl"

# ========== 关键修改：结果目录 ==========
RESULTS_DIR = "results_qwen25_7b"
# ========================================

VLLM_LOG_FILE = "vllm_profiler_qwen25_7b.log"
MONITOR_DATA_FILE = "monitoring_profile_qwen25_7b_temp.csv"
RESULTS_FILE = os.path.join(RESULTS_DIR, "performance_profile.csv")


# --- 辅助函数 ---

def log_message(message):
    """打印带有时间戳的日志信息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [PROFILER-QWEN25-7B] {message}")


def cleanup_zombies():
    """清理可能残留的 vLLM 服务器进程"""
    log_message("正在检查并清理残留的 vLLM 服务器进程...")
    try:
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], check=False)
        time.sleep(5)
    except Exception as e:
        log_message(f"清理进程时出错: {e}")


def wait_for_gpu_memory(required_free_gib=30.0, timeout=60):
    """等待直到 GPU 显存释放"""
    log_message(f"等待 GPU 显存释放 (需要 > {required_free_gib} GiB)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            free_memory_mib = float(result.stdout.strip())
            free_memory_gib = free_memory_mib / 1024.0
            
            if free_memory_gib >= required_free_gib:
                log_message(f"GPU 显存已释放: {free_memory_gib:.2f} GiB")
                return True
        except Exception as e:
            log_message(f"检查显存时出错: {e}")
        
        time.sleep(2)
    
    log_message("警告：等待 GPU 显存释放超时。")
    return False


def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    log_message("接收到中断信号，正在清理并退出...")
    cleanup_zombies()
    reset_gpu_frequency()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def set_gpu_frequency(freq_mhz):
    """使用 nvidia-smi 设置 GPU 核心频率 (DVFS)"""
    try:
        log_message(f"正在设置 GPU 核心频率为: {freq_mhz} MHz...")
        subprocess.run(["sudo", "nvidia-smi", "-lgc", "210,1410"], check=True, capture_output=True, text=True)
        subprocess.run(["sudo", "nvidia-smi", "-lgc", str(freq_mhz)], check=True, capture_output=True, text=True)
        log_message("GPU 频率设置成功。")
        time.sleep(2)
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"错误：设置 GPU 频率失败。返回码: {e.returncode}")
        log_message(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        log_message("错误：未找到 `nvidia-smi` 命令。")
        return False


def reset_gpu_frequency():
    """恢复 GPU 默认频率设置"""
    log_message("正在恢复 GPU 默认频率设置...")
    try:
        subprocess.run(["sudo", "nvidia-smi", "-rgc"], check=True, capture_output=True, text=True)
        log_message("GPU 频率已恢复。")
    except Exception as e:
        log_message(f"恢复频率设置失败: {e}")


def start_vllm_server():
    """在后台启动 vLLM API 服务器"""
    cleanup_zombies()
    
    if not wait_for_gpu_memory():
        log_message("显存不足，尝试强制清理...")
        cleanup_zombies()
        if not wait_for_gpu_memory():
            log_message("错误：显存仍然不足，无法启动 vLLM。")
            return None
    
    log_message(f"正在启动 vLLM 服务器 (模型: {MODEL_PATH})...")
    # Qwen2.5-7B 参数量小，使用与Llama相同的max-model-len
    cmd = f"{VLLM_ENV_PYTHON} {VLLM_SERVER_SCRIPT} --model {MODEL_PATH} --max-model-len 4096 --trust-remote-code"
    with open(VLLM_LOG_FILE, "w") as log_file:
        process = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    
    start_time = time.time()
    while time.time() - start_time < VLLM_TIMEOUT:
        try:
            with open(VLLM_LOG_FILE, "r") as f:
                content = f.read()
                if "Application startup complete" in content or "Uvicorn running on" in content:
                    log_message("vLLM 服务器已就绪。")
                    return process
        except FileNotFoundError:
            pass
        time.sleep(2)

    log_message("错误：vLLM 服务器在超时时间内未能成功启动。")
    kill_process_group(process)
    return None


def kill_process_group(process):
    """安全地终止一个进程及其所有子进程"""
    if process and psutil.pid_exists(process.pid):
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, 9)
            log_message(f"已终止进程组 (PGID: {pgid})。")
        except ProcessLookupError:
            pass


def run_batch_sender(batch_size, target_input_len, max_output_tokens, model_name):
    """运行批处理发送器并解析其输出"""
    log_message(f"正在执行批处理测试: Batch={batch_size}, InputLen≈{target_input_len}, MaxOutput={max_output_tokens}...")
    
    cmd = (f"{VLLM_ENV_PYTHON} {BATCH_SENDER_SCRIPT} {batch_size} {PROMPT_FILE} "
           f"--max_tokens {max_output_tokens} --target_input_len {target_input_len} --model {model_name}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout
        log_message(f"批处理测试完成。")
        
        total_time_match = re.search(r"完成整个批次的总耗时: ([\d.]+) 秒", output)
        avg_output_match = re.search(r"平均输出长度: ([\d.]+) tokens", output)
        
        if total_time_match:
            total_time = float(total_time_match.group(1))
            avg_output_len = float(avg_output_match.group(1)) if avg_output_match else max_output_tokens
            return total_time, avg_output_len
        else:
            log_message("警告：无法从输出中解析总耗时。")
            return None, None
            
    except subprocess.CalledProcessError as e:
        log_message(f"错误：运行批处理发送器失败。返回码: {e.returncode}")
        log_message(f"Stderr: {e.stderr}")
        return None, None


def start_monitor():
    """启动系统监控脚本"""
    log_message("正在启动系统监控...")
    if os.path.exists(MONITOR_DATA_FILE):
        os.remove(MONITOR_DATA_FILE)
        
    cmd = f"{VLLM_ENV_PYTHON} {MONITOR_SCRIPT} {MONITOR_DATA_FILE}"
    process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(1)
    return process


def analyze_power_consumption():
    """分析监控数据并计算平均功耗"""
    if not os.path.exists(MONITOR_DATA_FILE):
        log_message("警告：未找到监控数据文件。")
        return None
    try:
        df = pd.read_csv(MONITOR_DATA_FILE)
        avg_power = df['gpu_power_watts'].mean()
        log_message(f"计算出的平均 GPU 功耗为: {avg_power:.2f} W")
        return avg_power
    except Exception as e:
        log_message(f"错误：分析功耗数据时出错: {e}")
        return None


# --- 主执行逻辑 ---

def main():
    """主函数，编排整个性能分析流程"""
    log_message("=== 开始执行 Qwen2.5-7B-Instruct 性能建模分析 (DVFS 版本) ===")
    log_message(f"模型路径: {MODEL_PATH}")
    log_message(f"结果目录: {RESULTS_DIR}")
    
    # 创建结果目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    cleanup_zombies()
    
    if os.geteuid() != 0:
        log_message("警告：脚本未使用 sudo 权限运行。设置 GPU 频率的操作可能会失败。")

    # 计算总测试点数
    total_points = len(GPU_FREQUENCIES) * len(BATCH_SIZES) * len(TOKEN_LENGTH_CONFIGS)
    log_message(f"总测试配置数: {len(GPU_FREQUENCIES)} freq × {len(BATCH_SIZES)} batch × {len(TOKEN_LENGTH_CONFIGS)} length = {total_points} 个点")

    # 加载已有结果，支持断点续传
    existing_results = set()
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 4:
                        existing_results.add((int(row[0]), int(row[1]), int(float(row[2])), int(float(row[3]))))
            log_message(f"发现已有结果文件，包含 {len(existing_results)} 个数据点。将跳过已测试的配置。")
        except Exception as e:
            log_message(f"读取已有结果文件时出错: {e}。将重新开始。")
    else:
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "gpu_frequency_mhz", "batch_size", 
                "avg_input_len", "avg_output_len",
                "total_latency_sec", "avg_gpu_power_watts"
            ])

    all_results = []
    completed = len(existing_results)

    try:
        for freq in GPU_FREQUENCIES:
            if not set_gpu_frequency(freq):
                log_message(f"跳过频率 {freq} MHz，因为无法设置。")
                continue

            for bs in BATCH_SIZES:
                for input_len, output_len, length_name in TOKEN_LENGTH_CONFIGS:
                    if (freq, bs, input_len, output_len) in existing_results:
                        log_message(f"配置 (Freq={freq}, Batch={bs}, In={input_len}, Out={output_len}) 已存在，跳过。")
                        continue

                    completed += 1
                    log_message(f"--- [{completed}/{total_points}] 测试: Freq={freq}MHz, Batch={bs}, Length={length_name} ---")
                    
                    vllm_process = start_vllm_server()
                    if not vllm_process:
                        log_message("vLLM 启动失败，跳过当前测试。")
                        continue
                    
                    monitor_process = start_monitor()
                    
                    total_latency, actual_output_len = run_batch_sender(bs, input_len, output_len, MODEL_PATH)
                    
                    kill_process_group(monitor_process)
                    kill_process_group(vllm_process)
                    cleanup_zombies()
                    
                    time.sleep(2)
                    
                    avg_power = analyze_power_consumption()
                    
                    if total_latency is not None and avg_power is not None:
                        final_output_len = actual_output_len if actual_output_len else output_len
                        
                        result_row = [freq, bs, input_len, final_output_len, total_latency, avg_power]
                        all_results.append(result_row)
                        
                        with open(RESULTS_FILE, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(result_row)
                            
                        log_message(f"结果已保存: Freq={freq}MHz, Batch={bs}, In={input_len}, Out={final_output_len:.0f}, "
                                   f"Latency={total_latency:.2f}s, Power={avg_power:.2f}W")
                    else:
                        log_message("由于数据采集不完整，本次测试结果被丢弃。")

                    time.sleep(5)

    finally:
        reset_gpu_frequency()

    log_message("=== 所有性能分析测试已完成 ===")
    log_message(f"最终结果已保存在: {RESULTS_FILE}")
    
    if all_results:
        final_df = pd.DataFrame(all_results, columns=[
            "gpu_frequency_mhz", "batch_size", 
            "avg_input_len", "avg_output_len",
            "total_latency_sec", "avg_gpu_power_watts"
        ])
        print("\n--- 最终性能画像数据 ---")
        print(final_df)
        print("\n--------------------------")


if __name__ == "__main__":
    main()
