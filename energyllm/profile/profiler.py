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

GPU_FREQUENCIES = [705, 810, 900, 960, 1005, 1050, 1110, 1155, 1200, 1245, 1290, 1320, 1350, 1380, 1410]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
TOKEN_LENGTH_CONFIGS = [
    (50, 50, "short"),
    (250, 150, "medium"),
    (500, 300, "long"),
]

VLLM_TIMEOUT = 180
VLLM_ENV_PYTHON = "/home/vipuser/anaconda3/envs/vllm_env/bin/python"
VLLM_SERVER_SCRIPT = "-m vllm.entrypoints.openai.api_server"
MODEL_PATH = "/home/data/models/Meta-Llama-3.1-8B-Instruct"
BATCH_SENDER_SCRIPT = "/home/data/EnergyLLM/test/proposed/batch_sender.py"
MONITOR_SCRIPT = "/home/data/EnergyLLM/test/proposed/monitor_full_system.py"
PROMPT_FILE = "prompt_datasets_jsonl/sharegpt_en_mixed_all_buckets.jsonl"

RESULTS_DIR = "results"
VLLM_LOG_FILE = "vllm_profiler.log"
MONITOR_DATA_FILE = "monitoring_profile_temp.csv"
RESULTS_FILE = os.path.join(RESULTS_DIR, "performance_profile.csv")


def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [PROFILER-DVFS] {message}")


def cleanup_zombies():
    log_message("Checking and cleaning up remaining vLLM server processes...")
    try:
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], check=False)
        time.sleep(5)
    except Exception as e:
        log_message(f"Error cleaning up processes: {e}")


def wait_for_gpu_memory(required_free_gib=30.0, timeout=60):
    log_message(f"Waiting for GPU memory to be freed (need > {required_free_gib} GiB)...")
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
                log_message(f"GPU memory freed: {free_memory_gib:.2f} GiB")
                return True
        except Exception as e:
            log_message(f"Error checking memory: {e}")
        
        time.sleep(2)
    
    log_message("Warning: Timeout waiting for GPU memory to be freed.")
    return False


def signal_handler(sig, frame):
    log_message("Received interrupt signal, cleaning up and exiting...")
    cleanup_zombies()
    reset_gpu_frequency()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def set_gpu_frequency(freq_mhz):
    try:
        log_message(f"Setting GPU core frequency to: {freq_mhz} MHz...")
        subprocess.run(["sudo", "nvidia-smi", "-lgc", "210,1410"], check=True, capture_output=True, text=True)
        subprocess.run(["sudo", "nvidia-smi", "-lgc", str(freq_mhz)], check=True, capture_output=True, text=True)
        log_message("GPU frequency set successfully.")
        time.sleep(2)
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"Error: Failed to set GPU frequency. Return code: {e.returncode}")
        log_message(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        log_message("Error: `nvidia-smi` command not found.")
        return False


def reset_gpu_frequency():
    log_message("Restoring GPU default frequency settings...")
    try:
        subprocess.run(["sudo", "nvidia-smi", "-rgc"], check=True, capture_output=True, text=True)
        log_message("GPU frequency restored.")
    except Exception as e:
        log_message(f"Failed to restore frequency settings: {e}")


def start_vllm_server():
    cleanup_zombies()
    
    if not wait_for_gpu_memory():
        log_message("Insufficient memory, attempting forced cleanup...")
        cleanup_zombies()
        if not wait_for_gpu_memory():
            log_message("Error: Still insufficient memory, cannot start vLLM.")
            return None
    
    log_message("Starting vLLM server...")
    cmd = f"{VLLM_ENV_PYTHON} {VLLM_SERVER_SCRIPT} --model {MODEL_PATH} --max-model-len 4096"
    with open(VLLM_LOG_FILE, "w") as log_file:
        process = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    
    start_time = time.time()
    while time.time() - start_time < VLLM_TIMEOUT:
        try:
            with open(VLLM_LOG_FILE, "r") as f:
                content = f.read()
                if "Application startup complete" in content or "Uvicorn running on" in content:
                    log_message("vLLM server is ready.")
                    return process
        except FileNotFoundError:
            pass
        time.sleep(2)

    log_message("Error: vLLM server failed to start within timeout.")
    kill_process_group(process)
    return None


def kill_process_group(process):
    if process and psutil.pid_exists(process.pid):
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, 9)
            log_message(f"Terminated process group (PGID: {pgid}).")
        except ProcessLookupError:
            pass


def run_batch_sender(batch_size, target_input_len, max_output_tokens, model_name):
    log_message(f"Running batch test: Batch={batch_size}, InputLen≈{target_input_len}, MaxOutput={max_output_tokens}...")
    
    cmd = (f"{VLLM_ENV_PYTHON} {BATCH_SENDER_SCRIPT} {batch_size} {PROMPT_FILE} "
           f"--max_tokens {max_output_tokens} --target_input_len {target_input_len} --model {model_name}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout
        log_message(f"Batch test completed.")
        
        total_time_match = re.search(r"Total time for entire batch: ([\d.]+) seconds", output)
        avg_output_match = re.search(r"Average output length: ([\d.]+) tokens", output)
        
        if not total_time_match:
            total_time_match = re.search(r"total_time_match: ([\d.]+) 秒", output)
        if not avg_output_match:
            avg_output_match = re.search(r"avg len: ([\d.]+) tokens", output)
        
        if total_time_match:
            total_time = float(total_time_match.group(1))
            avg_output_len = float(avg_output_match.group(1)) if avg_output_match else max_output_tokens
            return total_time, avg_output_len
        else:
            log_message("Warning: Unable to parse total time from output.")
            return None, None
            
    except subprocess.CalledProcessError as e:
        log_message(f"Error: Failed to run batch sender. Return code: {e.returncode}")
        log_message(f"Stderr: {e.stderr}")
        return None, None


def start_monitor():
    log_message("Starting system monitoring...")
    if os.path.exists(MONITOR_DATA_FILE):
        os.remove(MONITOR_DATA_FILE)
        
    cmd = f"{VLLM_ENV_PYTHON} {MONITOR_SCRIPT} {MONITOR_DATA_FILE}"
    process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(1)
    return process


def analyze_power_consumption():
    if not os.path.exists(MONITOR_DATA_FILE):
        log_message("Warning: Monitoring data file not found.")
        return None
    try:
        df = pd.read_csv(MONITOR_DATA_FILE)
        avg_power = df['gpu_power_watts'].mean()
        log_message(f"Calculated average GPU power: {avg_power:.2f} W")
        return avg_power
    except Exception as e:
        log_message(f"Error: Failed to analyze power data: {e}")
        return None


def main():
    log_message("=== Starting Proposed Method Performance Profiling (DVFS Version) ===")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    cleanup_zombies()
    
    if os.geteuid() != 0:
        log_message("Warning: Script not running with sudo privileges. GPU frequency setting may fail.")

    total_points = len(GPU_FREQUENCIES) * len(BATCH_SIZES) * len(TOKEN_LENGTH_CONFIGS)
    log_message(f"Total test configurations: {len(GPU_FREQUENCIES)} freq × {len(BATCH_SIZES)} batch × {len(TOKEN_LENGTH_CONFIGS)} length = {total_points} points")

    existing_results = set()
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 4:
                        existing_results.add((int(row[0]), int(row[1]), int(float(row[2])), int(float(row[3]))))
            log_message(f"Found existing results file with {len(existing_results)} data points. Will skip tested configurations.")
        except Exception as e:
            log_message(f"Error reading existing results file: {e}. Will start from scratch.")
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
                log_message(f"Skipping frequency {freq} MHz due to setting failure.")
                continue

            for bs in BATCH_SIZES:
                for input_len, output_len, length_name in TOKEN_LENGTH_CONFIGS:
                    if (freq, bs, input_len, output_len) in existing_results:
                        log_message(f"Configuration (Freq={freq}, Batch={bs}, In={input_len}, Out={output_len}) already exists, skipping.")
                        continue

                    completed += 1
                    log_message(f"--- [{completed}/{total_points}] Testing: Freq={freq}MHz, Batch={bs}, Length={length_name} ---")
                    
                    vllm_process = start_vllm_server()
                    if not vllm_process:
                        log_message("vLLM startup failed, skipping current test.")
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
                            
                        log_message(f"Results saved: Freq={freq}MHz, Batch={bs}, In={input_len}, Out={final_output_len:.0f}, "
                                   f"Latency={total_latency:.2f}s, Power={avg_power:.2f}W")
                    else:
                        log_message("Test results discarded due to incomplete data collection.")

                    time.sleep(5)

    finally:
        reset_gpu_frequency()

    log_message("=== All profiling tests completed ===")
    log_message(f"Final results saved to: {RESULTS_FILE}")
    
    if all_results:
        final_df = pd.DataFrame(all_results, columns=[
            "gpu_frequency_mhz", "batch_size", 
            "avg_input_len", "avg_output_len",
            "total_latency_sec", "avg_gpu_power_watts"
        ])
        print("\n--- Final Performance Profile Data ---")
        print(final_df)
        print("\n--------------------------------------")


if __name__ == "__main__":
    main()
