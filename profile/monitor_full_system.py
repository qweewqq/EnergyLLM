import time
import csv
import sys
import os
import psutil
import argparse 

try:
    import nvidia_smi as pynvml
except ImportError:
    import pynvml

def run_vm_monitoring(output_csv_path, num_cpu_cores, sampling_interval=1):
    P_PHYSICAL_IDLE_CPU = 77
    PHYSICAL_CPU_THREADS = 96
    P_PHYSICAL_DYNAMIC_RANGE = 385 - 77 

    vm_resource_ratio = num_cpu_cores / PHYSICAL_CPU_THREADS
    p_vm_max_dynamic_cpu = P_PHYSICAL_DYNAMIC_RANGE * vm_resource_ratio
    
    print("\n" + "="*40)
    print("      Power Estimation Model Configured")
    print(f"      - Physical CPU Total Threads: {PHYSICAL_CPU_THREADS}")
    print(f"      - Experiment CPU Cores Limit: {num_cpu_cores}")
    print(f"      - Resource Ratio: {vm_resource_ratio:.2%}")
    print(f"      - Process Max Dynamic Power Contribution: {p_vm_max_dynamic_cpu:.2f} W")
    print(f"      - CPU Power Estimation Formula: P = {P_PHYSICAL_IDLE_CPU} + {p_vm_max_dynamic_cpu:.2f} * (Avg CPU Usage % / 100)")
    print("="*40 + "\n")

    P_BASE_RAM = 18
    P_ACTIVE_PER_GB_RAM = 0.34

    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        print("GPU initialization successful.")
    except Exception as e:
        print(f"Warning: Unable to initialize NVML (GPU monitoring): {e}. Will only monitor CPU/RAM usage.")
        gpu_handle = None
    
    psutil.cpu_percent(interval=None) 

    print(f"Starting VM system monitoring (with integrated power estimation)... Data will be saved to {output_csv_path}")

    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'gpu_utilization_percent', 'gpu_memory_used_mb', 'gpu_power_watts', 
                'gpu_sm_clock_mhz',
                'cpu_utilization_percent', 'ram_used_gb', 'ram_percent',
                'cpu_power_estimate_watts', 'ram_power_estimate_watts'
            ])
            csvfile.flush() 
            
            while True:
                timestamp = time.time()
            
                gpu_util, gpu_mem_used, gpu_power, gpu_sm_clock = None, None, None, None
                if gpu_handle:
                    try:
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                        gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                        gpu_mem_used = gpu_mem_info.used / (1024**2)
                        gpu_power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
                        gpu_sm_clock = pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_SM)
                    except pynvml.NVMLError as err:
                        print(f"Unable to read GPU data: {err}")
                
                cpu_util = psutil.cpu_percent(interval=None)
                ram_info = psutil.virtual_memory()
                ram_used_gb = ram_info.used / (1024**3)
                ram_percent = ram_info.percent
                
                cpu_power_estimate = P_PHYSICAL_IDLE_CPU + p_vm_max_dynamic_cpu * (cpu_util / 100)
                ram_power_estimate = P_BASE_RAM + P_ACTIVE_PER_GB_RAM * ram_used_gb
                
                writer.writerow([
                    timestamp, gpu_util, gpu_mem_used, gpu_power,
                    gpu_sm_clock, 
                    cpu_util, ram_used_gb, ram_percent,
                    cpu_power_estimate, ram_power_estimate
                ])
                csvfile.flush() 
                
                time.sleep(1) 
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped. Data saved to {output_csv_path}")
    finally:
        if gpu_handle:
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="VM system monitoring script with integrated power estimation.")
    parser.add_argument("output_file", help="Path to output CSV file.")
    parser.add_argument("--cpu-cores", type=int, default=psutil.cpu_count(), 
                        help="Number of CPU cores limited for this experiment. Defaults to all system cores.")
    
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    run_vm_monitoring(args.output_file, args.cpu_cores)
