import time
import csv
import sys
import os
import psutil
import argparse # 导入 argparse 库来处理命令行参数

try:
    # 优先使用新的库 nvidia-ml-py
    import nvidia_smi as pynvml
except ImportError:
    # 如果没安装新的，回退到旧的 pynvml
    import pynvml

def run_vm_monitoring(output_csv_path, num_cpu_cores, sampling_interval=1):
    """
    (最终增强版) VM 系统监控脚本，能根据传入的核心数动态调整功耗估算模型。
    """
    # <<< 关键修改：动态计算 CPU 功耗模型参数 >>>
    
    # 物理 CPU 的基本信息 (保持不变)
    P_PHYSICAL_IDLE_CPU = 77
    PHYSICAL_CPU_THREADS = 96
    P_PHYSICAL_DYNAMIC_RANGE = 385 - 77 # 308W

    # 根据传入的核心数，动态计算 VM 能贡献的最大动态功耗
    vm_resource_ratio = num_cpu_cores / PHYSICAL_CPU_THREADS
    p_vm_max_dynamic_cpu = P_PHYSICAL_DYNAMIC_RANGE * vm_resource_ratio
    
    print("\n" + "="*40)
    print("      功耗估算模型已动态配置")
    print(f"      - 物理 CPU 总线程数: {PHYSICAL_CPU_THREADS}")
    print(f"      - 本次实验限制核心数: {num_cpu_cores}")
    print(f"      - 资源占比: {vm_resource_ratio:.2%}")
    print(f"      - 进程最大动态功耗贡献: {p_vm_max_dynamic_cpu:.2f} W")
    print(f"      - CPU 功耗估算公式: P = {P_PHYSICAL_IDLE_CPU} + {p_vm_max_dynamic_cpu:.2f} * (Avg CPU Usage % / 100)")
    print("="*40 + "\n")

    # 内存参数 (保持不变)
    P_BASE_RAM = 18
    P_ACTIVE_PER_GB_RAM = 0.34
    # <<< 修改结束 >>>

    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        print("GPU 初始化成功。")
    except Exception as e:
        print(f"警告: 无法初始化 NVML (GPU监控): {e}. 将只监控 CPU/RAM 使用率。")
        gpu_handle = None
    
    psutil.cpu_percent(interval=None) # 初始调用

    print(f"开始 VM 系统监控 (已集成功耗估算)... 数据将保存到 {output_csv_path}")

    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'gpu_utilization_percent', 'gpu_memory_used_mb', 'gpu_power_watts', 
                'gpu_sm_clock_mhz', # <<< 新增列 >>>
                'cpu_utilization_percent', 'ram_used_gb', 'ram_percent',
                'cpu_power_estimate_watts', 'ram_power_estimate_watts'
            ])
            csvfile.flush() # <-- 关键修改：强制刷新缓冲区，确保表头写入硬盘
            
            while True:
                timestamp = time.time()
                
                # 1. 采集原始数据
                gpu_util, gpu_mem_used, gpu_power, gpu_sm_clock = None, None, None, None
                if gpu_handle:
                    try:
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                        gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                        gpu_mem_used = gpu_mem_info.used / (1024**2)
                        gpu_power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
                        gpu_sm_clock = pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_SM) # <<< 新增采集项 >>>
                    except pynvml.NVMLError as err:
                        print(f"无法读取 GPU 数据: {err}")
                
                cpu_util = psutil.cpu_percent(interval=None)
                ram_info = psutil.virtual_memory()
                ram_used_gb = ram_info.used / (1024**3)
                ram_percent = ram_info.percent
                
                # 2. 计算估算值 (现在使用动态计算的参数)
                cpu_power_estimate = P_PHYSICAL_IDLE_CPU + p_vm_max_dynamic_cpu * (cpu_util / 100)
                ram_power_estimate = P_BASE_RAM + P_ACTIVE_PER_GB_RAM * ram_used_gb
                
                # 3. 写入完整数据行
                writer.writerow([
                    timestamp, gpu_util, gpu_mem_used, gpu_power,
                    gpu_sm_clock, # <<< 写入新数据 >>>
                    cpu_util, ram_used_gb, ram_percent,
                    cpu_power_estimate, ram_power_estimate
                ])
                csvfile.flush() # <-- 关键修改：强制刷新缓冲区，确保每行数据都写入硬盘
                
                time.sleep(1) # 固定的采样间隔
    except KeyboardInterrupt:
        print(f"\n监控停止。数据已保存到 {output_csv_path}")
    finally:
        if gpu_handle:
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    # <<< 关键修改：使用 argparse 解析命令行参数 >>>
    parser = argparse.ArgumentParser(description="VM 系统监控脚本，集成功耗估算。")
    parser.add_argument("output_file", help="输出 CSV 文件的路径。")
    parser.add_argument("--cpu-cores", type=int, default=psutil.cpu_count(), 
                        help="本次实验限制的 CPU 核心数。默认为系统的所有核心数。")
    
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    run_vm_monitoring(args.output_file, args.cpu_cores)
    # <<< 修改结束 >>>