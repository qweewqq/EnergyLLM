# EnergyLLM: Energy-Efficient LLM Serving via RL-based DVFS and Batch Scheduling

EnergyLLM is an energy-efficient LLM serving system with joint optimization of GPU DVFS and batch scheduling.

EnergyLLM formulates the joint control of dynamic voltage and frequency scaling (DVFS) and batch scheduling as a Markov Decision Process (MDP), and learns an energy-aware policy using reinforcement learning to dynamically adapt to fluctuating workloads while satisfying Service Level Objectives (SLOs) at minimal energy cost.

## Motivation

The rapid expansion of Large Language Models (LLMs) has driven energy costs in data centers to unsustainable levels. While current serving systems focus on optimizing throughput and latency, energy efficiency is often treated as a secondary consideration, resulting in significant resource waste.

EnergyLLM aims to serve LLMs efficiently with joint DVFS-batch optimization. The key insight is that GPU frequency and batch size exhibit strong non-linear coupling effects on energy and latency, making independent optimization suboptimal. By jointly optimizing these two dimensions through reinforcement learning, EnergyLLM achieves **38.1% energy reduction** while maintaining **89.94% SLO attainment rate**.

## Installation

#### Prerequisites

EnergyLLM uses [vLLM](https://github.com/vllm-project/vllm) as the default inference engine. Please install vLLM first:

```bash
conda create -n energyllm python=3.8
conda activate energyllm
pip install vllm
```

#### Install EnergyLLM from source

```bash
git clone https://github.com/qweewqq/EnergyLLM.git
cd EnergyLLM
pip install -e .
pip install -r requirements.txt
```

#### Verify GPU DVFS support

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
```

Your GPU should support multiple frequency levels. For NVIDIA A100, the supported range is 210-1410 MHz.

## Getting Started

We get started with a simple example for energy-efficient LLM serving with EnergyLLM.

### Step 1: Profile GPU Performance

First, profile your GPU to collect performance data across different frequency and batch size configurations:

```bash
cd EnergyLLM/energyllm/profile
python profiler.py \
    --model_path /path/to/Meta-Llama-3.1-8B-Instruct \
    --output_dir ../results \
    --frequencies 705,810,960,1050,1110,1155,1200,1260,1305,1350,1380,1395,1410 \
    --batch_sizes 1,2,4,8,16,32,64
```

This will generate a `performance_profile.csv` file containing latency and power measurements for all frequency-batch combinations.

### Step 2: Build Performance Models

Train latency and power prediction models from profiled data:

```bash
cd EnergyLLM/energyllm/model_build
python model_builder.py \
    --profile_data ../results/performance_profile.csv \
    --output_dir ../results
```

This will generate `latency_model.joblib`, `power_model.joblib`, and `model_meta.joblib` files.

### Step 3: Prepare Dataset

Download and process the ShareGPT dataset:

```bash
cd EnergyLLM/datasets/LMSYS-Chat-1M
# Download ShareGPT dataset from HuggingFace
# Process the dataset
python process_lmsys.py
```

Or use the WildChat dataset:

```bash
cd EnergyLLM/datasets/WildChat
python process_wildchat.py
```

### Step 4: Train RL Policy

Train the RL scheduler using PPO:

```bash
cd EnergyLLM/energyllm
python train.py \
    --dataset_path ../datasets/LMSYS-Chat-1M/processed/lmsys_en_mixed_1000.jsonl \
    --latency_model results/latency_model.joblib \
    --power_model results/power_model.joblib \
    --model_meta results/model_meta.joblib \
    --output_dir checkpoints/ \
    --total_timesteps 500000
```

Training takes approximately 2 hours on CPU. The trained policy will be saved to `checkpoints/ppo_final.pt`.

### Step 5: Run Energy-Efficient Serving

Deploy the trained policy for energy-efficient LLM serving:

```bash
python scheduler.py \
    --model_path checkpoints/ppo_final.pt \
    --slo_latency_sec 11.0 \
    --dataset_path ../datasets/LMSYS-Chat-1M/processed/lmsys_en_mixed_1000.jsonl \
    --arrival_rate 0.8 \
    --duration_sec 60
```

The scheduler will automatically adjust GPU frequency and batch size based on real-time workload to minimize energy consumption while satisfying SLO constraints.

## End-to-End Evaluations

### Reproduce Main Results

Run the full evaluation on ShareGPT and WildChat datasets:

```bash
cd EnergyLLM/energyllm/experiments

# Evaluate on ShareGPT (Llama-3.1-8B)
python run_sharegpt_eval.py --runs 10 --duration 60

# Evaluate on WildChat (Llama-3.1-8B)
python run_wildchat_eval.py --runs 10 --duration 60

# Cross-model evaluation on Qwen2.5-7B
python run_sharegpt_eval_qwen25_7b.py --runs 10 --duration 60
python run_wildchat_eval_qwen25_7b.py --runs 10 --duration 60
```

### Run Ablation Studies

Evaluate different variants to understand component contributions:

```bash
# Ablation on ShareGPT
python run_sharegpt_ablation.py --runs 10 --duration 60

# Ablation on WildChat
python run_wildchat_ablation.py --runs 10 --duration 60
```

### Baseline Comparisons

Compare against baseline methods (Static-High, DVFS-Only, Batch-Only, Reactive-DVFS, Token-Aware, DynamoLLM):

```bash
cd test/baselines
python run_batch_experiments.py \
    --dataset sharegpt \
    --runs 10 \
    --duration 60
```

## Results

### Main Results (Llama-3.1-8B-Instruct on ShareGPT)

| Method | Efficiency↑ | SLO(%)↑ | Energy(J)↓ | Latency(s)↓ | P99(s)↓ | Throughput↑ |
|--------|------------|---------|-----------|------------|---------|-------------|
| Static-High | 10.75 | **97.59** | 907.94 | **7.88** | **10.75** | **0.858** |
| DVFS-Only | 12.22 | 78.28 | 643.04 | 9.55 | 13.35 | 0.808 |
| Batch-Only | 11.86 | 77.98 | 660.05 | 9.62 | 14.18 | 0.832 |
| Reactive-DVFS | 10.51 | 62.65 | 598.91 | 10.71 | 14.99 | 0.796 |
| Token-Aware | 7.21 | 64.56 | 903.58 | 10.40 | 15.39 | 0.781 |
| DynamoLLM | 8.50 | 51.30 | 612.52 | 11.43 | 15.60 | 0.782 |
| **EnergyLLM (Ours)** | **16.01** | 89.94 | **561.97** | 8.72 | 11.88 | 0.831 |

*Efficiency = SLO(%) / Energy × 100. Higher is better.*

**Key Findings:**
- **38.1% energy reduction** compared to Static-High baseline (561.97J vs 907.94J)
- **1.49× higher efficiency** compared to best baseline DVFS-Only (16.01 vs 12.22)
- **89.94% SLO attainment** maintained across diverse workloads
- **Strong cross-model generalization** to Qwen2.5-7B without retraining (35.6% energy reduction)

### Ablation Study Results

| Variant | Efficiency↑ | SLO(%)↑ | Energy(J)↓ |
|---------|------------|---------|-----------|
| w/o RL (Heuristic) | 13.13 | 87.01 | 665.53 |
| w/o Energy Reward | 12.08 | **97.05** | 803.69 |
| Fixed Freq (1200MHz) | 14.80 | 95.89 | 649.20 |
| Fixed Batch (16) | 12.00 | 72.46 | 605.16 |
| **EnergyLLM (Ours)** | **16.01** | 89.94 | **561.97** |

The ablation results demonstrate that:
- RL-based control outperforms hand-crafted heuristics by 18.0%
- Energy-aware reward is essential for balancing SLO and energy
- Joint optimization is critical (fixed frequency/batch hurts performance)

## Key Observations

Our work is motivated by three critical observations from profiling experiments:

**Observation 1: Strong coupling between DVFS and batch scheduling.**
At low frequency (≤800MHz), small batches achieve near-baseline performance but large batches suffer severe penalties (1.85× latency, 2.31× energy). At high frequency (>1100MHz), large batches become viable but consume excessive energy (3.09×). This coupling makes single-dimension optimization insufficient.

**Observation 2: Request heterogeneity drives content-aware energy adaptation.**
Short requests (0-100 tokens) consume minimal energy (~13J) and meet SLOs even at lowest GPU frequency. Long requests (500+ tokens) consume an order of magnitude more energy (~245J) and incur severe latency penalties at low frequencies. Optimal strategies must be content-aware.

**Observation 3: DVFS overhead necessitates stability.**
GPU frequency scaling introduces ~50ms hardware reconfiguration latency. For short requests (~15ms processing time), switching after every request reduces effective throughput to just 23% of ideal capacity. Optimal strategies must balance immediate energy savings against throughput loss from switching.

## TODO

- [ ] Add support for multi-GPU distributed serving
- [ ] Integrate with production serving frameworks (e.g., Ray Serve)
- [ ] Add online fine-tuning capabilities
- [ ] Support for more LLM architectures
