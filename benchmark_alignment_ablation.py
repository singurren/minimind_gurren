import torch
import torch.nn as nn
import pandas as pd
import time
import gc
from model.model_minimind import MiniMindConfig, MiniMindModel

# 屏蔽 transformers 的详细日志
import transformers
transformers.logging.set_verbosity_error()

def measure_peak_memory(name, setup_fn, train_step_fn, config, device):
    print(f"Testing {name}...", end="", flush=True)
    
    # Garbage Collection & Cache Clearing
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # 1. Setup Models
        models = setup_fn(config, device)
        
        # 2. Dummy Input
        batch_size = 2
        seq_len = 512
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        
        # 3. Training Step Simulation
        start_time = time.time()
        train_step_fn(models, input_ids)
        end_time = time.time()
        
        # 4. Measure
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
        
        print(f" Done. Peak: {peak_mem:.2f} MB")
        
        # Cleanup
        del models
        del input_ids
        
        return peak_mem
    except Exception as e:
        print(f" Failed: {e}")
        return 0.0

def benchmark_alignment():
    print("🚀 Starting Real-World Alignment Memory Benchmark")
    print("--------------------------------------------------")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ Warning: Running on CPU. Memory stats will be inaccurate (tracking RAM not VRAM).")

    # 使用较小的配置以确保在笔记本上能跑通对比，重点是相对比例
    config = MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        vocab_size=6400,
        max_position_embeddings=2048
    )
    
    results = []

    # ==========================================
    # 1. SimPO (Simple Preference Optimization)
    # ==========================================
    # 特点：No Reference Model, No Reward/Critic Model. 
    # Loss 包含 margin，直接在 Policy 上计算。
    def setup_simpo(config, device):
        policy_model = MiniMindModel(config).to(device)
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
        return policy_model, optimizer

    def step_simpo(models, input_ids):
        model, optimizer = models
        optimizer.zero_grad()
        # Forward
        outputs, _, _ = model(input_ids)
        logits = outputs  # [B, Seq, Vocab]
        # Fake SimPO Loss (Margin based, self-contained)
        loss = logits.mean() 
        loss.backward()
        optimizer.step()

    mem_simpo = measure_peak_memory("SimPO (Policy Only)", setup_simpo, step_simpo, config, device)
    results.append({"Strategy": "SimPO", "Models Loaded": "Policy", "Peak Memory (MB)": f"{mem_simpo:.2f}"})

    # ==========================================
    # 2. DPO / GRPO (Direct Preference / Group Relative)
    # ==========================================
    # 特点：需要 Reference Model 计算 KL 散度 (Ref 冻结，不占梯度显存，但占权重显存)。
    # GRPO 同样需要 Ref Model 计算 KL，且移除了 Critic。显存结构与 DPO 类似。
    def setup_dpo(config, device):
        policy_model = MiniMindModel(config).to(device)
        ref_model = MiniMindModel(config).to(device)
        ref_model.eval().requires_grad_(False) # Ref model is frozen
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
        return policy_model, ref_model, optimizer

    def step_dpo(models, input_ids):
        policy, ref, optimizer = models
        optimizer.zero_grad()
        
        # Policy Forward
        policy_out, _, _ = policy(input_ids)
        
        # Ref Forward (No Grad)
        with torch.no_grad():
            ref_out, _, _ = ref(input_ids)
            
        # Fake DPO/GRPO Loss (Policy Logprobs - Ref Logprobs)
        # 简单模拟依赖两个模型输出的 Loss
        loss = (policy_out.mean() - ref_out.mean()).abs()
        loss.backward()
        optimizer.step()

    mem_dpo = measure_peak_memory("DPO/GRPO (Policy + Ref)", setup_dpo, step_dpo, config, device)
    results.append({"Strategy": "DPO / GRPO", "Models Loaded": "Policy + Ref", "Peak Memory (MB)": f"{mem_dpo:.2f}"})

    # ==========================================
    # 3. PPO (Proximal Policy Optimization - Legacy)
    # ==========================================
    # 特点：Policy + Reference + Critic (Value Model)。
    # Critic 通常与 Policy 大小相近。这是最重的一套方案。
    def setup_ppo(config, device):
        policy_model = MiniMindModel(config).to(device)
        ref_model = MiniMindModel(config).to(device)
        critic_model = MiniMindModel(config).to(device) # Critic is usually a separate model or head
        
        ref_model.eval().requires_grad_(False)
        # Critic 需要训练
        optimizer = torch.optim.AdamW(list(policy_model.parameters()) + list(critic_model.parameters()), lr=1e-5)
        return policy_model, ref_model, critic_model, optimizer

    def step_ppo(models, input_ids):
        policy, ref, critic, optimizer = models
        optimizer.zero_grad()
        
        # 1. Policy Forward
        policy_out, _, _ = policy(input_ids)
        # 2. Ref Forward
        with torch.no_grad():
            ref_out, _, _ = ref(input_ids)
        # 3. Critic Forward (Value estimation)
        value_out, _, _ = critic(input_ids)
        
        # Fake PPO Loss (Actor Loss + Critic Loss)
        loss = policy_out.mean() + value_out.mean()
        loss.backward()
        optimizer.step()

    mem_ppo = measure_peak_memory("PPO (Policy + Ref + Critic)", setup_ppo, step_ppo, config, device)
    results.append({"Strategy": "PPO (Baseline)", "Models Loaded": "Policy + Ref + Critic", "Peak Memory (MB)": f"{mem_ppo:.2f}"})

    # ==========================================
    # Conclusion
    # ==========================================
    df = pd.DataFrame(results)
    print("\n📊 Real-World Memory Measurement Results:")
    print(df.to_string(index=False))
    
    print("\n📝 Engineer's Analysis:")
    if mem_simpo > 0 and mem_dpo > 0:
        saving_dpo = (mem_dpo - mem_simpo) / mem_dpo * 100
        print(f"1. [SimPO vs DPO]: SimPO 通过移除 Reference Model，显存占用降低了约 {saving_dpo:.1f}%。")
        print("   -> 关键优势：允许在同样硬件下使用更大的 Batch Size 或更长的 Sequence。")
    
    if mem_ppo > 0 and mem_dpo > 0:
        saving_ppo = (mem_ppo - mem_dpo) / mem_ppo * 100
        print(f"2. [GRPO vs PPO]: GRPO (同 DPO 架构) 相比传统的 PPO，去除了 Critic Model，显存节省了约 {saving_ppo:.1f}%。")
        print("   -> 关键优势：不需要训练独立的 Value Network，且利用 Group 采样去除了对 Critic 的依赖。")

if __name__ == "__main__":
    benchmark_alignment()
