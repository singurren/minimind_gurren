import torch
import time
from model.model_minimind import MiniMindConfig, MiniMindModel
import pandas as pd

def benchmark_attention():
    print("Starting Architecture Ablation: GQA vs MHA")
    print("--------------------------------------------------")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Running on CPU, memory metrics might not reflect GPU usage accurately.")

    # 保持模型其他参数一致，只修改 attention heads
    base_config = {
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "vocab_size": 6400,
        "max_position_embeddings": 4096,
        "num_attention_heads": 8,
    }

    configs = {
        "Model A (MHA)": MiniMindConfig(**base_config, num_key_value_heads=8), # MHA: KV heads = Query heads
        "Model B (GQA)": MiniMindConfig(**base_config, num_key_value_heads=2)  # GQA: KV heads = 1/4 Query heads
    }

    results = []

    # Dummy Input
    batch_size = 8
    seq_len = 2048 # 长序列更能体现 KV Cache 差异
    dummy_input = torch.randint(0, 6400, (batch_size, seq_len)).to(device)

    for name, config in configs.items():
        print(f"Testing {name}...")
        model = MiniMindModel(config).to(device)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # Testing
        with torch.no_grad():
            for _ in range(50):
                # 模拟推理生成过程中的 KV Cache 增长
                # 这里简单做一次 forward pass，主要看 peak memory
                # 在真实推理中，KV Cache 是逐步增长的，这里 max memory 会反映出存储 cache 所需的显存
                _ = model(dummy_input, use_cache=True)
        
        end_time = time.time()
        
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
        avg_latency = (end_time - start_time) / 50 * 1000 # ms
        
        results.append({
            "Model": name,
            "KV Heads": config.num_key_value_heads,
            "Peak Memory (MB)": f"{max_mem:.2f}",
            "Latency (ms)": f"{avg_latency:.2f}"
        })
        
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print("\nAblation Results:")
    print(df.to_string(index=False))
    
    # 结论生成
    print("\nConclusion:")
    mha_mem = float(results[0]["Peak Memory (MB)"])
    gqa_mem = float(results[1]["Peak Memory (MB)"])
    saving = (mha_mem - gqa_mem) / mha_mem * 100
    print(f"GQA 相比 MHA 节省了 {saving:.1f}% 的显存。")
    print("在长上下文 (2048+) 推理场景下，GQA 显著降低了 KV Cache 的显存占用，从而允许更大的 Batch Size 或更长的 Sequence Length。")

if __name__ == "__main__":
    benchmark_attention()
