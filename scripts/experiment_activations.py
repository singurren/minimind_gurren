import torch
import torch.nn as nn
import time
from transformers.activations import ACT2FN

# 📘📘📘 实验说明 📘📘📘
# 这个脚本用于对比标准MLP（Non-Gated）和SwiGLU（Gated）结构的参数量和推理速度。
# 我在选择FeedForward层结构时，纠结于使用经典的GELU MLP还是SwiGLU。
# 虽然SwiGLU参数量多了1/3（3个线性层 vs 2个），但文献表明其收敛性能更好。
# 此脚本用于量化二者的性能开销差距，以决定是否值得引入额外的参数量。

class StandardMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        # Standard: Up -> Act -> Down
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))

class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        # SwiGLU: (Gate -> Act) * Up -> Down
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def benchmark(model, x, n_iters=100):
    model.eval()
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    return (end - start) / n_iters

def main():
    hidden_size = 512
    # 为了保持参数量近似可比，StandardMLP的intermediate_size通常更大，
    # 但SwiGLU通常设为 8/3 * hidden_size。
    # 这里我们控制intermediate_size相同，直接看参数量增加的比例。
    intermediate_size = int(hidden_size * 4) 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    std_mlp = StandardMLP(hidden_size, intermediate_size).to(device)
    swiglu_mlp = SwiGLUMLP(hidden_size, intermediate_size).to(device)
    
    print(f"=== MLP 结构对比 (Hidden={hidden_size}, Inter={intermediate_size}) ===")
    
    # 1. 参数量对比
    p_std = count_parameters(std_mlp)
    p_swiglu = count_parameters(swiglu_mlp)
    print(f"Standard MLP 参数量: {p_std}")
    print(f"SwiGLU MLP 参数量:   {p_swiglu} (+{((p_swiglu - p_std)/p_std)*100:.2f}%)")
    
    # 2. 速度对比
    batch_size = 32
    seq_len = 128
    x = torch.randn(batch_size, seq_len, hidden_size).to(device)
    
    t_std = benchmark(std_mlp, x)
    t_swiglu = benchmark(swiglu_mlp, x)
    
    print(f"\n平均推理时间 (batch={batch_size}, seq={seq_len}):")
    print(f"Standard MLP: {t_std*1000:.4f} ms")
    print(f"SwiGLU MLP:   {t_swiglu*1000:.4f} ms (+{((t_swiglu - t_std)/t_std)*100:.2f}%)")
    
    print("\n结论：")
    print("SwiGLU引入了额外的Gate投影层，导致参数量和计算量均增加约50%（在相同intermediate_size下）。")
    print("但在实际应用中，我们通常会调整intermediate_size（例如从4h降到8/3h）来平衡参数量。")
    print("最终决定：采用SwiGLU，因为其带来的PPL收益通常优于单纯增加深度。")

if __name__ == "__main__":
    main()
