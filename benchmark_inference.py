import torch
import time
import pandas as pd
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# 屏蔽无关警告
import warnings
warnings.filterwarnings("ignore")

def benchmark_pytorch(config=None, model_path=None, batch_size=8, seq_len=512):
    print(f"正在进行原生 PyTorch 推理基准测试 (BS={batch_size}, Len={seq_len})...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载模型
    if model_path and os.path.exists(model_path):
        print(f"   正在从以下路径加载模型: {model_path}")
        try:
            # 尝试加载预训练模型（如果存在）
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
            # 更新配置以匹配加载的模型
            config = model.config
        except Exception as e:
            print(f"   从 {model_path} 加载模型时出错: {e}")
            print("   将回退到随机初始化。")
            model = MiniMindForCausalLM(config).to(device)
    else:
        print("   使用随机初始化（架构基准测试）。")
        model = MiniMindForCausalLM(config).to(device)
        
    model.eval()
    
    # 2. 准备模拟数据
    # 模拟输入：Batch 个 Prompt
    # 确保词表大小匹配
    vocab_size = getattr(config, 'vocab_size', 6400)
    input_ids = torch.randint(0, vocab_size, (batch_size, 32)).to(device)
    
    # 3. 预热
    print("   正在预热...")
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=10, do_sample=False)
        
    if device == "cuda":
        torch.cuda.synchronize()
    
    # 4. 测量
    print("   正在运行推理...")
    start_time = time.time()
    
    max_new_tokens = seq_len # 生成长度
    with torch.no_grad():
        # 如果未设置，pad_token_id 默认为 eos_token_id
        pad_token_id = getattr(config, 'eos_token_id', 2)
        _ = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            pad_token_id=pad_token_id
        )
        
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    total_tokens = batch_size * max_new_tokens
    throughput = total_tokens / total_time
    latency = total_time * 1000 # 单位: ms
    
    print(f"   完成。耗时: {total_time:.2f}s")
    return throughput, latency

def benchmark_vllm(model_path, batch_size=8, seq_len=512):
    print(f"\n正在进行 vLLM 推理基准测试 (BS={batch_size}, Len={seq_len})...")
    
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("   [错误] 未安装 vLLM。请通过 `pip install vllm` 进行安装。")
        return 0, 0

    if not model_path or not os.path.exists(model_path):
        print(f"   [错误] 模型路径 '{model_path}' 不存在。")
        print("   vLLM 需要磁盘上的有效模型路径（Transformers 格式）。")
        return 0, 0

    print(f"   正在从以下路径加载模型: {model_path}")
    try:
        # 1. 初始化 vLLM 引擎
        # 如果其他进程共享 GPU，将 gpu_memory_utilization 设置得稍低以避免 OOM
        llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1)
        
        # 2. 定义采样参数
        # ignore_eos=True 确保生成正好 max_tokens 个 token 以进行公平基准测试
        sampling_params = SamplingParams(max_tokens=seq_len, temperature=0, ignore_eos=True)
        
        # 3. 准备 Batch Prompt（模拟 Token ID）
        # 输入长度 32，与 PyTorch 基准测试相同
        input_token_ids = [[1] * 32 for _ in range(batch_size)]
        
        # 4. 预热
        print("   正在预热...")
        llm.generate(prompt_token_ids=input_token_ids, sampling_params=sampling_params, use_tqdm=False)
        
        # 5. 测量
        print("   正在运行推理...")
        start_time = time.time()
        
        outputs = llm.generate(prompt_token_ids=input_token_ids, sampling_params=sampling_params, use_tqdm=False)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        # 计算实际生成的 token 数量（虽然 ignore_eos=True 应该使其正好为 seq_len）
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        
        throughput = total_tokens / total_time
        latency = total_time * 1000 # 单位: ms
        
        print(f"   完成。耗时: {total_time:.2f}s")
        return throughput, latency

    except Exception as e:
        print(f"   [错误] vLLM 基准测试运行失败: {e}")
        return 0, 0

def run_benchmark():
    parser = argparse.ArgumentParser(description="MiniMind 推理基准测试")
    parser.add_argument("--framework", type=str, default="pytorch", choices=["pytorch", "vllm", "all"], help="要测试的框架")
    parser.add_argument("--model_path", type=str, default="./MiniMind2", help="模型路径（vLLM 必需，PyTorch 可选）")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch 大小")
    parser.add_argument("--seq_len", type=int, default=128, help="生成序列长度")
    
    args = parser.parse_args()
    
    print("开始推理性能基准测试")
    print("--------------------------------------------------")
    
    # PyTorch 随机初始化的基础配置（回退方案）
    config = MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        vocab_size=6400,
        max_position_embeddings=2048
    )
    
    results = []
    
    # 1. PyTorch 基准测试
    if args.framework in ["pytorch", "all"]:
        pt_throughput, pt_latency = benchmark_pytorch(config, args.model_path, args.batch_size, args.seq_len)
        results.append({"框架": "PyTorch", "吞吐量 (tok/s)": f"{pt_throughput:.2f}", "延迟 (ms)": f"{pt_latency:.2f}"})
    
    # 2. vLLM 基准测试
    if args.framework in ["vllm", "all"]:
        vllm_throughput, vllm_latency = benchmark_vllm(args.model_path, args.batch_size, args.seq_len)
        if vllm_throughput > 0:
            results.append({"框架": "vLLM", "吞吐量 (tok/s)": f"{vllm_throughput:.2f}", "延迟 (ms)": f"{vllm_latency:.2f}"})
    
    # 3. 报告
    if results:
        df = pd.DataFrame(results)
        print("\n推理性能对比:")
        print(df.to_string(index=False))
    else:
        print("\n未生成结果。")

    if args.framework == "pytorch":
        print("\n提示: 使用 '--framework vllm --model_path <路径>' 来测试 vLLM。")

if __name__ == "__main__":
    run_benchmark()
