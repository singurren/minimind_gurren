import torch
import os
import argparse
import time
import numpy as np
import onnxruntime as ort
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def export_onnx(model_path, output_path):
    print(f"🚀 Starting ONNX export...")
    print(f"   Model Path: {model_path}")
    print(f"   Output Path: {output_path}")

    device = "cpu" # Exporting on CPU is usually safer/sufficient for structure
    
    # 1. Load Model
    # 这里的配置应该与训练时保持一致，为演示目的我们使用默认小参数
    config = MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        vocab_size=6400,
        max_position_embeddings=2048
    )
    model = MiniMindForCausalLM(config)
    model.eval()
    
    # 如果有真实权重，应该在这里加载
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path, map_location=device))

    # 2. Define Dummy Input
    # Batch Size = 1, Seq Len = 64
    dummy_input = torch.randint(0, config.vocab_size, (1, 64)).to(device)

    # 3. Export to ONNX
    # 工业界部署通常需要支持动态 Batch 和动态 Sequence Length
    input_names = ["input_ids"]
    output_names = ["logits"]
    
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"}
    }

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14, # 较高的 opset 支持更多算子
        do_constant_folding=True # 优化图结构
    )
    print(f"✅ Model exported to {output_path}")

    # 4. Verify Export
    verify_onnx(model, output_path)

def verify_onnx(torch_model, onnx_path):
    print("\n🔍 Verifying ONNX model correctness...")
    
    # Create a test input different from dummy input
    test_input = torch.randint(0, 6400, (2, 128))
    
    # 1. PyTorch Output
    with torch.no_grad():
        torch_out = torch_model(test_input).logits.numpy()
        
    # 2. ONNX Runtime Output
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    # 3. Compare
    # 允许一定的精度误差 (fp32通常在1e-5级别)
    diff = np.max(np.abs(torch_out - ort_out))
    print(f"   Max Difference: {diff:.2e}")
    
    if diff < 1e-4: # 放宽一点点，考虑到不同后端的浮点差异
        print("✅ Export Verified! The ONNX model matches PyTorch outputs.")
        print("\n💡 Engineer's Note:")
        print("   ONNX (Open Neural Network Exchange) 是通往高性能推理引擎 (如 TensorRT) 的关键桥梁。")
        print("   通过将动态图 (PyTorch) 转换为静态计算图 (ONNX)，我们可以进行：")
        print("   1. 算子融合 (Operator Fusion): 减少 GPU Kernel 启动开销。")
        print("   2. 精度量化 (Quantization): 方便地转换为 FP16/INT8。")
        print("   3. 跨平台部署: 一次导出，到处运行 (Triton Server, Edge Devices)。")
    else:
        print("❌ Verification Failed! Difference is too large.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MiniMind to ONNX")
    parser.add_argument("--model_path", type=str, default=None, help="Path to PyTorch model weights")
    parser.add_argument("--output_path", type=str, default="minimind.onnx", help="Output ONNX file path")
    args = parser.parse_args()

    export_onnx(args.model_path, args.output_path)
