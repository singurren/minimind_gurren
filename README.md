# MiniMind-Research: 轻量级 LLM 的全链路手写复现与消融实验

> **核心定位：** `从零复现` · `消融实验` · `DeepSeek-GRPO` · `MinHash LSH` · `数据工程`

## 📖 项目背景与定位 (Motivation)

本项目是参考 [MiniMind](https://github.com/jingyaogong/minimind) 架构进行的**复现与重构**。

**我为什么要做这个项目？**
在学习大模型过程中，我意识到仅仅“跑通代码”无法触及 LLM 的核心细节。因此，我选择**从零手写（Hand-coded）** 核心模块：
*   **模型端：** 不依赖现成库，手动搭建 Transformer、RoPE 旋转位置编码及 GQA 注意力机制。
*   **训练端：** 独立实现 Pre-train、SFT 及 DPO/GRPO 的训练循环，而非直接调用 Trainer API。
*   **数据端：** 抛弃原有简易脚本，重写了基于 MinHash 的工业级数据清洗流水线。

在此基础上，我进一步开展了针对架构、优化器及对齐算法的**消融实验**，以量化探究工业界主流技术（如 DeepSeek-GRPO、SimPO）的具体收益与资源代价。

---

## 📊 核心贡献与实验结论

### 1. 架构消融：GQA vs MHA (显存优化)

针对推理阶段的 KV Cache 显存瓶颈，对比了分组查询注意力（GQA）与标准多头注意力（MHA）。

*   **测试条件：** Batch Size=8, Seq Len=2048, Hidden Size=512
*   **测试脚本：** `benchmark_arch_ablation.py`

| 模型配置 | Attention 类型 | KV Heads | 峰值显存 (MB) | 显存节省 |
| :--- | :---: | :---: | :---: | :---: |
| Model A | MHA | 8 | 995.03 | - |
| **Model B** | **GQA** | **2** | **598.16** | **📉 39.9%** |

**结论：** 实验证明，在长上下文推理场景下，**GQA 是低显存推理的最优解**。它在维持模型表达能力的同时，显著降低了 KV Cache 显存占用（~40%），允许在同等硬件下支持更长的上下文。

### 2. 对齐算法消融：PPO vs DPO vs SimPO (资源开销)

对比了主流偏好对齐算法在静态显存开销上的差异。

*   **测试脚本：** `benchmark_alignment_ablation.py`

| 算法策略 | 需加载模型组件 | 显存峰值 (MB) | 资源优势 |
| :--- | :--- | :---: | :--- |
| PPO (Baseline) | Policy + Ref + Critic | 1139.62 | 传统基准 |
| **DPO / GRPO** | Policy + Ref | 629.46 | 节省 **44.8%** (移除 Critic) |
| **SimPO** | **Policy Only** | **525.42** | 节省 **53.9%** (极致显存) |

**结论：**
*   **SimPO** 展现了极致的显存效率，适合单卡微调。
*   **GRPO** 通过移除 Value Network，显著降低了 RL 训练门槛，验证了其作为高效对齐方案的可行性。

### 3. 数据工程重构

引入 `data_process_pro.py`，构建了符合工业标准的预处理流水线：

*   **🛠️ MinHash LSH 去重：** 实现模糊去重，解决简单的 MD5 匹配无法识别的近义重复。
*   **🔍 质量过滤：** 增加基于启发式规则的清洗逻辑，从源头提升 Token 有效性。

---

## 🚀 快速复现 (Quick Start)

本项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理。

### 1. 环境准备
安装 `uv` 并同步项目环境：
```bash
uv sync
```

### 2. 运行消融实验测试

```bash
# 1. 架构消融测试 (GQA vs MHA)
uv run benchmark_arch_ablation.py

# 2. 优化器性能测试 (AdamW vs Lion)
uv run benchmark_optimizer_ablation.py

# 3. 对齐算法显存分析 (PPO/DPO/SimPO)
uv run benchmark_alignment_ablation.py
```

### 3. 运行数据处理工具
```bash
uv run data_process_pro.py --input dataset/pretrain_data.jsonl --output dataset/clean_data.jsonl
```

---

## 🤝 致谢

本项目基于 [MiniMind](https://github.com/jingyaogong/minimind) (by @jingyaogong) 进行二次开发。感谢原作者提供了如此优秀的极简 LLM 实现，为通过消融实验深入理解大模型原理提供了绝佳的基础。