# laya-lite 三值量化研究（Ternary / 1.58-bit）

> 目标：把 `laya-full`（ModernBERT-large，421M，FP16 843MB）压缩到三值权重 {-1, 0, +1}，体积降到约 1.7 bit/weight。
> 调研时间：2026-09-23

---

## 一、现状与压缩空间

| 项目 | 实测值 |
|---|---|
| 参数量 | 421.3M（206 个张量，205 个 F16 + 1 个 F32 `temperature`） |
| 当前体积 | 843 MB（FP16） |
| 架构 | ModernBERT-large，28 层，hidden 1024，vocab 50368 |
| 线性层 | `attn.Wqkv` [3072,1024]、`attn.Wo` [1024,1024]、`mlp.Wi` [5248,1024]、`mlp.Wo` [1024,2624] |
| 归一化 | `attn_norm` / `mlp_norm` / `final_norm` / `embeddings.norm`（**RMSNorm，无 bias**） |
| 偏置 | `attention_bias: false`、`mlp_bias: false` —— **全模型无 bias** |
| 决策头 | `act_head` + `temperature`[3]（RL 校准温度，不能量化） |

**两个天然利好**：全模型无 bias + 归一化是 RMSNorm —— 这正好命中 BitNet 系列「bias-free + RMSNorm」的前提条件。

**理论压缩目标**

| 格式 | bit/weight | 体积 | 相对 FP16 |
|---|---|---|---|
| FP16（现状） | 16 | 843 MB | 1.0x |
| Ternary g128（每 128 权重 1 个 FP16 scale） | 1.72 | **≈ 91 MB** | **9.3x** |
| 密集打包 trit（PTQ1_0 风格） | 1.75 | ≈ 92 MB | 9.0x |
| 2-bit slot 打包（PQ2_0 风格） | 2.13 | ≈ 112 MB | 7.5x |

---

## 二、三值量化原理

### 2.1 基础公式（BitNet b1.58，absmean 量化）

```
γ = (1/nm) · Σ|W_ij|                      # 逐张量（或逐组）绝对均值作为 scale
W_q = RoundClip(W / (γ + ε), -1, 1)       # 三值化：{-1, 0, +1}
Ŵ  = γ · W_q                              # 反量化
```

推理时 `W_q ∈ {-1,0,1}`，`y = W_q · x` 退化为**加减法 + 条件取反**，乘法被彻底消除。

### 2.2 工程增强（Bonsai / Prism ML 的做法，实测有效）

- **Ternary g128**：不是整张量一个 scale，而是每 **128 个权重共享 1 个 FP16 scale**，误差显著低于 per-tensor。
- **Blockwise Hadamard rotation**（block 1024，固定 ±1 符号）：量化前把权重矩阵做正交旋转，**把离群值抹平**；旋转折叠进权重存储（不额外占 bit），运行时对激活做同样的变换。这一招是 2-bit 以下精度不崩的关键。
- **打包格式**：`PTQ1_0`（密集 trit，5 trit 存 8 bit，1.75 bpw）或 `PQ2_0`（1 trit 占 2 bit，2.13 bpw，解包更快）。

---

## 三、三条技术路线（按成本从低到高）

### 路线 A：纯 PTQ，零训练 ⭐ 最推荐先试

**代表：CAT-Q / BitTern（Intel）**

- 两个组件：**Learnable Modulation (LM)** 调节权重分布与三值阈值 + **Softened Ternarization (ST)** 用可微过渡函数引导三值化收敛。
- 代价：**仅 512 条校准样本**，8~60 小时（8×A100）可量化 14B~235B 模型；1.7B~8B 模型更快。
- 效果：超过用 **100B token 训练的 BitNet b1.58 v1/v2**，训练 token 消耗降低约 **10 万倍**。
- 代码：`https://github.com/IntelChina-AI/BitTern`（GitHub 当前网络不可达，需代理）

**同路线其他方案**

| 方法 | 核心思路 | 备注 |
|---|---|---|
| **PTQTP**（2509.16989） | 权重矩阵分解为结构化 trit-planes（2×1.58-bit），渐进逼近保证全局一致性 | 数学推理保持率 82.4% vs 竞品 0%；量化仅需约 1 小时 |
| **QTEA**（2609.00224） | GPTQ 式逐列优化 + 显著权重做残差补偿（1:4 半结构化稀疏）+ error decay | 有效 1.7 bpw，Qwen3-14B 上比最强三值 PTQ 基线 +16.7% |
| **Tequila**（2509.23809） | 解决 **deadzone trapping**：把卡在死区边界的权重改造成 dynamic bias 重新激活 | ARC 上比 SOTA +4%；推理开销近乎为零 |
| **ExTernD**（2607.13511） | 扩展秩三值分解 A ≈ B·diag(D)·C，秩超过满秩用于纠错 | 精度可任意逼近 bf16，但 bpw 会升到 5.7（已非 1.58 bit） |
| **ScaleQ-1.58**（2608.01078） | CAT-Q + AYOT 校准（用模型自己的推理轨迹当校准上下文） | 面向推理型 LLM；纯分类任务收益有限 |

### 路线 B：PTQ + 轻量微调（需 GPU 训练）

**代表：An Extra RMSNorm is All You Need for Fine Tuning to 1.58 Bits（2505.08823）**

- 做法极简：**在每个线性投影前插入 RMSNorm**，配 **逐层渐进量化 schedule** + STE（直通估计），把 FP 权重稳定微调成三值模型。
- 优点：不需要复杂的知识蒸馏管线，就能追平/超过蒸馏方案。
- **对本模型的额外优势**：ModernBERT 已经是 RMSNorm 且无 bias，插入位置天然对齐。
- 代价：需要训练。本机 GTX 1650 4GB 显存跑 421M 模型微调**非常吃力**（需梯度检查点 + 小 batch，或租云卡）。

### 路线 C：从头 QAT（最贵，不推荐）

**代表：BitNet b1.58（2402.17764）、Spectra（2407.12327）、TernaryLM（2602.07374）**

- 从头训练三值模型，效果最好，但需要 **100B 级 token + 大规模算力**。
- 对本场景（已有训练好的 421M 检查点）性价比最低。

---

## 四、关键结论：2-bit 是道坎

**ParetoQ（2502.02631）** 的发现最重要：

> 在 2-bit 和 3-bit 之间存在 **learning transition**。3-bit 及以上，微调后模型仍贴近原预训练分布；**2-bit 及以下（含三值），表征会发生剧烈变化**。

这意味着：**纯 PTQ 直接三值化，精度必然掉**，必须靠「少量校准学习」（路线 A）或「微调」（路线 B）把表征重新拉回来。不要期待零成本直接 round 到 {-1,0,1} 就能用。

同时 ParetoQ 也证明：**三值 / 2-bit / 3-bit 在「体积-精度」权衡上表现相当，普遍优于 4-bit 和二值** —— 三值化是值得做的。

---

## 五、针对本模型（ModernBERT 分类器）的建议

1. **先做基准测量**：用 `eval/results.json` 里对应的任务跑一遍 FP16 基线，记录准确率，作为量化后的对照。
2. **只量化线性层权重**：`attn.Wqkv` / `attn.Wo` / `mlp.Wi` / `mlp.Wo` 四类共 112 个矩阵。
3. **不要量化**：
   - `temperature`[3] 与 `act_head`（RL 校准温度，动了概率校准就废了）
   - 所有 `*_norm.weight`（RMSNorm 缩放系数，只有 1024 维，占比极小但影响巨大）
   - `tok_embeddings.weight` 建议先保留 FP16（50368×1024 占 103MB，是单张量最大头；想省内存可用 g128 三值化，但精度风险最高，放最后再试）
4. **打包策略**：优先 **g128 分组 scale**，再考虑加 Hadamard rotation 抹平离群值。
5. **预期收益**：约 **843 MB → 91 MB**，压缩 **9.3x**。

---

## 六、必须知道的两个坑

1. **存储小 ≠ 推理快**。PyTorch 原生没有三值 GEMM kernel，反量化回 FP16 再算的话**速度反而更慢**。要真正加速，需要：
   - llama.cpp 生态的 `TQ1_0`/`TQ2_0` 三值 kernel（仅支持 GGUF/LLM 结构），或
   - Bonsai 那套 **定制 llama.cpp fork**（`PrismML-Eng/llama.cpp`，带 CUDA/Metal 三值混合注意力 kernel），或
   - 自写 lookup-table kernel（QTEA 报告可提速 7.2x）。
   - 若目标只是**减小体积**，则打包存储即可，无需 kernel。
2. **校准集必须来自真实分布**。laya 是「状态 + 类型化问题」的决策模型，校准样本应覆盖实际业务里的 state 文本（邮件/工单/JSON），否则量化误差会集中在长尾输入上。

---

## 七、参考来源

**论文（HuggingFace Papers）**
- [The Era of 1-bit LLMs: All LLMs are in 1.58 Bits (BitNet b1.58)](https://huggingface.co/papers/2402.17764) — 2402.17764
- [CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs](https://huggingface.co/papers/2606.26650) — 2606.26650 ⭐ 纯 PTQ，512 样本
- [An Extra RMSNorm is All You Need for Fine Tuning to 1.58 Bits](https://huggingface.co/papers/2505.08823) — 2505.08823 ⭐ 轻量微调
- [Tequila: Trapping-free Ternary Quantization](https://huggingface.co/papers/2509.23809) — 2509.23809
- [PTQTP: Post-Training Quantization to Trit-Planes](https://huggingface.co/papers/2509.16989) — 2509.16989
- [QTEA: Ternary LLMs with Sparse Residual Salient Weight](https://huggingface.co/papers/2609.00224) — 2609.00224
- [ExTernD: Expanded-Rank Ternary Decomposition](https://huggingface.co/papers/2607.13511) — 2607.13511
- [Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization](https://huggingface.co/papers/2601.07892) — 2601.07892
- [ScaleQ-1.58](https://huggingface.co/papers/2608.01078) — 2608.01078
- [ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization](https://huggingface.co/papers/2502.02631) — 2502.02631 ⭐ 2-bit 是道坎
- [Spectra: Ternary, Quantized, and FP16 Language Models](https://huggingface.co/papers/2407.12327) — 2407.12327
- [TernaryLM](https://huggingface.co/papers/2602.07374) — 2602.07374

**工程实现**
- [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) — Ternary g128 + Hadamard rotation，PTQ1_0 / PQ2_0 打包，27B→5.95GB（9.0x），保留 98.2% 智能
- [PrismML-Eng/Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) — 跑起来的唯一可信来源（含 whitepaper）
- [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) — 三值 kernel fork（CUDA/Metal）
- [IntelChina-AI/BitTern](https://github.com/IntelChina-AI/BitTern) — CAT-Q 官方代码
- [Tencent/AngelSlim](https://github.com/Tencent/AngelSlim) — Sherry 1.25-bit 代码

**本机环境**
- GPU：GTX 1650 4GB（PTQ 校准可跑；路线 B 微调需梯度检查点或云卡）
- torch 2.5.1+cu121 / transformers 5.14.1
- 网络：GitHub 全线不可达，HuggingFace 需走本地代理 `127.0.0.1:7890`
