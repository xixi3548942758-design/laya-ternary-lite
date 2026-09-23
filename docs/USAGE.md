# laya-lite 使用说明

把 `laya-full`（ModernBERT-large，421M，FP16 **843 MiB**）三值化成权重只有 {−1, 0, +1} 的
**87.6 MiB** 版本，等效 **1.74 bit/weight**，压缩 **9.2x**。

显存策略：打包权重常驻 **CPU 内存**，前向时按行分块解包，只把当前这一块送进 device，
算完即弃 —— device 上的常驻占用只有激活值本身。

---

## 目录结构

```
F:\HiTi_code\dev\laya\
├── runclaude.bat
├── laya-full\                     # 原始 FP16 模型（不动）
│   ├── model.safetensors          843 MiB
│   ├── encoder\  tokenizer\  eval\  assets\
│   ├── rl_common.py  rl_agent_api.py  rl_agent_config.json
│   └── multilingual\  typed-decisions\      # 另两个 checkpoint
└── laya-lite\                     # 本目录
    ├── README.md                  # 三值量化调研报告（含论文与工程实现索引）
    ├── USAGE.md                   # 本文件
    ├── ternary.py                 # 量化/打包核心：g128 absmean + 5 trits/byte
    ├── calib_data.py              # 校准语料构造
    ├── quantize.py                # 纯 PTQ 量化（基线，已被 calibrate.py 取代）
    ├── calibrate.py               # GPTQ 风格逐层校准
    ├── finetune.py                # STE 自蒸馏微调
    ├── pack.py                    # 打包 trits/scales -> out/
    ├── runtime.py                 # 低显存推理引擎（LayaLite）
    ├── test_fidelity.py           # 保真度 + 显存对比
    ├── work\                      # 中间产物（trits/scales npy，~411 MiB）
    ├── work_calib_backup\         # 校准结果的备份
    └── out\                       # 最终产物
        ├── weights.bin            80.6 MiB  三值权重，5 trits/byte
        ├── scales.bin              6.3 MiB  g128 的 FP16 scale
        ├── keep.bin                0.7 MiB  norm / bias / 决策头 / temperature
        └── manifest.json          偏移表与元信息
```

---

## 快速开始

```python
import sys
sys.path.insert(0, r"F:\HiTi_code\dev\laya\laya-lite")

from runtime import LayaLite

lite = LayaLite(device="cpu")          # 或 "cuda"

state = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
         "body": "Hi, we were billed twice for March. Please refund the duplicate today."}
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts",
                                "other": "everything else"}},
    "urgency":    {"type": "score", "instructions": "How urgent is this request?",
                   "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}

print(lite.system_one(state, questions))
```

`LayaLite.system_one` 的签名与返回结构与 `laya-full/rl_agent_api.py` 的 `RLAgent.system_one`
完全一致，可以直接替换。

---

## 量化流程

四个阶段，每一步都可中断续跑（产物按张量落盘，重跑自动跳过已完成项）。

```bash
cd F:\HiTi_code\dev\laya\laya-lite

# 1. GPTQ 风格逐层校准（约 42 分钟 @ GTX 1650）
python calibrate.py --samples 256 --batch 32 --device cuda

# 2. STE 自蒸馏微调（让模型适应三值权重）
python finetune.py --steps 2000 --samples 256 --batch 4 --device cuda

# 3. 打包
python pack.py

# 4. 保真度 + 显存对比
python test_fidelity.py --n 64 --device cuda
```

### 各阶段在做什么

**`calibrate.py` —— GPTQ 风格校准**
逐层用校准集的输入二阶矩 `H = XᵀX` 做误差补偿：沿输入维度每 128 个权重一块，
量化完一块后把残差按 `H⁻¹` 传播给尚未量化的列。
顺序校准：量化完第 i 层就写回模型，再收集第 i+1 层的激活。
H 过于病态时自动退回 `H=I`（等价普通 absmean），避免 scale 溢出 FP16。

**`finetune.py` —— STE 自蒸馏**
FP16 模型当 teacher，输出（logits + act logits）**预计算**后缓存，训练时 teacher 不占显存。
学生把每个 Linear 换成三值 STE 层：前向走量化权重，反向按直通估计回传，
权重以 FP32 保留并持续更新。损失 = logits 的 KL + act logits 的 KL。
用 FP32 参数 + 无状态 SGD + warmup/cosine —— FP16 权重配 AdamW 会在几十步内溢出成 NaN。

**`pack.py` —— 打包**
trits 按 5 个一组压进一个 uint8（`3^5 = 243 ≤ 256`，等效 1.6 bit/weight），
scale 按行分组存放，norm/bias 等小张量原精度搬运。

**`runtime.py` —— 推理**
`TernaryLinear.forward` 按输出维度分块解包，tile 默认 512 行
（最大单块 = 512×1024×2 = 1 MiB）。`TernaryEmbedding` 只解包序列里实际出现的 token 行，
不能按 `[min, max]` 整段解包 —— token id 散布在整个词表上，那样等于把 103 MiB 的 FP16 词表整个拉起来。

---

## 关键设计取舍

| 决策 | 原因 |
|---|---|
| 只量化 `numel ≥ 16K` 的 2D 权重矩阵 | norm / bias / `temperature` / `act_head` 占比 < 0.1%，动了反而伤概率校准 |
| `temperature` 与 `act_head` 保持 FP32 | 这两个是 RL 校准出来的概率温度，量化会让置信度失真 |
| `in_dim` 不是 128 倍数时补零列 | `mlp.Wo` 的 2624 不是 128 倍数；运行时把输入同样补零，乘积不变 |
| 权重常驻 CPU、分块解包 | device 上只有激活值，显存峰值由最大单块决定 |
| 打包用 5 trits/byte 而非 2-bit slot | 1.6 vs 2.13 bit/weight，87.6 MiB vs 112 MiB，直接决定能否进 100 MiB 预算 |

---

## 实测数据

### 体积

| 项 | 大小 |
|---|---|
| 原始 FP16 | 843 MiB |
| 三值产物 | **87.6 MiB**（weights 80.6 + scales 6.3 + keep 0.7） |
| 压缩比 | 9.17x |
| 等效位宽 | 1.738 bit/weight |

### 量化误差（权重空间重建）

| 阶段 | relMSE | corr |
|---|---|---|
| absmean（纯 PTQ） | 26–29% | 0.871 |
| + Hadamard rotation | 26% | 0.888 |
| GPTQ 校准 | 26–29% | 0.871（H 近对角，补偿量级仅 1e-5，几乎不生效） |

> 1.58 bit 对 421M 模型是信息论层面的极限：三值权重每个只承载 1.585 bit，
> 纯 PTQ 的 corr 天花板就在 0.89 附近。**ParetoQ 指出 2-bit 以下是「learning transition」，
> 表征必须重新学** —— 这正是 `finetune.py` 存在的理由。

### 保真度与显存（实测）

对比对象：FP16 原模型（`RLAgent`）vs 三值 lite，32 个 state / 67 个问题，同一批输入。

| 指标 | 纯 PTQ | GPTQ 校准（4GB 卡） | **GPTQ + STE 微调（4090）** |
|---|---|---|---|
| 一致率 | 33.3% | 33.3% | **80.6%** |
| 平均 KL | 0.44 | 0.3881 | **0.1714**（中位 0.1298） |
| 平均 \|Δp\| | 0.24 | 0.2449 | **0.1412** |
| score 平均误差 | 0.44 级 | 0.378 级 | **0.211 级** |

**显存与内存**

| 场景 | 占用 |
|---|---|
| `device="cuda"` 推理峰值分配 | **61–65 MiB** ✅ 满足「< 0.1 GB」 |
| `device="cpu"` 显存 | **0 MiB**（无卡部署） |
| `device="cpu"` 加载后 CPU 内存净增 | **126 MiB** |
| CPU 加载 / 推理耗时 | 8.4 s / 10.1 s |

> 用 meta device 建骨架（先不分配随机权重），把 CPU 内存净增从 1784 MiB 压到 126 MiB。

### 精度是怎么救回来的

**纯 PTQ 必然崩**：1.585 bit/weight 对 421M 参数是信息论极限，corr 天花板 0.89，
28 层累积后输出塌成均匀分布，一致率恒等于选项数倒数（3 选项 → 33.3%）。

已排除的原因：
- 打包/解包、模块替换、MHA 接管 —— 用原始 FP16 权重喂进 `LayaLite`，输出与基线**逐位一致**
- Hadamard rotation —— 只把 corr 从 0.871 抬到 0.888
- GPTQ 误差补偿 —— 本模型 H 近对角（offdiag 仅对角 2%），补偿量级 1e-5，实际等价 absmean

**真正的解法是 STE 自蒸馏微调，而且必须用大卡**：

| | GTX 1650 4GB | RTX 4090 24GB |
|---|---|---|
| batch | 4（显存所迫） | **32** |
| 优化器 | SGD（AdamW 会 OOM） | **AdamW** |
| 微调 loss | 0.10–0.34 **剧烈抖动** | **0.0008–0.0017 稳定** |
| 校准耗时 | 42 分钟 | **5.2 分钟** |
| 微调耗时 | 100 分钟跑不完 | **7.8 分钟** |

结论：**梯度噪声是精度的关键**。batch=4 时噪声压不住硬量化的不连续，loss 一直在跳；
4090 上 batch=32 + AdamW 后 loss 稳定收敛到 0.001，一致率随之从随机水平升到 80.6%。

### 复现命令（4090）

```bash
python calibrate.py --samples 256 --batch 128 --device cuda   # 5.2 分钟
python finetune.py --steps 2000 --samples 256 --batch 32 --device cuda --anneal 400 --optim adamw   # 7.8 分钟
python pack.py                                                 # 秒级
python test_fidelity.py --n 32 --device cuda                   # 出上表数字
```

---

## 已知限制

1. **校准语料是合成的**。`calib_data.py` 用模板构造了 256 条 (state, questions)，
   覆盖邮件/工单/对话/JSON/评论文本 + 三种题型。没有真实业务数据是这套方案最大的近似，
   换成真实日志后校准与微调效果应会更好。
2. **精度恢复仍在进行中**。STE 微调把训练集 KL 从 0.22 压到 0.09–0.13，
   但硬量化的 loss 曲面不连续，loss 持续抖动，需要跟踪最优快照而非赌收敛。
3. **打包权重不适合频繁随机访问**。按行分块解包在 CPU 上做，GPU 上会放大 PCIe 传输开销；
   若追求吞吐应改用 llama.cpp 的 TQ1_0/TQ2_0 或自写 kernel（见 README.md 第六节）。
4. **`multilingual/` 与 `typed-decisions/` 两个 checkpoint 未处理**，本方案只覆盖仓库根目录的英文 checkpoint。
