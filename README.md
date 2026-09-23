<div align="center">

# laya-ternary-lite

**Laya 决策模型的 1.58-bit 三值量化实现**

把 421M 参数的 [Laya](https://huggingface.co/convaiinnovations/laya) 从 FP16 压到
**1.738 bit/weight**，体积缩小 **9.17x**，推理显存压到 **78.8 MiB**。

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/base-laya%20(Apache%202.0)-green.svg)](https://huggingface.co/convaiinnovations/laya)
[![Bits](https://img.shields.io/badge/weights-1.738%20bit%2Fweight-orange.svg)](#-结果)
[![VRAM](https://img.shields.io/badge/VRAM-78.8%20MiB-success.svg)](#-结果)

</div>

---

## 📊 结果

| 指标 | FP16 原模型 | **laya-ternary-lite** | |
|---|---|---|---|
| 磁盘体积 | 843 MiB | **87.62 MiB** | **9.17x** ↓ |
| 有效位宽 | 16 bit | **1.738 bit/weight** | — |
| 推理显存 | 1609.2 MiB | **78.8 MiB** | **20.4x** ↓ |
| 无卡模式显存 | — | **0 MiB** | CPU 内存净增 126 MiB |
| 加载耗时 | 29.2 s | **0.9 s** | **32x** ↑ |
| 单次推理延迟 (p50) | 1129.2 ms | **1165.0 ms** | 1.0x（持平） |
| **答案一致率** | 100%（自身） | **81.2%** | — |
| 平均 KL 散度 | 0 | **0.1903** | — |

> 测试集：64 个 state / 149 个问题，与量化校准集**不同 seed**，避免过拟合。
> 硬件：GTX 1650 4GB + Windows 11。

**分任务类型的一致率**

| 题型 | 一致率 |
|---|---|
| `score`（有序打分） | **93.6%** (44/47) |
| `choice`（多选分类） | 75.5% (40/53) |
| `noul`（是非判断） | 75.5% (37/49) |

---

## 🚀 快速开始

```bash
pip install torch transformers safetensors numpy psutil
```

```python
import sys
sys.path.insert(0, "path/to/laya-ternary-lite")

from runtime import LayaLite

lite = LayaLite(device="cpu")        # 无卡：显存 0，内存 +126 MiB
# lite = LayaLite(device="cuda")     # 有卡：显存 78.8 MiB

state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts",
                     "other": "everything else"}
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
}

print(lite.system_one(state, questions))
```

`LayaLite.system_one()` 的签名与返回结构与原模型的 `RLAgent.system_one()` **完全一致**，可直接替换。

### 开箱即用

**推理不需要下载原模型。** 仓库自带 `base/`（3.5 MB），内含架构定义、tokenizer 和
`rl_common.py`，`LayaLite` 会自动使用它：

```
laya-ternary-lite/
├── base/           ← 随仓库自带，3.5 MB（架构 + tokenizer + rl_common）
├── out/            ← 三值权重，88 MB
└── *.py
```

```bash
git clone https://github.com/xixi3548942758-design/laya-ternary-lite.git
cd laya-ternary-lite
pip install -r requirements.txt
python -c "
import sys; sys.path.insert(0,'.')
from runtime import LayaLite
lite = LayaLite(device='cpu')
print(lite.system_one({'subject':'Billed twice, please refund.'},
      {'d':{'type':'choice','instructions':'Which department?',
           'criteria':{'billing':'invoices','technical':'bugs'}}}))
"
```

> **只有「重新量化」才需要完整的 `laya-full`**（含 `model.safetensors`），
> 因为要拿原始 FP16 权重当 teacher：
> ```bash
> pip install modelscope
> modelscope download --model convaiinnovations/laya --local_dir ./laya-full
> ```

---

## 🔬 工作原理

### 三值量化基础

权重取值限制在 `{-1, 0, +1}`，每 128 个权重共享一个 FP16 scale：

```
γ    = mean(|W_group|)                     # absmean，BitNet b1.58 的做法
W_q  = clamp(round(W / γ), -1, 1)          # 三值化
Ŵ    = γ · W_q                              # 反量化
```

`W_q · x` 理论上可以把乘法退化成加减法 —— 但见下方[「关于速度的实话」](#-关于速度的实话)。

### 打包格式

| 层 | 内容 |
|---|---|
| `weights.bin` | 80.6 MiB，**5 个 trit 压进 1 个 uint8**（3⁵ = 243 ≤ 256），等效 1.6 bit/weight |
| `scales.bin` | 6.3 MiB，每 128 权重 1 个 FP16 scale |
| `keep.bin` | 0.7 MiB，norm / bias / 决策头 / temperature 保持原精度 |

### 三阶段流水线

```
①  GPTQ 风格逐层校准          ②  STE 自蒸馏微调            ③  打包
   ─────────────────             ────────────────             ────────
   用校准集的激活二阶矩 H=XᵀX      FP16 模型当 teacher，          5 trits/byte
   逐块量化 + 残差补偿             预计算输出后学生模型            + g128 scale
   顺序校准：量化完第 i 层         用 STE 前向三值/反向直通
   再收集第 i+1 层激活             损失 = logits KL + act KL
```

**为什么必须做第 ② 步？** 见下方[「为什么纯 PTQ 救不回来」](#-为什么纯-ptq-救不回来)。

---

## 📐 关键设计取舍

| 决策 | 原因 |
|---|---|
| **只量化 `numel ≥ 16K` 的 2D 权重矩阵** | norm / bias / `temperature` / `act_head` 占比 < 0.1%，动了反而伤概率校准 |
| **`temperature` 与 `act_head` 保持原精度** | 这两个是 RL 校准出来的概率温度，量化会让置信度失真 |
| **`in_dim` 不是 128 倍数时补零列** | `mlp.Wo` 的 2624 不是 128 倍数；运行时把输入同样补零，乘积不变 |
| **打包用 5 trits/byte 而非 2-bit slot** | 1.6 vs 2.13 bit/weight → 87.6 MiB vs 112 MiB，直接决定能否进 100 MiB 预算 |
| **用 meta device 建模型骨架** | 普通 `nn.Linear` 随机初始化要占 1.7 GB，而那些层马上要被量化替换；meta 化后 CPU 内存净增从 1784 MiB 降到 **126 MiB** |
| **打包数据留内存，只把当前 tile 的 packed 字节传 GPU** | 传 packed 比传解包后的 FP16 少 10 倍 PCIe 流量，显存和速度兼得 |

---

## 💡 关于速度的实话

**存储小 ≠ 推理快。** PyTorch 没有三值 GEMM kernel，本项目**仍然是「解包成 FP16 → 普通矩阵乘法」**，并没有吃到「乘法退化成加法」的算力红利。

那为什么还能跑平 FP16？因为省下的是**访存**：权重只有 1/10 大小，搬运成本大幅下降。

早期版本比 FP16 **慢 12.2 倍**（9849 ms），根因是三处实现问题：

| # | 问题 | 修法 | 结果 |
|---|---|---|---|
| 1 | 解包用 5 轮 `%3` / `//3` 整数运算 | **256 项查表** `LUT[packed]` | 占 91% 时间的元凶 |
| 2 | 把解包后的 FP16 传 GPU（2MB/tile × 610 次） | 只传 **packed 字节**，到 GPU 再解包 | PCIe 流量降到 1/10 |
| 3 | 设备判断 `torch.device('cuda') == x.device` 恒为假 | `'cuda' != 'cuda:0'`，改比 `.type` | 之前根本没走 GPU 路径 |

**9849 ms → 747 ms，提速 13.2 倍。**

想要真正的算力加速，需要自写三值 kernel（参考 llama.cpp 的 `TQ1_0` / Prism ML 的 Bonsai fork）。

---

## 🧱 为什么纯 PTQ 救不回来

这是本项目最有价值的一条经验。

**纯 PTQ 会把模型彻底打崩**：一致率恒为 **33.3%**，正好是 3 选项瞎猜的概率。

```
FP16 输出:  [0.1164, 0.3271, 0.5565]   ← 有明确倾向
三值输出:  [0.3318, 0.3331, 0.3351]   ← 塌成均匀分布，argmax = 抛硬币
```

**逐项排除**（都不是 bug）：

| 怀疑对象 | 实测结论 |
|---|---|
| runtime 有 bug | ❌ 用原始 FP16 权重喂进 `LayaLite`，输出与基线**逐位一致**（score 1.4392 vs 1.44） |
| Hadamard rotation | ❌ 只把 corr 从 0.871 抬到 **0.888** |
| GPTQ 误差补偿 | ❌ 本模型 H 近对角（offdiag 仅对角的 2%），补偿量级 **1e-5**，实际等价 absmean |

**根本原因**：1.585 bit/weight 对 421M 参数是**信息论层面的极限**，纯 PTQ 的重建相关性天花板就在 0.89，28 层累积后必然崩。[ParetoQ](https://huggingface.co/papers/2502.02631) 明确指出 **2-bit 以下是 learning transition，表征必须重新学**。

**而救回来的关键，是 batch size：**

| | GTX 1650 4GB | RTX 4090 24GB |
|---|---|---|
| batch | 4（显存所迫） | **32** |
| 优化器 | SGD（AdamW 会 OOM） | **AdamW** |
| 微调 loss | 0.10–0.34 **剧烈抖动** | **0.0008–0.0017 稳定收敛** |
| **答案一致率** | **33.3%**（随机） | **81.2%** |
| 校准 + 微调耗时 | 42 分钟 + 100 分钟跑不完 | **5.2 + 7.8 分钟** |

> batch=4 的梯度噪声压不住硬量化的不连续，loss 一直在跳；batch=32 + AdamW 后 loss 稳定收敛，一致率直接从随机水平升到 81.2%。

---

## 🔁 复现

```bash
# 1. GPTQ 风格逐层校准（约 5 分钟 @ 4090）
python calibrate.py --samples 256 --batch 128 --device cuda

# 2. STE 自蒸馏微调（约 8 分钟 @ 4090）—— 这一步决定成败
python finetune.py --steps 2000 --samples 256 --batch 32 --device cuda \
                   --anneal 400 --optim adamw

# 3. 打包
python pack.py

# 4. 评测
python benchmark.py --n 64 --device cuda
```

**每一步都支持中断续跑**（产物按张量落盘，重跑自动跳过已完成项）。

### 显存/内存受限时

| 场景 | 建议 |
|---|---|
| 只有 4GB 显卡 | 微调可跑但**精度救不回来**，建议租用 ≥16GB 显存的机器做第 2 步 |
| 无卡部署 | `LayaLite(device="cpu")`，显存 0，内存 +126 MiB，单次推理约 10 s |
| 追求极致速度 | `LayaLite(device="cuda", packed_on_gpu=True)` → 610 ms，但显存 164 MiB |
| 追求极致省显存 | `LayaLite(device="cuda", packed_on_gpu=False)`（默认）→ 747 ms，显存 77.5 MiB |

---

## ⚠️ 已知限制

1. **精度不是 100%**。81.2% 的一致率意味着约 1/5 的答案与 FP16 原模型不同，高风险的自动化决策场景需要人工复核或置信度门控。
2. **校准语料是合成的**。`calib_data.py` 用模板构造了 256 条 `(state, questions)`，覆盖邮件 / 工单 / 对话 / JSON / 评论文本 + 三种题型。**没有真实业务数据是这套方案最大的近似** —— 换成真实日志后校准与微调效果应会更好。
3. **速度没有吃到三值红利**。见[「关于速度的实话」](#-关于速度的实话)。
4. **只处理了英文 checkpoint**。仓库里的 `multilingual/`（mmBERT-base）和 `typed-decisions/` 两个 checkpoint 未做量化。
5. **`benchmark.py` 的评分是自定的**。5 个维度各 10 分的加权方式带有主观性，请以自己的业务指标为准。

---

## 📁 项目结构

```
laya-ternary-lite/
├── LICENSE                 Apache-2.0
├── NOTICE                  归属声明 + 修改说明
├── README.md
├── requirements.txt
├── base/                   随仓库自带：架构 + tokenizer + rl_common（3.5 MB）
├── ternary.py              量化/打包核心：g128 absmean + 5 trits/byte
├── runtime.py              推理引擎（LayaLite / TernaryLinear / TernaryEmbedding）
├── calib_data.py           校准语料构造
├── calibrate.py            GPTQ 风格逐层校准
├── finetune.py             STE 自蒸馏微调
├── pack.py                 打包 trits/scales → out/
├── quantize.py             纯 PTQ 量化（基线，已被 calibrate.py 取代）
├── benchmark.py            综合基准：精度 + 延迟 + 显存 + 评分
├── test_fidelity.py        保真度对比
├── docs/
│   ├── RESEARCH.md         三值量化调研（12 篇论文 + 工程实现索引）
│   └── USAGE.md            详细用法与实测数据
└── out/
    ├── weights.bin         80.6 MiB  三值权重
    ├── scales.bin           6.3 MiB  g128 FP16 scale
    ├── keep.bin             0.7 MiB  未量化的小张量
    └── manifest.json       偏移表与元信息
```

---

## 🙏 致谢与许可

本项目是 [**Laya**](https://huggingface.co/convaiinnovations/laya)（Convai Innovations）的量化衍生作品，
基于 **Apache License 2.0** 发布。

| 上游项目 | 作者 | 许可证 |
|---|---|---|
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | Convai Innovations | Apache-2.0 |
| [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) | Answer.AI / LightOn | Apache-2.0 |

```
Based on Laya by Convai Innovations.
Original model and weights are licensed under Apache-2.0.
```

依据 Apache-2.0 第 4 条，本项目：
- ✅ 保留原作者的版权与归属声明（见 [`NOTICE`](NOTICE)）
- ✅ 在 [`NOTICE`](NOTICE) 中明确说明了对原模型的修改内容
- ✅ 衍生权重同样以 Apache-2.0 发布

### 引用

```bibtex
@misc{laya-ternary-lite,
  title  = {laya-ternary-lite: 1.58-bit ternary quantization of the Laya decision model},
  year   = {2026},
  note   = {Derived from convaiinnovations/laya (Apache-2.0)},
  url    = {https://github.com/<your-username>/laya-ternary-lite}
}
```

<div align="center">
<sub>如果这个项目对你有帮助，欢迎点个 ⭐</sub>
</div>
