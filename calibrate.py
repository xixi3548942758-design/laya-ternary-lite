"""GPTQ 风格的三值校准量化。

为什么需要它
------------
纯 absmean 三值化（quantize.py）在 421M 模型上重建 corr 只有 0.87、相对误差 27%，
28 层累积后输出直接塌成均匀分布——这正是 ParetoQ 说的「2-bit 以下表征会剧变」。
解法是让量化感知真实激活：逐层用校准集的输入二阶矩 H = XᵀX 做误差补偿。

算法
----
沿输入维度每 group(=128) 个权重一块，逐块三值化；量化完一块后把残差
err = W_b − Ŵ_b 按 H⁻¹ 传播给还没量化的列，使后面的块提前补偿掉前面的误差。
块的粒度对齐到 group，于是每组仍然只存一个 scale，打包体积不变。

顺序校准：量化完第 i 层就把它写回模型，再收集第 i+1 层的激活——
这样后面的层看到的是「前面已经量化过」的真实输入分布，而不是 FP16 的理想分布。

用法：
    python calibrate.py                  # 全量校准（支持中断续跑）
    python calibrate.py --samples 128    # 换校准集大小
    python calibrate.py --layers 1 --samples 16   # 冒烟测试
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import ternary as T
from calib_data import build_calibration_set

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "..", "laya-full")
if FULL not in sys.path:
    sys.path.insert(0, FULL)

from rl_common import QTYPES, build_model, build_sequence, collate_items, render_options  # noqa: E402

LAYER_KEYS = ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
# (state_dict 里的权重 key, 用来收集激活的模块名；None 表示该权重用 H=I，即普通 absmean)
EXTRA = [
    ("encoder.embeddings.tok_embeddings.weight", None),
    ("scorer.1.weight", "scorer.1"),
    ("head.layers.0.self_attn.in_proj_weight", "head.layers.0.self_attn"),
    ("head.layers.0.self_attn.out_proj.weight", "head.layers.0.self_attn.out_proj"),
    ("head.layers.0.linear1.weight", "head.layers.0.linear1"),
    ("head.layers.0.linear2.weight", "head.layers.0.linear2"),
    ("head.layers.1.self_attn.in_proj_weight", "head.layers.1.self_attn"),
    ("head.layers.1.self_attn.out_proj.weight", "head.layers.1.self_attn.out_proj"),
    ("head.layers.1.linear1.weight", "head.layers.1.linear1"),
    ("head.layers.1.linear2.weight", "head.layers.1.linear2"),
]


# --------------------------------------------------------------------------- GPTQ 核心
def gptq_ternary(W: np.ndarray, H: np.ndarray, group: int = 128, damp: float = 0.05):
    """W [out,in] + 激活二阶矩 H [in,in] -> (trits int8 [out,in_pad], scales fp16 [out,in_pad/group])。

    in 不是 group 整数倍时在右侧补零列（运行时把输入也补零，乘积不变）。

    H 在样本不足时会病态（校准序列里某些维度几乎没被激活），此时误差补偿会爆炸、
    scale 直接溢出 FP16。所以先查对角线跨度，太病态就退回 H=I（等价于普通 absmean）。
    """
    out_dim, in_dim = W.shape
    in_pad = ((in_dim + group - 1) // group) * group
    if in_pad != in_dim:
        W = np.concatenate([W, np.zeros((out_dim, in_pad - in_dim), np.float32)], axis=1)

    H = np.asarray(H, dtype=np.float64)
    if in_pad != in_dim:
        Hp = np.zeros((in_pad, in_pad), np.float64)
        Hp[:in_dim, :in_dim] = H
        fill = float(np.diag(H).mean()) * 1e-3 if in_dim else 1.0
        for i in range(in_dim, in_pad):
            Hp[i, i] = fill
        H = Hp

    d = np.diag(H)
    dmin = d.min() if d.size else 0.0
    spread = (d.max() / dmin) if dmin > 0 else np.inf
    ill_conditioned = (not np.isfinite(spread)) or spread > 1e8 or dmin <= 0

    if ill_conditioned:
        Hinv = np.eye(in_pad, dtype=np.float64)          # 无补偿，退化为 absmean
    else:
        Hc = H.copy()
        Hc[np.diag_indices_from(Hc)] += damp * float(d.mean())
        try:
            L = np.linalg.cholesky(Hc)
            Li = np.linalg.inv(L)
            Hinv = Li.T @ Li                              # = Hc⁻¹
        except np.linalg.LinAlgError:
            Hinv = np.eye(in_pad, dtype=np.float64)

    W = W.astype(np.float64).copy()
    gpr = in_pad // group
    trits = np.zeros((out_dim, in_pad), dtype=np.int8)
    scales = np.zeros((out_dim, gpr), dtype=np.float16)

    for gi in range(gpr):
        b0, b1 = gi * group, (gi + 1) * group
        Wb = W[:, b0:b1]
        s = np.abs(Wb).mean(axis=1)
        s = np.where(s > 0, s, 1e-8)
        q = np.clip(np.rint(Wb / s[:, None]), -1, 1)
        trits[:, b0:b1] = q.astype(np.int8)
        scales[:, gi] = s.astype(np.float16)
        if b1 < in_pad:
            upd = (Wb - q * s[:, None]) @ Hinv[b0:b1, b1:]
            W[:, b1:] -= upd
            if not np.isfinite(W[:, b1:]).all():          # 补偿失控就回滚并停手
                W[:, b1:] += upd
                Hinv = np.eye(in_pad, dtype=np.float64)
    return trits, scales


def dequant_rows(trits: np.ndarray, scales: np.ndarray, group: int = 128) -> np.ndarray:
    out_dim, in_pad = trits.shape
    t = trits.astype(np.float32).reshape(out_dim, -1, group)
    return (t * scales.astype(np.float32)[:, :, None]).reshape(out_dim, in_pad)


# --------------------------------------------------------------------------- 激活收集
class HCollector:
    """累积某一层输入的 XᵀX，而不是存下整个 X（几万行 × 1024 存不下）。"""

    def __init__(self, in_dim):
        self.H = np.zeros((in_dim, in_dim), dtype=np.float64)
        self.n = 0

    def __call__(self, module, args):
        x = args[0].detach().float()
        x = x.reshape(-1, x.shape[-1]).cpu().numpy()
        self.H += x.T @ x
        self.n += x.shape[0]
        return None


@torch.no_grad()
def collect_H_multi(model, batches, targets, dims, device):
    """一次前向同时收集同组所有模块的 H，避免「每个张量跑一遍全量前向」的浪费。

    targets: [(weight_key, hook_module_name 或 None)]；None 表示用 H=I（普通 absmean）。
    """
    collectors, handles = {}, []
    for key, mod_name in targets:
        if mod_name is None:
            collectors[key] = None
            continue
        c = HCollector(dims[key])
        collectors[key] = c
        handles.append(model.get_submodule(mod_name).register_forward_pre_hook(c))
    if handles:
        use_amp = device.type == "cuda"
        for b in batches:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                      b["marker_mask"].to(device), b["qtype"].to(device))
        for h in handles:
            h.remove()
    return {k: (np.eye(dims[k], dtype=np.float64) if c is None else c.H) for k, c in collectors.items()}


# --------------------------------------------------------------------------- 校准语料
def encode_calibration(samples, tok, cfg, max_items=4096):
    items = []
    for state, qs in samples:
        for q in qs:
            seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(q)):
                continue
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]],
                          "target": [0.0] * len(markers), "label": -1,
                          "episode": 0, "ep_step": 0, "ep_len": 1, "src": "calib"})
            if len(items) >= max_items:
                return items
    return items


def make_batches(items, pad_id, per_batch=16):
    items = sorted(items, key=lambda it: len(it["ids"]))
    return [collate_items([items[i:i + per_batch]], pad_id) for i in range(0, len(items), per_batch)]


# --------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--group", type=int, default=T.GROUP)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--layers", type=int, default=0, help="只校准前 N 层（冒烟测试）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--damp", type=float, default=0.05)
    ap.add_argument("--fresh", action="store_true", help="丢弃已有进度重跑")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    prog_path = os.path.join(args.work, "progress.json")
    prog = json.load(open(prog_path)) if (os.path.exists(prog_path) and not args.fresh) else {}
    t0 = time.time()
    dev = torch.device(args.device)

    with open(os.path.join(FULL, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(FULL, "tokenizer"))
    from safetensors.torch import load_file
    orig = load_file(os.path.join(FULL, "model.safetensors"))

    model = build_model(cfg, encoder_dir=os.path.join(FULL, "encoder"))
    model.load_state_dict(orig, strict=True)
    model.to(dev).eval()                      # 保持 FP32，前向用 autocast —— 与 RLAgent 行为一致
    model.encoder.config.reference_compile = False
    model_dtype = next(model.parameters()).dtype
    n_layers = len(model.encoder.layers)

    print(f"[校准] device={dev} 样本={args.samples} batch={args.batch} group={args.group} 层数={n_layers}")
    samples = build_calibration_set(args.samples)
    items = encode_calibration(samples, tok, cfg)
    batches = make_batches(items, tok.pad_token_id, args.batch)
    print(f"[校准] 编码 {len(items)} 个序列 -> {len(batches)} 个 batch"
          f" (平均 {np.mean([len(i['ids']) for i in items]):.0f} tokens)")

    n_use = min(n_layers, args.layers) if args.layers else n_layers
    # 按层分组：同层 4 个矩阵的输入激活一次前向就能全部拿到
    groups = [[("encoder.layers.%d.%s.weight" % (i, k), "encoder.layers.%d.%s" % (i, k))
               for k in LAYER_KEYS] for i in range(n_use)]
    extra = [(k, m) for k, m in EXTRA if args.layers == 0 or not k.startswith("head.layers.1")]
    if extra:
        groups.append(extra)          # head / scorer / embedding 都在同一次前向里，合并收集
    n_total = sum(len(g) for g in groups)

    done = skipped = 0
    for group in groups:
        pending = []
        for key, hook_mod in group:
            stem = key.replace(".", "_")
            if os.path.exists(os.path.join(args.work, stem + ".trits.npy")) and \
               os.path.exists(os.path.join(args.work, stem + ".scales.npy")) and not args.fresh:
                skipped += 1
            else:
                pending.append((key, hook_mod))
        if not pending:
            continue

        dims = {k: int(orig[k].shape[1]) for k, _ in pending}
        Hs = collect_H_multi(model, batches, pending, dims, dev)

        for key, hook_mod in pending:
            W = orig[key].float().numpy()
            trits, scales = gptq_ternary(W, Hs[key], args.group, args.damp)
            stem = key.replace(".", "_")
            np.save(os.path.join(args.work, stem + ".trits.npy"), trits)
            np.save(os.path.join(args.work, stem + ".scales.npy"), scales)
            prog[key] = {"in_pad": int(trits.shape[1]), "out": int(trits.shape[0]),
                         "in": int(W.shape[1])}
            json.dump(prog, open(prog_path, "w"), indent=1)

            # 写回模型，让后面的层看到量化后的真实输入分布。
            # 必须裁掉 in_pad 补出来的零列——模型里是标准 nn.Linear，不会 pad 输入。
            q_full = dequant_rows(trits, scales, args.group)
            q = torch.from_numpy(q_full[:, :W.shape[1]]).to(dev, model_dtype)
            if key.endswith(".weight"):
                model.get_submodule(key.rsplit(".", 1)[0]).weight.data = q
            else:                               # head 的 MHA 把 qkv 打包成 in_proj_weight
                model.get_submodule(key.rsplit(".", 1)[0]).in_proj_weight.data = q
            done += 1
            print(f"  [{done+skipped}/{n_total}] {key:<52} in_pad={trits.shape[1]:<5} "
                  f"nz={(trits != 0).mean():.3f}  {time.time()-t0:.0f}s", flush=True)

    print(f"\n[校准] 完成 {done} 个新量化，跳过 {skipped} 个已存在，用时 {time.time()-t0:.0f}s")
    print("[校准] 下一步: python pack.py   （把 work/ 里的 trits+scales 打包成 out/）")


if __name__ == "__main__":
    sys.exit(main())
