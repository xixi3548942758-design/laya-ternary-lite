"""STE 自蒸馏微调：让模型自己去适应三值权重。

为什么还需要这一步
------------------
纯 PTQ（无论 absmean 还是 GPTQ 补偿）在 1.58 bit 上都停在 relMSE≈26%、corr≈0.89，
28 层累积后输出塌成均匀分布——ParetoQ 指出 2-bit 以下是「learning transition」，
表征必须重新学，量化误差靠算不出来，只能靠训练消化。Hadamard rotation 也只把
corr 从 0.871 抬到 0.888，救不了。

做法
----
* 把 FP16 模型当 teacher，先用校准集把它的输出（logits + act logits）**预计算**存下来，
  于是训练时不需要 teacher 常驻显存。
* 学生模型把每个 Linear 换成三值 STE 层：前向走量化权重，反向按直通估计回传，
  权重本身以 FP16 保留并持续更新。
* 损失 = logits 的 KL + act logits 的 MSE，只更新三值层的权重，norm/bias 冻结。

用法：
    python finetune.py --steps 40 --samples 64          # 冒烟
    python finetune.py --steps 800 --samples 256        # 正式（支持中断续跑）
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ternary as T
from calib_data import build_calibration_set

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "..", "laya-full")
if FULL not in sys.path:
    sys.path.insert(0, FULL)

from rl_common import QTYPES, build_model, build_sequence, collate_items, render_options  # noqa: E402

LAYER_KEYS = ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
EXTRA = [
    ("scorer.1", "scorer.1"),
    ("head.layers.0.self_attn.out_proj", "head.layers.0.self_attn.out_proj"),
    ("head.layers.0.linear1", "head.layers.0.linear1"),
    ("head.layers.0.linear2", "head.layers.0.linear2"),
    ("head.layers.1.self_attn.out_proj", "head.layers.1.self_attn.out_proj"),
    ("head.layers.1.linear1", "head.layers.1.linear1"),
    ("head.layers.1.linear2", "head.layers.1.linear2"),
]


class TernarySTE(nn.Module):
    """前向三值、反向直通。权重以 FP16 保存并持续更新。"""

    def __init__(self, lin: nn.Linear, group: int = 128):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.group = group
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.mix = 0.0          # 0=全软(梯度平滑) 1=全硬(真三值)，训练中渐进推高
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def quantized(self):
        """per-row group absmean 三值化，返回 (wq [out,in_pad], trits int8, scales)。"""
        W = self.weight.float()
        out, in_f = W.shape
        g = self.group
        in_pad = ((in_f + g - 1) // g) * g
        if in_pad != in_f:
            W = F.pad(W, (0, in_pad - in_f))
        wb = W.reshape(out, in_pad // g, g)
        s = wb.abs().mean(-1, keepdim=True).clamp_min(1e-8)
        x = wb / s
        hard = x.round().clamp(-1, 1)
        # 硬量化的 loss 曲面不连续，直接训练会持续抖动；先用线性软化 x.clamp(-1,1)
        # 拿到平滑梯度，再随训练推进逐步推向真三值。导出时 mix=1，落盘的就是硬量化结果。
        q = x.clamp(-1, 1) + (hard - x.clamp(-1, 1)) * self.mix
        wq = (q * s).reshape(out, in_pad)
        return wq, hard.to(torch.int8), s.squeeze(-1)

    def forward(self, x):
        wq, _, _ = self.quantized()
        wq = wq[:, :self.in_features]                                   # 裁掉 group 对齐补的零列
        w = self.weight.float() + (wq - self.weight.float()).detach()   # STE
        return F.linear(x, w.to(x.dtype), self.bias)


def build_ste_model(cfg, dev):
    """保持 FP32 参数：FP16 权重配 AdamW 会在几十步内溢出成 NaN，而 SGD 在 FP16 上
    更新量也吃不住精度。FP32 参数 + 无状态 SGD 是这张 4GB 卡上唯一稳的组合。"""
    from safetensors.torch import load_file
    model = build_model(cfg, encoder_dir=os.path.join(FULL, "encoder"))
    model.load_state_dict(load_file(os.path.join(FULL, "model.safetensors")), strict=True)
    model.to(dev).eval()
    model.encoder.config.reference_compile = False

    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if name.startswith("act_head") or name.endswith("scorer.3"):
            continue                                   # 决策头的最后一层不动
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], TernarySTE(mod))
        n += 1
    return model, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--group", type=int, default=T.GROUP)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--accum", type=int, default=1, help="梯度累积步数，等效放大 batch")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--optim", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--anneal", type=int, default=400, help="软化->硬化的步数")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    ckpt_path = os.path.join(args.work, "ckpt.pt")
    dev = torch.device(args.device)
    t0 = time.time()

    with open(os.path.join(FULL, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(FULL, "tokenizer"))

    samples = build_calibration_set(args.samples)
    items = []
    for state, qs in samples:
        for q in qs:
            seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(q)):
                continue
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]],
                          "target": [0.0] * len(markers), "label": -1,
                          "episode": 0, "ep_step": 0, "ep_len": 1, "src": "ft"})
    items = sorted(items, key=lambda it: len(it["ids"]))
    batches = [collate_items([items[i:i + args.batch]], tok.pad_token_id)
               for i in range(0, len(items), args.batch)]
    print(f"[微调] {len(items)} 序列 / {len(batches)} batch  样本={args.samples} steps={args.steps} lr={args.lr}")

    # ---- teacher 输出预计算（这样训练时不需要 teacher 占显存）
    cache = os.path.join(args.work, "teacher.pt")
    if os.path.exists(cache) and not args.fresh:
        teacher = torch.load(cache, map_location="cpu")
        print(f"[微调] 复用 teacher 缓存 ({len(teacher)} 个 batch)")
    else:
        tm = build_model(cfg, encoder_dir=os.path.join(FULL, "encoder"))
        tm.load_state_dict(__import__("safetensors.torch", fromlist=["load_file"])
                           .load_file(os.path.join(FULL, "model.safetensors")), strict=True)
        tm.to(dev).eval()
        tm.encoder.config.reference_compile = False
        teacher = []
        with torch.no_grad():
            for b in batches:
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    lg, ac = tm(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                b["marker_mask"].to(dev), b["qtype"].to(dev))
                teacher.append((lg.float().cpu(), ac.float().cpu()))
        torch.save(teacher, cache)
        del tm
        torch.cuda.empty_cache()
        print(f"[微调] teacher 预计算完成 {len(teacher)} 个 batch  {time.time()-t0:.0f}s")

    # ---- 学生
    model, n_ste = build_ste_model(cfg, dev)
    params = [p for p in model.parameters() if p.requires_grad]
    for name, mod in model.named_modules():
        if isinstance(mod, TernarySTE):
            mod.weight.requires_grad_(True)
            if mod.bias is not None:
                mod.bias.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    print(f"[微调] 三值层 {n_ste} 个, 可训练参数 {n_par/1e6:.1f}M")

    if args.optim == "adamw":
        # 24GB 卡上直接用 AdamW：SGD 是 4GB 卡被显存逼出来的妥协，梯度噪声压不住硬量化
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    else:
        opt = torch.optim.SGD(params, lr=args.lr)
    # 恒定 lr 会要么震荡（1e-4）要么不动（2e-5），所以配 warmup + cosine：
    # 前期快速压 loss，后期收窄步长稳住。
    def _lr_at(step):
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_at)
    start = 0
    if os.path.exists(ckpt_path) and not args.fresh:
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"[微调] 从第 {start} 步续跑")

    def export_ternary(dest):
        os.makedirs(dest, exist_ok=True)
        for md in model.modules():
            if isinstance(md, TernarySTE):
                md.mix = 1.0
        with torch.no_grad():
            for nm, md in model.named_modules():
                if not isinstance(md, TernarySTE):
                    continue
                _, q, sc = md.quantized()
                st = (nm + ".weight").replace(".", "_")
                np.save(os.path.join(dest, st + ".trits.npy"),
                        q.reshape(md.out_features, -1).cpu().numpy().astype(np.int8))
                np.save(os.path.join(dest, st + ".scales.npy"), sc.cpu().numpy().astype(np.float16))

    # STE 是硬量化，loss 曲面不连续，loss 会持续抖动；与其赌收敛，
    # 不如一路跟踪最优并单独落盘，训练结束后直接用 best 打包。
    best_dir = os.path.join(args.work, "best")
    best_kl = float("inf")
    model.train()
    for step in range(start, args.steps):
        mix = min(1.0, (step + 1) / max(1, args.anneal))
        for md in model.modules():
            if isinstance(md, TernarySTE):
                md.mix = mix
        b = batches[step % len(batches)]
        t_lg, t_ac = teacher[step % len(batches)]
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            s_lg, s_ac = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                               b["marker_mask"].to(dev), b["qtype"].to(dev))
        mask = b["marker_mask"].to(dev)
        m = mask.float()
        t_lg, t_ac = t_lg.to(dev), t_ac.to(dev)
        # marker 位置上做 softmax 后 KL，保证概率分布形状被对齐
        tl = t_lg.masked_fill(~mask, -1e4).float()
        sl = s_lg.masked_fill(~mask, -1e4).float()
        loss_kl = (F.kl_div(F.log_softmax(sl, -1), F.softmax(tl, -1), reduction="none") * m).sum() / m.sum().clamp_min(1)
        # act_logits 是未过 softmax 的 logits，量级上千（teacher 实测 max≈4680），
        # 直接算 MSE 会让损失尺度爆炸、梯度变 NaN，所以同样在 softmax 后取 KL。
        loss_act = F.kl_div(F.log_softmax(s_ac.float(), -1), F.softmax(t_ac.float(), -1), reduction="batchmean")
        loss = loss_kl + loss_act
        (loss / args.accum).backward()
        if (step + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step(step + 1)
            opt.zero_grad(set_to_none=True)

        cur = loss_kl.item()
        if cur < best_kl:
            best_kl = cur
            export_ternary(best_dir)
        if (step + 1) % 40 == 0 or step == start:
            print(f"  step {step+1:>4}/{args.steps}  kl={loss_kl.item():.4f}  act={loss_act.item():.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, ckpt_path)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": args.steps}, ckpt_path)

    # ---- 导出 trits / scales（与 calibrate.py 同格式，供 pack.py 使用）
    model.eval()
    n_out = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, TernarySTE):
                continue
            key = name + ".weight"
            wq, q, s = mod.quantized()
            trits = q.reshape(mod.out_features, -1).cpu().numpy().astype(np.int8)
            scales = s.cpu().numpy().astype(np.float16)
            stem = key.replace(".", "_")
            np.save(os.path.join(args.work, stem + ".trits.npy"), trits)
            np.save(os.path.join(args.work, stem + ".scales.npy"), scales)
            n_out += 1
    # 进度表（pack.py 依赖）。合并而非覆盖——embedding 和 MHA 的 in_proj_weight
    # 不在 STE 训练范围内，它们的 npy 要保留 calibrate.py 的产出。
    prog_path = os.path.join(args.work, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {}
    for name, mod in model.named_modules():
        if isinstance(mod, TernarySTE):
            prog[name + ".weight"] = {"in_pad": int(mod.in_features + (-mod.in_features) % mod.group),
                                      "out": int(mod.out_features), "in": int(mod.in_features)}
    json.dump(prog, open(prog_path, "w"), indent=1)
    print(f"\n[微调] 导出 {n_out} 个三值张量到 {args.work}")
    print("[微调] 下一步: python pack.py --work %s" % args.work)


if __name__ == "__main__":
    sys.exit(main())
