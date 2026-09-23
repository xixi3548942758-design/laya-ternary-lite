"""保真度对比：三值 lite 模型 vs FP16 原模型。

laya 的输出是「类型化答案 + 校准概率」，所以对齐度比 perplexity 更有意义：
    choice -> 选项是否选得一样、概率分布的 KL
    score  -> 分数差、概率分布的 KL
    noul   -> 概率差
同时报 device 上的峰值显存，用来核对 0.1 GB 预算。

用法：
    python test_fidelity.py                       # CPU，对比 32 个样本
    python test_fidelity.py --device cuda --n 64
    python test_fidelity.py --mem                 # 只测显存
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "..", "laya-full")
sys.path.insert(0, HERE)
sys.path.insert(0, FULL)

from calib_data import build_calibration_set          # noqa: E402


def to_api(qs):
    """内部 question 格式 -> laya/Jev API 格式。"""
    out = {}
    for i, q in enumerate(qs):
        d = {"type": q["t"], "instructions": q["ins"]}
        if q["t"] in ("choice", "score"):
            d["criteria"] = q["crit"]
        out["q%d" % i] = d
    return out


def probs_of(ans, qtype):
    if qtype == "noul":
        return np.array([1.0 - ans["noul"], ans["noul"]])
    return np.array([ans["probabilities"][k] for k in ans["probabilities"]])


def kl(p, q, eps=1e-9):
    p = np.clip(np.asarray(p, float), eps, 1)
    q = np.clip(np.asarray(q, float), eps, 1)
    return float((p * np.log(p / q)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lite", default=os.path.join(HERE, "out"))
    ap.add_argument("--mem", action="store_true")
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    import torch
    from rl_agent_api import RLAgent
    from runtime import LayaLite

    samples = build_calibration_set(args.n, seed=args.seed)
    api_qs = [(st, to_api(qs)) for st, qs in samples]

    if args.mem:
        _mem_report(args, api_qs, torch)
        return 0

    t0 = time.time()
    ref = RLAgent(FULL, device=args.device)
    print(f"[FP16] 加载完成 {time.time()-t0:.1f}s")
    t0 = time.time()
    lite = LayaLite(args.lite, device=args.device)
    print(f"[lite] 加载完成 {time.time()-t0:.1f}s")

    agree = tot = 0
    kls, dprob, dscore = [], [], []
    t_fp = t_lite = 0.0
    for state, qs in api_qs:
        t0 = time.time(); a = ref.system_one(state, qs); t_fp += time.time() - t0
        t0 = time.time(); b = lite.system_one(state, qs); t_lite += time.time() - t0
        for qid, qa in a["answers"].items():
            qb = b["answers"][qid]
            qt = qa["type"]
            pa, pb = probs_of(qa, qt), probs_of(qb, qt)
            kls.append(kl(pa, pb))
            dprob.append(float(np.abs(pa - pb).mean()))
            tot += 1
            if qt == "choice":
                agree += int(qa["choice"] == qb["choice"])
            elif qt == "score":
                dscore.append(abs(qa["score"] - qb["score"]))
                agree += int(abs(qa["score"] - qb["score"]) < 0.5)
            else:
                agree += int(abs(qa["noul"] - qb["noul"]) < 0.2)

    print(f"\n=== 保真度 ({tot} 个问题, {len(api_qs)} 个 state) ===")
    print(f"  一致率          {agree/tot:6.1%}")
    print(f"  平均 KL         {np.mean(kls):6.4f}   (中位 {np.median(kls):.4f})")
    print(f"  平均 |Δp|       {np.mean(dprob):6.4f}")
    if dscore:
        print(f"  score 平均误差  {np.mean(dscore):6.3f} 级")
    print(f"\n  延迟: FP16 {t_fp:.1f}s | lite {t_lite:.1f}s  ({t_lite/max(t_fp,1e-9):.1f}x)")

    if args.device == "cuda":
        print(f"\n  device 峰值显存 {torch.cuda.max_memory_allocated()/2**20:.1f} MiB "
              f"(reserved {torch.cuda.max_memory_reserved()/2**20:.1f} MiB)")
    return 0


def _mem_report(args, api_qs, torch):
    from runtime import LayaLite
    lite = LayaLite(args.lite, device=args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    state, qs = api_qs[0]
    lite.system_one(state, qs)
    import psutil
    proc = psutil.Process()
    print(f"CPU RSS          {proc.memory_info().rss/2**20:.1f} MiB")
    if args.device == "cuda":
        print(f"device 峰值显存  {torch.cuda.max_memory_allocated()/2**20:.1f} MiB")
        print(f"device 保留显存  {torch.cuda.max_memory_reserved()/2**20:.1f} MiB")
        print(f"CUDA context    {torch.cuda.memory_reserved()-torch.cuda.max_memory_allocated()/1:.0f} (含框架开销)")


if __name__ == "__main__":
    sys.exit(main())
