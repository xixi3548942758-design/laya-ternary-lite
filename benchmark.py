"""laya-lite 综合基准：精度保真度 + 延迟 + 显存，最后给出评分。

用法：
    python benchmark.py --n 64 --device cuda
    python benchmark.py --n 64 --device cpu
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


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lite", default=os.path.join(HERE, "out"))
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()

    import torch
    import psutil
    from rl_agent_api import RLAgent
    from runtime import LayaLite

    samples = build_calibration_set(args.n, seed=args.seed)
    api_qs = [(st, to_api(qs)) for st, qs in samples]
    n_q = sum(len(qs) for _, qs in api_qs)
    print(f"测试集: {len(api_qs)} 个 state / {n_q} 个问题 (seed={args.seed}, 与校准集不同)")

    proc = psutil.Process()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---------------- 精度：FP16 基线 vs lite ----------------
    t0 = time.time()
    ref = RLAgent(FULL, device=args.device)
    t_fp_load = time.time() - t0
    t0 = time.time()
    lite = LayaLite(args.lite, device=args.device)
    t_lite_load = time.time() - t0
    fp16_vram = torch.cuda.memory_allocated() / 2**20 if args.device == "cuda" else 0.0

    agree = tot = 0
    kls, dprob, dscore = [], [], []
    lat_fp, lat_lite = [], []
    by_type = {"choice": [0, 0], "score": [0, 0], "noul": [0, 0]}
    for state, qs in api_qs:
        t0 = time.time(); a = ref.system_one(state, qs); lat_fp.append(time.time() - t0)
        t0 = time.time(); b = lite.system_one(state, qs); lat_lite.append(time.time() - t0)
        for qid, qa in a["answers"].items():
            qb = b["answers"][qid]
            qt = qa["type"]
            pa, pb = probs_of(qa, qt), probs_of(qb, qt)
            kls.append(kl(pa, pb)); dprob.append(float(np.abs(pa - pb).mean()))
            tot += 1
            by_type[qt][1] += 1
            if qt == "choice":
                hit = int(qa["choice"] == qb["choice"])
            elif qt == "score":
                hit = int(abs(qa["score"] - qb["score"]) < 0.5)
                dscore.append(abs(qa["score"] - qb["score"]))
            else:
                hit = int(abs(qa["noul"] - qb["noul"]) < 0.2)
            agree += hit
            by_type[qt][0] += hit

    # 单独测 lite 的显存：先把 FP16 模型请出去，否则峰值里混着它的 1.7 GB
    if args.device == "cuda":
        del ref
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        st0, q0 = api_qs[0]
        lite.system_one(st0, q0)
        lite_vram_peak = torch.cuda.max_memory_allocated() / 2**20
    else:
        lite_vram_peak = 0.0
    cpu_rss = proc.memory_info().rss / 2**20

    print(f"\n{'='*62}\n  精度保真度（vs FP16 原模型）\n{'='*62}")
    print(f"  总体一致率        {agree/tot:6.1%}   ({agree}/{tot})")
    for k, (h, t) in by_type.items():
        if t:
            print(f"    {k:<8}        {h/t:6.1%}   ({h}/{t})")
    print(f"  平均 KL           {np.mean(kls):6.4f}   (中位 {np.median(kls):.4f}, p95 {pct(kls,95):.4f})")
    print(f"  平均 |Δp|         {np.mean(dprob):6.4f}")
    if dscore:
        print(f"  score 平均误差    {np.mean(dscore):6.3f} 级  (最大 {max(dscore):.2f})")

    print(f"\n{'='*62}\n  延迟（每次 system_one 调用，含全部问题）\n{'='*62}")
    print(f"  FP16   p50 {pct(lat_fp,50)*1000:8.1f} ms   p95 {pct(lat_fp,95)*1000:8.1f} ms")
    print(f"  lite   p50 {pct(lat_lite,50)*1000:8.1f} ms   p95 {pct(lat_lite,95)*1000:8.1f} ms")
    print(f"  慢 {np.mean(lat_lite)/max(np.mean(lat_fp),1e-9):.1f}x")
    print(f"  加载: FP16 {t_fp_load:.1f}s | lite {t_lite_load:.1f}s")

    print(f"\n{'='*62}\n  资源占用\n{'='*62}")
    if args.device == "cuda":
        print(f"  FP16 模型显存     {fp16_vram:8.1f} MiB")
        print(f"  lite 峰值显存     {lite_vram_peak:8.1f} MiB   (已排除 FP16 模型)")
    else:
        print(f"  lite 显存          0.0 MiB  (无卡模式)")
    print(f"  进程 CPU RSS      {cpu_rss:8.1f} MiB")
    tot_bytes = json.load(open(os.path.join(args.lite, "manifest.json")))["sizes"]["total"]
    print(f"  磁盘体积          {tot_bytes/2**20:8.2f} MiB   (原 843 MiB)")

    # ---------------- 评分 ----------------
    print(f"\n{'='*62}\n  评分\n{'='*62}")
    scores = {}
    # 体积：9x 满分，低于 3x 不及格
    ratio = 843 * 2**20 / tot_bytes
    scores["体积压缩"] = min(10, max(0, (ratio - 3) / 6 * 10))
    # 显存：<100MiB 满分，线性衰减到 1GB
    vram = lite_vram_peak if args.device == "cuda" else 0.0
    scores["显存占用"] = 10.0 if vram < 100 else max(0, 10 - (vram - 100) / 900 * 10)
    # 精度：一致率，80% 满分，33%(随机) 零分
    acc = agree / tot
    scores["精度保真"] = min(10, max(0, (acc - 0.33) / (0.80 - 0.33) * 10))
    # 延迟：相对 FP16，1x 满分，20x 零分
    sp = np.mean(lat_lite) / max(np.mean(lat_fp), 1e-9)
    scores["推理速度"] = min(10, max(0, (20 - sp) / 19 * 10))
    # 部署门槛：无卡可用 + 内存低
    scores["部署门槛"] = 10.0 if cpu_rss < 1500 else max(0, 10 - (cpu_rss - 1500) / 2000 * 10)

    for k, v in scores.items():
        bar = "#" * int(round(v)) + "." * (10 - int(round(v)))
        print(f"  {k:<10} {v:5.1f}/10  [{bar}]")
    overall = np.mean(list(scores.values()))
    print(f"\n  综合评分     {overall:5.1f}/10")
    verdict = ("优秀，可直接投产" if overall >= 8 else
               "良好，可投产但需接受精度折损" if overall >= 6.5 else
               "可用，仅建议用于低风险场景" if overall >= 5 else
               "不达标，需继续优化")
    print(f"  结论         {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
