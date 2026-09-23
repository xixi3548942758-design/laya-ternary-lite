"""把 laya-full 的 FP16 检查点三值化并打包成 laya-lite。

产出（默认 ./out/）：
    weights.bin    所有三值权重，5 trits/byte 密集打包
    scales.bin     每组 128 权重共享的 FP16 scale
    keep.bin       保持 FP16/F32 的小张量（norm、bias、决策头、temperature）
    manifest.json  偏移表与元信息

用法：
    python quantize.py                     # 全量量化
    python quantize.py --limit 4           # 只量化前 4 层（冒烟测试）
    python quantize.py --group 256         # 换分组大小
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from safetensors import safe_open

import ternary as T

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "..", "laya-full")


# --------------------------------------------------------------------------- 张量分类
def is_quantizable(name: str, shape) -> bool:
    """只量化大的权重矩阵；norm / bias / temperature / 决策头一律保留原精度。"""
    if name.endswith(".bias"):
        return False
    if "norm" in name or name == "temperature":
        return False
    if name.startswith("act_head") or name == "type_emb.weight":
        return False                      # RL 概率校准头 + 类型嵌入，动了会影响置信度
    if len(shape) != 2:
        return False
    return int(np.prod(shape)) >= 1 << 14  # 至少 16K 参数的矩阵才值得三值化


# --------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(FULL, "model.safetensors"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--group", type=int, default=T.GROUP)
    ap.add_argument("--limit", type=int, default=0, help="只量化前 N 层（冒烟测试）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    with safe_open(args.model, framework="numpy") as f:
        names = list(f.keys())
        plan_q, plan_k = [], []
        for n in names:
            sl = f.get_slice(n)
            shape = tuple(sl.get_shape())
            (plan_q if is_quantizable(n, shape) else plan_k).append((n, shape))
        if args.limit:                       # 冒烟测试：只量化前 N 个矩阵，其余按保留处理
            plan_q, plan_k = plan_q[:args.limit], plan_k + plan_q[args.limit:]

        n_q_params = sum(int(np.prod(s)) for _, s in plan_q)
        n_k_params = sum(int(np.prod(s)) for _, s in plan_k)
        print(f"模型: {len(names)} 个张量 | 量化 {len(plan_q)} 个 ({n_q_params/1e6:.1f}M 参数)"
              f" | 保留 {len(plan_k)} 个 ({n_k_params/1e6:.3f}M 参数)")

        est = T.ternary_bytes(n_q_params, args.group) + n_k_params * 2
        print(f"预计产物: {est/2**20:.1f} MiB  (原始 {os.path.getsize(args.model)/2**20:.1f} MiB,"
              f" 压缩 {os.path.getsize(args.model)/est:.1f}x)")

        # ---- 逐张量量化并流式写入（内存里始终只有一个张量）
        fw = open(os.path.join(args.out, "weights.bin"), "wb")
        fs = open(os.path.join(args.out, "scales.bin"), "wb")
        fk = open(os.path.join(args.out, "keep.bin"), "wb")
        man = {"format": "laya-ternary-v1", "group": args.group,
               "trits_per_byte": T.TRITS_PER_BYTE, "source": os.path.basename(args.model),
               "quantized": {}, "kept": {}}
        w_off = s_off = k_off = 0
        layers_done = 0

        for idx, (n, shape) in enumerate(plan_q):
            w = f.get_tensor(n)
            trits, scales = T.quantize_grouped(w, args.group)
            packed = T.pack_trits(trits)
            sb = scales.tobytes()
            pb = packed.tobytes()
            fw.write(pb)
            fs.write(sb)
            man["quantized"][n] = {
                "shape": list(shape), "numel": int(w.size), "n_pad": int(trits.size),
                "packed_off": w_off, "packed_len": len(pb),
                "scale_off": s_off, "scale_len": len(sb),
                "scale_dtype": "F16", "recon_mse": float(((T.dequantize_grouped(trits, scales, args.group, w.size)
                                                           - w.astype(np.float32).reshape(-1)) ** 2).mean()),
            }
            w_off += len(pb)
            s_off += len(sb)
            layers_done += 1
            if layers_done % 32 == 0 or layers_done == len(plan_q):
                print(f"  [{layers_done}/{len(plan_q)}] {n:<52} mse={man['quantized'][n]['recon_mse']:.3e}"
                      f"  {time.time()-t0:.0f}s", flush=True)

        for n, shape in plan_k:
            w = f.get_tensor(n)
            b = np.ascontiguousarray(w).tobytes()
            fk.write(b)
            man["kept"][n] = {"shape": list(shape), "dtype": str(w.dtype), "off": k_off, "len": len(b)}
            k_off += len(b)

        fw.close(); fs.close(); fk.close()

    man["sizes"] = {"weights_bin": w_off, "scales_bin": s_off, "keep_bin": k_off,
                    "total": w_off + s_off + k_off}
    with open(os.path.join(args.out, "manifest.json"), "w") as fp:
        json.dump(man, fp, indent=1)

    tot = man["sizes"]["total"]
    print(f"\n完成: {tot/2**20:.2f} MiB  ({tot/1e6:.1f} MB)  用时 {time.time()-t0:.0f}s")
    print(f"  weights.bin {w_off/2**20:.2f} MiB | scales.bin {s_off/2**20:.2f} MiB | keep.bin {k_off/2**20:.2f} MiB")
    print(f"  实际 bit/weight = {tot*8/n_q_params:.3f}")
    print(f"  压缩比 {os.path.getsize(args.model)/tot:.2f}x  (原始 {os.path.getsize(args.model)/2**20:.1f} MiB)")


if __name__ == "__main__":
    sys.exit(main())
