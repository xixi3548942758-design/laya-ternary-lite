"""把 calibrate.py 产出的 trits/scales 打包成运行时用的 out/ 目录。

产出：
    weights.bin    所有三值权重，5 trits/byte
    scales.bin     每组 128 个权重共享的 FP16 scale（按行分组，每行 in_pad/128 组）
    keep.bin       未量化的小张量（norm / bias / 决策头 / temperature），原精度
    manifest.json  偏移表与元信息

用法：
    python pack.py                 # 打包 work/ -> out/
    python pack.py --check         # 只报告体积，不写文件
"""
import argparse
import json
import os
import sys

import numpy as np
from safetensors import safe_open

import ternary as T

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "..", "laya-full")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--model", default=os.path.join(FULL, "model.safetensors"))
    ap.add_argument("--group", type=int, default=T.GROUP)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    prog_path = os.path.join(args.work, "progress.json")
    if not os.path.exists(prog_path):
        print("找不到 %s —— 先跑 python calibrate.py" % prog_path)
        return 1
    prog = json.load(open(prog_path))

    os.makedirs(args.out, exist_ok=True)
    man = {"format": "laya-ternary-gptq-v1", "group": args.group,
           "trits_per_byte": T.TRITS_PER_BYTE, "quantized": {}, "kept": {}}

    fw = open(os.path.join(args.out, "weights.bin"), "wb")
    fs = open(os.path.join(args.out, "scales.bin"), "wb")
    fk = open(os.path.join(args.out, "keep.bin"), "wb")
    w_off = s_off = k_off = 0
    n_q = n_qp = 0

    for key in sorted(prog):
        stem = key.replace(".", "_")
        trits = np.load(os.path.join(args.work, stem + ".trits.npy"))
        scales = np.load(os.path.join(args.work, stem + ".scales.npy"))
        out_dim, in_pad = trits.shape
        gpr = scales.shape[1]
        assert gpr == in_pad // args.group, (key, gpr, in_pad)

        pb = T.pack_trits(trits.reshape(-1)).tobytes()
        sb = np.ascontiguousarray(scales).tobytes()
        fw.write(pb)
        fs.write(sb)
        man["quantized"][key] = {
            "shape": [out_dim, prog[key].get("in", in_pad)], "in_pad": int(in_pad),
            "groups_per_row": int(gpr), "out_features": int(out_dim),
            "packed_off": w_off, "packed_len": len(pb),
            "scale_off": s_off, "scale_len": len(sb), "scale_dtype": "F16",
        }
        w_off += len(pb)
        s_off += len(sb)
        n_q += 1
        n_qp += out_dim * in_pad

    # 未量化的张量原样搬运
    with safe_open(args.model, framework="numpy") as f:
        for key in f.keys():
            if key in man["quantized"]:
                continue
            arr = np.ascontiguousarray(f.get_tensor(key))
            b = arr.tobytes()
            fk.write(b)
            man["kept"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                                "off": k_off, "len": len(b)}
            k_off += len(b)

    fw.close(); fs.close(); fk.close()

    total = w_off + s_off + k_off
    man["sizes"] = {"weights_bin": w_off, "scales_bin": s_off, "keep_bin": k_off, "total": total}
    man["stats"] = {"quantized_tensors": n_q, "quantized_params": n_qp,
                    "effective_bits_per_weight": total * 8 / max(n_qp, 1)}
    if args.check:
        for p in ("weights.bin", "scales.bin", "keep.bin"):
            fp = os.path.join(args.out, p)
            if os.path.exists(fp):
                os.remove(fp)
    else:
        with open(os.path.join(args.out, "manifest.json"), "w") as fp:
            json.dump(man, fp, indent=1)

    src = os.path.getsize(args.model)
    print(f"量化张量 {n_q} 个 / {n_qp/1e6:.1f}M 参数")
    print(f"weights.bin {w_off/2**20:7.2f} MiB   scales.bin {s_off/2**20:6.2f} MiB   "
          f"keep.bin {k_off/2**20:6.2f} MiB")
    print(f"总计 {total/2**20:.2f} MiB ({total/1e6:.1f} MB)   压缩 {src/total:.2f}x   "
          f"等效 {total*8/max(n_qp,1):.3f} bit/weight")
    if total > 100 * 1024**2:
        print(f"!! 超过 100 MiB 预算 {total/2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
