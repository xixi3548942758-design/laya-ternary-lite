"""laya-lite 推理引擎：从打包的三值权重直接前向，显存占用压到最低。

显存策略
--------
打包权重（~87 MiB）常驻 **CPU 内存**，前向时按输出维度分块解包，只把当前这一块
的 FP16 权重送进 device，算完即弃。于是 device 上的常驻占用只有激活值本身，
峰值由最大单块权重决定（tile 行数 × in_features × 2 字节）。

    tile=512 时，最大单块 = 512×1024×2 = 1 MiB。

接口与 laya-full/rl_agent_api.py 的 RLAgent.system_one 保持一致。
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "..", "laya-full")
if FULL not in sys.path:
    sys.path.insert(0, FULL)

from rl_common import (QTYPES, build_model, build_sequence, collate_items,      # noqa: E402
                       confidence_from_probs, render_options, temp_bucket)

TRITS_PER_BYTE = 5
# 查表法：一个 uint8 的 256 种取值 → 5 个 trit 全部预先算好，
# 比 5 轮 "%3 / //3" 整数运算快一个量级，且能直接在 GPU 上跑。
_LUT_NP = np.array([[(i // (3 ** k)) % 3 - 1 for k in range(TRITS_PER_BYTE)]
                    for i in range(256)], dtype=np.int8)
_LUT_T = None
_NP_DTYPE = {"F16": "float16", "F32": "float32", "F64": "float64",
             "I64": "int64", "I32": "int32", "U8": "uint8", "I8": "int8"}


def np_dtype(name: str) -> np.dtype:
    """manifest 里可能是 safetensors 的 'F16' 或 numpy 的 'float16'，两种都认。"""
    return np.dtype(_NP_DTYPE.get(name, name))


# --------------------------------------------------------------------------- 解包
def unpack_np(packed: np.ndarray) -> np.ndarray:
    """uint8 memmap -> int8 {-1,0,1}，长度 = 5 × len(packed)。

    用 256 项查表一次搞定：早先的 5 轮 "%3 / //3" 整数运算是整条链路上最慢的一环
    （实测占单次推理 91% 的时间），查表快一个量级。
    """
    return _LUT_NP[np.asarray(packed)].reshape(-1)


def _lut_torch():
    global _LUT_T
    if _LUT_T is None:
        _LUT_T = torch.from_numpy(_LUT_NP)
    return _LUT_T


def unpack_torch(packed):
    """uint8 tensor -> int8 {-1,0,1} tensor，长度 = 5 x len(packed)，全程在 packed 所在设备上完成。"""
    lut = _lut_torch().to(packed.device)
    return lut[packed.long()].reshape(-1)


class TernaryStore:
    """打包权重的只读视图；解包结果按行返回 FP16 numpy 数组。"""

    def __init__(self, lite_dir):
        with open(os.path.join(lite_dir, "manifest.json")) as f:
            self.man = json.load(f)
        self.group = self.man["group"]
        self.weights = np.memmap(os.path.join(lite_dir, "weights.bin"), dtype=np.uint8, mode="r")
        self.scales = np.memmap(os.path.join(lite_dir, "scales.bin"), dtype=np.float16, mode="r")
        self.keep = np.memmap(os.path.join(lite_dir, "keep.bin"), dtype=np.uint8, mode="r")
        self.quantized = self.man["quantized"]
        self.kept = self.man["kept"]
        self.scale_itemsize = np.dtype(np_dtype(next(iter(self.quantized.values()))["scale_dtype"])).itemsize

    # -- 三值权重 ----------------------------------------------------------
    def rows(self, name: str, r0: int, r1: int) -> np.ndarray:
        """解包张量 name 的第 r0..r1 行，返回 [r1-r0, in_pad] 的 FP16 数组。

        分组按行对齐：行 r 占用 group 区间 [r*gpr, (r+1)*gpr)，gpr = in_pad/group。
        """
        info = self.quantized[name]
        d = info["in_pad"]
        a, b = r0 * d, r1 * d
        g = self.group
        g0, g1 = a // g, (b + g - 1) // g
        t0, t1 = g0 * g, g1 * g
        p0, p1 = t0 // TRITS_PER_BYTE, (t1 + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
        packed = self.weights[info["packed_off"] + p0: info["packed_off"] + p1]
        trits = unpack_np(packed)
        off = t0 - p0 * TRITS_PER_BYTE
        trits = trits[off: off + (t1 - t0)].reshape(-1, g).astype(np.float32)
        # scale_off 在 manifest 里是字节偏移，而 scales memmap 的索引单位是 FP16 元素
        s0 = info["scale_off"] // self.scale_itemsize
        sc = self.scales[s0 + g0: s0 + g1].astype(np.float32)
        w = (trits * sc[:, None]).reshape(-1)[: b - a]
        return w.reshape(r1 - r0, d).astype(np.float16)

    def full(self, name: str) -> np.ndarray:
        info = self.quantized[name]
        return self.rows(name, 0, info["out_features"])

    def to(self, device):
        """把打包权重整块搬上 device（约 84 MiB）。之后解包在 device 上做，
        省掉「CPU 解包 -> 逐块传 GPU」这条最慢的路径。"""
        self._dev = torch.device(device)
        self._w = torch.from_numpy(np.array(self.weights)).to(self._dev)     # 拷贝成可写
        self._s = torch.from_numpy(np.array(self.scales)).to(self._dev)
        return self

    def rows_torch(self, name, r0, r1, dtype=torch.float16):
        """在 device 上解包第 r0..r1 行，返回 [r1-r0, in_pad] 的 tensor。"""
        info = self.quantized[name]
        d = info["in_pad"]
        a, b = r0 * d, r1 * d
        g = self.group
        g0, g1 = a // g, (b + g - 1) // g
        t0, t1 = g0 * g, g1 * g
        p0, p1 = t0 // TRITS_PER_BYTE, (t1 + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
        packed = self._w[info["packed_off"] + p0: info["packed_off"] + p1]
        trits = unpack_torch(packed)
        off = t0 - p0 * TRITS_PER_BYTE
        trits = trits[off: off + (t1 - t0)].view(-1, g).to(dtype)
        s0 = info["scale_off"] // self.scale_itemsize
        sc = self._s[s0 + g0: s0 + g1].to(dtype)
        return (trits * sc[:, None]).reshape(-1)[: b - a].view(r1 - r0, d)

    def rows_hybrid(self, name, r0, r1, device, dtype=torch.float16):
        """折中方案：打包数据留在内存，只把这一小块 packed 字节传上 device 再解包。

        传的是 packed（1/5 大小）而不是解包后的 FP16，PCIe 流量降到 1/10，
        device 上也只留 tile 级临时量，显存和速度可以兼得。
        """
        info = self.quantized[name]
        d = info["in_pad"]
        a, b = r0 * d, r1 * d
        g = self.group
        g0, g1 = a // g, (b + g - 1) // g
        t0, t1 = g0 * g, g1 * g
        p0, p1 = t0 // TRITS_PER_BYTE, (t1 + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
        pb = np.ascontiguousarray(self.weights[info["packed_off"] + p0: info["packed_off"] + p1])
        packed = torch.from_numpy(pb).to(device)
        trits = unpack_torch(packed)
        off = t0 - p0 * TRITS_PER_BYTE
        trits = trits[off: off + (t1 - t0)].view(-1, g).to(dtype)
        s0 = info["scale_off"] // self.scale_itemsize
        sc = torch.from_numpy(np.ascontiguousarray(
            self.scales[s0 + g0: s0 + g1])).to(device, dtype)
        return (trits * sc[:, None]).reshape(-1)[: b - a].view(r1 - r0, d)

    def keep_tensor(self, name: str) -> np.ndarray:
        info = self.kept[name]
        buf = self.keep[info["off"]: info["off"] + info["len"]]
        return np.frombuffer(buf.tobytes(), dtype=np_dtype(info["dtype"])).reshape(info["shape"]).copy()


# --------------------------------------------------------------------------- 模块
class TernaryLinear(nn.Module):
    """权重常驻 CPU 打包格式；forward 时按行分块解包，用一块丢一块。"""

    def __init__(self, store, name, bias=None, tile=512):
        super().__init__()
        self.store = store
        self.name = name
        info = store.quantized[name]
        self.out_features = info["out_features"]
        self.in_features = info["shape"][1]
        self.in_pad = info["in_pad"]
        self.tile = tile
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        dtype = x.dtype
        if self.in_pad != self.in_features:      # 量化时补的零列，输入同样补零
            x = F.pad(x, (0, self.in_pad - self.in_features))
        _d = getattr(self.store, "_dev", None)
        on_dev = _d is not None and _d.type == x.device.type
        outs = []
        for r0 in range(0, self.out_features, self.tile):
            r1 = min(r0 + self.tile, self.out_features)
            if on_dev:
                w = self.store.rows_torch(self.name, r0, r1, dtype)
            elif x.device.type == "cuda":
                w = self.store.rows_hybrid(self.name, r0, r1, x.device, dtype)
            else:
                w = torch.from_numpy(self.store.rows(self.name, r0, r1)).to(x.device, dtype)
            outs.append(F.linear(x, w))
        y = torch.cat(outs, -1) if len(outs) > 1 else outs[0]
        if self.bias is not None:
            y = y + self.bias.to(dtype)
        return y


class TernaryEmbedding(nn.Module):
    """只解包实际用到的行——序列里几百个 token，远小于 50368 的完整词表。

    注意不能按 [min(row), max(row)] 整段解包：token id 散布在整个词表上，
    那样等于把 103 MiB 的 FP16 词表整个拉起来，正好毁掉省显存的目的。
    """

    def __init__(self, store, name):
        super().__init__()
        self.store = store
        self.name = name
        info = store.quantized[name]
        self.num_embeddings = info["out_features"]
        self.embedding_dim = info["shape"][1]
        self.in_pad = info["in_pad"]
        self.out_dtype = torch.float32          # 由 LayaLite 按模型实际 dtype 覆盖

    def forward(self, idx):
        flat = idx.reshape(-1).to(idx.device)
        rows = torch.unique(flat)
        _d = getattr(self.store, "_dev", None)
        if _d is not None and _d.type == idx.device.type:
            # 只解包用到的那些行：按行号 gather 出对应的 packed 区间再一次性解包
            lo, hi = int(rows.min()), int(rows.max()) + 1
            if (hi - lo) * self.in_pad <= 64 * self.in_pad:      # 行号集中 -> 整段解包更划算
                w = self.store.rows_torch(self.name, lo, hi, self.out_dtype)
                lut = w[rows - lo]
            else:
                lut = torch.stack([self.store.rows_torch(self.name, int(r), int(r) + 1, self.out_dtype)[0]
                                   for r in rows])
            pos = torch.searchsorted(rows, flat)
            return lut[pos][:, :self.embedding_dim].view(*idx.shape, self.embedding_dim)
        flat_cpu = flat.cpu()
        rows_cpu = torch.unique(flat_cpu)
        lut = np.empty((rows_cpu.numel(), self.in_pad), dtype=np.float16)
        for i, r in enumerate(rows_cpu.tolist()):
            lut[i] = self.store.rows(self.name, r, r + 1)[0]
        lut = torch.from_numpy(np.ascontiguousarray(lut[:, :self.embedding_dim]))
        pos = torch.searchsorted(rows_cpu, flat_cpu)
        return lut[pos].to(idx.device, self.out_dtype).view(*idx.shape, self.embedding_dim)


class _MHAUnpack:
    """nn.MultiheadAttention 把 qkv 打包成一个 in_proj_weight 参数，而且内部是
    F.linear(x, self.out_proj.weight) 直接取权重、不走模块 forward，所以这两个都得手动接管。"""

    def __init__(self, store, in_key, out_key, tile=1024):
        self.store, self.in_key, self.out_key = store, in_key, out_key

    def __call__(self, module, args):
        ref = module.in_proj_bias if module.in_proj_bias is not None else module.out_proj.bias
        dev, dt = ref.device, ref.dtype
        i_in = self.store.quantized[self.in_key]["shape"][1]
        o_in = self.store.quantized[self.out_key]["shape"][1]
        module.in_proj_weight = nn.Parameter(
            torch.from_numpy(np.ascontiguousarray(self.store.full(self.in_key)[:, :i_in])).to(dev, dt),
            requires_grad=False)
        module.out_proj.weight = nn.Parameter(
            torch.from_numpy(np.ascontiguousarray(self.store.full(self.out_key)[:, :o_in])).to(dev, dt),
            requires_grad=False)
        return None


# --------------------------------------------------------------------------- 引擎
class LayaLite:
    def __init__(self, lite_dir=None, full_dir=FULL, device="cpu", tile=512, verbose=False,
                 packed_on_gpu=None):
        lite_dir = lite_dir or os.path.join(HERE, "out")
        self.store = TernaryStore(lite_dir)
        self.device = torch.device(device)
        self.tile = tile

        with open(os.path.join(full_dir, "rl_agent_config.json")) as f:
            self.cfg = json.load(f)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(os.path.join(full_dir, "tokenizer"))

        # 在 meta device 上建骨架：普通 nn.Linear 的随机初始化要占 1.7 GB，
        # 而我们要量化掉的正是这些层，先分配再释放纯属浪费（无卡部署时更是致命）。
        # meta tensor 不占存储，等替换完再靠 load_state_dict(assign=True) 灌真值。
        with torch.device("meta"):
            model = build_model(self.cfg, encoder_dir=os.path.join(full_dir, "encoder"))
        self.model = self._install(model)
        self._load_keep()
        # packed_on_gpu=True  : 84 MiB 打包数据常驻显存 -> 610 ms 但显存 164 MiB，超 0.1 GB 预算
        # packed_on_gpu=False : 打包数据留内存，只把当前 tile 的 packed 字节传上 GPU 再解包
        #                       -> 747 ms、显存 77 MiB，两个指标同时达标，故设为默认
        if packed_on_gpu is None:
            packed_on_gpu = False
        if self.device.type == "cuda" and packed_on_gpu:
            self.store.to(self.device)
        self.model.to(self.device).eval()
        self.model.encoder.config.reference_compile = False
        mdtype = next(self.model.parameters()).dtype
        for m in self.model.modules():
            if isinstance(m, TernaryEmbedding):
                m.out_dtype = mdtype

        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.dtype = torch.float16
        if verbose:
            print(f"[laya-lite] device={self.device} tile={tile} "
                  f"打包权重 {sum(v['len'] for v in self.store.kept.values())/2**20:.2f} MiB(keep) + "
                  f"{self.store.weights.nbytes/2**20:.2f} MiB(trits) + {self.store.scales.nbytes/2**20:.2f} MiB(scales)")

    # -- 装配 --------------------------------------------------------------
    def _install(self, model):
        st, n_lin, n_emb, n_mha = self.store, 0, 0, 0
        for name, module in list(model.named_modules()):
            key = name + ".weight"
            if key not in st.quantized:
                continue
            if ".self_attn.out_proj" in name:
                continue                     # MHA 内部直接取 .weight，交给 _MHAUnpack
            if isinstance(module, nn.Linear):
                # 占位 buffer：meta 下取不到真值，但形状要对，_load_keep 会把真值灌进来
                ph = torch.empty(module.out_features) if module.bias is not None else None
                parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                setattr(parent, name.rsplit(".", 1)[-1], TernaryLinear(st, key, ph, self.tile))
                n_lin += 1
            elif isinstance(module, nn.Embedding):
                parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                setattr(parent, name.rsplit(".", 1)[-1], TernaryEmbedding(st, key))
                n_emb += 1
        for name, module in list(model.named_modules()):
            in_key, out_key = name + ".in_proj_weight", name + ".out_proj.weight"
            if in_key in st.quantized and isinstance(module, nn.MultiheadAttention):
                module.in_proj_weight = nn.Parameter(torch.zeros(0), requires_grad=False)
                if out_key in st.quantized:
                    module.out_proj.weight = nn.Parameter(torch.zeros(0), requires_grad=False)
                module.register_forward_pre_hook(_MHAUnpack(st, in_key, out_key))
                n_mha += 1
        print(f"[laya-lite] 已接管 {n_lin} 个 Linear + {n_emb} 个 Embedding + {n_mha} 个 MHA")
        return model

    def _fix_rope(self):
        """ModernBERT 的 RoPE inv_freq 是 non-persistent buffer，不随 state_dict 走，
        meta 化后没人给它赋值。好在它只依赖 config，按同一公式重算即可。"""
        rope = getattr(self.model.encoder, "rotary_emb", None)
        if rope is None or not hasattr(rope, "layer_types"):
            return
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        for lt in rope.layer_types:
            params = rope.config.rope_parameters.get(lt)
            if params is None:
                continue
            fn = rope.compute_default_rope_parameters
            if params["rope_type"] != "default":
                fn = ROPE_INIT_FUNCTIONS[params["rope_type"]]
            inv, scaling = fn(rope.config, layer_type=lt)
            rope.register_buffer("%s_inv_freq" % lt, inv, persistent=False)
            rope.register_buffer("%s_original_inv_freq" % lt, inv.clone(), persistent=False)
            setattr(rope, "%s_attention_scaling" % lt, scaling)

    def _load_keep(self):
        """把 norm / bias / 决策头 / temperature 等小张量按原名灌回去。"""
        sd = self.model.state_dict()
        loaded = 0
        for name, info in self.store.kept.items():
            if name in sd:
                # assign=True 是整块替换、不做 dtype 转换（copy_ 才会转），
                # 而 keep.bin 里存的是原始 F16/F32，必须显式对齐到模型 dtype
                sd[name] = torch.from_numpy(self.store.keep_tensor(name)).to(torch.float32)
                loaded += 1
        missing = [k for k in sd if k not in self.store.kept and k not in self.store.quantized]
        # assign=True：meta tensor 没有存储，copy_ 不进去，只能整块替换
        self.model.load_state_dict(sd, strict=False, assign=True)
        self._fix_rope()
        if missing:
            print(f"[laya-lite] 警告: {len(missing)} 个张量既未量化也未保留: {missing[:5]}")

    # -- 推理 --------------------------------------------------------------
    @staticmethod
    def _to_internal(qdef):
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        ins = qdef["instructions"]
        return {"t": t, "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": crit}

    @torch.no_grad()
    def system_one(self, state, questions):
        ids, items = list(questions.keys()), []
        for qid in ids:
            q = self._to_internal(questions[qid])
            seq, markers = build_sequence(self.tok, state, q, self.cfg["max_len"], self.cfg["head_max_len"])
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r: options do not fit in head_max_len=%d tokens"
                                 % (qid, self.cfg["head_max_len"]))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": [0.0] * len(markers),
                          "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "api"})
        b = collate_items([items], self.tok.pad_token_id)
        logits, act = self.model(b["input_ids"].to(self.device), b["attention_mask"].to(self.device),
                                 b["marker_pos"].to(self.device), b["marker_mask"].to(self.device),
                                 b["qtype"].to(self.device))
        logits, act = logits.float().cpu().numpy(), torch.softmax(act.float(), -1).cpu().numpy()
        answers, n_tokens = {}, int(b["attention_mask"].sum())
        for r, qid in enumerate(ids):
            q = self._to_internal(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            z = logits[r, :k] / self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            p = np.exp(z - z.max())
            p = p / p.sum()
            ext = {"act_probability": float(act[r, 0])}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                                "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
            elif q["t"] == "score":
                answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                                "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4), "rl_agent": ext}
        return {"model": "laya-lite-ternary", "answers": answers,
                "usage": {"input_tokens": n_tokens, "output_tokens": 0}}

    @torch.no_grad()
    def raw(self, batch):
        """直接跑一个 collate 好的 batch，返回 (logits, act)，供保真度对比用。"""
        return self.model(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device),
                          batch["marker_pos"].to(self.device), batch["marker_mask"].to(self.device),
                          batch["qtype"].to(self.device))
