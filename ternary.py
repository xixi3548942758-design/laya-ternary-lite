"""三值量化核心：分组 absmean 量化 + 5-trit/byte 密集打包。

设计要点
--------
* 权重取值 {-1, 0, +1}，每 GROUP 个权重共享一个 scale（Bonsai 的 ternary g128 思路）。
* 打包用 5 trits/byte（3^5 = 243 <= 256），等效 1.6 bit/weight，逼近 log2(3)=1.585 的理论下限。
* 展平后按 lcm(5, GROUP) 对齐 padding，保证 pack/group 边界都整齐。
"""
import numpy as np

TRITS_PER_BYTE = 5           # 3^5 = 243，一个 uint8 装 5 个 trit
GROUP = 128                  # 每 128 个权重共享 1 个 scale
_ALIGN = 640                 # lcm(5, 128)
_POW3 = np.array([1, 3, 9, 27, 81], dtype=np.int32)


def pad_to_align(n: int) -> int:
    """展平长度向上对齐到 640 的倍数。"""
    return ((n + _ALIGN - 1) // _ALIGN) * _ALIGN


# --------------------------------------------------------------------------- 量化
def quantize_grouped(w: np.ndarray, group: int = GROUP):
    """FP 权重 -> (trits int8{-1,0,1}, scales float16)。展平后按 group 分组，absmean 求 scale。"""
    flat = np.asarray(w, dtype=np.float32).reshape(-1)
    n = flat.size
    n_pad = pad_to_align(n)
    if n_pad != n:
        flat = np.concatenate([flat, np.zeros(n_pad - n, dtype=np.float32)])
    g = flat.reshape(-1, group)
    scales = np.abs(g).mean(axis=1).astype(np.float16)          # absmean，BitNet b1.58 的 γ
    denom = scales.astype(np.float32)
    denom[denom == 0] = 1e-8
    q = np.clip(np.rint(g / denom[:, None]), -1, 1).astype(np.int8)
    return q.reshape(-1)[:n_pad], scales


def dequantize_grouped(trits: np.ndarray, scales: np.ndarray, group: int = GROUP,
                       n: int = None) -> np.ndarray:
    """trits/scales -> FP32 权重（推理时可分块调用以省内存）。n 给定时截断掉 padding。"""
    t = trits.astype(np.float32).reshape(-1, group)
    out = (t * scales.astype(np.float32)[:, None]).reshape(-1)
    return out if n is None else out[:n]


# --------------------------------------------------------------------------- 打包
def pack_trits(q: np.ndarray) -> np.ndarray:
    """int8 {-1,0,1} 数组（长度须为 5 的倍数）-> uint8，每字节 5 个 trit。"""
    n = q.size
    if n % TRITS_PER_BYTE:
        q = np.concatenate([q, np.zeros(TRITS_PER_BYTE - n % TRITS_PER_BYTE, dtype=np.int8)])
    t = q.reshape(-1, TRITS_PER_BYTE).astype(np.int32) + 1        # {-1,0,1} -> {0,1,2}
    return (t * _POW3).sum(axis=1).astype(np.uint8)


def unpack_trits(packed: np.ndarray, n: int) -> np.ndarray:
    """uint8 -> int8 {-1,0,1}，返回前 n 个。"""
    b = packed.astype(np.int32).copy()
    out = np.empty((b.size, TRITS_PER_BYTE), dtype=np.int8)
    for i in range(TRITS_PER_BYTE):
        out[:, i] = (b % 3) - 1
        b //= 3
    return out.reshape(-1)[:n]


# --------------------------------------------------------------------------- 体积核算
def ternary_bytes(numel: int, group: int = GROUP, scale_bytes: int = 2) -> int:
    """给定参数量，返回三值化后的字节数（packed trits + scales）。"""
    n_pad = pad_to_align(numel)
    return n_pad // TRITS_PER_BYTE + (n_pad // group) * scale_bytes


def fp16_bytes(numel: int) -> int:
    return numel * 2
