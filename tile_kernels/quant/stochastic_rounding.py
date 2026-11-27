from tile_kernels.config import is_ascend

if is_ascend():
    from tilelang.ascend import language as T
    from tilelang.ascend.language import simd as S
else:
    from tilelang import language as T

from typing import Literal

_MAGIC = [T.uint32(0x5671D42B), T.uint32(0x9995E499), T.uint32(0xACE1B8A5), T.uint32(0xE153538D)]
_MAGIC_A, _MAGIC_B, _MAGIC_C = T.uint32(0x7FEB352D), T.uint32(0x846CA68B), T.uint32(0xD35A2D97)


@T.macro
def rbits_hash_cuda(values, quad_idx: T.uint32, num_rbits: Literal[1, 2]):
    h = T.alloc_var(T.uint32, quad_idx)
    res = T.alloc_local((num_rbits,), T.uint32)
    h += T.reinterpret(values[0], T.uint32) * _MAGIC[0]
    h += T.reinterpret(values[1], T.uint32) * _MAGIC[1]
    h += T.reinterpret(values[2], T.uint32) * _MAGIC[2]
    h += T.reinterpret(values[3], T.uint32) * _MAGIC[3]
    h = h ^ (h >> 23)
    h = h * _MAGIC_A
    h = h ^ (h >> 16)
    res[0] = h * _MAGIC_B
    res[0] = res[0] ^ (res[0] >> 11)
    if num_rbits == 2:
        res[1] = h * _MAGIC_C
        res[1] = res[1] ^ (res[1] >> 11)
    return res


@T.macro
def _rbits_hash_calc_h_asc(values: T.LocalBuffer[(4,), T.float32x64], quad_offsets: T.uint32x64):
    h = S.alloc_var(T.uint32)
    weighted = S.alloc_local((2,), T.uint32)
    weighted[0] = quad_offsets
    S.vmula(weighted[0], T.reinterpret(values[0], 'uint32x64'), S.vdup(_MAGIC[0], T.uint32))
    S.vmula(weighted[0], T.reinterpret(values[1], 'uint32x64'), S.vdup(_MAGIC[1], T.uint32))
    weighted[1] = S.vmuls(T.reinterpret(values[2], 'uint32x64'), _MAGIC[2])
    S.vmula(weighted[1], T.reinterpret(values[3], 'uint32x64'), S.vdup(_MAGIC[3], T.uint32))
    h = S.vadd(weighted[0], weighted[1])

    h = S.vxor(S.vshrs(h, 23), h)
    h = S.vmuls(h, _MAGIC_A)
    h = S.vxor(S.vshrs(h, 16), h)

    return h


@T.macro
def rbits_hash_bf16_asc(values: T.LocalBuffer[(4,), T.float32x64], quad_idx: T.uint32):
    tmp = S.alloc_local((4,), T.float32)
    res = S.alloc_local((2,), T.uint32)
    # permutate to make 4 values in the same group align
    tmp[0], tmp[1] = S.vdintlv(values[0], values[1])
    tmp[2], tmp[3] = S.vdintlv(values[2], values[3])
    tmp[0], tmp[2] = S.vdintlv(tmp[0], tmp[2])
    tmp[1], tmp[3] = S.vdintlv(tmp[1], tmp[3])
    quad_offsets = T.reinterpret(S.vci(T.reinterpret(quad_idx, T.int32), T.int32), 'uint32x64')

    h = _rbits_hash_calc_h_asc(tmp, quad_offsets)
    res[0] = S.vmuls(h, _MAGIC_B)
    res[0] = S.vxor(res[0], S.vshrs(res[0], 11))
    res[1] = S.vmuls(h, _MAGIC_C)
    res[1] = S.vxor(res[1], S.vshrs(res[1], 11))
    res[0], res[1] = S.vintlv(res[0], res[1])

    return res


@T.macro
def rbits_hash_fp8_asc(values: T.LocalBuffer[(4,), T.float32x64], quad_offsets: T.uint32x64):
    h = _rbits_hash_calc_h_asc(values, quad_offsets)
    h_b = S.vmuls(h, _MAGIC_B)
    return S.vxor(S.vshrs(h_b, 11), h_b)


@T.macro
def stochastic_round_fp8_asc(values: T.LocalBuffer[(4,), T.float32x64], quad_offsets: T.uint32x64):
    # Align FP32 and E4M3 subnormals, then retain 16 probability bits.
    low = S.alloc_local((2,), T.uint16)
    high = S.alloc_local((2,), T.uint16)
    for i in T.unroll(2, explicit=True):
        aligned = S.alloc_local((2,), T.uint32)
        for j in T.unroll(2, explicit=True):
            scaled = S.vmuls(values[i + j * 2], 2.0**-120)
            aligned[j] = S.vshrs(T.reinterpret(scaled, 'uint32x64'), 4)
        low[i], high[i] = S.vdintlv(T.reinterpret(aligned[0], 'uint16x128'), T.reinterpret(aligned[1], 'uint16x128'))

    # Use all 128 halfwords of the hash; swap the two bytes for the odd positions.
    rbits = S.alloc_local((2,), T.uint16)
    rbits[0] = T.reinterpret(rbits_hash_fp8_asc(values, quad_offsets), 'uint16x128')
    rbits[1] = S.vor(S.vshls(rbits[0], 8), S.vshrs(rbits[0], 8))
    res = S.alloc_local((2,), T.uint16)
    for i in T.unroll(2, explicit=True):
        res[i] = high[i]
        carry = S.vcmp(rbits[i], low[i], op='lt')
        res[i] = S.vadds(res[i], 1, carry, mode='MODE_MERGING')
        # 0000|s|0000|e4|m3
        negative = S.vcmps(res[i], 1 << 11, op='ge')
        res[i] = S.vadds(res[i], (1 << 7) - (1 << 11), negative, mode='MODE_MERGING')
    S.vmula(res[0], res[1], S.vdup(256, T.uint16))
    return T.reinterpret(res[0], 'float8_e4m3fnx256')
