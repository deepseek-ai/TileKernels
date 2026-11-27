from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

# One vector register holds 64 fp32 lanes: either 64 fp32 elements loaded directly, or
# 64 bf16 elements unpacked with `UNPK_B16` + `vcvt(..., part=0)`.
_VEC = 64
# A vector register is 256B regardless of dtype, i.e. 128 bf16 or 64 fp32 elements.
_VREG_BYTES = 256
# Ascend hidden-tile alignment granularity (matches _VEC).
_H_BLK_ALIGN = 64


@T.macro
def _load_f32(src: T.Ref):
    """Load 64 contiguous BF16 or FP32 elements into an FP32 vector register."""
    if src.dtype == T.bfloat16:
        return S.vcvt(S.vld(src, dist='UNPK_B16'), T.float32, part=0)
    return S.vld(src)


@T.macro
def _store_f32(dst: T.Ref, val):
    """Store an FP32 vector register into 64 contiguous BF16 or FP32 elements."""
    if dst.dtype == T.bfloat16:
        S.vsts(dst, S.vcvt(val, T.bfloat16), dist='PK_B32')
    else:
        S.vsts(dst, val)
