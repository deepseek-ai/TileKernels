from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


@T.macro
def load_f32(src: T.Ref):
    """Load 64 contiguous elements as FP32."""
    if src.dtype == T.bfloat16:
        return S.vcvt(S.vld(src, dist='UNPK_B16'), T.float32, part=0)
    return S.vld(src)
