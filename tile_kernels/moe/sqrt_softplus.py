from tilelang import language as T

from tile_kernels.config import is_ascend

if is_ascend():
    from tilelang.ascend.language import simd as S


_G0 = 0.8325546383857727
_FORWARD_COEFFS = (
    0.000471802573883906,
    -0.0004226923920214176,
    -8.536428504157811e-05,
    0.002192076062783599,
    0.000292900949716568,
    0.004986039828509092,
    0.011144610121846199,
    0.03287079185247421,
    0.11599518358707428,
)


@T.macro
def sqrt_softplus_forward_cuda(x: T.Ref):
    return T.sqrt(T.Select(x > 20.0, x, T.log1p(T.exp(x))))


# P(t) is evaluated in Horner order, with coefficients stored from t^8 down.
@T.macro
def _horner_Pt(t):
    poly = S.alloc_var(T.float32)
    poly = S.vdup(_FORWARD_COEFFS[0], T.float32)
    S.vmadd(poly, t, S.vdup(_FORWARD_COEFFS[1], T.float32))
    S.vmadd(poly, t, S.vdup(_FORWARD_COEFFS[2], T.float32))
    S.vmadd(poly, t, S.vdup(_FORWARD_COEFFS[3], T.float32))
    S.vmadd(poly, t, S.vdup(_FORWARD_COEFFS[4], T.float32))
    S.vmadd(poly, t, S.vdup(_FORWARD_COEFFS[5], T.float32))
    S.vmadd(poly, t, S.vdup(_FORWARD_COEFFS[6], T.float32))
    S.vmadd(poly, t, S.vdup(_FORWARD_COEFFS[7], T.float32))
    S.vmadd(poly, t, S.vdup(_FORWARD_COEFFS[8], T.float32))
    return poly


# For the selected positive softplus branch, x >= log(2). Newton and the
# FMA residual correction refine the hardware sqrt/reciprocal estimates.
# The reciprocal is only an initial estimate; separately rounding it is redundant.
@T.macro
def _vsqrt_rne(x):
    root0 = S.vsqrt(x)
    recip0 = S.vdiv(S.vdup(1.0, T.float32), root0, precision='ftz_true')
    root1 = S.alloc_var(T.float32)
    root1 = S.vmuls(S.vadd(root0, S.vmul(x, recip0)), 0.5)
    resid = S.alloc_var(T.float32)
    resid = S.vneg(x)
    S.vmula(resid, root1, root1)  # resid = root1^2 - x, exact FMA
    corr = S.vmuls(recip0, -0.5)
    root2 = S.alloc_var(T.float32)
    root2 = root1
    S.vmula(root2, resid, corr)  # root2 = root1 - resid * recip0 / 2
    return root2


# s = x + l^2 with a single rounding, then y = sqrt(s).
@T.macro
def _assemble_sqrt(x, l):
    s = S.alloc_var(T.float32)
    s = S.vrelu(x)
    S.vmula(s, l, l)  # s = max(x,0) + l^2, one rounding
    return _vsqrt_rne(s)


# Match master in the negative tail: preserve subnormal exp outputs.
@T.macro
def _exp_neg_half_abs(x):
    neg_half_abs = S.vmuls(S.vabs(x), -0.5)
    return S.vexp(neg_half_abs, precision='ftz_false')


# l = a*g = sqrt(log1p(q)) with q = a^2, g = G0 + t*P(t) ~= sqrt(log1p(q)/q), t = 1 - q.
@T.macro
def _sqrt_log1p(a):
    t = S.alloc_var(T.float32)
    t = S.vsub(S.vdup(1.0, T.float32), S.vmul(a, a))
    p = _horner_Pt(t)
    S.vmadd(t, p, S.vdup(_G0, T.float32))  # g = G0 + t*P(t), single rounding
    return S.vmul(a, t)


# y = sqrt(softplus(x)): x > 0 -> sqrt(x + l^2), else -> l.
@T.macro
def _assemble_softsqrt(x, l):
    y_pos = _assemble_sqrt(x, l)
    return S.vsel(y_pos, l, S.vcmps(x, 0.0, None, 'gt'))


@T.macro
def sqrt_softplus_forward_asc(logit):
    a = _exp_neg_half_abs(logit)
    l = _sqrt_log1p(a)
    return _assemble_softsqrt(logit, l)


# Q(u) coefficients in descending degree (Horner order), shared by both backends.
# For u=y*y<1, dy/dx = (y/2) * (1 + u*Q(u)).
_BACKWARD_COEFFS = (
    0.000129130945,
    -0.00130112655,
    0.00827621203,
    -0.041647464,
    0.166663632,
    -0.499999821,
)


@T.macro
def sqrt_softplus_backward_cuda(y: T.Ref):
    """Recover dy/dx from y=sqrt(softplus(x)); use the polynomial only for y²<1."""
    u = y * y
    half_y = 0.5 * y
    q = T.alloc_var(T.float32, init=_BACKWARD_COEFFS[0])
    q = q * u + _BACKWARD_COEFFS[1]
    q = q * u + _BACKWARD_COEFFS[2]
    q = q * u + _BACKWARD_COEFFS[3]
    q = q * u + _BACKWARD_COEFFS[4]
    q = q * u + _BACKWARD_COEFFS[5]
    return T.if_then_else(u < 1.0, (u * q) * half_y + half_y, 0.5 * ((1.0 - T.exp(-u)) / y))


@T.macro
def sqrt_softplus_backward_asc(y):
    """Recover dy/dx from y=sqrt(softplus(x)); use the polynomial only for y²<1."""
    u = S.vmul(y, y)
    half_y = S.vmuls(y, 0.5)
    q = T.alloc_var(y.dtype, init=S.vdup(_BACKWARD_COEFFS[0], T.float32))
    S.vmadd(q, u, S.vdup(_BACKWARD_COEFFS[1], T.float32))
    S.vmadd(q, u, S.vdup(_BACKWARD_COEFFS[2], T.float32))
    S.vmadd(q, u, S.vdup(_BACKWARD_COEFFS[3], T.float32))
    S.vmadd(q, u, S.vdup(_BACKWARD_COEFFS[4], T.float32))
    S.vmadd(q, u, S.vdup(_BACKWARD_COEFFS[5], T.float32))
    small = T.alloc_var(y.dtype, init=S.vmul(u, q))
    S.vmadd(small, half_y, half_y)
    numerator = S.vsub(S.vdup(1.0, T.float32), S.vexp(S.vneg(u), precision='ftz_true'))
    large = S.vmuls(S.vdiv(numerator, y, precision='ftz_true'), 0.5)
    return S.vsel(small, large, S.vcmps(u, 1.0, op='lt'))


@T.macro
def _horner_Pt_pto(t):
    poly = _FORWARD_COEFFS[0]
    for coeff in _FORWARD_COEFFS[1:]:
        poly = poly * t + coeff
    return poly


# For the selected positive softplus branch, x >= log(2). Newton and the
# FMA residual correction refine the hardware sqrt/reciprocal estimates.
@T.macro
def _vsqrt_rne_pto(x):
    root0 = T.sqrt(x)
    recip0 = 1.0 / root0
    root1 = (root0 + x * recip0) * 0.5
    resid = root1 * root1 - x
    corr = recip0 * -0.5
    return root1 + resid * corr


# s = x + l^2, then y = sqrt(s).
@T.macro
def _assemble_sqrt_pto(x, l):
    s = l * l + T.max(x, 0.0)
    return _vsqrt_rne_pto(s)


# Match master in the negative tail: preserve subnormal exp outputs.
@T.macro
def _exp_neg_half_abs_pto(x):
    return T.exp(T.abs(x) * -0.5)


# l = a*g = sqrt(log1p(q)) with q = a^2, g = G0 + t*P(t) ~= sqrt(log1p(q)/q), t = 1 - q.
@T.macro
def _sqrt_log1p_pto(a):
    t = 1.0 - a * a
    p = _horner_Pt_pto(t)
    t = t * p + _G0
    return a * t


# y = sqrt(softplus(x)): x > 0 -> sqrt(x + l^2), else -> l.
@T.macro
def _assemble_softsqrt_pto(x, l):
    y_pos = _assemble_sqrt_pto(x, l)
    return T.Select(x > 0.0, y_pos, l)


@T.macro
def sqrt_softplus_forward_pto(logit):
    a = _exp_neg_half_abs_pto(logit)
    l = _sqrt_log1p_pto(a)
    return _assemble_softsqrt_pto(logit, l)


@T.macro
def sqrt_softplus_backward_pto(y):
    """Recover dy/dx from y=sqrt(softplus(x)); use the polynomial only for y²<1."""
    u = y * y
    q = _BACKWARD_COEFFS[0]
    for coeff in _BACKWARD_COEFFS[1:]:
        q = q * u + coeff
    half_y = y * 0.5
    small = (u * q) * half_y + half_y
    numerator = 1.0 - T.exp(-u)
    large = (numerator / y) * 0.5
    return T.Select(u < 1.0, small, large)
