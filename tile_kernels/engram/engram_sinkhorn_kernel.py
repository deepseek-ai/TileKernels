import torch
from tilelang import language as T

from tile_kernels.config import get_num_sms, get_num_vec_cores, is_ascend
from tile_kernels.engram.engram_sinkhorn_cuda import (
    get_engram_sinkhorn_finalize_kernel_cuda,
    get_engram_sinkhorn_momentum_kernel_cuda,
    get_engram_sinkhorn_step_kernel_cuda,
    get_engram_sinkhorn_step_reduce_kernel_cuda,
)


def engram_sinkhorn_momentum_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta1: float,
    nesterov: bool = True,
) -> torch.Tensor:
    """Update momentum buffer in-place and produce the fp32 tensor consumed by Sinkhorn normalization.

    Args:
        grad: Gradient tensor, shape (numel,), bfloat16 or float32. numel must be a multiple of 128.
        momentum_buffer: Momentum buffer, shape (numel,), bfloat16 or float32. Modified in-place.
        beta1: Momentum decay coefficient.
        nesterov: If True, write the Nesterov lookahead into ``sinkhorn_input``; otherwise write the updated momentum.

    Returns:
        sinkhorn_input: Shape (numel,), float32.
    """
    numel = grad.numel()
    assert grad.dtype in (torch.bfloat16, torch.float32), f'unsupported grad dtype: {grad.dtype}'
    assert momentum_buffer.dtype in (torch.bfloat16, torch.float32), f'unsupported momentum dtype: {momentum_buffer.dtype}'
    assert numel % 128 == 0
    alignment = 256 if numel % 256 == 0 else 128

    sinkhorn_input = torch.empty_like(momentum_buffer, dtype=torch.float32)

    if is_ascend():
        from tile_kernels.engram.engram_sinkhorn_asc import get_engram_sinkhorn_momentum_kernel_asc

        kernel = get_engram_sinkhorn_momentum_kernel_asc(
            T.dtype(grad.dtype),
            T.dtype(momentum_buffer.dtype),
            nesterov,
            beta1,
            alignment,
            get_num_vec_cores(),
        )
    else:
        kernel = get_engram_sinkhorn_momentum_kernel_cuda(
            T.dtype(grad.dtype),
            T.dtype(momentum_buffer.dtype),
            nesterov,
            beta1,
            alignment,
            use_pdl=False,
        )  # PDL regression: >=2us slower, keep PDL off.

    if numel > 0:
        num_rows = numel // alignment
        kernel(
            grad.view(num_rows, alignment),
            momentum_buffer.view(num_rows, alignment),
            sinkhorn_input.view(num_rows, alignment),
        )

    return sinkhorn_input


def engram_sinkhorn_step(
    matrix: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize one Sinkhorn iterate and write per-CTA column partials.

    Args:
        matrix: Base matrix of shape (num_rows, head_dim), float32.
        row_scale: Per-row scale of shape (num_rows,), float32. Updated in-place.
        col_scale: Per-column scale of shape (head_dim,), float32.
        eps: Epsilon added to each row norm for numerical stability.

    Returns:
        A tuple of ``row_norm`` with shape (num_rows,) and ``col_sumsq_partial``
        with shape (num_ctas, head_dim), both float32.
    """
    head_dim = matrix.shape[1]
    row_norm = torch.empty_like(row_scale)

    if is_ascend():
        from tile_kernels.engram.engram_sinkhorn_asc import get_engram_sinkhorn_step_kernel_asc

        num_partials = get_num_vec_cores()
        kernel = get_engram_sinkhorn_step_kernel_asc(head_dim, eps, num_partials)
    else:
        num_partials = get_num_sms() * 3
        kernel = get_engram_sinkhorn_step_kernel_cuda(
            head_dim,
            eps,
            num_partials,
            use_pdl=False,
        )  # PDL regression: >=2us slower, keep PDL off.

    col_sumsq_partial = matrix.new_empty((num_partials, head_dim))
    kernel(matrix, row_scale, col_scale, row_norm, col_sumsq_partial)

    return row_norm, col_sumsq_partial


def engram_sinkhorn_step_reduce(
    col_sumsq_partial: torch.Tensor,
    col_sumsq: torch.Tensor,
) -> None:
    """Reduce Sinkhorn column partials in a fixed order.

    Args:
        col_sumsq_partial: Per-worker partials, shape (num_partials, head_dim), float32.
        col_sumsq: Output buffer, shape (head_dim,), float32.
    """
    num_partials, head_dim = col_sumsq_partial.shape

    if is_ascend():
        from tile_kernels.engram.engram_sinkhorn_asc import get_engram_sinkhorn_step_reduce_kernel_asc

        kernel = get_engram_sinkhorn_step_reduce_kernel_asc(
            head_dim,
            num_partials,
            get_num_vec_cores(),
        )
    else:
        kernel = get_engram_sinkhorn_step_reduce_kernel_cuda(
            head_dim,
            num_partials,
            use_pdl=False,
        )  # PDL regression: >=2us slower, keep PDL off.

    kernel(col_sumsq_partial, col_sumsq)


def engram_sinkhorn_finalize(
    matrix: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    eps: float,
    alpha: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the final row-normalized Sinkhorn matrix.

    Args:
        matrix: Base matrix of shape (num_rows, head_dim), float32.
        row_scale: Per-row scale of shape (num_rows,), float32.
        col_scale: Per-column scale of shape (head_dim,), float32.
        eps: Epsilon added to each row norm for numerical stability.
        alpha: Scalar applied after row normalization.
        out: Optional pre-allocated output tensor with the same shape, dtype as ``matrix``.

    Returns:
        Final matrix of shape (num_rows, head_dim), float32, scaled by ``alpha``.
    """
    rows, head_dim = matrix.shape
    if out is None:
        out = torch.empty_like(matrix)

    if is_ascend():
        from tile_kernels.engram.engram_sinkhorn_asc import get_engram_sinkhorn_finalize_kernel_asc

        kernel = get_engram_sinkhorn_finalize_kernel_asc(
            head_dim,
            eps,
            alpha,
            get_num_vec_cores(),
        )
    else:
        kernel = get_engram_sinkhorn_finalize_kernel_cuda(head_dim, eps, alpha, use_pdl=False)  # PDL regression: >=2us slower, keep PDL off.

    if rows > 0:
        kernel(matrix, row_scale, col_scale, out)

    return out
