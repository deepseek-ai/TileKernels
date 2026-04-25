import torch
from torch.utils.checkpoint import checkpoint

from tile_kernels.mhc.norm_fn_kernel import (
    _mhc_fn_normw_merge_bwd,
    _mhc_fn_normw_merge_fwd,
    _mhc_pre_norm_fn_bwd_mul,
    _mhc_pre_norm_fn_bwd_norm,
    _mhc_pre_norm_fn_fwd_mul,
    _mhc_pre_norm_fn_fwd_norm,
    round_to_tf32,
)


class _MHCFnNormwMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx: '_MHCFnNormwMerge', fn: torch.Tensor, normw: torch.Tensor) -> torch.Tensor:
        ctx.fn_main_grad = getattr(fn, 'main_grad', None)
        ctx.normw_main_grad = getattr(normw, 'main_grad', None)
        ctx.save_for_backward(fn, normw)
        out_fn = torch.empty_like(fn)
        _mhc_fn_normw_merge_fwd(*fn.shape)(fn, normw, out_fn)
        return out_fn

    @staticmethod
    def backward(ctx: '_MHCFnNormwMerge', out_fn_grad: torch.Tensor) -> tuple[None, None]:
        fn, normw = ctx.saved_tensors

        fn_grad: torch.Tensor = ctx.fn_main_grad
        if fn_grad is None:
            fn_grad = torch.zeros_like(fn)

        normw_grad: torch.Tensor = ctx.normw_main_grad
        if normw_grad is None:
            normw_grad = torch.zeros_like(normw)

        _mhc_fn_normw_merge_bwd(*fn.shape)(fn, normw, out_fn_grad, fn_grad, normw_grad)

        if ctx.fn_main_grad is not None:
            fn_grad = None

        if ctx.normw_main_grad is not None:
            normw_grad = None

        return fn_grad, normw_grad


class MHCPreNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: 'MHCPreNormFn',
        x: torch.Tensor,
        fn: torch.Tensor,
        norm_eps: float,
        fuse_grad_acc: bool,
        n_splits: int,
    ) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        assert fn.dtype == torch.float32

        mhc_mult3, mhc_hidden_size = fn.shape
        assert x.shape[-1] * x.shape[-2] == mhc_hidden_size
        outer_shape = x.shape[:-2]

        out_mul_splitted = torch.empty(
            n_splits,
            *outer_shape,
            1,
            mhc_mult3,
            dtype=torch.float32,
            device=x.device,
        )
        sqrsum_splitted = torch.empty(
            n_splits,
            *outer_shape,
            1,
            dtype=torch.float32,
            device=x.device,
        )
        out = torch.empty(
            *outer_shape,
            mhc_mult3,
            dtype=torch.float32,
            device=x.device,
        )

        # TileLang implementation doesn't support split-k, so we set n_splits to 1
        # You may want to adopt the DeepGEMM implementation with split-k for better performance
        n_splits = 1
        out_mul_splitted = out_mul_splitted[:1]
        sqrsum_splitted = sqrsum_splitted[:1]

        fn = round_to_tf32(fn)

        fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, 1, mhc_hidden_size)
        fwd_mul_kernel(
            x.view(-1, mhc_hidden_size),
            fn,
            out_mul_splitted.view(-1, 1, mhc_mult3),
            sqrsum_splitted.view(-1, 1),
        )
        # END of TileLang implementation of pre-norm-fn forward matmul

        out_mul = torch.empty_like(out_mul_splitted[0])
        sqrsum = torch.empty_like(sqrsum_splitted[0])

        fwd_norm_kernel = _mhc_pre_norm_fn_fwd_norm(
            mhc_mult3,
            1,
            mhc_hidden_size,
            norm_eps,
            n_splits,
        )
        fwd_norm_kernel(
            out_mul_splitted.view(n_splits, -1, 1, mhc_mult3),
            sqrsum_splitted.view(n_splits, -1, 1),
            out_mul.view(-1, 1, mhc_mult3),
            sqrsum.view(-1, 1),
            out.view(-1, mhc_mult3),
        )

        ctx.save_for_backward(x, fn, out_mul, sqrsum)
        ctx.norm_eps = norm_eps
        ctx.fuse_grad_acc = fuse_grad_acc

        return out

    @staticmethod
    def backward(
        ctx: 'MHCPreNormFn',
        out_grad: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor, None, None, None]:
        x, fn, out_mul, sqrsum = ctx.saved_tensors
        norm_eps = ctx.norm_eps

        mhc_mult3, mhc_hidden_size = fn.shape

        out_mul_grad = torch.empty_like(out_mul)
        sqrsum_grad = torch.empty_like(sqrsum)
        bwd_norm_kernel = _mhc_pre_norm_fn_bwd_norm(mhc_mult3, 1, mhc_hidden_size, norm_eps)
        bwd_norm_kernel(
            out_grad.view(-1, mhc_mult3),
            out_mul.view(-1, 1, mhc_mult3),
            sqrsum.view(-1, 1),
            out_mul_grad.view(-1, 1, mhc_mult3),
            sqrsum_grad.view(-1, 1),
        )

        if ctx.fuse_grad_acc:
            x_grad: torch.Tensor = x.untyped_storage().grad_from_mhc_post.view_as(x)
        else:
            x_grad = torch.zeros_like(x)
        fn_grad = torch.empty_like(fn)

        out_mul_grad = round_to_tf32(out_mul_grad)

        bwd_mul_kernel = _mhc_pre_norm_fn_bwd_mul(mhc_mult3, 1, mhc_hidden_size)
        bwd_mul_kernel(
            out_mul_grad.view(-1, 1, mhc_mult3),
            sqrsum_grad.view(-1, 1),
            x.view(-1, mhc_hidden_size),
            fn,
            x_grad.view(-1, mhc_hidden_size),
            fn_grad,
        )

        if ctx.fuse_grad_acc:
            del x.untyped_storage().grad_from_mhc_post
            return None, fn_grad, None, None, None
        return x_grad, fn_grad, None, None, None


def mhc_pre_norm_fn(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
    fuse_grad_acc: bool = True,
    n_splits: int = 16,
) -> torch.Tensor:
    if mhc_norm_weight is not None:
        mhc_fn = _MHCFnNormwMerge.apply(mhc_fn, mhc_norm_weight)
    return MHCPreNormFn.apply(
        residual,
        mhc_fn,
        mhc_norm_eps,
        fuse_grad_acc,
        n_splits,
    )
