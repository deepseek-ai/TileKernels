from functools import partial

import tilelang
from tilelang import language as T

from tile_kernels.config import get_max_smem_per_sm


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
    },
)
def get_engram_gate_bwd_kernel_cuda(
    hidden_size: int,
    scalar: float,
    num_persistent_blocks: int,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
    has_image_token_mask: bool = False,
    use_pdl: bool = False,
):
    """Backward kernel: 8 warps per CTA, 2 warps per head.

    grad_out and v cached in shared memory. grad_w accumulates in registers.
    Cross-warp dldg reduction via shared memory.
    """
    assert hc_mult == 4
    num_tokens = T.dynamic('num_tokens')
    warp_size = 32
    warps_per_head = 2
    num_warps = hc_mult * warps_per_head
    threads = warp_size * num_warps
    threads_per_head = warp_size * warps_per_head
    assert hidden_size % threads == 0
    elems_per_thread = hidden_size // threads
    elems_per_warp_pair = hidden_size // threads_per_head
    # NOTE: Keep go vec size aligned with x vec size for memory order
    go_vec_size = 4
    x_vec_size = 4
    v_copy_vec_size = 8

    # NOTE Performance only tuned for hidden_size in {4096, 7168}
    def _choose_v_vec_size(elems_per_thread):
        for vec in [4, 2, 1]:
            if elems_per_thread % vec == 0:
                return vec

    def _choose_v_copy_threads(hidden_size, vec_size):
        num_vectors = hidden_size // vec_size
        for candidate in (256, 128, 64, 32):
            if num_vectors % candidate == 0:
                return candidate
        raise ValueError(f'No valid vectorized v copy layout for hidden_size={hidden_size}, vec_size={vec_size}')

    def _choose_go_blk_d(hidden_size, go_tile):
        go_blk_d = None
        for blk in range(go_tile, hidden_size // 2 + 1, go_tile):
            if hidden_size % blk == 0:
                go_blk_d = blk
        if go_blk_d is None:
            raise ValueError(f'No valid backward grad_out tile for hidden_size={hidden_size}')
        return go_blk_d

    def _choose_x_blk_d(hidden_size, x_tile, hc_mult, warps_per_head):
        """Largest multiple of x_tile that divides hidden_size, <= hidden_size // 2, fits in smem."""
        smem_fixed = (hc_mult + 1) * hidden_size * 2 + hc_mult * warps_per_head * 4
        smem_per_x = 2 * hc_mult * (2 + 2 + 4)
        x_smem_limit = (get_max_smem_per_sm() - smem_fixed) // smem_per_x
        x_limit = min(x_smem_limit, hidden_size // 2)
        result = x_tile
        for blk in range(x_tile, x_limit + 1, x_tile):
            if hidden_size % blk == 0:
                result = blk
        return result

    v_vec_size = _choose_v_vec_size(elems_per_thread)
    v_copy_threads = _choose_v_copy_threads(hidden_size, v_copy_vec_size)
    go_blk_d = _choose_go_blk_d(hidden_size, threads_per_head * go_vec_size)
    x_blk_d = _choose_x_blk_d(hidden_size, threads_per_head * x_vec_size, hc_mult, warps_per_head)

    assert hidden_size % x_blk_d == 0
    assert x_blk_d % (threads_per_head * x_vec_size) == 0
    assert go_blk_d + x_blk_d <= hidden_size

    def smem_layout(i, j, vs):
        thread_id = i * threads_per_head + (j // vs) % threads_per_head
        local_id = (j % vs) + (j // (threads_per_head * vs)) * vs
        return thread_id, local_id

    def v_smem_layout(j):
        thread_id = (j // v_copy_vec_size) % v_copy_threads
        local_id = (j % v_copy_vec_size) + (j // (v_copy_threads * v_copy_vec_size)) * v_copy_vec_size
        return thread_id, local_id

    @T.prim_func
    def engram_gate_bwd_kernel_cuda(
        grad_out: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float],
        dot_in: T.Tensor[(num_tokens, hc_mult), T.float],
        gate_in: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_x_in: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_k_in: T.Tensor[(num_tokens, hc_mult), T.float],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        grad_x: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        grad_kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        grad_w_partial: T.Tensor[(num_persistent_blocks, hc_mult, hidden_size), T.float],
    ):
        with T.Kernel(num_persistent_blocks, threads=threads) as pid_p:
            if use_pdl:
                T.pdl_sync()
            tid = T.get_thread_binding()
            warp_id = T.get_warp_idx()
            lane_id = T.get_lane_idx()
            head_id = warp_id // warps_per_head
            sub_warp_id = warp_id % warps_per_head

            go_local = T.alloc_local((go_vec_size,), T.float)
            v_local = T.alloc_local((go_vec_size,), T.float)
            go_v_local = T.alloc_local((v_vec_size,), T.float)
            go_x_local = T.alloc_local((x_vec_size,), T.float)
            x_local = T.alloc_local((x_vec_size,), T.float)
            k_local = T.alloc_local((x_vec_size,), T.float)
            w_fused_local = T.alloc_local((x_vec_size,), T.float)
            grad_v_partial = T.alloc_local((v_vec_size,), T.float)
            grad_w_local = T.alloc_local((elems_per_warp_pair,), T.float)

            dldg_local = T.alloc_local((1,), T.float)
            dldg_r = T.alloc_local((1,), T.float)

            gate_local = T.alloc_local((hc_mult,), T.float)
            gate_local_hc = T.alloc_var(T.float)
            rstd_x_local = T.alloc_var(T.float)
            rstd_k_local = T.alloc_var(T.float)
            dot_in_local = T.alloc_var(T.float)
            dot_x_local = T.alloc_var(T.float)
            dot_k_local = T.alloc_var(T.float)

            go_copy_layout = T.Fragment((hc_mult, go_blk_d), forward_fn=partial(smem_layout, vs=go_vec_size))
            x_copy_layout = T.Fragment((hc_mult, x_blk_d), forward_fn=partial(smem_layout, vs=x_vec_size))
            v_copy_layout = T.Fragment((hidden_size,), forward_fn=v_smem_layout)

            go_smem = T.alloc_shared((hc_mult, hidden_size), T.bfloat16)
            v_smem = T.alloc_shared((hidden_size,), T.bfloat16)
            x_smem = T.alloc_shared((2, hc_mult, x_blk_d), T.bfloat16)
            k_smem = T.alloc_shared((2, hc_mult, x_blk_d), T.bfloat16)
            w_smem = T.alloc_shared((2, hc_mult, x_blk_d), T.float)
            dldg_smem = T.alloc_shared((hc_mult, warps_per_head), T.float)

            per_block = T.ceildiv(num_tokens, num_persistent_blocks)
            t_start = T.min(per_block * pid_p, num_tokens)
            t_end = T.min(per_block * (pid_p + 1), num_tokens)

            T.clear(grad_w_local)

            for i_s in T.serial(t_start, t_end):
                if has_image_token_mask and image_token_mask[i_s]:
                    if i_s + 1 < t_end:
                        if not image_token_mask[i_s + 1]:
                            T.async_copy(kv[i_s + 1, hc_mult, :], v_smem, loop_layout=v_copy_layout)
                            T.async_copy(
                                grad_out[i_s + 1, :, :go_blk_d],
                                go_smem[:, :go_blk_d],
                                loop_layout=go_copy_layout,
                            )

                    for i_h, i_d in T.Parallel(hc_mult, hidden_size):
                        grad_x[i_s, i_h, i_d] = grad_out[i_s, i_h, i_d]
                    for i_h, i_d in T.Parallel(hc_mult + 1, hidden_size):
                        grad_kv[i_s, i_h, i_d] = 0.0

                    if i_s + 1 < t_end:
                        if not image_token_mask[i_s + 1]:
                            T.ptx_wait_group(1)
                            T.sync_threads()
                    continue

                # === Prologue: load grad_out and v into smem ===
                if i_s == t_start:
                    T.async_copy(kv[i_s, hc_mult, :], v_smem, loop_layout=v_copy_layout)
                    T.async_copy(grad_out[i_s, :, :go_blk_d], go_smem[:, :go_blk_d], loop_layout=go_copy_layout)
                    T.ptx_wait_group(1)
                    # ensure v_smem is readable by all threads
                    T.sync_threads()

                T.copy(gate_in[i_s, :], gate_local)

                # Per-head scalar loads
                gate_local_hc = gate_in[i_s, head_id]
                rstd_x_local = rstd_x_in[i_s, head_id]
                rstd_k_local = rstd_k_in[i_s, head_id]
                dot_in_local = dot_in[i_s, head_id]

                T.clear(dldg_local)

                go_sub_blks = go_blk_d // (threads_per_head * go_vec_size)
                num_go_tiles = hidden_size // go_blk_d

                # === Pass 1a: dldg — two warps per head ===
                for i_b in T.serial(1, num_go_tiles):
                    T.async_copy(
                        grad_out[i_s, :, i_b * go_blk_d : (i_b + 1) * go_blk_d],
                        go_smem[:, i_b * go_blk_d : (i_b + 1) * go_blk_d],
                        loop_layout=go_copy_layout,
                    )
                    T.ptx_wait_group(1)
                    for i_sub in T.serial(go_sub_blks):
                        go_base = (i_b - 1) * go_blk_d + i_sub * threads_per_head * go_vec_size + sub_warp_id * warp_size * go_vec_size
                        for i_k in T.vectorized(go_vec_size):
                            go_local[i_k] = go_smem[head_id, go_base + lane_id * go_vec_size + i_k]
                            v_local[i_k] = v_smem[go_base + lane_id * go_vec_size + i_k]
                        for i_k in T.serial(go_vec_size):
                            dldg_local[0] += go_local[i_k] * v_local[i_k]

                # Epilogue: process last go tile
                T.ptx_wait_group(0)
                for i_sub in T.serial(go_sub_blks):
                    go_base = (num_go_tiles - 1) * go_blk_d + i_sub * threads_per_head * go_vec_size + sub_warp_id * warp_size * go_vec_size
                    for i_k in T.vectorized(go_vec_size):
                        go_local[i_k] = go_smem[head_id, go_base + lane_id * go_vec_size + i_k]
                        v_local[i_k] = v_smem[go_base + lane_id * go_vec_size + i_k]
                    for i_k in T.serial(go_vec_size):
                        dldg_local[0] += go_local[i_k] * v_local[i_k]

                # Cross-warp dldg reduction via smem
                dldg_local[0] = T.warp_reduce_sum(dldg_local[0])
                if lane_id == 0:
                    dldg_smem[head_id, sub_warp_id] = dldg_local[0]
                T.sync_threads()

                # Prefetch next token's v
                if i_s + 1 < t_end:
                    if not has_image_token_mask or not image_token_mask[i_s + 1]:
                        T.async_copy(kv[i_s + 1, hc_mult, :], v_smem, loop_layout=v_copy_layout)

                T.async_copy(hidden_states[i_s, :, :x_blk_d], x_smem[0, :, :], loop_layout=x_copy_layout)
                T.async_copy(kv[i_s, :hc_mult, :x_blk_d], k_smem[0, :, :], loop_layout=x_copy_layout)
                T.async_copy(weight_fused[:, :x_blk_d], w_smem[0, :, :], loop_layout=x_copy_layout)

                # Gate derivative
                dldg_r[0] = dldg_smem[head_id, 0] + dldg_smem[head_id, 1]
                dldg_r[0] = T.Select(
                    T.abs(dot_in_local) * scalar * rstd_x_local * rstd_k_local < clamp_value,
                    0.0,
                    dldg_r[0] * gate_local_hc * (1.0 - gate_local_hc) * 0.5 * T.sqrt(scalar * rstd_x_local * rstd_k_local / T.abs(dot_in_local)),
                )

                # === Pass 1b: grad_v ===
                for i in T.serial(elems_per_thread // v_vec_size):
                    T.clear(grad_v_partial)
                    for i_h in T.unroll(hc_mult):
                        for i_k in T.vectorized(v_vec_size):
                            go_v_local[i_k] = go_smem[i_h, i * threads * v_vec_size + tid * v_vec_size + i_k]
                        for i_k in T.vectorized(v_vec_size):
                            grad_v_partial[i_k] += go_v_local[i_k] * gate_local[i_h]
                    for i_k in T.vectorized(v_vec_size):
                        grad_kv[i_s, hc_mult, i * threads * v_vec_size + tid * v_vec_size + i_k] = grad_v_partial[i_k]

                dot_x_local = dot_in_local * rstd_x_local * rstd_x_local / hidden_size
                dot_k_local = dot_in_local * rstd_k_local * rstd_k_local / hidden_size

                # === Pass 2: grad_x, grad_k, grad_w — pipelined x/k, 2 warps per head ===
                x_sub_blks = x_blk_d // (threads_per_head * x_vec_size)
                num_x_tiles = hidden_size // x_blk_d

                for i_b in T.unroll(1, num_x_tiles):
                    phase = i_b % 2
                    prev = (i_b - 1) % 2

                    T.async_copy(hidden_states[i_s, :, i_b * x_blk_d : (i_b + 1) * x_blk_d], x_smem[phase, :, :], loop_layout=x_copy_layout)
                    T.async_copy(kv[i_s, :hc_mult, i_b * x_blk_d : (i_b + 1) * x_blk_d], k_smem[phase, :, :], loop_layout=x_copy_layout)
                    T.async_copy(weight_fused[:, i_b * x_blk_d : (i_b + 1) * x_blk_d], w_smem[phase, :, :], loop_layout=x_copy_layout)

                    T.ptx_wait_group(3)

                    for i_sub in T.unroll(x_sub_blks):
                        sub_off = i_sub * (threads_per_head * x_vec_size) + sub_warp_id * (warp_size * x_vec_size)
                        global_base = (i_b - 1) * x_blk_d + sub_off
                        for i_k in T.vectorized(x_vec_size):
                            go_x_local[i_k] = go_smem[head_id, global_base + lane_id * x_vec_size + i_k]
                            x_local[i_k] = x_smem[prev, head_id, sub_off + lane_id * x_vec_size + i_k]
                            k_local[i_k] = k_smem[prev, head_id, sub_off + lane_id * x_vec_size + i_k]
                            w_fused_local[i_k] = w_smem[prev, head_id, sub_off + lane_id * x_vec_size + i_k]
                        for i_k in T.vectorized(x_vec_size):
                            grad_x[i_s, head_id, global_base + lane_id * x_vec_size + i_k] = go_x_local[i_k] + dldg_r[0] * (
                                k_local[i_k] * w_fused_local[i_k] - x_local[i_k] * dot_x_local
                            )
                            grad_kv[i_s, head_id, global_base + lane_id * x_vec_size + i_k] = dldg_r[0] * (
                                x_local[i_k] * w_fused_local[i_k] - k_local[i_k] * dot_k_local
                            )
                        for i_k in T.vectorized(x_vec_size):
                            grad_w_local[((i_b - 1) * x_sub_blks + i_sub) * x_vec_size + i_k] += dldg_r[0] * x_local[i_k] * k_local[i_k]

                # Epilogue: process last x/k/w tile + prefetch next token's go
                T.ptx_wait_group(0)

                # ensure v_smem is ready and go_smem[:go_blk_d] is clean
                T.sync_threads()
                if i_s + 1 < t_end:
                    if not has_image_token_mask or not image_token_mask[i_s + 1]:
                        T.async_copy(
                            grad_out[i_s + 1, :, :go_blk_d],
                            go_smem[:, :go_blk_d],
                            loop_layout=go_copy_layout,
                        )

                for i_sub in T.unroll(x_sub_blks):
                    sub_off = i_sub * (threads_per_head * x_vec_size) + sub_warp_id * (warp_size * x_vec_size)
                    global_base = (num_x_tiles - 1) * x_blk_d + sub_off
                    for i_k in T.vectorized(x_vec_size):
                        go_x_local[i_k] = go_smem[head_id, global_base + lane_id * x_vec_size + i_k]
                        x_local[i_k] = x_smem[(num_x_tiles - 1) % 2, head_id, sub_off + lane_id * x_vec_size + i_k]
                        k_local[i_k] = k_smem[(num_x_tiles - 1) % 2, head_id, sub_off + lane_id * x_vec_size + i_k]
                        w_fused_local[i_k] = w_smem[(num_x_tiles - 1) % 2, head_id, sub_off + lane_id * x_vec_size + i_k]
                    for i_k in T.vectorized(x_vec_size):
                        grad_x[i_s, head_id, global_base + lane_id * x_vec_size + i_k] = go_x_local[i_k] + dldg_r[0] * (
                            k_local[i_k] * w_fused_local[i_k] - x_local[i_k] * dot_x_local
                        )
                        grad_kv[i_s, head_id, global_base + lane_id * x_vec_size + i_k] = dldg_r[0] * (
                            x_local[i_k] * w_fused_local[i_k] - k_local[i_k] * dot_k_local
                        )
                    for i_k in T.vectorized(x_vec_size):
                        grad_w_local[((num_x_tiles - 1) * x_sub_blks + i_sub) * x_vec_size + i_k] += dldg_r[0] * x_local[i_k] * k_local[i_k]

            # Write grad_w_local to global
            for i_reg in T.unroll(elems_per_warp_pair // x_vec_size):
                global_off = (i_reg * threads_per_head + sub_warp_id * warp_size + lane_id) * x_vec_size
                for i_k in T.vectorized(x_vec_size):
                    grad_w_partial[pid_p, head_id, global_off + i_k] = grad_w_local[i_reg * x_vec_size + i_k]
            if use_pdl:
                T.pdl_trigger()

    return engram_gate_bwd_kernel_cuda
