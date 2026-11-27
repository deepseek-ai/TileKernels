import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_engram_hash_kernel_cuda(
    max_ngram_size: int = 3,
    num_ngram_layers: int = 2,
    num_embed_table_per_ngram: int = 8,
    has_reciprocal: bool = False,
    has_image_token_mask: bool = False,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    threads = 32
    blk_m = threads
    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram

    @T.prim_func
    def engram_hash_kernel_cuda(
        ngram_token_ids: T.Tensor[(num_tokens, max_ngram_size), T.int32],
        multipliers: T.Tensor[(num_ngram_layers, max_ngram_size), T.int64],
        vocab_sizes: T.Tensor[(num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), T.int32],
        reciprocal: T.Tensor[(num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), T.uint64],
        offsets: T.Tensor[(num_ngram_layers, num_out_cols), T.int32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        output: T.Tensor[(num_ngram_layers, num_tokens, num_out_cols), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), num_ngram_layers, threads=threads) as (pid_s, pid_h):
            if use_pdl:
                T.pdl_sync()
            tid = T.get_thread_binding()
            token_idx = pid_s * blk_m + tid
            output_local = T.alloc_local((num_out_cols,), T.int32)
            hash_local = T.alloc_local((1,), T.uint64)
            remainder = T.alloc_local((1,), T.uint32)

            if not has_reciprocal:
                reciprocal_shared = T.alloc_shared((num_out_cols,), T.uint64)
                for i in T.serial(T.ceildiv(num_out_cols, threads)):
                    col = i * threads + tid
                    if col < num_out_cols:
                        divisor = vocab_sizes[pid_h, col // num_embed_table_per_ngram, col % num_embed_table_per_ngram]
                        reciprocal_shared[col] = T.floordiv(T.uint64(0xFFFFFFFFFFFFFFFF), T.cast(divisor, T.uint64))
                T.sync_threads()
            if token_idx >= num_tokens:
                T.thread_return()

            if has_image_token_mask and image_token_mask[token_idx]:
                T.fill(output_local, -1)
            else:
                hash_local[0] = T.cast(ngram_token_ids[token_idx, 0], T.uint64) * T.cast(multipliers[pid_h, 0], T.uint64)
                for ngram_idx in T.unroll(1, max_ngram_size):
                    hash_local[0] = T.bitwise_xor(
                        hash_local[0],
                        T.cast(ngram_token_ids[token_idx, ngram_idx], T.uint64) * T.cast(multipliers[pid_h, ngram_idx], T.uint64),
                    )
                    for j in T.unroll(num_embed_table_per_ngram):
                        col = (ngram_idx - 1) * num_embed_table_per_ngram + j
                        divisor = T.cast(vocab_sizes[pid_h, ngram_idx - 1, j], T.uint32)
                        mu = reciprocal[pid_h, ngram_idx - 1, j] if has_reciprocal else reciprocal_shared[col]
                        quotient = T.call_pure_extern(T.uint64, '__umul64hi', hash_local[0], mu)
                        remainder[0] = T.cast(hash_local[0], T.uint32) - T.cast(quotient, T.uint32) * divisor
                        if remainder[0] >= divisor:
                            remainder[0] -= divisor
                        output_local[col] = T.cast(remainder[0], T.int32) + offsets[pid_h, col]
            T.copy(output_local, output[pid_h, token_idx, :])
            if use_pdl:
                T.pdl_trigger()

    return engram_hash_kernel_cuda
