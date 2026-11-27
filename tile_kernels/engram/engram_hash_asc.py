import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


@T.macro
def estimate_quotient_lo32(a_lo, a_hi, b_lo, b_hi):
    """Low 32 bits of a quotient estimate, at most two below a // d.

    Requires 0 <= a < 2**63, 0 < d < 2**31, b = (2**64 - 1) // d.
    """
    p01_lo, p01_hi = S.vmull(a_lo, b_hi)
    p10_lo, p10_hi = S.vmull(a_hi, b_lo)
    carry, _ = S.vaddc(p01_lo, p10_lo)
    _, high = S.vaddcs(p01_hi, p10_hi, carry)
    quotient = S.alloc_local((1,), T.uint32)
    quotient[0] = high
    S.vmula(quotient[0], a_hi, b_hi)
    return quotient[0]


@tilelang.jit
def get_engram_hash_kernel_asc(
    max_ngram_size: int,
    num_ngram_layers: int,
    num_embed_table_per_ngram: int,
    has_reciprocal: bool,
    has_image_token_mask: bool,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram
    vec_size = 64

    @T.prim_func
    def engram_hash_kernel_asc(
        ngram_token_ids: T.Tensor[(num_tokens, max_ngram_size), T.int32],
        multipliers: T.Tensor[(num_ngram_layers, max_ngram_size), T.int64],
        vocab_sizes: T.Tensor[(num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), T.int32],
        reciprocal: T.Tensor[(num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), T.uint64],
        offsets: T.Tensor[(num_ngram_layers, num_out_cols), T.int32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        output: T.Tensor[(num_ngram_layers, num_tokens, num_out_cols), T.int32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            token_ids_ub = T.alloc_shared((vec_size, max_ngram_size), T.int32)
            multipliers_ub = T.alloc_shared((num_ngram_layers, max_ngram_size * 2), T.uint32)
            vocab_sizes_ub = T.alloc_shared((num_ngram_layers, num_out_cols), T.uint32)
            reciprocal_ub = T.alloc_shared((num_ngram_layers, num_out_cols), T.uint64)
            offsets_ub = T.alloc_shared((num_ngram_layers, num_out_cols), T.int32)
            image_token_mask_ub = T.alloc_shared((vec_size,), T.uint8)
            output_ub = T.alloc_shared((num_ngram_layers, vec_size, num_out_cols), T.int32)
            reciprocal_words = T.view(reciprocal_ub, (num_ngram_layers, num_out_cols * 2), T.uint32)
            T.annotate_buffer_versions({token_ids_ub: 1, image_token_mask_ub: 1, output_ub: 1})

            T.copy(T.view(multipliers, (num_ngram_layers, max_ngram_size * 2), T.uint32), multipliers_ub)
            T.copy(T.view(vocab_sizes, (num_ngram_layers, num_out_cols), T.uint32), vocab_sizes_ub)
            T.copy(offsets, offsets_ub)
            if has_reciprocal:
                T.copy(T.view(reciprocal, (num_ngram_layers, num_out_cols)), reciprocal_ub)
            else:
                with T.SimtVF(threads=128):
                    for layer, col in T.Parallel(num_ngram_layers, num_out_cols):
                        reciprocal_ub[layer, col] = T.floordiv(T.uint64(0xFFFFFFFFFFFFFFFF), T.cast(vocab_sizes_ub[layer, col], T.uint64))

            for tile_id in T.Persistent(
                [T.ceildiv(num_tokens, vec_size)],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=0,
            ):
                row = tile_id * vec_size
                T.copy(ngram_token_ids[row : row + vec_size, :], token_ids_ub)
                if has_image_token_mask:
                    T.copy(T.view(image_token_mask, (num_tokens,), T.uint8)[row : row + vec_size], image_token_mask_ub)

                with T.SimdVF():
                    lanes = T.reinterpret(S.vci(0, T.int32), 'uint32x64')
                    token_indices = S.vmuls(lanes, max_ngram_size)
                    output_indices = S.vmuls(lanes, num_out_cols)
                    hash_words = S.alloc_local((2,), T.uint32)
                    product_high = S.alloc_local((1,), T.uint32)
                    remainder = S.alloc_local((1,), T.uint32)
                    result = S.alloc_local((1,), T.int32)
                    if has_image_token_mask:
                        image = T.reinterpret(S.vld(image_token_mask_ub[0], dist='UNPK4_B8'), 'uint32x64')
                        is_image = S.vcmps(image, 0, op='ne')
                    token0 = T.reinterpret(S.vgather2(token_ids_ub[0, 0], token_indices), 'uint32x64')
                    for layer in T.unroll(num_ngram_layers):
                        multiplier0_lo = S.vld(multipliers_ub[layer, 0], dist='BRC_B32')
                        multiplier0_hi = S.vld(multipliers_ub[layer, 1], dist='BRC_B32')
                        initial_lo, initial_hi = S.vmull(token0, multiplier0_lo)
                        hash_words[0] = initial_lo
                        hash_words[1] = initial_hi
                        S.vmula(hash_words[1], token0, multiplier0_hi)
                        for ngram in T.unroll(1, max_ngram_size, explicit=True):
                            token = T.reinterpret(S.vgather2(token_ids_ub[0, ngram], token_indices), 'uint32x64')
                            multiplier_lo = S.vld(multipliers_ub[layer, ngram * 2], dist='BRC_B32')
                            multiplier_hi = S.vld(multipliers_ub[layer, ngram * 2 + 1], dist='BRC_B32')
                            product_lo, product_hi = S.vmull(token, multiplier_lo)
                            hash_words[0] = S.vxor(hash_words[0], product_lo)
                            product_high[0] = product_hi
                            S.vmula(product_high[0], token, multiplier_hi)
                            hash_words[1] = S.vxor(hash_words[1], product_high[0])
                            for table in T.serial(num_embed_table_per_ngram):
                                col = (ngram - 1) * num_embed_table_per_ngram + table
                                divisor = S.vld(vocab_sizes_ub[layer, col], dist='BRC_B32')
                                mu_lo = S.vld(reciprocal_words[layer, col * 2], dist='BRC_B32')
                                mu_hi = S.vld(reciprocal_words[layer, col * 2 + 1], dist='BRC_B32')
                                quotient = estimate_quotient_lo32(hash_words[0], hash_words[1], mu_lo, mu_hi)
                                remainder[0] = S.vsub(hash_words[0], S.vmul(quotient, divisor))
                                remainder[0] = S.vmin(remainder[0], S.vsub(remainder[0], divisor))
                                remainder[0] = S.vmin(remainder[0], S.vsub(remainder[0], divisor))
                                result[0] = S.vadd(T.reinterpret(remainder[0], 'int32x64'), S.vld(offsets_ub[layer, col], dist='BRC_B32'))
                                if has_image_token_mask:
                                    result[0] = S.vsel(S.vdup(-1, T.int32), result[0], is_image)
                                S.vscatter(result[0], output_ub[layer, 0, col], output_indices)

                T.copy(output_ub, output[:, row : row + vec_size, :])

    return engram_hash_kernel_asc
