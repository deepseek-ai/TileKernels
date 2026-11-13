import tilelang
from tilelang.ascend import language as T


@tilelang.jit(target='pto')
def get_engram_grad_w_reduce_kernel_parallel(
    hidden_size: int,
    num_partials: int,
    hc_mult: int,
    num_vec_cores: int,
):
    vec_size = 64
    max_shards_per_head = 16
    assert hidden_size % vec_size == 0

    num_chunks = hidden_size // vec_size
    shards_per_head = max(divisor for divisor in range(1, min(max_shards_per_head, num_chunks) + 1) if num_chunks % divisor == 0)
    block_width = hidden_size // shards_per_head
    num_block_chunks = block_width // vec_size

    def sum_partials(buffer, index):
        value = T.float32(0.0)
        for partial in range(num_partials):
            value = value + buffer[partial, index]
        return value

    @T.prim_func
    def engram_grad_w_reduce_kernel_parallel(
        grad_w_partial: T.Tensor[(num_partials, hc_mult, hidden_size), T.float32],
        weight_hidden: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        weight_embed: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        grad_weight_hidden: T.Tensor[(hc_mult, hidden_size), T.float32],
        grad_weight_embed: T.Tensor[(hc_mult, hidden_size), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            partial_ub = T.alloc_shared((num_partials, block_width), T.float32)
            weight_hidden_ub = T.alloc_shared((block_width,), T.bfloat16)
            weight_embed_ub = T.alloc_shared((block_width,), T.bfloat16)
            grad_hidden_ub = T.alloc_shared((block_width,), T.float32)
            grad_embed_ub = T.alloc_shared((block_width,), T.float32)

            for shard, head in T.Persistent(
                [shards_per_head, hc_mult],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=0,
            ):
                offset = shard * block_width

                T.copy(grad_w_partial[:, head, offset : offset + block_width], partial_ub)
                T.copy(weight_hidden[head, offset : offset + block_width], weight_hidden_ub)
                T.copy(weight_embed[head, offset : offset + block_width], weight_embed_ub)
                T.copy(grad_weight_hidden[head, offset : offset + block_width], grad_hidden_ub)
                T.copy(grad_weight_embed[head, offset : offset + block_width], grad_embed_ub)

                with T.SimdVF(lanes=128):
                    for i in T.Parallel(block_width):
                        # Preserve the VMI partial accumulation order.
                        total = sum_partials(partial_ub, i)
                        wh = T.cast(weight_hidden_ub[i], 'float32')
                        we = T.cast(weight_embed_ub[i], 'float32')
                        grad_hidden_ub[i] += total * we
                        grad_embed_ub[i] += total * wh

                T.copy(grad_hidden_ub, grad_weight_hidden[head, offset : offset + block_width])
                T.copy(grad_embed_ub, grad_weight_embed[head, offset : offset + block_width])

    return engram_grad_w_reduce_kernel_parallel
