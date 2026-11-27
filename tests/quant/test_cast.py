import math
import os

import pytest
import torch

from tile_kernels.config import is_ascend
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level
from tile_kernels.testing.numeric import assert_equal, count_bytes, check_bias
from tile_kernels.testing.quant import quant_level_rank
from tile_kernels.quant import cast

os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

DTYPES = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'e4m3': torch.float8_e4m3fn,
}
SR_PAIRS = [('fp32', 'bf16'), ('fp32', 'e4m3'), ('bf16', 'e4m3')]


def generate_test_params(level: int) -> list[dict]:
    LARGE_CAST_NUMEL = 3 * 2**31  # 6,442,450,944 elements (~6.44G)
    # Level 0 is shared by correctness and performance tests; keep it 128-aligned.
    # Offsets at higher levels exercise the remaining n_mod branches.
    sizes = {
        num_tokens * hidden_size + offset
        for num_tokens in generate_num_tokens(level)
        for hidden_size in generate_hidden_sizes(level, ['main'], align=128)
        for offset in ((0,) if level == 0 else (0, 256, 128, 32, 4, 1))
    }
    sizes.update((2**27, 2**27 + 128, LARGE_CAST_NUMEL, LARGE_CAST_NUMEL + 128))
    return [
        {
            'shape': (n,),
            'in_dtype': in_dtype,
            'out_dtype': out_dtype,
            'stochastic_rounding': stochastic_rounding,
        }
        for n in sorted(sizes)
        for in_dtype in (('fp32',) if is_ascend() else DTYPES)
        for out_dtype in (('bf16',) if is_ascend() else DTYPES)
        for stochastic_rounding in ((True,) if is_ascend() else (False, True))
        if not stochastic_rounding or ((in_dtype, out_dtype) in SR_PAIRS and n % 4 == 0)
    ]


def generate_test_data(params):
    shape = params['shape']
    in_dtype = params['in_dtype']
    out_dtype = params['out_dtype']

    x = randn(shape, dtype=torch.float32)
    x = x.to(DTYPES[in_dtype])
    base_args = dict(x=x, dtype=DTYPES[out_dtype])
    return x, base_args


def cast_ref(x, dtype):
    # FP8 T.cast saturates finite overflow and infinities, and preserves NaNs.
    if dtype == torch.float8_e4m3fn:
        max_value = torch.finfo(dtype).max
        x = x.float().clamp(-max_value, max_value)
    return x.to(dtype)


@pytest.mark.parametrize('params', [p for p in generate_test_params(get_test_level()) if not p['stochastic_rounding']], ids=make_param_id)
def test_cast(params):
    # Test round-to-nearest conversion with FP8 saturation.
    x, base_args = generate_test_data(params)
    func = lambda: cast(**base_args, stochastic_rounding=False)
    func_ref = lambda x: cast_ref(x, base_args['dtype'])
    x_casted = func()

    # Bound reference memory for multi-billion-element inputs.
    for start in range(0, x.numel(), 2**20):
        ref = x[start : start + 2**20]
        x_casted_ref = func_ref(ref)
        assert_equal(x_casted[start : start + 2**20], x_casted_ref)


@pytest.mark.parametrize('params', [p for p in generate_test_params(get_test_level()) if p['stochastic_rounding']], ids=make_param_id)
def test_cast_stochastic(params):
    x, base_args = generate_test_data(params)
    func_stochastic = lambda: cast(**base_args, stochastic_rounding=True)
    func_nearest = lambda x: cast_ref(x, base_args['dtype'])

    # Check determinism.
    x1_casted = func_stochastic()
    x2_casted = func_stochastic()
    assert_equal(x1_casted, x2_casted)

    if x.numel() == 0:
        return

    # Check the complete output in chunks to bound temporary memory.
    exist_different = False
    mantissa_bits = {'bf16': 7, 'e4m3': 3}[params['out_dtype']]
    step = 2.0 ** (-mantissa_bits)
    for start in range(0, x.numel(), 2**20):
        ref = x[start : start + 2**20].float()
        stochastic_chunk = x1_casted[start : start + 2**20]
        nearest_chunk = func_nearest(x[start : start + 2**20])

        # Check if there are differences between SR and RNE
        if not exist_different and not torch.equal(stochastic_chunk.view(torch.uint8), nearest_chunk.view(torch.uint8)):
            exist_different = True

        # Check rounding direction and magnitude bias.
        stochastic = stochastic_chunk.float()
        check_bias(stochastic, ref)
        scale = ref.abs().mean()
        mean_error = (stochastic - ref).mean().abs()
        allowed_mean_error = 10 * step * scale / math.sqrt(ref.numel())
        assert mean_error <= allowed_mean_error, f'{mean_error=} {allowed_mean_error=}'

        # Check SR and RNE pick numerically adjacent quant levels.
        rank_stochastic = quant_level_rank(stochastic_chunk, params['out_dtype'])
        rank_nearest = quant_level_rank(nearest_chunk, params['out_dtype'])
        assert ((rank_stochastic - rank_nearest).abs() <= 1).all(), (
            f'SR output is not adjacent to RNE: max rank gap={(rank_stochastic - rank_nearest).abs().max().item()}'
        )

    # if numel is large, there should exist atleast one different
    if x.numel() >= 65536:
        assert exist_different, 'SR output is identical to RNE'


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_cast_benchmark(benchmark_timer, benchmark_record, params):
    x, base_args = generate_test_data(params)
    func = lambda: cast(**base_args, stochastic_rounding=params['stochastic_rounding'])
    out = func()
    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, out)
    benchmark_record(
        kernel='cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
