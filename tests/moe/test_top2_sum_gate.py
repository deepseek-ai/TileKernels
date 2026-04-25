import os

import torch
import torch.nn.functional as F
import pytest

import tile_kernels
from tile_kernels.moe.scoring import ScoringFunc
from tile_kernels.testing.generator import generate_num_tokens
from tile_kernels.testing.numeric import count_bytes, assert_equal
from tile_kernels.testing.bench import make_param_id

from tile_kernels.torch import topk_sum_and_topk_group_idx as torch_topk_sum_and_topk_group_idx
from tile_kernels.torch import top2_sum_gate as torch_top2_sum_gate
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# HIP fixes: T.sync_warp() now supported (compiler memory fence).
#            T.alloc_var(init=0) now generates initialization code.


_CONFIGS = [
    (0, 0, 72, 1, 6),
    (0, 0, 32, 2, 6),
    (0, 0, 64, 2, 6),
    (0, 0, 96, 2, 6),
    (0, 0, 16, 2, 6),
    (0, 0, 36, 2, 6),
    (0, 0, 108, 2, 6),
    (0, 0, 128, 2, 6),
    (0, 0, 144, 2, 6),
    (8, 8, 256, 2, 8),
    (8, 4, 256, 2, 8),
]


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {
            'num_groups': num_groups,
            'num_topk_groups': num_topk_groups,
            'num_routed_experts': num_routed_experts,
            'num_shared_experts': num_shared_experts,
            'num_topk': num_topk,
        }
        for num_groups, num_topk_groups, num_routed_experts, num_shared_experts, num_topk in _CONFIGS
    ]
    if is_benchmark:
        scoring_funcs = [sf for sf in ScoringFunc if sf != ScoringFunc.IDENTITY]
        params = [
            {**p, 'scoring_func': sf}
            for p in params
            for sf in scoring_funcs
        ]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_top2_sum_gate(params):
    num_groups = params['num_groups']
    num_topk_groups = params['num_topk_groups']
    num_routed_experts = params['num_routed_experts']
    num_shared_experts = params['num_shared_experts']
    num_topk = params['num_topk']

    num_extra_experts = 32  # Only enabled for `use_shared_as_routed`
    num_group_sum_topk = 2
    routed_scaling_factor = 1.5
    num_ep_ranks, num_tp_ranks = 4, 2
    ep_rank = 0
    assert num_routed_experts % num_ep_ranks == 0 and (num_routed_experts + num_extra_experts) % num_ep_ranks == 0

    def get_group_masked_scores(scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        scores.add_(bias.unsqueeze(0))
        if num_topk_groups != num_groups:
            group_idx = torch_topk_sum_and_topk_group_idx(scores.view(num_tokens, num_groups, num_routed_experts // num_groups), num_group_sum_topk, num_topk_groups)
            group_mask = scores.new_ones(num_tokens, num_groups, dtype=torch.bool).scatter_(1, group_idx, False)
            score_mask = group_mask.unsqueeze(-1).expand(num_tokens, num_groups, num_routed_experts // num_groups).reshape(scores.size(0), num_routed_experts)
            scores = scores.masked_fill(score_mask, float('-inf'))
        return scores

    def get_scores(logits: torch.Tensor, scoring_func: ScoringFunc) -> torch.Tensor:
        if scoring_func == ScoringFunc.SIGMOID:
            return logits.sigmoid()
        elif scoring_func == ScoringFunc.SOFTMAX:
            return logits.softmax(dim=-1)
        elif scoring_func == ScoringFunc.SQRTSOFTPLUS:
            return F.softplus(logits).sqrt()
        else:
            raise ValueError(f'Unknown scoring function: {scoring_func}')

    def construct_input(scoring_func: ScoringFunc):
        while True:
            logits = torch.randn((num_tokens + num_padded_tokens, num_routed_experts), dtype=torch.float, device='cuda')
            bias = torch.randn(num_routed_experts, dtype=torch.float, device='cuda')
            scores = get_scores(logits[:num_tokens], scoring_func)

            # NOTES: We expect the top-k group to be stable, and since the internal kernel uses low-precision operations,
            # the generated data needs to have a sufficient gap between the k-th largest and
            # the (k+1)-th largest values to ensure that the selected top-k groups are definitive
            if num_topk_groups != num_groups:
                # NOTES: softmax has different behavior than other functions, so we need to handle it separately
                if scoring_func != ScoringFunc.SOFTMAX:
                    scores_ref = scores + bias.unsqueeze(0)
                else:
                    scores_ref = logits[:num_tokens] + bias.unsqueeze(0)

                group_scores_ref = scores_ref.view(num_tokens, num_groups, num_routed_experts // num_groups).topk(num_group_sum_topk, dim=-1, sorted=False).values.sum(-1)
                group_scores_ref, _ = group_scores_ref.sort(dim=-1, descending=True)
                not_equal = group_scores_ref[:, num_topk_groups - 1] - group_scores_ref[:, num_topk_groups] > 1e-6
                if not torch.all(not_equal):
                    continue

            scores = get_group_masked_scores(scores, bias)
            topk_weights_ref, _ = torch.topk(scores, k=num_topk + 1, dim=-1, sorted=True)

            # NOTES: We expect the top-k results to be stable, and since the internal kernel uses low-precision operations,
            # the generated data needs to have a sufficient gap between the top-k values.
            not_equal = topk_weights_ref[:, : num_topk - 1] - topk_weights_ref[:, 1:num_topk] > 2e-6
            if torch.all(not_equal):
                break
        return logits, bias

    # noinspection PyShadowingNames
    def get_kwargs(use_shared_as_routed: bool, num_extra_experts: int, num_tokens: int, num_padded_tokens: int, fix_routing: bool):
        mask = None
        if num_padded_tokens > 0:
            mask = torch.ones(num_tokens + num_padded_tokens, dtype=torch.bool, device='cuda')
            mask[-num_padded_tokens:] = False

        to_physical_map = None
        logical_count = None
        fix_routing_mask = None
        unmapped_topk_idx = torch.zeros((num_tokens + num_padded_tokens, num_topk), dtype=torch.int64, device='cuda')

        # Test to physical map
        if use_shared_as_routed:
            assert num_shared_experts <= num_extra_experts
            to_physical_map = torch.arange(0, num_routed_experts + num_shared_experts, dtype=torch.int, device='cuda')
            to_physical_map = to_physical_map.view(-1, 1).expand(-1, num_extra_experts + 1).contiguous()
            logical_count = torch.ones((num_routed_experts + num_shared_experts,), dtype=torch.int, device='cuda')

        # Test fix routing mask
        if fix_routing:
            fix_routing_mask = torch.ones((num_tokens + num_padded_tokens,), dtype=torch.bool, device='cuda')
            # NOTES: Use separate generator to generate the same value for same (num_tokens, num_topk)
            unmapped_topk_idx_generator = torch.Generator(device='cuda').manual_seed(42)
            unmapped_topk_idx = torch.randint(
                0,
                num_routed_experts,
                (num_tokens + num_padded_tokens, num_topk),
                generator=unmapped_topk_idx_generator,
                dtype=torch.int64,
                device='cuda',
            )
        return dict(
            mask=mask,
            fix_routing_mask=fix_routing_mask,
            to_physical_map=to_physical_map,
            logical_count=logical_count,
            unmapped_topk_idx=unmapped_topk_idx,
        )

    # Correctness test
    use_shared_as_routed_valid = num_topk % num_shared_experts == 0 and num_routed_experts % (num_topk // num_shared_experts) == 0
    for num_tokens in generate_num_tokens():
        for num_padded_tokens in (0, 10):
            for tp_rank in range(0, num_tp_ranks):
                for use_shared_as_routed in (False, True) if use_shared_as_routed_valid else (False,):
                    for scoring_func in ScoringFunc:
                        if scoring_func == ScoringFunc.IDENTITY:
                            continue
                        for fix_routing in (False, True):
                            logits, bias = construct_input(scoring_func)
                            args = (
                                logits,
                                bias,
                                num_topk,
                                num_topk_groups,
                                num_groups,
                                use_shared_as_routed,
                                num_shared_experts,
                                routed_scaling_factor,
                                ep_rank,
                                num_ep_ranks,
                                tp_rank,
                                num_tp_ranks,
                                str(scoring_func),
                            )
                            kwargs = get_kwargs(use_shared_as_routed, num_extra_experts, num_tokens, num_padded_tokens, fix_routing)
                            kwargs_ref = get_kwargs(use_shared_as_routed, num_extra_experts, num_tokens, num_padded_tokens, fix_routing)

                            topk_idx, topk_weights= tile_kernels.moe.top2_sum_gate(*args, **kwargs)
                            topk_idx_ref, topk_weights_ref = torch_top2_sum_gate(*args, **kwargs_ref)

                            unmapped_topk_idx = kwargs['unmapped_topk_idx']
                            unmapped_topk_idx_ref = kwargs_ref['unmapped_topk_idx']

                            sorted_topk_idx, _ = topk_idx.sort(dim=1)
                            sorted_topk_idx_ref, _ = topk_idx_ref.sort(dim=1)

                            sorted_unmapped_topk_idx = unmapped_topk_idx.sort(dim=1)[0]
                            sorted_unmapped_topk_idx_ref = unmapped_topk_idx_ref.sort(dim=1)[0]

                            sorted_topk_weights, _ = topk_weights.sort(dim=1)
                            sorted_topk_weights_ref, _ = topk_weights_ref.sort(dim=1)

                            assert_equal(sorted_topk_idx, sorted_topk_idx_ref)
                            assert_equal(sorted_unmapped_topk_idx, sorted_unmapped_topk_idx_ref)
                            assert torch.allclose(sorted_topk_weights, sorted_topk_weights_ref), (
                                f'{sorted_topk_weights=}\n'
                                f'{sorted_topk_weights_ref=}\n'
                                f'{scoring_func=}, {num_groups=}, {num_topk_groups=}, {num_routed_experts=}, {num_shared_experts=}, {num_topk=}, {fix_routing=}\n'
                                f'Different topk weights: \n'
                                f'{[(sorted_topk_weights[i], sorted_topk_weights_ref[i]) for i in range(topk_weights.size(0)) if not torch.equal(sorted_topk_weights[i], sorted_topk_weights_ref[i])]}'
                            )

    # Check sort stability
    tp_rank = 0
    logits = torch.zeros((num_tokens + num_padded_tokens, num_routed_experts), dtype=torch.float, device='cuda')
    bias = torch.zeros(num_routed_experts, dtype=torch.float, device='cuda')
    args = (
        logits,
        bias,
        num_topk,
        num_topk_groups,
        num_groups,
        use_shared_as_routed,
        num_shared_experts,
        routed_scaling_factor,
        ep_rank,
        num_ep_ranks,
        tp_rank,
        num_tp_ranks,
        str(ScoringFunc.SIGMOID),
    )
    fix_routing = False
    kwargs = get_kwargs(use_shared_as_routed, num_extra_experts, num_tokens, num_padded_tokens, fix_routing)
    topk_idx, topk_weights = tile_kernels.moe.top2_sum_gate(*args, **kwargs)
    num_experts_on_rank = min(num_routed_experts // num_ep_ranks, num_topk)
    assert torch.all(
        topk_idx[:num_tokens, :num_experts_on_rank] == torch.arange(0, num_experts_on_rank, dtype=topk_idx.dtype, device=topk_idx.device)
    )


def generate_benchmark_test_case(params):
    num_groups = params['num_groups']
    num_topk_groups = params['num_topk_groups']
    num_routed_experts = params['num_routed_experts']
    num_shared_experts = params['num_shared_experts']
    num_topk = params['num_topk']

    num_extra_experts = 32
    num_group_sum_topk = 2
    routed_scaling_factor = 1.5
    num_ep_ranks, num_tp_ranks = 4, 2
    ep_rank = 0

    use_shared_as_routed_valid = num_topk % num_shared_experts == 0 and num_routed_experts % (num_topk // num_shared_experts) == 0
    use_shared_as_routed = use_shared_as_routed_valid

    num_tokens, num_padded_tokens = 32, 4
    tp_rank = num_tp_ranks - 1
    fix_routing = False

    logits = torch.randn((num_tokens + num_padded_tokens, num_routed_experts), dtype=torch.float, device='cuda')
    bias = torch.randn(num_routed_experts, dtype=torch.float, device='cuda')

    # noinspection PyShadowingNames
    def get_kwargs(use_shared_as_routed, num_extra_experts, num_tokens, num_padded_tokens, fix_routing):
        mask = None
        if num_padded_tokens > 0:
            mask = torch.ones(num_tokens + num_padded_tokens, dtype=torch.bool, device='cuda')
            mask[-num_padded_tokens:] = False

        to_physical_map = None
        logical_count = None
        fix_routing_mask = None
        unmapped_topk_idx = torch.zeros((num_tokens + num_padded_tokens, num_topk), dtype=torch.int64, device='cuda')

        if use_shared_as_routed:
            assert num_shared_experts <= num_extra_experts
            to_physical_map = torch.arange(0, num_routed_experts + num_shared_experts, dtype=torch.int, device='cuda')
            to_physical_map = to_physical_map.view(-1, 1).expand(-1, num_extra_experts + 1).contiguous()
            logical_count = torch.ones((num_routed_experts + num_shared_experts,), dtype=torch.int, device='cuda')

        if fix_routing:
            fix_routing_mask = torch.ones((num_tokens + num_padded_tokens,), dtype=torch.bool, device='cuda')
            unmapped_topk_idx_generator = torch.Generator(device='cuda').manual_seed(42)
            unmapped_topk_idx = torch.randint(
                0,
                num_routed_experts,
                (num_tokens + num_padded_tokens, num_topk),
                generator=unmapped_topk_idx_generator,
                dtype=torch.int64,
                device='cuda',
            )
        return dict(
            mask=mask,
            fix_routing_mask=fix_routing_mask,
            to_physical_map=to_physical_map,
            logical_count=logical_count,
            unmapped_topk_idx=unmapped_topk_idx,
        )

    kwargs = get_kwargs(use_shared_as_routed, num_extra_experts, num_tokens, num_padded_tokens, fix_routing)

    return {
        'logits': logits,
        'bias': bias,
        'num_topk': num_topk,
        'num_topk_groups': num_topk_groups,
        'num_groups': num_groups,
        'use_shared_as_routed_valid': use_shared_as_routed_valid,
        'num_shared_experts': num_shared_experts,
        'routed_scaling_factor': routed_scaling_factor,
        'ep_rank': ep_rank,
        'num_ep_ranks': num_ep_ranks,
        'tp_rank': tp_rank,
        'num_tp_ranks': num_tp_ranks,
        'kwargs': kwargs,
    }


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_top2_sum_gate_benchmark(benchmark_timer, benchmark_record, params):
    scoring_func = params['scoring_func']

    tc = generate_benchmark_test_case(params)
    logits = tc['logits']
    bias = tc['bias']
    kwargs = tc['kwargs']

    args = (
        logits,
        bias,
        tc['num_topk'],
        tc['num_topk_groups'],
        tc['num_groups'],
        tc['use_shared_as_routed_valid'],
        tc['num_shared_experts'],
        tc['routed_scaling_factor'],
        tc['ep_rank'],
        tc['num_ep_ranks'],
        tc['tp_rank'],
        tc['num_tp_ranks'],
        str(scoring_func),
    )

    t_us = benchmark_timer(lambda: tile_kernels.moe.top2_sum_gate(*args, **kwargs))
    num_bytes = count_bytes(logits, bias)
    bandwidth_gbs = num_bytes / t_us / 1e3

    benchmark_record(
        kernel='top2_sum_gate',
        operation='fwd',
        params={**params, 'scoring_func': str(scoring_func)},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
