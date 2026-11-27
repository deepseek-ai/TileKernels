import random
import os
from itertools import product
from typing import Iterable

import torch
from tile_kernels.utils import align
from tile_kernels.config import get_device_num_sms, get_device
from tile_kernels.rand import randn


def get_test_level() -> int:
    """0=core, 1=default, 2=full. Default 1."""
    level = int(os.getenv('TK_TEST_LEVEL', '1'))
    if level not in (0, 1, 2):
        raise ValueError(f'TK_TEST_LEVEL must be 0, 1 or 2, got {level}')
    return level


def generate_samples(level: int, **dimensions):
    """Yield the full Cartesian product at level 2, else a small zipped subset.

    Use to sample independent parameter dimensions in tests so the lower levels
    cover each value at least once without exploding into the full product.
    """
    keys = list(dimensions.keys())
    values_by_key = [list(dimensions[k]) for k in keys]
    if level >= 2:
        for combo in product(*values_by_key):
            yield dict(zip(keys, combo))
        return
    max_len = max(len(v) for v in values_by_key)
    for i in range(max_len):
        yield {k: values_by_key[idx][i % len(values_by_key[idx])] for idx, k in enumerate(keys)}


def generate_num_tokens(level: int, alignment: int = 1, backward_only: bool = False) -> list[int]:
    if level == 0 and backward_only:
        tokens = [8001]
    else:
        tokens = [512, 8001]
    if level >= 2:
        tokens = [0, 1] + tokens
    return [align(num_tokens, alignment) for num_tokens in tokens]


def generate_hidden_sizes(level: int, usage: list[str], align: int = 64) -> list[int]:
    """Return sorted, unique widths for the requested model usages. Note
    that align only takes effect for level 2 corner cases.
    """
    sizes_by_usage = {
        # Model backbone hidden width (D).
        'main': [5120],
        # FFN intermediate width (F)
        'ffn': [2304],
        # Query low-rank projection width.
        'q_lora': [1280],
        # Per-head width for indexer and attention Q/K.
        'head': [128, 512],
        # Wv output width within each group.
        'group': [1024],
        # Indexer head count.
        'indexer': [32],
        # Attention heads per Wv group times head width; Wv weight input width.
        'wv_input': [4096],
        # Wv group count times per-group output width.
        'wv': [8192],
        # Attention head count times head width.
        'attn_context': [32768],
        # Engram embedding width per hash head.
        'engram_head': [256],
        # Concatenated Engram embeddings fed into the fused KV projection.
        'engram_input': [6144],
        # Vision encoder hidden width.
        'vision': [1024],
    }
    sizes = set()
    for expression in usage:
        products = {1}
        for factor in expression.split('*'):
            factor = factor.strip()
            if factor in sizes_by_usage:
                factor_sizes = sizes_by_usage[factor]
            elif factor.isascii() and factor.isdecimal() and int(factor) > 0:
                factor_sizes = [int(factor)]
            else:
                raise ValueError(f'Invalid hidden size factor {factor!r} in {expression!r}')
            products = {left * right for left in products for right in factor_sizes}
        sizes.update(products)
    if level >= 2:
        corner_sizes = [32, 96, 128, 192, 384]
        sizes.update(size for size in corner_sizes if size % align == 0)
    return sorted(sizes)


def generate_cast_config(level: int, fmt: str) -> list[tuple[bool, bool, bool]]:
    if fmt in ('fp32', 'bf16'):
        return [(False, False, False)]
    configs = [(True, True, True), (False, True, True)]
    if level >= 1:
        configs = [(False, True, False), (False, False, False)] + configs
    return configs


def generate_num_sms(level: int) -> list[int]:
    device_num_sms = get_device_num_sms()
    sms = [device_num_sms]
    if level >= 1:
        sms = [device_num_sms - 19] + sms
    if level >= 2:
        sms = [1] + sms
    return sms


def generate_moe_params(level: int) -> Iterable[dict]:
    if level == 0:
        yield {'num_send_tokens': 4001, 'num_topk': 6, 'num_experts': 48, 'num_ep_ranks': 8}
        yield {'num_send_tokens': 4001, 'num_topk': 8, 'num_experts': 4, 'num_ep_ranks': 64}
        return

    if level >= 2:
        # Corner-case seeds
        yield {'num_send_tokens': 0, 'num_topk': 1, 'num_experts': 1, 'num_ep_ranks': 1}
        yield {'num_send_tokens': 1, 'num_topk': 8, 'num_experts': 256, 'num_ep_ranks': 8}

    extra_num_topk_list = (1, 7) if level >= 2 else ()
    extra_num_experts_list = (256, 288) if level >= 2 else ()
    extra_num_ep_ranks_list = (1, 72, 256) if level >= 2 else ()

    for num_tokens in (4001,):
        for num_topk in (2, 6, 8, 9) + extra_num_topk_list:
            for num_experts in (72, 384) + extra_num_experts_list:
                for num_ep_ranks in (8, 64) + extra_num_ep_ranks_list:
                    if num_experts % num_ep_ranks == 0:
                        yield {
                            'num_send_tokens': num_tokens,
                            'num_topk': num_topk,
                            'num_experts': num_experts // num_ep_ranks,
                            'num_ep_ranks': num_ep_ranks,
                        }


def generate_topk_idx(params: dict) -> torch.Tensor:
    num_send_tokens = params['num_send_tokens']
    num_experts = params['num_experts']
    num_topk = params['num_topk']
    num_ep_ranks = params['num_ep_ranks']

    if num_send_tokens == 0:
        return torch.empty((0, num_topk), dtype=torch.int64, device=get_device())
    scores = torch.rand((num_send_tokens * num_ep_ranks, num_experts * num_ep_ranks), dtype=torch.bfloat16, device=get_device())
    _, topk_idx = torch.topk(scores, k=num_topk, dim=-1, sorted=False)
    # NOTE: Free large tensor
    del scores
    mask = topk_idx >= num_experts
    topk_idx[mask] = -1
    mask = mask.all(dim=1)
    topk_idx = topk_idx[~mask]
    return topk_idx


def generate_psum_layout(params: dict) -> tuple[torch.Tensor, torch.Tensor]:
    num_send_tokens = params['num_send_tokens']
    num_experts = params['num_experts']
    num_topk = params['num_topk']
    num_ep_ranks = params['num_ep_ranks']
    alignment = params['alignment']

    if num_send_tokens == 0:
        return torch.empty((0,), dtype=torch.bool, device=get_device()), torch.zeros((num_experts,), dtype=torch.int32, device=get_device())
    scores = torch.rand((num_send_tokens * num_ep_ranks, num_experts * num_ep_ranks), dtype=torch.bfloat16, device=get_device())
    _, topk_idx = torch.topk(scores, k=num_topk, dim=-1, sorted=False)
    # NOTE: Free large tensor
    del scores
    mask = topk_idx < num_experts
    valid_topk = topk_idx[mask]
    counts = torch.bincount(valid_topk, minlength=num_experts)
    aligned_counts = (counts + alignment - 1) // alignment * alignment
    prefix_sum = torch.cumsum(aligned_counts, dim=0, dtype=torch.int32)
    prefix_sum_list = prefix_sum.cpu().tolist()
    counts_list = counts.cpu().tolist()
    num_expanded_tokens = prefix_sum_list[-1]
    mask = torch.zeros(num_expanded_tokens, dtype=torch.bool, device=get_device())
    for c, offset in zip(counts_list, [0] + prefix_sum_list):
        if c > 0:
            mask[offset : offset + c] = 1
    return mask, (prefix_sum - aligned_counts + counts).to(torch.int32)


def generate_rand_float(shape: tuple[int, ...]) -> torch.Tensor:
    # We want to sample from a uniform distribution over the exponent of sf
    exp = random.randint(-110, 126)
    sf = float(2**exp)
    float_tensor = randn(shape, dtype=torch.float32, device=get_device()) * sf

    mask = torch.logical_or(torch.isnan(float_tensor), torch.isinf(float_tensor))
    if mask.any():
        num_values = mask.to(torch.int32).sum().item()
        normal_values = randn((num_values,), dtype=torch.float32, device=get_device())
        float_tensor[mask] = normal_values

    max_value = torch.finfo(torch.float32).max / 8
    return torch.clamp(float_tensor, -max_value, max_value)
