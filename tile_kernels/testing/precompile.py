"""Record and replay the compile-time arguments used by correctness tests."""

import dataclasses
import importlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def resolve(name):
    module, qualname = name.split(':', 1)
    if not module.startswith('tile_kernels.'):
        raise ValueError(f'Not a tile_kernels entry: {name}')
    value = importlib.import_module(module)
    for part in qualname.split('.'):
        value = getattr(value, part)
    return value


def encode(value):
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, tuple):
        return {'$tuple': [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: encode(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = f'{type(value).__module__}:{type(value).__qualname__}'
        return {'$dataclass': name, 'fields': {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    import torch
    from tilelang import language as T

    if isinstance(value, torch.dtype):
        return {'$torch_dtype': str(value).removeprefix('torch.')}
    if isinstance(value, T.dtype):
        return {'$tl_dtype': str(value)}
    raise TypeError(f'Unsupported compile-time argument: {type(value)}')


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if '$tuple' in value:
        return tuple(decode(v) for v in value['$tuple'])
    if '$torch_dtype' in value:
        import torch

        return getattr(torch, value['$torch_dtype'])
    if '$tl_dtype' in value:
        from tilelang import language as T

        return T.dtype(value['$tl_dtype'])
    if '$dataclass' in value:
        return resolve(value['$dataclass'])(**{k: decode(v) for k, v in value['fields'].items()})
    return {k: decode(v) for k, v in value.items()}


def capture(jit, args, kwargs):
    func = jit.func
    name = f'{func.orig_func.__module__}:{func.orig_func.__qualname__}'
    if resolve(name) is not jit:
        raise ValueError(f'JIT entry is not importable: {name}')
    # TileLang's binder extracts eager T.const values from tensor shapes/strides.
    key, tensors, arguments = func._parse_phase1_key(*args, **kwargs)
    # capture() runs after the JIT call has populated the template cache.
    template = func.p1_cache[key]
    values = template._parse_phase2_key(**tensors, **arguments)
    constants = {symbol.orig_name: value for symbol, value in zip(template.matcher or (), values, strict=True)}
    return {'kernel': name, 'arguments': encode(arguments | constants)}


def replay(kernel, configs, jobs):
    """Compile one JIT entry's recipes; return the number of failed recipes."""
    # Rebuild current code; TileLang owns argument binding and cache keys.
    jit = resolve(kernel)
    configs = [decode(config) for config in configs]
    return sum(compiled is None for compiled in jit.par_compile(configs, num_workers=min(jobs, len(configs)), ignore_error=True))


def profile_path(level):
    from tile_kernels.config import is_ascend

    backend = 'ascend' if is_ascend() else 'cuda'
    return Path(__file__).resolve().parents[2] / 'tests/precompile_profiles' / backend / f'level{level}.jsonl'


def read_profile(path):
    records = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        records.setdefault(canonical(row['recipe']), set()).update(row['tests'])
    return records


def write_profile(path, records):
    rows = [canonical({'recipe': json.loads(recipe), 'tests': sorted(tests)}) for recipe, tests in sorted(records.items())]
    Path(path).write_text(''.join(row + '\n' for row in rows))
