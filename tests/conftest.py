# Root-level conftest
#
# Loads the benchmark plugin (CLI options, markers, fixtures).
# The plugin lives in a file deliberately NOT named conftest.py to
# avoid pluggy's duplicate-registration error.

from tilelang.utils.target import determine_target

pytest_plugins = [
    'tests.pytest_random_plugin',
    'tests.pytest_benchmark_plugin',
]

# Condition variable: True when running on AMD/HIP (e.g. MI350), False on NVIDIA/CUDA.
# Used by individual test files to filter out NV-only features (FP4/e2m1, TMA-aligned SF,
# packed UE8M0, get_warp_idx) that are not supported on HIP targets.
IS_HIP: bool = determine_target(return_object=True).kind.name == 'hip'
