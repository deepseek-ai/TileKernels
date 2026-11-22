# Root-level conftest
#
# Loads pytest plugins (CLI options, markers, fixtures).
# Plugins ship with the tile_kernels package in tile_kernels/testing/pytest/
# and are loaded here by name; a non-conftest name avoids pluggy's
# duplicate-registration error.

# ---------------------------------------------------------------------------
# GPU binding for pytest-xdist workers (MUST happen before any import that
# triggers CUDA driver initialization, e.g. tilelang).
# ---------------------------------------------------------------------------
import os
import re
import signal
import subprocess
import math

import pytest

# Point the shared pytest plugins at this repository's data directories so
# their default fallback (<tests-dir>/...) matches where this repo keeps its
# benchmark baselines and GPU memory profiles.
_TESTS_DIR = os.path.dirname(__file__)
os.environ.setdefault('TK_BENCHMARK_TESTS_DIR', _TESTS_DIR)
os.environ.setdefault('TK_BENCHMARK_BASELINES_DIR', os.path.join(_TESTS_DIR, 'benchmark_baselines'))
os.environ.setdefault('TK_GPU_MEM_PROFILES_DIR', os.path.join(_TESTS_DIR, 'gpu_mem_profiles'))
os.environ.setdefault('TK_BENCHMARK_FAIL_MODE', 'regression')
os.environ.setdefault('TK_BENCHMARK_REGRESSION_THRESHOLD', '0.05')
os.environ.setdefault('TK_BENCHMARK_MIN_DELTA_US', '0.8')


def _ignore_sigint_in_xdist_workers():
    """Make Ctrl+C interrupt pytest-xdist runs cleanly.

    When the user presses Ctrl+C, the terminal delivers SIGINT to the whole
    foreground process group — the xdist controller AND every worker.  Each
    worker then dies from KeyboardInterrupt mid-test, the controller reports
    the in-flight test as failed ('F') and *restarts* the worker (xdist
    allows up to 4 x num_workers restarts by default), so a single Ctrl+C
    does not stop the run.

    Ignoring SIGINT in workers leaves interruption to the controller, which
    handles KeyboardInterrupt by tearing down the session and terminating
    all workers via execnet (SIGTERM, then SIGKILL after a timeout).
    """
    if os.environ.get('PYTEST_XDIST_WORKER') is None:
        return
    signal.signal(signal.SIGINT, signal.SIG_IGN)


_ignore_sigint_in_xdist_workers()


def _bind_worker_gpu():
    """Set CUDA_VISIBLE_DEVICES for this xdist worker before CUDA driver init.

    tilelang (imported transitively by tile_kernels) initializes the CUDA
    driver at import time.  Once the driver is initialized, changes to
    CUDA_VISIBLE_DEVICES are ignored.  Therefore we must set the variable
    here — at the very top of conftest.py — before pytest_plugins triggers
    any tile_kernels imports.
    """
    worker_id = os.environ.get('PYTEST_XDIST_WORKER')
    if worker_id is None:
        return

    if os.path.exists('/dev/davinci_manager'):
        npu_id = int(worker_id.replace('gw', ''))

        visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
        if visible is not None and visible.strip():
            npu_list = visible.split(',')
        else:
            # Detect available NPU count without importing torch.
            try:
                result = subprocess.run(
                    ['npu-smi', 'info', '-l'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    # `npu-smi info -l` prints a "Total Count : N" header followed by a
                    # multi-line record per NPU, so count the header, not the lines.
                    match = re.search(r'Total\s+Count\s*:\s*(\d+)', result.stdout)
                    npu_list = [str(i) for i in range(int(match.group(1)))] if match else ['0']
                else:
                    npu_list = ['0']
            except (FileNotFoundError, subprocess.TimeoutExpired):
                npu_list = ['0']

        num_npus = len(npu_list)
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = npu_list[npu_id % num_npus]

    else:
        gpu_id = int(worker_id.replace('gw', ''))

        # Determine total GPU count without importing torch/tilelang (which would
        # initialize the CUDA driver and defeat the purpose).
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        gpu_list = []
        if visible is not None and visible.strip():
            gpu_list = visible.split(',')
        else:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
                    gpu_list = lines
                else:
                    gpu_list = [0]
            except (FileNotFoundError, subprocess.TimeoutExpired):
                gpu_list = [0]

        num_gpus = len(gpu_list)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list[gpu_id % num_gpus])

        # Restrict each worker's GPU memory to (total - 10 GB) / workers_per_gpu.
        # PYTEST_XDIST_WORKER_COUNT is set by pytest-xdist automatically.
        total_workers = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', '1'))
        workers_per_gpu = math.ceil(total_workers / num_gpus)
        _reserve_bytes = 10 * (1024**3)  # 10 GB reserved for system / frameworks

        import torch

        total_mem = torch.cuda.mem_get_info(0)[1]
        usable_mem = max(total_mem - _reserve_bytes, 0)
        mem_per_worker = usable_mem / workers_per_gpu
        fraction = mem_per_worker / total_mem
        fraction = max(min(fraction, 1.0), 0.0)
        torch.cuda.set_per_process_memory_fraction(fraction)


_bind_worker_gpu()
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_tile_kernels_runtime_config():
    from tile_kernels.config import reset_runtime_config

    reset_runtime_config()
    yield
    reset_runtime_config()


pytest_plugins = [
    'tile_kernels.testing.pytest.benchmark',
    'tile_kernels.testing.pytest.precompile',
    'tile_kernels.testing.pytest.random',
    'tile_kernels.testing.pytest.tilelang_warning',
    'tile_kernels.testing.pytest.xdist_failfast',
    'tile_kernels.testing.pytest.gpu_mem',
]
