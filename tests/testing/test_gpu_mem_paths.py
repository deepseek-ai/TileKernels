"""CPU-only tests for reusable GPU-memory profile storage."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


PLUGIN_DIR = Path(__file__).resolve().parents[2] / 'tile_kernels' / 'testing' / 'pytest'


def load_storage_plugin():
    # Import the storage code without importing accelerator kernels or requiring
    # torch. The GPU measurement fixture is not invoked by these tests.
    modules = {}
    for name in ('tile_kernels', 'tile_kernels.testing', 'tile_kernels.testing.pytest'):
        modules[name] = types.ModuleType(name)
    modules['torch'] = types.ModuleType('torch')
    with patch.dict(sys.modules, modules):
        for name in ('common', 'gpu_mem'):
            qualified = f'tile_kernels.testing.pytest.{name}'
            spec = importlib.util.spec_from_file_location(qualified, PLUGIN_DIR / f'{name}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
        return module


class Config:
    def __init__(self, output):
        self.options = {'--refresh-mem': True, '--gpu-mem-output': str(output), '--gpu-mem-threshold': 15000.0}

    def getoption(self, name, default=None):
        return self.options.get(name, default)


class GpuMemPathsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = load_storage_plugin()

    def test_refresh_uses_tests_root_fallback(self):
        for variable in ('TK_TESTS_DIR', 'TK_BENCHMARK_TESTS_DIR'):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tests_dir = root / 'tests'
                records = root / 'measurements.jsonl'
                record = {'test_id': 'quant/test_cast.py::test_cast[n=2]', 'peak_gpu_mb': 32.0}
                records.write_text(json.dumps(record) + '\n')
                with patch.dict(os.environ, {variable: str(tests_dir)}, clear=True):
                    summary = self.plugin._update_gpu_mem_profiles(Config(records))
                    shard = tests_dir / 'gpu_mem_profiles' / 'quant' / 'test_cast.jsonl'
                    self.assertEqual(summary['num_added'], 1)
                    self.assertEqual(json.loads(shard.read_text()), record)
                    self.assertEqual(self.plugin._load_gpu_mem_profiles(summary['path']), {record['test_id']: {'peak_gpu_mb': 32.0}})

    def test_override_takes_precedence(self):
        with patch.dict(os.environ, {'TK_GPU_MEM_PROFILES_DIR': '/custom', 'TK_TESTS_DIR': '/tests'}, clear=True):
            self.assertEqual(self.plugin._gpu_mem_profiles_dir(None), '/custom')

    def test_new_tests_root_takes_precedence_over_legacy(self):
        with patch.dict(os.environ, {'TK_TESTS_DIR': '/new', 'TK_BENCHMARK_TESTS_DIR': '/old'}, clear=True):
            self.assertEqual(self.plugin._gpu_mem_profiles_dir(None), os.path.join('/new', 'gpu_mem_profiles'))

    def test_empty_override_uses_fallback(self):
        with patch.dict(os.environ, {'TK_GPU_MEM_PROFILES_DIR': '', 'TK_TESTS_DIR': '/tests'}, clear=True):
            self.assertEqual(self.plugin._gpu_mem_profiles_dir(None), os.path.join('/tests', 'gpu_mem_profiles'))

    def test_no_configuration_stays_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(self.plugin._gpu_mem_profiles_dir(None))


if __name__ == '__main__':
    unittest.main()
