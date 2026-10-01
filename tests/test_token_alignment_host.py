"""Host validation tests for global alignment state; no kernels are compiled."""

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class TokenAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Native torch modules must be loaded before sys.modules isolation.
        __import__('torch')
        root = Path(__file__).resolve().parents[1]
        language = ModuleType('tilelang.language')
        language.float8_e4m3fn, language.float4_e2m1fn = 'e4m3', 'e2m1'
        language.max_value = lambda dtype: 448 if dtype == 'e4m3' else 6
        package = ModuleType('tilelang')
        package.language = language
        self.stack.enter_context(patch.dict(sys.modules, {'tilelang': package, 'tilelang.language': language}))
        spec = importlib.util.spec_from_file_location('_config_host', root / 'tile_kernels/config.py')
        self.config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.config)
        spec = importlib.util.spec_from_file_location('_utils_host', root / 'tile_kernels/utils.py')
        self.utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.utils)

    def assert_invalid_preserves_previous(self, value, error):
        self.config.set_token_alignment(64)
        with self.assertRaises(error):
            self.config.set_token_alignment(value)
        self.assertEqual(self.config.get_token_alignment(), 64)
        self.assertEqual(self.utils.align(65, self.config.get_token_alignment()), 128)

    def test_zero_is_rejected_before_state_mutation(self):
        self.assert_invalid_preserves_previous(0, ValueError)

    def test_negative_is_rejected_before_state_mutation(self):
        self.assert_invalid_preserves_previous(-8, ValueError)

    def test_non_integral_float_is_rejected(self):
        self.assert_invalid_preserves_previous(1.5, TypeError)

    def test_string_is_rejected(self):
        self.assert_invalid_preserves_previous('64', TypeError)

    def test_positive_integer_changes_alignment(self):
        self.config.set_token_alignment(256)
        self.assertEqual(self.config.get_token_alignment(), 256)
        self.assertEqual(self.utils.align(257, self.config.get_token_alignment()), 512)

    def test_index_protocol_is_supported_and_normalized(self):
        class IndexValue:
            def __index__(self):
                return 64

        self.config.set_token_alignment(IndexValue())
        self.assertIs(type(self.config.get_token_alignment()), int)
        self.assertEqual(self.config.get_token_alignment(), 64)

    def test_reset_restores_default(self):
        self.config.set_token_alignment(256)
        self.config.reset_runtime_config()
        self.assertEqual(self.config.get_token_alignment(), 128)


if __name__ == '__main__':
    unittest.main()
