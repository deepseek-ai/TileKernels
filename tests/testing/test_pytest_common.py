"""Unit tests for the shared pytest-plugin path helper.

Only the CPU-side helper is covered here: the GPU memory and benchmark
refresh paths both key their shards through `item_source_file`, so a wrong
key silently drops every shard instead of failing.
"""

import os
from types import SimpleNamespace

import pytest

from tile_kernels.testing.pytest.common import item_source_file


def _item(path):
    return SimpleNamespace(path=path)


def test_item_source_file_is_tests_root_relative(tmp_path, monkeypatch):
    tests_root = tmp_path / 'tests'
    test_file = tests_root / 'quant' / 'test_cast.py'
    test_file.parent.mkdir(parents=True)
    test_file.write_text('')

    monkeypatch.setenv('TK_TESTS_DIR', str(tests_root))
    monkeypatch.delenv('TK_BENCHMARK_TESTS_DIR', raising=False)

    assert item_source_file(_item(test_file)) == os.path.join('quant', 'test_cast.py')


def test_item_source_file_ignores_items_outside_the_tests_root(tmp_path, monkeypatch):
    tests_root = tmp_path / 'tests'
    tests_root.mkdir()
    outside = tmp_path / 'other' / 'test_cast.py'
    outside.parent.mkdir()
    outside.write_text('')

    monkeypatch.setenv('TK_TESTS_DIR', str(tests_root))
    monkeypatch.delenv('TK_BENCHMARK_TESTS_DIR', raising=False)

    assert item_source_file(_item(outside)) is None


def test_item_source_file_resolves_symlinked_tests_root(tmp_path, monkeypatch):
    """A symlinked tests root must not silently disable shard keys.

    pytest reports `item.path` with symlinks resolved, while the tests root
    may be exported in its symlinked form (for example
    `TK_TESTS_DIR=$PWD/tests` from a checkout under a symlinked path).
    """
    real_root = tmp_path / 'real'
    tests_root = real_root / 'tests'
    test_file = tests_root / 'quant' / 'test_cast.py'
    test_file.parent.mkdir(parents=True)
    test_file.write_text('')

    link_root = tmp_path / 'link'
    try:
        os.symlink(real_root, link_root)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform dependent
        pytest.skip(f'symlinks unavailable: {exc}')

    monkeypatch.setenv('TK_TESTS_DIR', str(link_root / 'tests'))
    monkeypatch.delenv('TK_BENCHMARK_TESTS_DIR', raising=False)

    assert item_source_file(_item(test_file)) == os.path.join('quant', 'test_cast.py')
