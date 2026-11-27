"""Record normal correctness-test JIT calls; recording errors fail the refresh."""

from pathlib import Path

import pytest

from tile_kernels.testing.precompile import canonical, capture, read_profile, write_profile
from .common import item_source_file


def pytest_addoption(parser):
    parser.addoption('--record-precompile', metavar='FILE', help='Record JIT recipes')


def pytest_configure(config):
    if config.getoption('--record-precompile'):
        config.pluginmanager.register(Recorder(config), 'precompile-recorder')


class Recorder:
    def __init__(self, config):
        self.config = config
        self.records = {}
        self.current = None
        self.failed = False
        self.patch = pytest.MonkeyPatch()

    def pytest_sessionstart(self, session):
        if not hasattr(self.config, 'workerinput') and self.config.getoption('numprocesses', default=0):
            return
        from tilelang.jit import JITImpl

        original = JITImpl.__call__

        def recorded(jit, *args, **kwargs):
            result = original(jit, *args, **kwargs)
            if self.current:
                try:
                    recipe = canonical(capture(jit, args, kwargs))
                except Exception as error:
                    raise RuntimeError(f'Precompile recording failed for {jit.func.orig_func.__qualname__} in {self.current}: {error}') from error
                self.records.setdefault(recipe, set()).add(self.current)
            return result

        # __call__ includes memory-cache hits; observing compile alone misses them.
        self.patch.setattr(JITImpl, '__call__', recorded)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item):
        source = item_source_file(item)
        if source is not None and not item.get_closest_marker('benchmark'):
            self.current = source + '::' + item.nodeid.split('::', 1)[1].split('[', 1)[0]
        try:
            yield
        finally:
            self.current = None

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error):
        records = getattr(node, 'workeroutput', {}).get('precompile')
        if error or records is None:
            self.failed = True
            return
        for recipe, tests in records.items():
            self.records.setdefault(recipe, set()).update(tests)

    def pytest_sessionfinish(self, session, exitstatus):
        self.patch.undo()
        if hasattr(self.config, 'workerinput'):
            self.config.workeroutput['precompile'] = {recipe: sorted(tests) for recipe, tests in self.records.items()}
        elif self.failed:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        elif exitstatus in (0, 5):
            # Merge the regular/large-memory stages into this refresh's temporary
            # file. refresh_all.sh publishes it only after both stages succeed.
            path = Path(self.config.getoption('--record-precompile'))
            previous = read_profile(path) if path.exists() else {}
            for recipe, tests in previous.items():
                self.records.setdefault(recipe, set()).update(tests)
            write_profile(path, self.records)
