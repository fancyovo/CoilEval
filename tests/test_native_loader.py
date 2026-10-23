from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import pytest

from coil_eval import EvaluationConfig, Evaluator
from coil_eval import _native_loader as loader


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, '_DIRECTORY', tmp_path / '_native')
    monkeypatch.setattr(loader, '_READY', False)
    monkeypatch.setattr(loader, '_HANDLES', [])
    calls = []
    monkeypatch.setattr(loader.ctypes, 'CDLL', lambda path, **kw: calls.append(path) or object())
    monkeypatch.setattr(loader.metadata, 'distribution', lambda name:
                        SimpleNamespace(locate_file=lambda path: tmp_path/path))
    return calls


def test_explicit_library_preserves_custom_runtime(runtime, tmp_path):
    path = tmp_path/'custom.so'
    assert loader.resolve_library(path) == path.resolve()
    assert runtime == []


def test_missing_packaged_backend_has_installation_guidance(runtime):
    with pytest.raises(FileNotFoundError, match='binary wheel'):
        loader.resolve_library()
    assert runtime == []


def test_packaged_runtime_is_loaded_once_across_threads(runtime):
    loader._DIRECTORY.mkdir()
    lib=loader._DIRECTORY/'libcoil_eval_native.so'
    lib.touch()
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths=list(pool.map(lambda _:loader.resolve_library(),range(8)))
    assert paths == [lib]*8
    assert [Path(p).name for p in runtime] == [p[2] for p in loader._RUNTIME_LIBRARIES]


def test_missing_runtime_is_actionable_and_retryable(runtime, monkeypatch):
    loader._DIRECTORY.mkdir()
    (loader._DIRECTORY/'libcoil_eval_native.so').touch()
    def missing(name):
        raise loader.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(loader.metadata, 'distribution', missing)
    with pytest.raises(RuntimeError, match='dependencies enabled'):
        loader.resolve_library()
    assert not loader._READY


def test_evaluator_uses_packaged_library(runtime, monkeypatch):
    from coil_eval import _backend
    loader._DIRECTORY.mkdir()
    lib=loader._DIRECTORY/'libcoil_eval_native.so'
    lib.touch()
    seen=[]
    monkeypatch.setattr(_backend, 'NativeBackend',lambda path,device_id:seen.append((path,device_id)))
    Evaluator(config=EvaluationConfig(target_volume=.05),device_id=2)
    assert seen == [(lib,2)]


def test_backend_injection_needs_no_native_library(runtime):
    backend=object()
    assert Evaluator(config=EvaluationConfig(target_volume=.05),backend=backend)._backend is backend
    assert runtime == []
