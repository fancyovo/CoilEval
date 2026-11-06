"""Exercise the editable tutorial workflow with real SciPy and a small fake field."""
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def example(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    path = Path(__file__).resolve().parents[1] / "examples/example2.py"
    spec = importlib.util.spec_from_file_location("user_example2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A different coil count/order exercises public parameter packing and saving.
    coefficients = np.array([[1., .1, .2], [1.1, -.1, .3]])
    coils = module.CoilSet(coefficients, coefficients * .2, coefficients * .3,
                           np.array([100., 120.]), 2)
    source = tmp_path / "coils.json"
    coils.write_json(source)
    for key, value in dict(COILS=source, DEVICE="cpu", N_STARTS=3, MAX_STEPS=2,
                           OUTPUT=tmp_path / "output").items():
        monkeypatch.setattr(module, key, value)
    return module, coils


def test_screen_failure_then_real_bfgs_progress_and_outputs(example, monkeypatch, capsys):
    import torch
    module, coils = example
    candidates = []

    class Evaluator:
        def __init__(self, **kwargs):
            self.calls = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def evaluate(self, x):
            self.calls += 1
            if self.calls <= module.N_STARTS:
                candidates.append(x.detach().clone())
            if self.calls == 1:
                raise module.InvalidEvaluationError("test candidate has no surface")
            def constant(value):
                return torch.as_tensor(value, dtype=x.dtype)
            m = SimpleNamespace(qs_volume=.01 + x[:, :-1].square().mean(),
                iota_boundary=constant(.47), coil_gap_min=constant(.11),
                coil_curvature_max=constant(5.), coil_length_mean=constant(3.5),
                relative_normal_drift_p95=constant(.001))
            return SimpleNamespace(metrics=m)

    monkeypatch.setattr(module, "TensorEvaluator", Evaluator)
    module.main()
    output = capsys.readouterr().out
    assert sum(line.startswith("screen ") for line in output.splitlines()) == 3
    assert "unscorable: test candidate" in output
    assert "from 2/3 valid starts" in output
    assert "step 001:" in output and "Optimizer success:" in output and "total:" in output
    rows = list(csv.DictReader((module.OUTPUT / "history.csv").open()))
    assert rows and all(np.isfinite(float(row["loss"])) for row in rows)
    saved = module.CoilSet.read_json(module.OUTPUT / "optimized_coils.json")
    assert saved.n_base_coils == 2 and saved.n_coefficients == 3 and saved.nfp == 2
    base = module.pack_coils(coils, device="cpu")
    assert len(candidates) == 3
    for candidate in candidates:
        torch.testing.assert_close(candidate[:, -1], base[:, -1], rtol=0, atol=0)
        torch.testing.assert_close(torch.sign(candidate[:, :-1]), torch.sign(base[:, :-1]))
    assert not torch.equal(candidates[0][:, :-1], base[:, :-1])


def test_all_invalid_candidates_report_failure_without_starting_optimizer(example, monkeypatch, capsys):
    module, _ = example

    class Evaluator:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def evaluate(self, x):
            raise module.InvalidEvaluationError("no surface")

    def unexpected_optimizer(*args, **kwargs):
        pytest.fail("Cannot optimize without a valid start")

    monkeypatch.setattr(module, "TensorEvaluator", Evaluator)
    monkeypatch.setattr(module, "minimize", unexpected_optimizer)
    module.main()
    output = capsys.readouterr().out
    assert "No valid start" in output and "Total:" in output
    assert not (module.OUTPUT / "optimized_coils.json").exists()
