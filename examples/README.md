# Examples

[evaluate.py](evaluate.py) loads [coils.json](coils.json) and evaluates its metrics.
[custom_loss.py](custom_loss.py) composes a scalar objective and returns its full physical gradient.
Pass `--library /path/to/libcoil_eval_native.so` to either script. Use `--help` for options.

[autograd.py](autograd.py) evaluates a PyTorch objective and its gradient. Add `--batch` for independent geometries. Install the `torch` extra to run it.
