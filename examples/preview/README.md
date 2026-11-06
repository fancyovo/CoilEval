# Example input and historical validation

[简体中文](README.zh-CN.md)

[example1.json](example1.json) supplies four basic coils, Fourier order4 and nfp=3 field periods.
Here nfp counts repeated field periods around the device; coefficients are metres and currents amperes.

The independent entry points are [example1.py](../example1.py) and [example2.py](../example2.py):

```sh
python examples/example1.py
python examples/example2.py
```

Edit script settings and Example2's objective directly; see the [example guide](../README.md).
Version 0.11.1 replaces the old sharded reproduction program with a one-run teaching example.
The older report's116 accepted updates used a different complete objective and optimization policy;
they are not an expected trajectory for the current teaching script.
Those calculations and full physical reports remain [historical validation](../../docs/preview_validation.md). Input values are unchanged.
