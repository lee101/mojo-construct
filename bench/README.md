# bench

`pixi run bench` is the source of truth and must be run through the flocked task.
The scripts here are support tools, kept because they settle questions the table
cannot.

| Script | Settles |
|---|---|
| `kernels.py` | kernel time on preallocated buffers, for A/B of two builds |
| `crossover.py` | kernel time across a size sweep (parallel threshold) |
| `parallel_sweep.py` + `build_variants.sh` | serial vs 1/2/4/8 workers on real codecs |

## Why the table moves the kernel and barely the rows

The table times the Python caller: list/NumPy conversion, ctypes, and
`ListContainer` construction around each FFI call. Converting 1M items with
`.tolist()` alone costs ~56 ms on this machine, while the whole integer decode
kernel is ~0.6 ms. A 3x kernel win is under 2% of the row. That is the expected
outcome, not a failed optimisation: the caller is the bottleneck for every row.

## `kernels.py` - kernel-only time

Measures the exported functions on preallocated buffers, so the wrapper
disappears. Run it against two library builds to compare kernels without caller
noise:

```bash
MOJO_CONSTRUCT_LIB=/path/to/old/libmojo-construct.so PYTHONPATH=python \
  pixi run python bench/kernels.py
```

Use it as a filter, not a tie-break: a regression here that also shows in
`bench.py` is real; a few percent either way at these sub-millisecond sizes is
not.


When `pixi run bench` cannot separate a change from run-to-run noise, keep the
change only if both hold:

1. `kernels.py` shows a clear win at the real caller size (the sizes here:
   1M Int32, 750k Float64, 1.5M Byte, 500k VarInt), and
2. the benchmark row does not get worse beyond its own noise.

Otherwise revert. Do not keep a change because it should be faster.

Kernel-only numbers recorded while shipping this phase (best of several runs;
load average was 20-27 during the session, so the ranges are wide):

| Kernel | Before | After |
|---|---:|---:|
| parse 1M Int32ul | 1.63-1.99 ms | 0.59-0.76 ms |
| parse 1.5M Byte | 2.43-2.99 ms | 0.75-0.98 ms |
| build 1M Int32ul | 0.44-0.54 ms | 0.45-0.54 ms |
| parse/build 750k Float64b | 0.57-0.92 ms | 0.57-1.06 ms |
| parse 500k VarInt | 2.42-2.94 ms | 2.42-2.64 ms |
| build 500k VarInt | 1.59-1.93 ms | 1.58-1.67 ms |
## `parallel_sweep.py` and `build_variants.sh` - the parallelism experiment

`build_variants.sh` compiles the codec library with the range split across a
forced worker count (`pixi run bash bench/build_variants.sh 1:1024 2:1024 4:1024 8:1024`)
and `parallel_sweep.py` times serial against each of those, checking every
parallel result against the serial one as it goes. The result on this host is
in the README: an empty `parallelize` call from a ctypes host costs ~5 ms, so
the split loses below ~8M elements and gains only 1.2-1.3x at 16M. The path was
reverted; the scripts stay so a toolchain with a cheaper launch can be measured
the same way.

## Tie-break rule

When `pixi run bench` cannot separate a change from run-to-run noise, keep the
change only if both hold:

1. `kernels.py` shows a clear win at the real caller size (the sizes here:
   1M Int32, 750k Float64, 1.5M Byte, 500k VarInt), and
2. the benchmark row does not get worse beyond its own noise.

Otherwise revert. Do not keep a change because it should be faster.

Kernel-only numbers recorded while shipping this phase (best of several runs;
load average was 20-27 during the session, so the ranges are wide):

| Kernel | Before | After |
|---|---:|---:|
| parse 1M Int32ul | 1.63-1.99 ms | 0.59-0.76 ms |
| parse 1.5M Byte | 2.43-2.99 ms | 0.75-0.98 ms |
| build 1M Int32ul | 0.44-0.54 ms | 0.45-0.54 ms |
| parse/build 750k Float64b | 0.57-0.92 ms | 0.57-1.06 ms |
| parse 500k VarInt | 2.42-2.94 ms | 2.42-2.69 ms |
| build 500k VarInt | 1.59-1.93 ms | 1.58-1.79 ms |
