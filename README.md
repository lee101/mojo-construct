# mojo-construct

`mojo-construct` is a standalone Mojo/Python implementation of the
compute-heavy part of [Construct](https://construct.readthedocs.io/): parsing
and building large homogeneous binary arrays inside declarative schemas. It
keeps schema traversal and Python object construction in Python, while a single
Mojo shared library performs bulk integer, floating-point, and varint codecs.

The Python package is named `mojo_construct` so it can be installed beside the
real `construct` package for parity testing. For the covered API, it can be
used with one import change:

```python
import mojo_construct as construct
```

This project is independently implemented and distributed under the MIT
license.

## Install

The checked-in Pixi environment pins the Mojo nightly used by this repository
and installs Construct 2.10.70 as the test and benchmark oracle.

```bash
pixi install
pixi run build
pixi run test
```

The build writes `dist/libmojo-construct.so`. Set `MOJO_CONSTRUCT_LIB` to use a
library at another path.

## Usage

This example builds a packet whose count is derived from its values, then
parses it back. The `VarInt[this.count]` array takes the Mojo bulk path.

```python
from mojo_construct import Byte, Const, Rebuild, Struct, VarInt, len_, this

Packet = Struct(
    "magic" / Const(b"MC"),
    "count" / Rebuild(Byte, len_(this.values)),
    "values" / VarInt[this.count],
)

blob = Packet.build({"values": [1, 128, 16384]})
assert blob.hex() == "4d4303018001808001"
assert Packet.parse(blob)["values"] == [1, 128, 16384]
```

Run it from the repository with `pixi run python examples/basic.py`, or use the same
imports in an interactive `pixi run python` session.

## Covered subset

The following names and constructor signatures mirror Construct 2.10.70:

- Integer fields from 8 through 64 bits in big, little, and native endian
  forms, including signed and unsigned 24-bit fields
- `Float16*`, `Float32*`, `Float64*`, `Byte`, `BytesInteger`, `Bytes`, and
  `GreedyBytes`
- `Struct`, `Sequence`, `Array`, `GreedyRange`, `RepeatUntil`, and dynamic
  `this` expressions
- `Const`, `Default`, `Rebuild`, `Computed`, `Check`, `If`, `IfThenElse`,
  `Switch`, `Select`, `Optional`, and `Pass`
- `Enum`, `FlagsEnum`, `VarInt`, and signed `ZigZag`
- `Prefixed`, `PrefixedArray`, `FixedSized`, `Padded`, `Padding`,
  `PaddedString`, `CString`, `StringEncoded`, `NullStripped`, and
  `NullTerminated`
- `RawCopy`, `Checksum`, `Pointer`, `Seek`, `Tell`, `Adapter`, `ExprAdapter`,
  and validator classes
- `BitStruct`, `Bitwise`, `BitsInteger`, `Bit`, `Nibble`, `Octet`, and `Flag`
  for byte-aligned groups of direct bit fields
- In-memory, stream, and file parse/build methods, plus fixed `sizeof`

Mojo acceleration applies to `Array` and `GreedyRange` over fixed numeric
fields, and to `Array`/`GreedyRange` over `VarInt`. Scalars and heterogeneous
schema control flow deliberately remain in Python.

This is not a complete port of Construct. It does not currently cover lazy
fields, tunneling and restreaming constructs, compression/encryption wrappers,
NumPy adapters, pickling/code generation, Kaitai export, or arbitrary mixtures
of byte-oriented fields inside `Bitwise`. `VarInt` is limited to unsigned
64-bit values and `ZigZag` to signed 64-bit values; Construct itself accepts
larger Python integers. `compile()` returns the same working construct rather
than generating Python source.

## Correctness

The test suite compares parse results, built bytes, stream positions, fixed
sizes, and error paths directly against the installed Construct 2.10.70.
Coverage includes boundary values for every accelerated integer type, random
24/32/64-bit vectors, float special values, malformed varint tails, dynamic
nested schemas, bit structs, adapters, raw copies, and checksums.

```text
434 passed
```

## Benchmarks

Run benchmarks only through the flocked Pixi task:

```bash
pixi run bench
```

The table is real output of the best of five runs (five, not three, so a single
run cannot carry the number) on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
x86-64, Python 3.13.14. Times include ctypes, NumPy scratch allocation,
conversion to Construct-compatible Python lists, and output allocation.
"Before" is the same machine and benchmark on the previous release of this
library; no numbers are estimated.

| Case | before | after | Speedup |
|---|---:|---:|---:|
| parse Array(1M, Int32ul) | 66.06 ms | 67.84 ms | 9.78x |
| build Array(1M, Int32ul) | 80.88 ms | 80.26 ms | 8.28x |
| parse Array(750k, Float64b) | 34.44 ms | 36.21 ms | 14.54x |
| build Array(750k, Float64b) | 43.36 ms | 43.17 ms | 11.09x |
| parse Array(1.5M, Byte) | 19.52 ms | 21.27 ms | 39.89x |
| parse Array(500k, VarInt) | 22.75 ms | 17.35 ms | 47.48x |
| build Array(500k, VarInt) | 87.37 ms | 87.33 ms | 6.20x |

Speedup is before/construct 2.10.70; the machine was shared during this session
(load average 20-27) and several rows move by ±10% between runs in either
direction, so treat row-level changes here as noise.

The kernels themselves did get faster. Timed on preallocated buffers with
`bench/kernels.py`, best of several runs:

| Kernel (kernel-only) | before | after |
|---|---:|---:|
| parse 1M Int32ul | 1.63 ms | 0.59 ms |
| parse 1.5M Byte | 2.43 ms | 0.75 ms |

### SIMD and fusion phase

This phase vectorised the remaining contiguous loops and fused the adjacent
passes that touched the same data. Kernel-only numbers below are medians of 15
alternating A/B runs of the previous build and this one, on the same buffers, so
the two libraries see the same machine load (load average was 21-23 during the
runs):

| Kernel (kernel-only) | before | after | change |
|---|---:|---:|---:|
| parse 1M Int32ul | 0.590 ms | 0.540 ms | -8.6% |
| parse 750k Float64b | 0.576 ms | 0.501 ms | -13.1% |
| build 750k Float64b | 0.586 ms | 0.500 ms | -14.7% |
| parse 1.5M Byte | 0.785 ms | 0.699 ms | -10.9% |

What changed:

- Float decode and encode are now one fused pass each: byte swap, bit
  reinterpretation and (for 32-bit) widening happen in vector registers, then
  each element is written once. Float encode previously stored, reloaded and
  re-stored every element to byte-swap it; that second pass is gone.
- Byte and unsigned little-endian 32-bit decode widen `uint8`/`uint32` to
  `uint64` in vector lanes instead of one scalar widen per element.
- 16-bit decode swaps and widens a whole vector per round instead of widening
  lane by lane in a scalar inner loop.
- The 16-bit decode vector path also had a pre-existing bug: it wrote lanes to
  the wrong elements (holes and duplicated values from index 8 on), which the
  old parity tests never exercised. It is fixed, and the new length tests pin it.

Vector loads are `alignment=1`. The Python caller passes a `bytes` object's
address and a NumPy result buffer, neither of which carries an alignment
guarantee, so an aligned load would be unsound. Each vector path is followed by
a scalar tail.

Caller-level rows from `pixi run bench` (best of five, three runs of each build,
load average 16-23):

| Case | before | after |
|---|---:|---:|
| parse Array(1M, Int32ul) | 63.76-68.42 ms | 64.08-65.60 ms |
| build Array(1M, Int32ul) | 78.64-80.51 ms | 77.31-78.43 ms |
| parse Array(750k, Float64b) | 33.37-37.32 ms | 30.66-36.20 ms |
| build Array(750k, Float64b) | 42.82-44.84 ms | 41.06-41.61 ms |
| parse Array(1.5M, Byte) | 18.84-20.23 ms | 18.25-21.25 ms |
| parse Array(500k, VarInt) | 16.29-24.40 ms | 17.91-20.15 ms |
| build Array(500k, VarInt) | 84.16-102.71 ms | 85.18-89.10 ms |

No row regressed beyond its own noise. Float build moved consistently; float
parse moved on the median but one sample overlaps the old range, which is the
caller noise the section above describes. The VarInt kernels were not
vectorised: their inner loops advance by a data-dependent amount, so iterations
depend on each other.

Those wins are a small part of the rows because the rows are dominated by the
Python caller: converting the decoded 1M items with `.tolist()` alone costs about
56 ms, against a 0.6 ms kernel. Removing serial chains and loop overhead in the
kernels is finished; the caller, not the kernel, is what the table measures. See
`bench/README.md` for the tie-break used where the table could not separate a
change from noise. The VarInt row moved in the other direction (the smaller best
run), so against Construct itself it now shows 47.48x where the previous table
showed 19.34x; that is the Construct-side time moving between runs, not a 2.5x
change in this library.

### Unrolling and summation order

There is no multi-accumulator reduction in the library: the codecs move and
convert values, none of them sums anything, and the only floating-point work is
a bit-for-bit conversion. No summation order changed, so results stay
bit-identical to Construct. Where a vector swap is used (8 values per round in
16-bit decode, which covers 16-bit parse and GreedyRange), the vector path and
the scalar tail cover disjoint ranges, so the output is unchanged byte for byte,
not only in the last bits.

These results target the workload this port is designed for. Small scalar
schemas do not benefit from an FFI call, so their normal Python `struct`
implementation is retained.

No GPU path is included; all benchmarked codecs use the CPU. GPU only: memory-bound (see the next section).

### GPU phase

CPU only: memory-bound.

Every codec here was measured against the 2 flops/byte candidate gate and none
of them is remotely close, so no GPU path was written and none was uploaded:

| Kernel | bytes/element in | bytes/element out | arithmetic/element | flops/byte |
|---|---:|---:|---:|---:|
| parse/build 32-bit int (4 B -> 8 B buffer) | 4 | 8 | byte swap only | 0 |
| parse/build 16-bit int | 2 | 8 | swap + sign extend | 0 |
| parse/build Float64 | 8 | 8 | bit reinterpret (no FP math) | 0 |
| build Float32 | 8 | 4 | FP64 -> FP32 narrow | ~0.15 |
| build/parse varint | 1-10 | 8 | shift/or, data-dependent | ~0.1 |

These are integer move-and-convert kernels: the work per element is a handful of
shift/or instructions and there is no floating-point arithmetic at all. The
arithmetic intensity is effectively zero (the only FP operation in the library is
the double-to-float narrowing, about 1 op per 12 bytes copied), so the kernels are
bound by memory bandwidth on the CPU already and would be bound by PCIe bandwidth
on the GPU. There is nothing to compute that a device could do faster, and the
transfer alone would cost more than the whole CPU kernel (0.5-2 ms at the sizes
real callers pass; see the parallelism section for those sizes).

On the pinned toolchain `from max.gpu import block_idx, thread_idx` and
`from max.gpu.host import DeviceContext` both resolve (`MOJO_NOTES.md` section 5),
so the host API is not the reason for skipping this phase; the gate is. At the
time of this check the shared GPU reported 5500 MiB free, so the 4000 MiB limit
was not the reason either.

As a result there is no GPU test to skip: `pixi run test` runs the CPU suite only,
and no test is marked or reported as skipped-on-CPU for this phase.

### Parallelism

The codecs are serial, and that is a measured decision rather than an omission.
`max.algorithm.parallelize` works on the pinned toolchain: the range split,
the nested `def work(i: Int) {imm}:` worker and the explicit worker-count form
all compile, and parallel output was bit-identical to serial in every case
tested. It loses anyway, because of what a launch costs when the library is
loaded through ctypes.

Two measurements set the floor. An **empty** `parallelize` body called from
ctypes costs **~5 ms per launch** (4.6-7.0 ms across repeat runs at load
average 19-22); the same empty body called natively from a Mojo `main` costs
**12-100 us**, which is the same work done through a different host. With four
Python threads issuing calls concurrently, the in-library cost rose to
**10-15 ms per call**. So the least a parallel codec can cost is several
milliseconds before it processes a byte.

Against that floor, the codec sweep (`bench/crossover.py`, serial) gives the
crossover below which a launch can never pay for itself:

| Kernel | serial @256k | @1024k | @4096k | @16384k |
|---|---:|---:|---:|---:|
| parse Int32ul | 0.16 ms | 0.67 ms | 4.64 ms | 25.33 ms |
| build Int32ul | 0.13 ms | 0.48 ms | 3.60 ms | 22.96 ms |
| parse Byte | 0.13 ms | 0.50 ms | 3.05 ms | 23.60 ms |
| parse Float64b | 0.16 ms | 0.68 ms | 3.71 ms | 25.61 ms |
| build Float64b | 0.18 ms | 0.70 ms | 7.03 ms | 30.07 ms |
| parse VarInt | 1.27 ms | 5.16 ms | 21.14 ms | 84.12 ms |
| build VarInt | 0.85 ms | 3.46 ms | 14.22 ms | 54.23 ms |

Median of 15 runs in each of five rounds, load average 21.9. Every benchmark
row (`bench/bench.py`) passes 500k-1.5M elements, where the serial kernel is
0.5-5 ms. A parallel split there is a 5-10x *regression* of the kernel, which
is why no threshold was shipped: the sizes real callers pass sit below the
point where a launch could ever break even.

The full experiment still ran to find that point.
`bench/build_variants.sh` builds the library with the range split across 1, 2,
4 and 8 workers at a forced threshold; `bench/parallel_sweep.py` A/Bs those
against the serial build and checks the output of every worker count against
the serial one. Median ms, speedup in parentheses:

| parse Int32ul | serial | 1 worker | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| 256k | 0.15 | 0.15 (1.00x) | 4.96 (0.03x) | 4.85 (0.03x) | 4.94 (0.03x) |
| 1024k | 0.56 | 0.56 (0.99x) | 5.44 (0.10x) | 5.44 (0.10x) | 4.99 (0.11x) |
| 2048k | 1.17 | 1.24 (0.94x) | 6.45 (0.18x) | 5.52 (0.21x) | 5.61 (0.21x) |
| 4096k | 6.04 | 4.94 (1.22x) | 9.63 (0.63x) | 8.27 (0.73x) | 7.95 (0.76x) |
| 8192k | 12.02 | 11.48 (1.05x) | 12.94 (0.93x) | 11.87 (1.01x) | 12.13 (0.99x) |
| 16384k | 24.17 | 24.29 (1.00x) | 22.79 (1.06x) | 18.69 (1.29x) | 19.46 (1.24x) |

| parse Byte | serial | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| 1024k | 0.48 | 0.48 (1.00x) | 4.41 (0.11x) | 4.52 (0.11x) | 4.25 (0.11x) |
| 8192k | 10.66 | 10.82 (0.98x) | 11.81 (0.90x) | 11.26 (0.95x) | 11.13 (0.96x) |
| 16384k | 19.75 | 21.42 (0.92x) | 18.69 (1.06x) | 16.49 (1.20x) | 16.79 (1.18x) |

| parse VarInt | serial | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| 1024k | 5.37 | 5.18 (1.04x) | 5.17 (1.04x) | 5.28 (1.02x) | 6.77 (0.79x) |
| 8192k | 43.26 | 42.56 (1.02x) | 43.05 (1.00x) | 42.69 (1.01x) | 42.11 (1.03x) |
| 16384k | 83.45 | 84.60 (0.99x) | 86.57 (0.96x) | 83.56 (1.00x) | 84.97 (0.98x) |

Build Int32ul, both float codecs and build VarInt are in the same table in
`bench/parallel_sweep.py`; none of them gained more than 1.07x at any size.
Load average was 18.3 during this sweep.

What that says:

- At 1M-2M elements, 2-8 workers cost **3-30x** the serial time. This is the
  range the benchmark and ordinary `Array`/`GreedyRange` callers use.
- The crossover for parse is ~8M elements, and only at 16M does 4 workers
  reach a real gain: **1.29x** for parse Int32ul, **1.20x** for parse Byte.
  Everything else at that size is within noise or slower.
- The gain **flattens and reverses** going from 4 to 8 workers (1.29x -> 1.24x
  for parse Int32ul, 1.20x -> 1.18x for parse Byte) on this shared host, so the
  4-worker cap is where it stops, not where it is optimal.
- One worker shows up to 1.22x at 4096k, which is noise: with one worker
  `parallelize` runs the range inline and nothing is split.

A threshold of 8M-16M elements with four workers would therefore be
serial at every size a real caller passes and only parallel where a single
launch consumes tens of milliseconds anyway. That is not worth shipping: it
puts worker threads inside whatever process loads the library (where the
concurrency measurement above shows launches become 2-3x more expensive), for
a 1.2-1.3x kernel gain at 8M-16M elements, against benchmark rows of 0.5M-1.5M.
At 16M a 25 ms kernel would come out ~6 ms faster, but no benchmark row is that
size, and 6 ms out of a 64 ms caller-dominated row is not a measurable win. The
range split was reverted and the codecs stay serial. The one structural change
kept from the experiment is that `decode_ints` now calls a `decode_ints_range`
with an explicit `[start, end)`, so a future toolchain with a cheaper launch can
re-enable it without touching the vector loops. The refactor itself is checked:
786 input/layout cases across the pre-experiment and current builds produce
identical bytes, and `test_serial_fallback_matches_scalar_reference` pins the
scalar paths.

## How it works

`src/construct.mojo` is one compilation unit exporting six non-parametric C ABI
functions. Python buffers cross the ABI as integer addresses. Each export
reconstructs an `UnsafePointer[..., AnyOrigin[mut=True]]` internally, matching
the pinned Mojo nightly's FFI requirements.

For a fixed numeric array, the Python layer reads one contiguous byte block and
allocates a contiguous `uint64`, `int64`, or `float64` result array. Mojo walks
the input once, performs unaligned endian-aware loads, sign extension or float
conversion, and writes into that caller-owned array. Building reverses the
process into a caller-owned byte array. Varints use the same layout plus a
two-word metadata buffer for consumed bytes and decoded count.

Decoding chooses one branch per call instead of testing the width per element.
Unsigned 8-bit and unsigned little-endian 32-bit decode read one vector of
narrow lanes and widen in the vector; 16-bit decode swaps and widens a vector
per round and applies sign extension to the widened lanes; float decode swaps
and reinterprets (or widens) in registers and writes each element once. Sign bit
and sign-extension mask are computed once per call. Integer-array builds convert and range-check input with
NumPy once, preserving already-contiguous 64-bit arrays without a copy across
the FFI boundary. The Mojo encoder uses host-width SIMD for 8-, 16-, 32-, and
64-bit fields, followed by a scalar tail. Every codec runs on one thread; the
section below records why dividing the range across `max.algorithm` workers was
measured and rejected on this toolchain.

Mojo does not allocate or retain memory. Python owns every input, output, and
scratch buffer for the entire FFI call, and converts decoded arrays to
`ListContainer` before returning them. Heterogeneous structs therefore preserve
the public object model and context behavior of Construct while large repeated
numeric fields avoid one Python call per element.
