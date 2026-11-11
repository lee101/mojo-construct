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

No GPU path is included; all benchmarked codecs use the CPU.

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
64-bit fields, followed by a scalar tail. Contiguous arrays of at least four
million items are divided across four workers; smaller arrays remain serial to
avoid launch overhead.

Mojo does not allocate or retain memory. Python owns every input, output, and
scratch buffer for the entire FFI call, and converts decoded arrays to
`ListContainer` before returning them. Heterogeneous structs therefore preserve
the public object model and context behavior of Construct while large repeated
numeric fields avoid one Python call per element.
