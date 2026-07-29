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
62 passed
```

## Benchmarks

Run benchmarks only through the flocked Pixi task:

```bash
pixi run bench
```

The table below is real output from the best of three runs on an Intel Xeon
E5-2697 v4 at 2.30 GHz, Linux x86-64, Python 3.13.14. Times include ctypes,
NumPy scratch allocation, conversion to Construct-compatible Python lists, and
output allocation.

| Case | mojo-construct | construct 2.10.70 | Speedup |
|---|---:|---:|---:|
| parse Array(1M, Int32ul) | 61.83 ms | 652.44 ms | 10.55x |
| build Array(1M, Int32ul) | 115.50 ms | 841.69 ms | 7.29x |
| parse Array(750k, Float64b) | 49.82 ms | 671.77 ms | 13.48x |
| build Array(750k, Float64b) | 44.74 ms | 514.29 ms | 11.50x |
| parse Array(1.5M, Byte) | 41.08 ms | 990.45 ms | 24.11x |
| parse Array(500k, VarInt) | 41.90 ms | 810.36 ms | 19.34x |
| build Array(500k, VarInt) | 89.38 ms | 568.29 ms | 6.36x |

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

Integer-array builds convert and range-check input with NumPy once, preserving
already-contiguous 64-bit arrays without a copy across the FFI boundary. The
Mojo encoder uses host-width SIMD for 8-, 16-, 32-, and 64-bit fields, followed
by a scalar tail. Contiguous arrays of at least four million items are divided
across four workers; smaller arrays remain serial to avoid launch overhead.

Mojo does not allocate or retain memory. Python owns every input, output, and
scratch buffer for the entire FFI call, and converts decoded arrays to
`ListContainer` before returning them. Heterogeneous structs therefore preserve
the public object model and context behavior of Construct while large repeated
numeric fields avoid one Python call per element.
