"""Serial vs parallel sweep for the codec kernels (worker scaling + crossover).

Runs the same FFI calls against several builds of the shared library that differ
only in PARALLEL_THRESHOLD / PARALLEL_WORKERS, checks every parallel result is
bit-identical to the serial one, and prints median times per size.

Build the variants first:  pixi run bash bench/build_variants.sh 1:1024 2:1024 4:1024 8:1024
Then:                     PYTHONPATH=python pixi run python bench/parallel_sweep.py
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIAL = os.path.join(ROOT, "dist", "libmojo-construct.so")
VARIANTS = sorted(glob.glob("/tmp/parwork/lib-w*.so"))

REPEAT = 9
OUTER = 5
SIZES = [1 << k for k in (18, 20, 21, 22, 23, 24)]

I = ctypes.c_int64
SIGNATURES = {
    "mc_decode_ints": ([I] * 7, None),
    "mc_encode_ints": ([I] * 6, None),
    "mc_decode_floats": ([I] * 6, None),
    "mc_encode_floats": ([I] * 6, None),
    "mc_decode_varints": ([I] * 5, I),
    "mc_encode_varints": ([I] * 4, I),
}


def load(path):
    lib = ctypes.CDLL(path)
    for name, (argtypes, restype) in SIGNATURES.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


def median_time(fn):
    samples = []
    for _ in range(OUTER):
        fn()
        best = float("inf")
        for _ in range(REPEAT):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
        samples.append(best)
    samples.sort()
    return samples[len(samples) // 2]


def addr(a):
    return int(a.ctypes.data)


def main():
    libs = {"serial": load(SERIAL)}
    for path in VARIANTS:
        libs[os.path.basename(path).split("-")[1].lstrip("w")] = load(path)
    rng = np.random.default_rng(11)
    limit = max(SIZES)

    int_bytes = rng.integers(0, 2**32, size=limit, dtype=np.uint32)
    int_out = np.empty(limit, dtype=np.uint64)
    int_u64 = int_bytes.astype(np.uint64)
    int_enc = np.empty(limit * 4, dtype=np.uint8)
    flt_in = rng.normal(size=limit)
    flt_out = np.empty(limit, dtype=np.float64)
    flt_enc = np.empty(limit * 8, dtype=np.uint8)
    byte_in = rng.integers(0, 256, size=limit, dtype=np.uint8)
    byte_out = np.empty(limit, dtype=np.uint64)
    var_u64 = rng.integers(0, 1 << 20, size=limit, dtype=np.uint64)
    var_enc = np.empty(limit * 10, dtype=np.uint8)
    var_out = np.empty(limit, dtype=np.uint64)
    meta = np.zeros(2, dtype=np.int64)
    var_nbytes = {}

    def measure_varints(n):
        if n not in var_nbytes:
            var_nbytes[n] = int(
                libs["serial"].mc_encode_varints(addr(var_u64), n, addr(var_enc), limit * 10)
            )
        return var_nbytes[n]

    # (label, buffer compared, byte length of that buffer, call)
    cases = [
        ("parse Int32ul", int_out, lambda n: n * 8,
         lambda lib, n: lib.mc_decode_ints(addr(int_bytes), addr(int_out), n, 4, 4, 0, 0)),
        ("build Int32ul", int_enc, lambda n: n * 4,
         lambda lib, n: lib.mc_encode_ints(addr(int_u64), addr(int_enc), n, 4, 4, 0)),
        ("parse Byte", byte_out, lambda n: n * 8,
         lambda lib, n: lib.mc_decode_ints(addr(byte_in), addr(byte_out), n, 1, 1, 0, 0)),
        ("parse Float64b", flt_out, lambda n: n * 8,
         lambda lib, n: lib.mc_decode_floats(addr(flt_enc), addr(flt_out), n, 8, 8, 1)),
        ("build Float64b", flt_enc, lambda n: n * 8,
         lambda lib, n: lib.mc_encode_floats(addr(flt_in), addr(flt_enc), n, 8, 8, 1)),
        ("parse VarInt", var_out, lambda n: n * 8,
         lambda lib, n: lib.mc_decode_varints(
             addr(var_enc), measure_varints(n), addr(var_out), n, addr(meta))),
        ("build VarInt", var_enc, lambda n: None,
         lambda lib, n: lib.mc_encode_varints(addr(var_u64), n, addr(var_enc), limit * 10)),
    ]

    libs["serial"].mc_encode_floats(addr(flt_in), addr(flt_enc), limit, 8, 8, 1)
    for n in SIZES:
        measure_varints(n)

    print(f"load average {os.getloadavg()[0]:.1f}; median ms, (serial/wN speedup)",
          flush=True)

    for label, buffer, view_len, call in cases:
        print(f"\n### {label}\n", flush=True)
        print("| size | " + " | ".join(libs) + " |")
        print("|---|" + "---:|" * len(libs))
        for n in SIZES:
            buffer.fill(0xA5)
            written_ref = call(libs["serial"], n)
            if view_len(n) is None:
                reference = buffer[: written_ref].copy()
            else:
                reference = buffer[: view_len(n)].copy()
            serial_ms = median_time(lambda: call(libs["serial"], n)) * 1000
            cells = [f"{serial_ms:.2f}"]
            for name, lib in libs.items():
                if name == "serial":
                    continue
                buffer.fill(0xA5)
                written = call(lib, n)
                if view_len(n) is None:
                    same = written == written_ref and np.array_equal(
                        buffer[:written], reference)
                else:
                    same = np.array_equal(buffer[: view_len(n)], reference)
                when = median_time(lambda lib=lib: call(lib, n))
                if not same:
                    cells.append("MISMATCH")
                else:
                    cells.append(f"{when * 1000:.2f} ({serial_ms / (when * 1000):.2f}x)")
            print(f"| {n // 1024}k | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
