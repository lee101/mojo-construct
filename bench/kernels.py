"""Kernel-only timing: FFI calls on preallocated buffers, no Python wrapper work.

Same data shapes as bench.py, so a row here is the ceiling for the matching row
there.  Run with:  pixi run python bench/kernels.py
Set MOJO_CONSTRUCT_LIB to time a different build of the shared library.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_construct import _lib  # noqa: E402

REPEAT = 7


def best(fn):
    for _ in range(2):
        fn()
    best_time = float("inf")
    for _ in range(REPEAT):
        start = time.perf_counter()
        fn()
        best_time = min(best_time, time.perf_counter() - start)
    return best_time


def main():
    lib = _lib.lib()
    addr = _lib._addr
    rng = np.random.default_rng(42)

    n_int = 1_000_000
    int_bytes = rng.integers(0, 2**32, size=n_int, dtype=np.uint32)
    int_out = np.empty(n_int, dtype=np.uint64)
    int_u64 = int_bytes.astype(np.uint64)
    int_encoded = np.empty(n_int * 4, dtype=np.uint8)

    n_flt = 750_000
    flt_in = rng.normal(size=n_flt)
    flt_out = np.empty(n_flt, dtype=np.float64)
    flt_encoded = np.empty(n_flt * 8, dtype=np.uint8)

    n_byte = 1_500_000
    byte_in = rng.integers(0, 256, size=n_byte, dtype=np.uint8)
    byte_out = np.empty(n_byte, dtype=np.uint64)

    n_var = 500_000
    var_u64 = rng.integers(0, 1 << 20, size=n_var, dtype=np.uint64)
    var_encoded = np.empty(n_var * 10, dtype=np.uint8)
    var_out = np.empty(n_var, dtype=np.uint64)
    meta = np.zeros(2, dtype=np.int64)

    n_var_bytes = int(lib.mc_encode_varints(
        addr(var_u64), n_var, addr(var_encoded), len(var_encoded)))
    lib.mc_encode_floats(addr(flt_in), addr(flt_encoded), n_flt, 8, 8, 1)

    cases = [
        ("kernel parse Array(1M, Int32ul)",
         lambda: lib.mc_decode_ints(addr(int_bytes), addr(int_out), n_int, 4, 4, 0, 0)),
        ("kernel build Array(1M, Int32ul)",
         lambda: lib.mc_encode_ints(addr(int_u64), addr(int_encoded), n_int, 4, 4, 0)),
        ("kernel parse Array(750k, Float64b)",
         lambda: lib.mc_decode_floats(addr(flt_encoded), addr(flt_out), n_flt, 8, 8, 1)),
        ("kernel build Array(750k, Float64b)",
         lambda: lib.mc_encode_floats(addr(flt_in), addr(flt_encoded), n_flt, 8, 8, 1)),
        ("kernel parse Array(1.5M, Byte)",
         lambda: lib.mc_decode_ints(addr(byte_in), addr(byte_out), n_byte, 1, 1, 0, 0)),
        ("kernel parse Array(500k, VarInt)",
         lambda: lib.mc_decode_varints(
             addr(var_encoded), n_var_bytes, addr(var_out), n_var, addr(meta))),
        ("kernel build Array(500k, VarInt)",
         lambda: lib.mc_encode_varints(
             addr(var_u64), n_var, addr(var_encoded), len(var_encoded))),
    ]

    print("| Case | kernel-only |")
    print("|---|---:|")
    for name, fn in cases:
        print(f"| {name} | {best(fn) * 1000:.2f} ms |")


if __name__ == "__main__":
    main()
