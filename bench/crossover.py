"""Size sweep for the parallel threshold: median kernel time per size.

Times the raw FFI calls on preallocated buffers (no Python wrapper), sweeping
the element count so the serial/parallel crossover is visible. The first column
is serial by construction (above the harness threshold), the second uses
whatever the library's own threshold decides, or forces a worker count:
N_WORKERS=0 disables the parallel path.

    PYTHONPATH=python pixi run python bench/crossover.py
    N_WORKERS=1 PYTHONPATH=python pixi run python bench/crossover.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_construct import _lib  # noqa: E402

REPEAT = 15
OUTER = 5

SIZES = [1 << k for k in (14, 16, 18, 19, 20, 21, 22, 23, 24)]


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


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def main():
    lib = _lib.lib()
    addr = _lib._addr
    rng = np.random.default_rng(7)
    limit = max(SIZES)

    int_bytes = rng.integers(0, 2**32, size=limit, dtype=np.uint32)
    int_out = np.empty(limit, dtype=np.uint64)
    int_u64 = int_bytes.astype(np.uint64)
    int_encoded = np.empty(limit * 4, dtype=np.uint8)

    flt_in = rng.normal(size=limit)
    flt_out = np.empty(limit, dtype=np.float64)
    flt_encoded = np.empty(limit * 8, dtype=np.uint8)

    byte_in = rng.integers(0, 256, size=limit, dtype=np.uint8)
    byte_out = np.empty(limit, dtype=np.uint64)

    var_u64 = rng.integers(0, 1 << 20, size=limit, dtype=np.uint64)
    var_encoded = np.empty(limit * 10, dtype=np.uint8)
    var_out = np.empty(limit, dtype=np.uint64)
    meta = np.zeros(2, dtype=np.int64)
    var_bytes = {
        n: int(lib.mc_encode_varints(addr(var_u64), n, addr(var_encoded), limit * 10))
        for n in SIZES
    }

    kernels = {
        "parse Int32ul": (
            lambda n: lib.mc_decode_ints(addr(int_bytes), addr(int_out), n, 4, 4, 0, 0),
            int_out,
        ),
        "build Int32ul": (
            lambda n: lib.mc_encode_ints(addr(int_u64), addr(int_encoded), n, 4, 4, 0),
            int_encoded,
        ),
        "parse Byte": (
            lambda n: lib.mc_decode_ints(addr(byte_in), addr(byte_out), n, 1, 1, 0, 0),
            byte_out,
        ),
        "parse Float64b": (
            lambda n: lib.mc_decode_floats(addr(flt_encoded), addr(flt_out), n, 8, 8, 1),
            flt_out,
        ),
        "build Float64b": (
            lambda n: lib.mc_encode_floats(addr(flt_in), addr(flt_encoded), n, 8, 8, 1),
            flt_encoded,
        ),
        "parse VarInt": (
            lambda n: lib.mc_decode_varints(
                addr(var_encoded), var_bytes[n], addr(var_out), n, addr(meta)
            ),
            var_out,
        ),
        "build VarInt": (
            lambda n: lib.mc_encode_varints(addr(var_u64), n, addr(var_encoded), limit * 10),
            var_encoded,
        ),
    }

    print(f"Machine: {cpu_name()}; load average {os.getloadavg()[0]:.1f}")
    print(f"Library: {_lib.LIB_PATH}")
    print()
    print("| Kernel | " + " | ".join(f"{n // 1024}k" for n in SIZES) + " |")
    print("|---|" + "---:|" * len(SIZES))
    for name, (call, _buffer) in kernels.items():
        cells = []
        for n in SIZES:
            cells.append(f"{median_time(lambda n=n: call(n)) * 1000:.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
