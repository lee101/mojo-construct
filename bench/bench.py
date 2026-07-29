from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import construct as ref  # noqa: E402
import mojo_construct as mc  # noqa: E402


def best_time(fn, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rng = np.random.default_rng(42)
    integers = rng.integers(0, 2**32, size=1_000_000, dtype=np.uint64).tolist()
    integer_blob = ref.Int32ul[len(integers)].build(integers)
    floats = rng.normal(size=750_000).tolist()
    float_blob = ref.Float64b[len(floats)].build(floats)
    bytes_values = rng.integers(0, 256, size=1_500_000, dtype=np.uint8).tolist()
    byte_blob = bytes(bytes_values)
    varints = rng.integers(0, 1 << 20, size=500_000, dtype=np.uint64).tolist()
    varint_blob = ref.VarInt[len(varints)].build(varints)

    cases = [
        ("parse Array(1M, Int32ul)",
         lambda: mc.Int32ul[len(integers)].parse(integer_blob),
         lambda: ref.Int32ul[len(integers)].parse(integer_blob)),
        ("build Array(1M, Int32ul)",
         lambda: mc.Int32ul[len(integers)].build(integers),
         lambda: ref.Int32ul[len(integers)].build(integers)),
        ("parse Array(750k, Float64b)",
         lambda: mc.Float64b[len(floats)].parse(float_blob),
         lambda: ref.Float64b[len(floats)].parse(float_blob)),
        ("build Array(750k, Float64b)",
         lambda: mc.Float64b[len(floats)].build(floats),
         lambda: ref.Float64b[len(floats)].build(floats)),
        ("parse Array(1.5M, Byte)",
         lambda: mc.Byte[len(bytes_values)].parse(byte_blob),
         lambda: ref.Byte[len(bytes_values)].parse(byte_blob)),
        ("parse Array(500k, VarInt)",
         lambda: mc.VarInt[len(varints)].parse(varint_blob),
         lambda: ref.VarInt[len(varints)].parse(varint_blob)),
        ("build Array(500k, VarInt)",
         lambda: mc.VarInt[len(varints)].build(varints),
         lambda: ref.VarInt[len(varints)].build(varints)),
    ]

    mc.Byte[1].parse(b"\x00")
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}; Python {platform.python_version()}")
    print()
    print("| Case | mojo-construct | construct 2.10.70 | Speedup |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        ours_time, ours_value = best_time(ours)
        ref_time, ref_value = best_time(theirs)
        if isinstance(ours_value, bytes):
            assert ours_value == ref_value
        else:
            assert ours_value == ref_value
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | "
            f"{ref_time * 1000:.2f} ms | {ref_time / ours_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
