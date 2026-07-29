from __future__ import annotations

import ctypes
import operator
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.environ.get(
    "MOJO_CONSTRUCT_LIB",
    os.path.join(ROOT, "dist", "libmojo-construct.so"),
)

I = ctypes.c_int64
_I64_MAX = (1 << 63) - 1

_SIGNATURES = {
    "mc_decode_ints": ([I] * 7, None),
    "mc_encode_ints": ([I] * 6, None),
    "mc_decode_floats": ([I] * 6, None),
    "mc_encode_floats": ([I] * 6, None),
    "mc_decode_varints": ([I] * 5, I),
    "mc_encode_varints": ([I] * 4, I),
}

_LIB: ctypes.CDLL | None = None


def build() -> str:
    if not os.path.exists(LIB_PATH):
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
        )
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIB, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIB


def _addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if not address:
        raise ValueError("cannot pass a null buffer to Mojo")
    return address


def _layout(count: int, width: int, stride: int | None, allowed_widths) -> tuple[int, int]:
    if not isinstance(count, (int, np.integer)):
        raise TypeError("count must be an integer")
    if not isinstance(width, (int, np.integer)):
        raise TypeError("width must be an integer")
    count = int(count)
    width = int(width)
    stride = width if stride is None else stride
    if not isinstance(stride, (int, np.integer)):
        raise TypeError("stride must be an integer")
    stride = int(stride)
    if count < 0:
        raise ValueError("count must be non-negative")
    if width not in allowed_widths:
        raise ValueError(f"unsupported element width: {width}")
    if stride < width:
        raise ValueError("stride must be at least the element width")
    required = 0 if count == 0 else (count - 1) * stride + width
    if required > _I64_MAX:
        raise OverflowError("buffer layout exceeds the C ABI integer range")
    return stride, required


def _byte_source(data) -> np.ndarray:
    try:
        source = np.frombuffer(data, dtype=np.uint8)
    except (TypeError, ValueError, BufferError) as exc:
        raise TypeError("data must be a contiguous bytes-like object") from exc
    return source


def _integer_source(values) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1:
        raise ValueError("integer input must be one-dimensional")
    if source.dtype.kind not in "iub":
        try:
            items = [operator.index(value) for value in values]
        except TypeError as exc:
            raise TypeError("integer input must contain integers") from exc
        dtype = np.int64 if any(value < 0 for value in items) else np.uint64
        try:
            return np.ascontiguousarray(items, dtype=dtype)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("integer input exceeds 64-bit range") from exc
    dtype = np.int64 if source.dtype.kind == "i" else np.uint64
    try:
        return np.ascontiguousarray(source, dtype=dtype)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("integer input exceeds 64-bit range") from exc


def decode_ints(
    data: bytes, count: int, width: int, *, stride: int | None = None,
    big: bool = False, signed: bool = False,
) -> np.ndarray:
    stride, required = _layout(count, width, stride, (1, 2, 3, 4, 8))
    source = _byte_source(data)
    if source.nbytes < required:
        raise ValueError(
            f"input buffer is too short: need {required} bytes, found {source.nbytes}"
        )
    result = np.empty(count, dtype=np.int64 if signed else np.uint64)
    if count:
        lib().mc_decode_ints(
            _addr(source), _addr(result), count, width, stride, big, signed
        )
    return result


def encode_ints(
    values, width: int, *, stride: int | None = None, big: bool = False,
) -> np.ndarray:
    source = _integer_source(values)
    stride, _ = _layout(len(source), width, stride, (1, 2, 3, 4, 8))
    output_size = len(source) * stride
    if output_size > _I64_MAX:
        raise OverflowError("output buffer exceeds the C ABI integer range")
    result = (
        np.empty(output_size, dtype=np.uint8)
        if stride == width
        else np.zeros(output_size, dtype=np.uint8)
    )
    if len(source):
        lib().mc_encode_ints(
            _addr(source), _addr(result), len(source), width, stride, big
        )
    return result


def decode_floats(
    data: bytes, count: int, width: int, *, stride: int | None = None,
    big: bool = False,
) -> np.ndarray:
    stride, required = _layout(count, width, stride, (4, 8))
    source = _byte_source(data)
    if source.nbytes < required:
        raise ValueError(
            f"input buffer is too short: need {required} bytes, found {source.nbytes}"
        )
    result = np.empty(count, dtype=np.float64)
    if count:
        lib().mc_decode_floats(
            _addr(source), _addr(result), count, width, stride, big
        )
    return result


def encode_floats(
    values, width: int, *, stride: int | None = None, big: bool = False,
) -> bytes:
    unconverted = np.asarray(values)
    if unconverted.ndim != 1:
        raise ValueError("float input must be one-dimensional")
    if unconverted.dtype.kind not in "fiub":
        raise TypeError("float input must contain real numbers")
    try:
        source = np.ascontiguousarray(unconverted, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError("float input must contain real numbers") from exc
    stride, _ = _layout(len(source), width, stride, (4, 8))
    output_size = len(source) * stride
    if output_size > _I64_MAX:
        raise OverflowError("output buffer exceeds the C ABI integer range")
    result = np.zeros(output_size, dtype=np.uint8)
    if len(source):
        lib().mc_encode_floats(
            _addr(source), _addr(result), len(source), width, stride, big
        )
    return result.tobytes()


def decode_varints(data: bytes, max_count: int) -> tuple[np.ndarray, int, int]:
    if not isinstance(max_count, (int, np.integer)):
        raise TypeError("max_count must be an integer")
    max_count = int(max_count)
    if max_count < 0:
        raise ValueError("max_count must be non-negative")
    if max_count > _I64_MAX:
        raise OverflowError("max_count exceeds the C ABI integer range")
    source = _byte_source(data)
    if source.nbytes > _I64_MAX:
        raise OverflowError("input buffer exceeds the C ABI integer range")
    result = np.empty(max_count, dtype=np.uint64)
    meta = np.zeros(2, dtype=np.int64)
    if data and max_count:
        status = int(lib().mc_decode_varints(
            _addr(source), len(source), _addr(result), max_count, _addr(meta)
        ))
    else:
        status = 0
    return result[: int(meta[1])], int(meta[0]), status


def encode_varints(values) -> bytes:
    source = _integer_source(values)
    if source.dtype.kind == "i" and len(source) and source.min() < 0:
        raise ValueError("varints must be unsigned 64-bit integers")
    source = np.ascontiguousarray(source, dtype=np.uint64)
    result = np.empty(max(1, len(source) * 10), dtype=np.uint8)
    if not len(source):
        return b""
    written = int(lib().mc_encode_varints(
        _addr(source), len(source), _addr(result), len(result)
    ))
    if written < 0:
        raise ValueError("varint output buffer exhausted")
    return result[:written].tobytes()
