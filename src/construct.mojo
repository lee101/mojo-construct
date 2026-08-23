"""Bulk binary codecs for the Python declarative layer."""

from std.memory import stack_allocation
from std.sys import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def swap16(v: UInt16) -> UInt16:
    return (v << 8) | (v >> 8)


def swap32(v: UInt32) -> UInt32:
    return (
        (v << 24)
        | ((v << 8) & 0x00FF0000)
        | ((v >> 8) & 0x0000FF00)
        | (v >> 24)
    )


def swap64(v: UInt64) -> UInt64:
    return (
        (v << 56)
        | ((v << 40) & 0x00FF000000000000)
        | ((v << 24) & 0x0000FF0000000000)
        | ((v << 8) & 0x000000FF00000000)
        | ((v >> 8) & 0x00000000FF000000)
        | ((v >> 24) & 0x0000000000FF0000)
        | ((v >> 40) & 0x000000000000FF00)
        | (v >> 56)
    )


@always_inline
def encode_int(
    src: U64Ptr,
    dst: BPtr,
    i: Int,
    width: Int,
    stride: Int,
    big: Int,
):
    var value = src[i]
    var p = dst + i * stride
    if width == 1:
        p[0] = UInt8(value)
    elif width == 2:
        var v = UInt16(value)
        p.bitcast[UInt16]().store[alignment=1](
            0, swap16(v) if big else v
        )
    elif width == 3:
        if big:
            p[0] = UInt8(value >> 16)
            p[1] = UInt8(value >> 8)
            p[2] = UInt8(value)
        else:
            p[0] = UInt8(value)
            p[1] = UInt8(value >> 8)
            p[2] = UInt8(value >> 16)
    elif width == 4:
        var v = UInt32(value)
        p.bitcast[UInt32]().store[alignment=1](
            0, swap32(v) if big else v
        )
    elif width == 8:
        p.bitcast[UInt64]().store[alignment=1](
            0, swap64(value) if big else value
        )


def encode_ints_range(
    src: U64Ptr,
    dst: BPtr,
    start: Int,
    end: Int,
    width: Int,
    stride: Int,
    big: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    var vector_end = end - (end - start) % W
    if stride == width and width == 1:
        while i < vector_end:
            var values = src.load[width=W](i).cast[DType.uint8]()
            dst.store(i, values)
            i += W
    elif stride == width and width == 2:
        var target = dst.bitcast[UInt16]()
        while i < vector_end:
            var values = src.load[width=W](i).cast[DType.uint16]()
            if big:
                values = (values << 8) | (values >> 8)
            target.store(i, values)
            i += W
    elif stride == width and width == 4:
        var target = dst.bitcast[UInt32]()
        while i < vector_end:
            var values = src.load[width=W](i).cast[DType.uint32]()
            if big:
                values = (
                    (values << 24)
                    | ((values << 8) & 0x00FF0000)
                    | ((values >> 8) & 0x0000FF00)
                    | (values >> 24)
                )
            target.store(i, values)
            i += W
    elif stride == width and width == 8:
        var target = dst.bitcast[UInt64]()
        while i < vector_end:
            var values = src.load[width=W](i)
            if big:
                values = (
                    (values << 56)
                    | ((values << 40) & 0x00FF000000000000)
                    | ((values << 24) & 0x0000FF0000000000)
                    | ((values << 8) & 0x000000FF00000000)
                    | ((values >> 8) & 0x00000000FF000000)
                    | ((values >> 24) & 0x0000000000FF0000)
                    | ((values >> 40) & 0x000000000000FF00)
                    | (values >> 56)
                )
            target.store(i, values)
            i += W
    for tail in range(i, end):
        encode_int(src, dst, tail, width, stride, big)


@export("mc_decode_ints")
def decode_ints(
    src_addr: Int,
    dst_addr: Int,
    count: Int,
    width: Int,
    stride: Int,
    big: Int,
    signed: Int,
) abi("C"):
    if count <= 0:
        return
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = U64Ptr(unsafe_from_address=dst_addr)
    for i in range(count):
        var p = src + i * stride
        var value = UInt64(0)
        if width == 1:
            value = UInt64(p[0])
        elif width == 2:
            var v = p.bitcast[UInt16]().load[alignment=1]()
            value = UInt64(swap16(v) if big else v)
        elif width == 3:
            if big:
                value = (
                    (UInt64(p[0]) << 16)
                    | (UInt64(p[1]) << 8)
                    | UInt64(p[2])
                )
            else:
                value = (
                    UInt64(p[0])
                    | (UInt64(p[1]) << 8)
                    | (UInt64(p[2]) << 16)
                )
        elif width == 4:
            var v = p.bitcast[UInt32]().load[alignment=1]()
            value = UInt64(swap32(v) if big else v)
        elif width == 8:
            var v = p.bitcast[UInt64]().load[alignment=1]()
            value = swap64(v) if big else v
        if signed and width < 8:
            var bits = UInt64(width * 8)
            var signbit = UInt64(1) << (bits - 1)
            if value & signbit:
                value |= ~((UInt64(1) << bits) - UInt64(1))
        dst[i] = value


@export("mc_encode_ints")
def encode_ints(
    src_addr: Int,
    dst_addr: Int,
    count: Int,
    width: Int,
    stride: Int,
    big: Int,
) abi("C"):
    if count <= 0:
        return
    var src = U64Ptr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    encode_ints_range(src, dst, 0, count, width, stride, big)


@export("mc_decode_floats")
def decode_floats(
    src_addr: Int,
    dst_addr: Int,
    count: Int,
    width: Int,
    stride: Int,
    big: Int,
) abi("C"):
    if count <= 0:
        return
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = F64Ptr(unsafe_from_address=dst_addr)
    var scratch = stack_allocation[1, DType.uint64]()
    for i in range(count):
        var p = src + i * stride
        if width == 4:
            if big:
                var raw = swap32(p.bitcast[UInt32]().load[alignment=1]())
                scratch.bitcast[UInt32]()[0] = raw
                dst[i] = Float64(
                    scratch.bitcast[Float32]()[0]
                )
            else:
                dst[i] = Float64(
                    p.bitcast[Float32]().load[alignment=1]()
                )
        elif width == 8:
            if big:
                var raw = swap64(p.bitcast[UInt64]().load[alignment=1]())
                scratch[0] = raw
                dst[i] = scratch.bitcast[Float64]()[0]
            else:
                dst[i] = p.bitcast[Float64]().load[alignment=1]()


@export("mc_encode_floats")
def encode_floats(
    src_addr: Int,
    dst_addr: Int,
    count: Int,
    width: Int,
    stride: Int,
    big: Int,
) abi("C"):
    if count <= 0:
        return
    var src = F64Ptr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    for i in range(count):
        var p = dst + i * stride
        if width == 4:
            p.bitcast[Float32]().store[alignment=1](0, Float32(src[i]))
            if big:
                var raw = p.bitcast[UInt32]().load[alignment=1]()
                p.bitcast[UInt32]().store[alignment=1](0, swap32(raw))
        elif width == 8:
            p.bitcast[Float64]().store[alignment=1](0, src[i])
            if big:
                var raw = p.bitcast[UInt64]().load[alignment=1]()
                p.bitcast[UInt64]().store[alignment=1](0, swap64(raw))


@export("mc_decode_varints")
def decode_varints(
    src_addr: Int,
    n: Int,
    dst_addr: Int,
    max_count: Int,
    meta_addr: Int,
) abi("C") -> Int:
    if n < 0 or max_count < 0:
        return -3
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = U64Ptr(unsafe_from_address=dst_addr)
    var meta = I64Ptr(unsafe_from_address=meta_addr)
    var pos = 0
    var count = 0
    while pos < n and count < max_count:
        var item_start = pos
        var value = UInt64(0)
        var shift = 0
        var complete = False
        for _ in range(10):
            if pos >= n:
                break
            var byte = src[pos]
            pos += 1
            if shift == 63 and (byte & 0x7E):
                meta[0] = Int64(item_start)
                meta[1] = Int64(count)
                return -2
            value |= UInt64(byte & 0x7F) << UInt64(shift)
            if byte < 0x80:
                complete = True
                break
            shift += 7
        if not complete:
            meta[0] = Int64(item_start)
            meta[1] = Int64(count)
            return -1
        dst[count] = value
        count += 1
    meta[0] = Int64(pos)
    meta[1] = Int64(count)
    return 0


@export("mc_encode_varints")
def encode_varints(
    src_addr: Int,
    count: Int,
    dst_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if count < 0 or capacity < 0:
        return -1
    var src = U64Ptr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var pos = 0
    for i in range(count):
        var value = src[i]
        while value >= 0x80:
            if pos >= capacity:
                return -1
            dst[pos] = UInt8(value) | UInt8(0x80)
            pos += 1
            value >>= 7
        if pos >= capacity:
            return -1
        dst[pos] = UInt8(value)
        pos += 1
    return pos
