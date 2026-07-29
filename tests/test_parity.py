import io
import math
import random
import zlib

import construct as ref
import numpy as np
import pytest

import mojo_construct as mc
from mojo_construct import _lib


def visible(value):
    if isinstance(value, dict):
        return {
            key: visible(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key.startswith("_"))
        }
    if isinstance(value, (list, tuple)):
        return [visible(item) for item in value]
    if hasattr(value, "intvalue"):
        return (str(value), int(value.intvalue))
    return value


INTEGER_FIELDS = [
    "Byte", "Int8sb", "Int16ub", "Int16sb", "Int16ul", "Int16sl",
    "Int24ub", "Int24sb", "Int24ul", "Int24sl", "Int32ub", "Int32sb",
    "Int32ul", "Int32sl", "Int64ub", "Int64sb", "Int64ul", "Int64sl",
    "Int8un", "Int8sn", "Int16un", "Int16sn", "Int32un", "Int32sn",
    "Int64un", "Int64sn",
]


@pytest.mark.parametrize("name", INTEGER_FIELDS)
def test_integer_scalar_and_array_parity(name):
    ours = getattr(mc, name)
    theirs = getattr(ref, name)
    width = ours.sizeof()
    signed = name.endswith(("sb", "sl", "sn"))
    bits = width * 8
    low = -(1 << (bits - 1)) if signed else 0
    high = (1 << (bits - (1 if signed else 0))) - 1
    values = [low, low + 1, 0, 1, high - 1, high]
    encoded = theirs[len(values)].build(values)
    assert ours[len(values)].build(values) == encoded
    assert ours[len(values)].parse(encoded) == theirs[len(values)].parse(encoded)
    for value in values:
        assert ours.parse(theirs.build(value)) == theirs.parse(ours.build(value))


@pytest.mark.parametrize(
    "name", [
        "Float16b", "Float16l", "Float32b", "Float32l", "Float32n",
        "Float64b", "Float64l", "Float64n",
    ]
)
def test_float_scalar_and_array_parity(name):
    ours = getattr(mc, name)
    theirs = getattr(ref, name)
    values = [-123.5, -0.0, 0.25, 1.5, 1000.0]
    encoded = theirs[len(values)].build(values)
    assert ours[len(values)].build(values) == encoded
    got = ours[len(values)].parse(encoded)
    expected = theirs[len(values)].parse(encoded)
    assert got == pytest.approx(expected, rel=1e-6)


def test_dynamic_struct_parse_and_build_parity():
    ours = mc.Struct(
        "magic" / mc.Const(b"MC"),
        "length" / mc.Byte,
        "payload" / mc.Bytes(mc.this.length),
        "tail" / mc.If(mc.this.length > 2, mc.Int16ul),
        "double_length" / mc.Computed(mc.this.length * 2),
        mc.Check(mc.this.length == mc.len_(mc.this.payload)),
    )
    theirs = ref.Struct(
        "magic" / ref.Const(b"MC"),
        "length" / ref.Byte,
        "payload" / ref.Bytes(ref.this.length),
        "tail" / ref.If(ref.this.length > 2, ref.Int16ul),
        "double_length" / ref.Computed(ref.this.length * 2),
        ref.Check(ref.this.length == ref.len_(ref.this.payload)),
    )
    blob = b"MC\x04data\x34\x12"
    assert visible(ours.parse(blob)) == visible(theirs.parse(blob))
    obj = dict(length=4, payload=b"data", tail=0x1234)
    assert ours.build(obj) == theirs.build(obj) == blob


def test_nested_context_and_keyword_struct_parity():
    ours = mc.Struct(
        "count" / mc.Byte,
        "inner" / mc.Struct(data=mc.Bytes(mc.this._.count)),
    )
    theirs = ref.Struct(
        "count" / ref.Byte,
        "inner" / ref.Struct(data=ref.Bytes(ref.this._.count)),
    )
    blob = b"\x03abc"
    assert visible(ours.parse(blob)) == visible(theirs.parse(blob))
    obj = {"count": 3, "inner": {"data": b"abc"}}
    assert ours.build(obj) == theirs.build(obj)


def test_sequence_and_index_context_parity():
    ours = mc.Sequence(mc.Byte, mc.Array(4, mc.Byte))
    theirs = ref.Sequence(ref.Byte, ref.Array(4, ref.Byte))
    blob = bytes(range(5))
    assert visible(ours.parse(blob)) == visible(theirs.parse(blob))
    assert ours.build([0, [1, 2, 3, 4]]) == theirs.build([0, [1, 2, 3, 4]])


def test_greedy_fixed_fields_and_partial_tail_parity():
    ours = mc.GreedyRange(mc.Int16ub)
    theirs = ref.GreedyRange(ref.Int16ub)
    stream1 = io.BytesIO(b"\x00\x01\x00\x02x")
    stream2 = io.BytesIO(b"\x00\x01\x00\x02x")
    assert ours.parse_stream(stream1) == theirs.parse_stream(stream2)
    assert stream1.tell() == stream2.tell() == 4


def test_varint_and_zigzag_parity():
    unsigned = [0, 1, 127, 128, 16_384, (1 << 63) - 1, (1 << 64) - 1]
    assert mc.VarInt[len(unsigned)].build(unsigned) == ref.VarInt[len(unsigned)].build(unsigned)
    encoded = ref.VarInt[len(unsigned)].build(unsigned)
    assert mc.VarInt[len(unsigned)].parse(encoded) == ref.VarInt[len(unsigned)].parse(encoded)
    signed = [-1_000_000, -1, 0, 1, 1_000_000]
    for value in signed:
        assert mc.ZigZag.build(value) == ref.ZigZag.build(value)
        assert mc.ZigZag.parse(ref.ZigZag.build(value)) == value


def test_greedy_varint_parity():
    values = [0, 127, 128, 999_999, 1 << 40]
    blob = ref.GreedyRange(ref.VarInt).build(values)
    assert mc.GreedyRange(mc.VarInt).parse(blob) == ref.GreedyRange(ref.VarInt).parse(blob)
    assert mc.GreedyRange(mc.VarInt).build(values) == blob


def test_greedy_varint_rewinds_incomplete_tail_like_upstream():
    blob = ref.GreedyRange(ref.VarInt).build([1, 300]) + b"\x80"
    ours_stream = io.BytesIO(blob)
    ref_stream = io.BytesIO(blob)
    assert mc.GreedyRange(mc.VarInt).parse_stream(ours_stream) == \
        ref.GreedyRange(ref.VarInt).parse_stream(ref_stream)
    assert ours_stream.tell() == ref_stream.tell() == len(blob) - 1


def test_enum_and_flags_parity():
    ours = mc.Struct(
        "kind" / mc.Enum(mc.Byte, ping=1, pong=2),
        "mode" / mc.FlagsEnum(mc.Byte, read=1, write=2, execute=4),
    )
    theirs = ref.Struct(
        "kind" / ref.Enum(ref.Byte, ping=1, pong=2),
        "mode" / ref.FlagsEnum(ref.Byte, read=1, write=2, execute=4),
    )
    blob = b"\x02\x05"
    assert visible(ours.parse(blob)) == visible(theirs.parse(blob))
    obj = {"kind": "ping", "mode": {"read": True, "write": True}}
    assert ours.build(obj) == theirs.build(obj) == b"\x01\x03"


def test_default_rebuild_and_prefixed_array_parity():
    ours = mc.Struct(
        "count" / mc.Rebuild(mc.Byte, mc.len_(mc.this.items)),
        "items" / mc.Array(mc.this.count, mc.Int16ul),
        "version" / mc.Default(mc.Byte, 7),
    )
    theirs = ref.Struct(
        "count" / ref.Rebuild(ref.Byte, ref.len_(ref.this.items)),
        "items" / ref.Array(ref.this.count, ref.Int16ul),
        "version" / ref.Default(ref.Byte, 7),
    )
    obj = {"items": [10, 20, 30]}
    assert ours.build(obj) == theirs.build(obj)
    blob = ours.build(obj)
    assert visible(ours.parse(blob)) == visible(theirs.parse(blob))
    assert mc.PrefixedArray(mc.Byte, mc.Int16ub).build([1, 2, 3]) == \
        ref.PrefixedArray(ref.Byte, ref.Int16ub).build([1, 2, 3])


def test_strings_padding_and_prefix_parity():
    cases = [
        (mc.PaddedString(8, "utf8"), ref.PaddedString(8, "utf8"), "mojo"),
        (mc.CString("utf8"), ref.CString("utf8"), "binary \N{GREEK SMALL LETTER PI}"),
        (mc.Padded(8, mc.Bytes(3), pattern=b"\xff"),
         ref.Padded(8, ref.Bytes(3), pattern=b"\xff"), b"abc"),
        (mc.Prefixed(mc.Int16ub, mc.GreedyBytes),
         ref.Prefixed(ref.Int16ub, ref.GreedyBytes), b"payload"),
    ]
    for ours, theirs, value in cases:
        assert ours.build(value) == theirs.build(value)
        blob = theirs.build(value)
        assert ours.parse(blob) == theirs.parse(blob)


def test_repeat_until_and_select_parity():
    predicate = lambda obj, items, ctx: obj == 0
    ours = mc.RepeatUntil(predicate, mc.Byte)
    theirs = ref.RepeatUntil(predicate, ref.Byte)
    blob = b"\x03\x02\x01\x00"
    assert ours.parse(blob) == theirs.parse(blob)
    assert ours.build([3, 2, 1, 0]) == theirs.build([3, 2, 1, 0])
    ours_select = mc.Select(mc.Const(b"AB"), mc.Const(b"CD"))
    theirs_select = ref.Select(ref.Const(b"AB"), ref.Const(b"CD"))
    assert ours_select.parse(b"CD") == theirs_select.parse(b"CD")


def test_adapters_and_validators_parity():
    decoder = lambda obj, ctx: obj * 10
    encoder = lambda obj, ctx: obj // 10
    ours = mc.ExprAdapter(mc.Byte, decoder, encoder)
    theirs = ref.ExprAdapter(ref.Byte, decoder, encoder)
    assert ours.parse(b"\x07") == theirs.parse(b"\x07") == 70
    assert ours.build(70) == theirs.build(70) == b"\x07"
    assert mc.OneOf(mc.Byte, {1, 2, 3}).parse(b"\x02") == \
        ref.OneOf(ref.Byte, {1, 2, 3}).parse(b"\x02")
    with pytest.raises(mc.ValidationError):
        mc.NoneOf(mc.Byte, {9}).parse(b"\x09")


def test_null_wrappers_parity():
    ours = mc.NullTerminated(mc.GreedyBytes, term=b"\x00\x00")
    theirs = ref.NullTerminated(ref.GreedyBytes, term=b"\x00\x00")
    blob = b"a\x00b\x00\x00\x00tail"
    stream1, stream2 = io.BytesIO(blob), io.BytesIO(blob)
    assert ours.parse_stream(stream1) == theirs.parse_stream(stream2)
    assert stream1.tell() == stream2.tell()
    stripped = b"payloadXYZXYZ"
    assert mc.NullStripped(mc.GreedyBytes, pad=b"XYZ").parse(stripped) == \
        ref.NullStripped(ref.GreedyBytes, pad=b"XYZ").parse(stripped)
    assert mc.StringEncoded(mc.Bytes(2), "ascii").build("ok") == \
        ref.StringEncoded(ref.Bytes(2), "ascii").build("ok")


def test_rawcopy_checksum_parity():
    ours = mc.Struct(
        "payload" / mc.RawCopy(mc.Bytes(5)),
        "checksum" / mc.Checksum(mc.Int32ub, zlib.crc32, mc.this.payload.data),
    )
    theirs = ref.Struct(
        "payload" / ref.RawCopy(ref.Bytes(5)),
        "checksum" / ref.Checksum(ref.Int32ub, zlib.crc32, ref.this.payload.data),
    )
    obj = {"payload": {"value": b"hello"}}
    assert ours.build(obj) == theirs.build(obj)
    blob = theirs.build(obj)
    assert visible(ours.parse(blob)) == visible(theirs.parse(blob))
    with pytest.raises(mc.ChecksumError):
        ours.parse(blob[:-1] + bytes([blob[-1] ^ 1]))


def test_pointer_seek_and_tell_parity():
    ours = mc.Struct(
        "start" / mc.Tell,
        "value" / mc.Pointer(4, mc.Int16ub),
        mc.Seek(2),
        "middle" / mc.Int16ub,
    )
    theirs = ref.Struct(
        "start" / ref.Tell,
        "value" / ref.Pointer(4, ref.Int16ub),
        ref.Seek(2),
        "middle" / ref.Int16ub,
    )
    blob = b"\x00\x00\x12\x34\x56\x78"
    assert visible(ours.parse(blob)) == visible(theirs.parse(blob))


def test_bitstruct_parity():
    ours = mc.BitStruct(
        "version" / mc.BitsInteger(3),
        "enabled" / mc.Flag,
        "kind" / mc.Nibble,
        "size" / mc.Octet,
    )
    theirs = ref.BitStruct(
        "version" / ref.BitsInteger(3),
        "enabled" / ref.Flag,
        "kind" / ref.Nibble,
        "size" / ref.Octet,
    )
    obj = {"version": 5, "enabled": True, "kind": 9, "size": 200}
    assert ours.build(obj) == theirs.build(obj)
    blob = theirs.build(obj)
    assert visible(ours.parse(blob)) == visible(theirs.parse(blob))
    assert ours.sizeof() == theirs.sizeof() == 2


def test_random_array_vectors_match_upstream():
    rng = random.Random(17)
    for name in ("Int24ub", "Int24sl", "Int32ub", "Int64sl"):
        ours, theirs = getattr(mc, name), getattr(ref, name)
        bits = ours.sizeof() * 8
        signed = name.endswith(("sb", "sl"))
        low = -(1 << (bits - 1)) if signed else 0
        high = (1 << (bits - (1 if signed else 0))) - 1
        values = [rng.randint(low, high) for _ in range(1003)]
        blob = theirs[len(values)].build(values)
        assert ours[len(values)].parse(blob) == values
        assert ours[len(values)].build(values) == blob


def test_simd_integer_encode_tails_match_upstream():
    values = np.arange(20, dtype=np.uint64)[1:]
    for name in ("Byte", "Int16ub", "Int24ul", "Int32ub", "Int64ul"):
        ours, theirs = getattr(mc, name), getattr(ref, name)
        assert ours[len(values)].build(values) == theirs[len(values)].build(values.tolist())
    signed = list(range(-9, 10))
    assert mc.Int32sl[len(signed)].build(signed) == ref.Int32sl[len(signed)].build(signed)


def test_parallel_integer_encode_threshold_matches_expected_bytes():
    for count in (3_999_999, 4_000_003):
        values = np.arange(count + 1, dtype=np.uint64)[1:]
        expected = values.astype("<u4").tobytes()
        assert mc.Int32ul[count].build(values) == expected


def test_float_special_values_match_upstream():
    values = [float("-inf"), -0.0, 0.0, float("inf"), float("nan")]
    for name in ("Float32b", "Float32l", "Float64b", "Float64l"):
        ours, theirs = getattr(mc, name), getattr(ref, name)
        blob = theirs[len(values)].build(values)
        got = ours[len(values)].parse(blob)
        assert got[:4] == theirs[len(values)].parse(blob)[:4]
        assert math.isnan(got[-1])


def test_error_paths_are_construct_errors():
    with pytest.raises(mc.StreamError):
        mc.Int32ub.parse(b"\x00")
    with pytest.raises(mc.FormatFieldError):
        mc.Int16ub[2].build([0, 70_000])
    with pytest.raises(mc.ConstError):
        mc.Const(b"OK").parse(b"NO")
    with pytest.raises(mc.CheckError):
        mc.Check(False).parse(b"")
    with pytest.raises(mc.SizeofError):
        mc.VarInt.sizeof()


def test_ffi_helpers_reject_invalid_layouts_before_native_call():
    with pytest.raises(ValueError, match="too short"):
        _lib.decode_ints(b"\x00\x01", 2, 2)
    with pytest.raises(ValueError, match="stride"):
        _lib.decode_ints(b"\x00" * 8, 2, 4, stride=3)
    with pytest.raises(ValueError, match="unsupported"):
        _lib.decode_floats(b"\x00" * 4, 1, 2)
    with pytest.raises(ValueError, match="non-negative"):
        _lib.decode_varints(b"\x00", -1)


def test_ffi_helpers_support_padded_strides_without_overread():
    encoded = _lib.encode_ints([0x1234, 0x5678], 2, stride=4, big=True)
    assert encoded.tobytes() == b"\x12\x34\x00\x00\x56\x78\x00\x00"
    assert _lib.decode_ints(encoded, 2, 2, stride=4, big=True).tolist() == [
        0x1234, 0x5678,
    ]


def test_bulk_integer_and_varint_builds_reject_silent_narrowing():
    with pytest.raises(mc.FormatFieldError):
        mc.Int16ub[1].build([1.5])
    with pytest.raises(TypeError, match="integers"):
        _lib.encode_ints([1.5], 2)
    with pytest.raises(TypeError, match="integers"):
        _lib.encode_varints([1.5])
    with pytest.raises(ValueError, match="unsigned"):
        _lib.encode_varints([-1])


def test_ffi_helpers_reject_non_contiguous_and_complex_inputs():
    non_contiguous = memoryview(b"\x00\x01\x02\x03")[::2]
    with pytest.raises(TypeError, match="contiguous"):
        _lib.decode_ints(non_contiguous, 1, 1)
    with pytest.raises(TypeError, match="real"):
        _lib.encode_floats([1 + 2j], 8)


def test_documented_control_and_sizing_constructs_have_parity():
    ours = mc.Struct(
        "kind" / mc.Byte,
        "value" / mc.Switch(mc.this.kind, {1: mc.Int16ub}, default=mc.Int32ul),
        "optional" / mc.Optional(mc.Byte),
        mc.Pass,
    )
    theirs = ref.Struct(
        "kind" / ref.Byte,
        "value" / ref.Switch(ref.this.kind, {1: ref.Int16ub}, default=ref.Int32ul),
        "optional" / ref.Optional(ref.Byte),
        ref.Pass,
    )
    for blob in (b"\x01\x12\x34\x07", b"\x02\x78\x56\x34\x12"):
        assert visible(ours.parse(blob)) == visible(theirs.parse(blob))

    sized_ours = mc.FixedSized(4, mc.Bytes(2))
    sized_theirs = ref.FixedSized(4, ref.Bytes(2))
    assert sized_ours.build(b"OK") == sized_theirs.build(b"OK")
    assert sized_ours.parse(b"OKxx") == sized_theirs.parse(b"OKxx")
    assert mc.Padding(3, pattern=b"x").build(None) == \
        ref.Padding(3, pattern=b"x").build(None)


def test_documented_file_methods_and_compile(tmp_path):
    schema = mc.Int32ub[3]
    path = tmp_path / "values.bin"
    schema.build_file([1, 2, 3], path)
    assert schema.parse_file(path) == [1, 2, 3]
    assert schema.compile() is schema


def test_malformed_bulk_varint_reports_failure_without_partial_array():
    overflow = b"\xff" * 9 + b"\x02"
    values, consumed, status = _lib.decode_varints(overflow, 1)
    assert values.tolist() == []
    assert consumed == 0
    assert status == -2
    with pytest.raises(mc.StreamError):
        mc.VarInt[1].parse(overflow)
