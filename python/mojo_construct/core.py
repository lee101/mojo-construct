from __future__ import annotations

import io
import operator
import struct
from collections import OrderedDict

import numpy as np

from . import _lib


class ConstructError(Exception):
    pass


class StreamError(ConstructError):
    pass


class FormatFieldError(ConstructError):
    pass


class IntegerError(ConstructError):
    pass


class StringError(ConstructError):
    pass


class ConstError(ConstructError):
    pass


class CheckError(ConstructError):
    pass


class RangeError(ConstructError):
    pass


class SizeofError(ConstructError):
    pass


class ValidationError(ConstructError):
    pass


class PaddingError(ConstructError):
    pass


class RawCopyError(ConstructError):
    pass


class ChecksumError(CheckError):
    pass


class Container(OrderedDict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __repr__(self):
        body = ", ".join(
            f"{k}={v!r}" for k, v in self.items()
            if not (isinstance(k, str) and k.startswith("_"))
        )
        return f"Container({body})"

    def __eq__(self, other):
        if not isinstance(other, dict):
            return False
        visible = lambda d: {
            k: v for k, v in d.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
        return visible(self) == visible(other)


class ListContainer(list):
    def __repr__(self):
        return f"ListContainer({list.__repr__(self)})"


def _read(stream, length: int) -> bytes:
    if length < 0:
        raise StreamError("negative read length")
    data = stream.read(length)
    if len(data) != length:
        raise StreamError(
            f"stream read less than specified amount, expected {length}, found {len(data)}"
        )
    return data


def _write(stream, data) -> None:
    if isinstance(data, np.ndarray) and not isinstance(stream, io.BytesIO):
        data = data.tobytes()
    if stream.write(data) != len(data):
        raise StreamError("stream wrote less than specified amount")


def _eval(value, context):
    return value(context) if callable(value) else value


class Expr:
    def __init__(self, fn, text="this"):
        self.fn = fn
        self.text = text

    def __call__(self, context):
        return self.fn(context)

    def __getattr__(self, name):
        return Expr(
            lambda c: (
                self(c)[name]
                if isinstance(self(c), dict) and name in self(c)
                else getattr(self(c), name)
            ),
            f"{self.text}.{name}",
        )

    def __getitem__(self, key):
        return Expr(lambda c: self(c)[_eval(key, c)], f"{self.text}[{key!r}]")

    def _binary(self, other, op, symbol):
        return Expr(lambda c: op(self(c), _eval(other, c)),
                    f"({self.text} {symbol} {other!r})")

    def _rbinary(self, other, op, symbol):
        return Expr(lambda c: op(_eval(other, c), self(c)),
                    f"({other!r} {symbol} {self.text})")

    def __add__(self, x): return self._binary(x, operator.add, "+")
    def __radd__(self, x): return self._rbinary(x, operator.add, "+")
    def __sub__(self, x): return self._binary(x, operator.sub, "-")
    def __rsub__(self, x): return self._rbinary(x, operator.sub, "-")
    def __mul__(self, x): return self._binary(x, operator.mul, "*")
    def __rmul__(self, x): return self._rbinary(x, operator.mul, "*")
    def __truediv__(self, x): return self._binary(x, operator.truediv, "/")
    def __floordiv__(self, x): return self._binary(x, operator.floordiv, "//")
    def __mod__(self, x): return self._binary(x, operator.mod, "%")
    def __and__(self, x): return self._binary(x, operator.and_, "&")
    def __or__(self, x): return self._binary(x, operator.or_, "|")
    def __xor__(self, x): return self._binary(x, operator.xor, "^")
    def __lshift__(self, x): return self._binary(x, operator.lshift, "<<")
    def __rshift__(self, x): return self._binary(x, operator.rshift, ">>")
    def __eq__(self, x): return self._binary(x, operator.eq, "==")
    def __ne__(self, x): return self._binary(x, operator.ne, "!=")
    def __lt__(self, x): return self._binary(x, operator.lt, "<")
    def __le__(self, x): return self._binary(x, operator.le, "<=")
    def __gt__(self, x): return self._binary(x, operator.gt, ">")
    def __ge__(self, x): return self._binary(x, operator.ge, ">=")
    def __neg__(self): return Expr(lambda c: -self(c), f"-{self.text}")
    def __invert__(self): return Expr(lambda c: ~self(c), f"~{self.text}")
    def __repr__(self): return self.text


this = Expr(lambda c: c)
obj_ = Expr(lambda c: c.get("_obj"))
list_ = Expr(lambda c: c.get("_list"))


def len_(value):
    return Expr(lambda c: len(_eval(value, c)), f"len_({value!r})")


class Construct:
    name = None
    flagbuildnone = False

    def parse(self, data, **contextkw):
        return self.parse_stream(io.BytesIO(data), **contextkw)

    def parse_stream(self, stream, **contextkw):
        context = Container(contextkw)
        context._ = context
        context._root = context
        context._parsing = True
        context._building = False
        return self._parse(stream, context)

    def parse_file(self, filename, **contextkw):
        with open(filename, "rb") as stream:
            return self.parse_stream(stream, **contextkw)

    def build(self, obj=None, **contextkw):
        stream = io.BytesIO()
        self.build_stream(obj, stream, **contextkw)
        return stream.getvalue()

    def build_stream(self, obj, stream, **contextkw):
        context = Container(contextkw)
        context._ = context
        context._root = context
        context._parsing = False
        context._building = True
        self._build(obj, stream, context)

    def build_file(self, obj, filename, **contextkw):
        with open(filename, "w+b") as stream:
            self.build_stream(obj, stream, **contextkw)

    def sizeof(self, **contextkw):
        context = Container(contextkw)
        context._ = context
        context._root = context
        try:
            return self._sizeof(context)
        except (KeyError, AttributeError, TypeError) as exc:
            raise SizeofError("cannot calculate size") from exc

    def compile(self, filename=None):
        return self

    def __rtruediv__(self, name):
        if not isinstance(name, str):
            return NotImplemented
        return Renamed(self, name)

    def __getitem__(self, count):
        return Array(count, self)

    def __repr__(self):
        suffix = f" {self.name}" if self.name else ""
        return f"<{type(self).__name__}{suffix}>"


class Renamed(Construct):
    def __init__(self, subcon, name):
        self.subcon = subcon
        self.name = name
        self.flagbuildnone = subcon.flagbuildnone

    def _parse(self, stream, context):
        return self.subcon._parse(stream, context)

    def _build(self, obj, stream, context):
        return self.subcon._build(obj, stream, context)

    def _sizeof(self, context):
        return self.subcon._sizeof(context)

    def __getattr__(self, name):
        return getattr(self.subcon, name)


class Subconstruct(Construct):
    def __init__(self, subcon):
        if not isinstance(subcon, Construct):
            raise TypeError("subcon should be a Construct field")
        self.subcon = subcon
        self.flagbuildnone = subcon.flagbuildnone

    def _parse(self, stream, context):
        return self.subcon._parse(stream, context)

    def _build(self, obj, stream, context):
        return self.subcon._build(obj, stream, context)

    def _sizeof(self, context):
        return self.subcon._sizeof(context)


class Adapter(Subconstruct):
    def _decode(self, obj, context):
        raise NotImplementedError

    def _encode(self, obj, context):
        raise NotImplementedError

    def _parse(self, stream, context):
        return self._decode(self.subcon._parse(stream, context), context)

    def _build(self, obj, stream, context):
        encoded = self._encode(obj, context)
        self.subcon._build(encoded, stream, context)
        return obj


class ExprAdapter(Adapter):
    def __init__(self, subcon, decoder, encoder):
        super().__init__(subcon)
        self.decoder = decoder
        self.encoder = encoder

    def _decode(self, obj, context):
        return self.decoder(obj, context)

    def _encode(self, obj, context):
        return self.encoder(obj, context)


class Validator(Adapter):
    def _validate(self, obj, context):
        raise NotImplementedError

    def _decode(self, obj, context):
        if not self._validate(obj, context):
            raise ValidationError(f"validation failed for {obj!r}")
        return obj

    def _encode(self, obj, context):
        if not self._validate(obj, context):
            raise ValidationError(f"validation failed for {obj!r}")
        return obj


class ExprValidator(Validator):
    def __init__(self, subcon, validator):
        super().__init__(subcon)
        self.validator = validator

    def _validate(self, obj, context):
        return self.validator(obj, context)


class OneOf(ExprValidator):
    def __init__(self, subcon, valids):
        self.valids = valids
        super().__init__(subcon, lambda obj, ctx: obj in self.valids)


class NoneOf(ExprValidator):
    def __init__(self, subcon, invalids):
        self.invalids = invalids
        super().__init__(subcon, lambda obj, ctx: obj not in self.invalids)


class FormatField(Construct):
    def __init__(self, endianity, format):
        if endianity not in "=<>" or format not in "fdBHLQbhlqe?":
            raise FormatFieldError("invalid struct format")
        self.fmtstr = endianity + format
        self.length = struct.calcsize(self.fmtstr)
        self.format = format
        self.big = endianity == ">"

    def _parse(self, stream, context):
        if isinstance(stream, _BitReader):
            raw = stream.read_bits(self.length * 8)
            data = raw.to_bytes(self.length, "big")
        else:
            data = _read(stream, self.length)
        try:
            return struct.unpack(self.fmtstr, data)[0]
        except struct.error as exc:
            raise FormatFieldError(str(exc)) from exc

    def _build(self, obj, stream, context):
        try:
            data = struct.pack(self.fmtstr, obj)
        except (struct.error, TypeError, OverflowError) as exc:
            raise FormatFieldError(str(exc)) from exc
        if isinstance(stream, _BitWriter):
            stream.write_bits(int.from_bytes(data, "big"), self.length * 8)
        else:
            _write(stream, data)
        return obj

    def _sizeof(self, context):
        return self.length

    def _bulk(self):
        if self.format in "BHLQbhlq":
            return ("int", self.length, self.big, self.format.islower())
        if self.format in "fd":
            return ("float", self.length, self.big, False)
        return None


class BytesInteger(Construct):
    def __init__(self, length, signed=False, swapped=False):
        self.length = length
        self.signed = signed
        self.swapped = swapped

    def _length(self, context):
        return int(_eval(self.length, context))

    def _parse(self, stream, context):
        length = self._length(context)
        data = _read(stream, length)
        return int.from_bytes(data, "little" if self.swapped else "big",
                              signed=self.signed)

    def _build(self, obj, stream, context):
        length = self._length(context)
        try:
            data = int(obj).to_bytes(
                length, "little" if self.swapped else "big", signed=self.signed
            )
        except (OverflowError, ValueError) as exc:
            raise IntegerError(str(exc)) from exc
        _write(stream, data)
        return obj

    def _sizeof(self, context):
        return self._length(context)

    def _bulk(self):
        if isinstance(self.length, int) and self.length in (1, 2, 3, 4, 8):
            return ("int", self.length, not self.swapped, self.signed)


class Bytes(Construct):
    def __init__(self, length):
        self.length = length

    def _length(self, context):
        return int(_eval(self.length, context))

    def _parse(self, stream, context):
        return _read(stream, self._length(context))

    def _build(self, obj, stream, context):
        length = self._length(context)
        if isinstance(obj, int):
            obj = bytes(length) if obj == 0 else obj.to_bytes(length, "big")
        if not isinstance(obj, (bytes, bytearray)):
            raise StringError("expected bytes")
        if len(obj) != length:
            raise StreamError(f"expected {length} bytes, found {len(obj)}")
        _write(stream, bytes(obj))
        return obj

    def _sizeof(self, context):
        return self._length(context)


class GreedyBytesType(Construct):
    def _parse(self, stream, context):
        return stream.read()

    def _build(self, obj, stream, context):
        if not isinstance(obj, (bytes, bytearray)):
            raise StringError("expected bytes")
        _write(stream, bytes(obj))
        return obj

    def _sizeof(self, context):
        raise SizeofError("greedy field has no fixed size")


class Array(Construct):
    def __init__(self, count, subcon, discard=False):
        self.count = count
        self.subcon = subcon
        self.discard = discard

    def _count(self, context):
        count = int(_eval(self.count, context))
        if count < 0:
            raise RangeError("negative array count")
        return count

    def _parse(self, stream, context):
        count = self._count(context)
        bulk = getattr(self.subcon, "_bulk", lambda: None)()
        if bulk and not isinstance(stream, _BitReader):
            kind, width, big, signed = bulk
            data = _read(stream, count * width)
            if kind == "int":
                values = _lib.decode_ints(
                    data, count, width, big=big, signed=signed
                )
            else:
                values = _lib.decode_floats(data, count, width, big=big)
            return ListContainer() if self.discard else ListContainer(values.tolist())
        if self.subcon is VarInt and not isinstance(stream, _BitReader):
            start = stream.tell()
            data = stream.read()
            values, consumed, status = _lib.decode_varints(data, count)
            if status or len(values) != count:
                stream.seek(start)
                raise StreamError("could not read enough varints")
            stream.seek(start + consumed)
            return ListContainer() if self.discard else ListContainer(values.tolist())
        result = ListContainer()
        old_index = context.get("_index")
        for index in range(count):
            context._index = index
            value = self.subcon._parse(stream, context)
            if not self.discard:
                result.append(value)
        if old_index is not None:
            context._index = old_index
        return result

    def _build(self, obj, stream, context):
        count = self._count(context)
        if obj is None or len(obj) != count:
            raise RangeError(f"expected {count} items")
        bulk = getattr(self.subcon, "_bulk", lambda: None)()
        if bulk and not isinstance(stream, _BitWriter):
            kind, width, big, signed = bulk
            if kind == "int":
                low = -(1 << (width * 8 - 1)) if signed else 0
                high = (1 << (width * 8 - (1 if signed else 0))) - 1
                try:
                    source = _lib._integer_source(obj)
                    if count and (source.min() < low or source.max() > high):
                        raise ValueError
                    values = np.ascontiguousarray(
                        source, dtype=np.int64 if signed else np.uint64
                    )
                except (OverflowError, TypeError, ValueError) as exc:
                    raise FormatFieldError("integer outside field range") from exc
                encoded = _lib.encode_ints(values, width, big=big)
            else:
                encoded = _lib.encode_floats(obj, width, big=big)
            _write(stream, encoded)
            return obj
        if self.subcon is VarInt and not isinstance(stream, _BitWriter):
            values = [int(value) for value in obj]
            if any(value < 0 or value > (1 << 64) - 1 for value in values):
                raise IntegerError("varint outside uint64 range")
            _write(stream, _lib.encode_varints(values))
            return obj
        for index, value in enumerate(obj):
            context._index = index
            self.subcon._build(value, stream, context)
        return obj

    def _sizeof(self, context):
        return self._count(context) * self.subcon._sizeof(context)


class GreedyRange(Construct):
    def __init__(self, subcon, discard=False):
        self.subcon = subcon
        self.discard = discard

    def _parse(self, stream, context):
        bulk = getattr(self.subcon, "_bulk", lambda: None)()
        if bulk and not isinstance(stream, _BitReader):
            kind, width, big, signed = bulk
            start = stream.tell()
            data = stream.read()
            count, remainder = divmod(len(data), width)
            if remainder:
                stream.seek(start + count * width)
            if kind == "int":
                values = _lib.decode_ints(
                    data[:count * width], count, width, big=big, signed=signed
                )
            else:
                values = _lib.decode_floats(
                    data[:count * width], count, width, big=big
                )
            return ListContainer() if self.discard else ListContainer(values.tolist())
        if self.subcon is VarInt and not isinstance(stream, _BitReader):
            start = stream.tell()
            data = stream.read()
            values, consumed, status = _lib.decode_varints(data, len(data))
            if status:
                stream.seek(start + consumed)
            return ListContainer() if self.discard else ListContainer(values.tolist())
        result = ListContainer()
        index = 0
        while True:
            pos = stream.tell()
            try:
                context._index = index
                value = self.subcon._parse(stream, context)
            except ConstructError:
                stream.seek(pos)
                break
            if not self.discard:
                result.append(value)
            index += 1
        return result

    def _build(self, obj, stream, context):
        return Array(len(obj), self.subcon, self.discard)._build(obj, stream, context)

    def _sizeof(self, context):
        raise SizeofError("greedy range has no fixed size")


class Struct(Construct):
    def __init__(self, *subcons, **subconskw):
        self.subcons = list(subcons)
        self.subcons.extend(Renamed(value, key) for key, value in subconskw.items())

    def _child(self, context, values=None):
        child = Container(values or {})
        child._ = context
        child._root = context.get("_root", context)
        child._parsing = context.get("_parsing", False)
        child._building = context.get("_building", False)
        return child

    def _parse(self, stream, context):
        result = self._child(context)
        for subcon in self.subcons:
            value = subcon._parse(stream, result)
            if subcon.name:
                result[subcon.name] = value
        return result

    def _build(self, obj, stream, context):
        if obj is None:
            obj = {}
        if not isinstance(obj, dict):
            obj = vars(obj)
        child = self._child(context, obj)
        for subcon in self.subcons:
            if subcon.name and subcon.name in child:
                value = child[subcon.name]
            elif subcon.flagbuildnone:
                value = None
            elif subcon.name:
                raise KeyError(subcon.name)
            else:
                value = None
            built = subcon._build(value, stream, child)
            if subcon.name:
                child[subcon.name] = built
        return child

    def _sizeof(self, context):
        child = self._child(context)
        return sum(subcon._sizeof(child) for subcon in self.subcons)


class Sequence(Construct):
    def __init__(self, *subcons, **subconskw):
        self.subcons = list(subcons)
        self.subcons.extend(Renamed(value, key) for key, value in subconskw.items())

    def _parse(self, stream, context):
        result = ListContainer()
        context._list = result
        for index, subcon in enumerate(self.subcons):
            context._index = index
            result.append(subcon._parse(stream, context))
        return result

    def _build(self, obj, stream, context):
        if len(obj) != len(self.subcons):
            raise RangeError("sequence length mismatch")
        for index, (subcon, value) in enumerate(zip(self.subcons, obj)):
            context._index = index
            subcon._build(value, stream, context)
        return obj

    def _sizeof(self, context):
        return sum(subcon._sizeof(context) for subcon in self.subcons)


class Const(Construct):
    def __init__(self, value, subcon=None):
        self.value = value
        self.subcon = subcon or Bytes(len(value))
        self.flagbuildnone = True

    def _parse(self, stream, context):
        value = self.subcon._parse(stream, context)
        if value != self.value:
            raise ConstError(f"expected {self.value!r}, found {value!r}")
        return value

    def _build(self, obj, stream, context):
        if obj is not None and obj != self.value:
            raise ConstError(f"expected {self.value!r}, found {obj!r}")
        return self.subcon._build(self.value, stream, context)

    def _sizeof(self, context):
        return self.subcon._sizeof(context)


class Computed(Construct):
    flagbuildnone = True

    def __init__(self, func):
        self.func = func

    def _parse(self, stream, context):
        return _eval(self.func, context)

    def _build(self, obj, stream, context):
        return _eval(self.func, context)

    def _sizeof(self, context):
        return 0


class Check(Computed):
    def _parse(self, stream, context):
        if not _eval(self.func, context):
            raise CheckError("check failed during parsing")

    def _build(self, obj, stream, context):
        if not _eval(self.func, context):
            raise CheckError("check failed during building")


class Default(Construct):
    flagbuildnone = True

    def __init__(self, subcon, value):
        self.subcon = subcon
        self.value = value

    def _parse(self, stream, context):
        return self.subcon._parse(stream, context)

    def _build(self, obj, stream, context):
        return self.subcon._build(
            _eval(self.value, context) if obj is None else obj, stream, context
        )

    def _sizeof(self, context):
        return self.subcon._sizeof(context)


class Rebuild(Default):
    def _build(self, obj, stream, context):
        return self.subcon._build(_eval(self.value, context), stream, context)


class IfThenElse(Construct):
    def __init__(self, condfunc, thensubcon, elsesubcon):
        self.condfunc = condfunc
        self.thensubcon = thensubcon
        self.elsesubcon = elsesubcon
        self.flagbuildnone = thensubcon.flagbuildnone and elsesubcon.flagbuildnone

    def _select(self, context):
        return self.thensubcon if _eval(self.condfunc, context) else self.elsesubcon

    def _parse(self, stream, context):
        return self._select(context)._parse(stream, context)

    def _build(self, obj, stream, context):
        return self._select(context)._build(obj, stream, context)

    def _sizeof(self, context):
        return self._select(context)._sizeof(context)


class Switch(Construct):
    def __init__(self, keyfunc, cases, default=None):
        self.keyfunc = keyfunc
        self.cases = cases
        self.default = Pass if default is None else default
        self.flagbuildnone = all(
            item.flagbuildnone for item in [*cases.values(), self.default]
        )

    def _select(self, context):
        return self.cases.get(_eval(self.keyfunc, context), self.default)

    def _parse(self, stream, context):
        return self._select(context)._parse(stream, context)

    def _build(self, obj, stream, context):
        return self._select(context)._build(obj, stream, context)

    def _sizeof(self, context):
        return self._select(context)._sizeof(context)


class PassType(Construct):
    flagbuildnone = True

    def _parse(self, stream, context): return None
    def _build(self, obj, stream, context): return None
    def _sizeof(self, context): return 0


class TellType(PassType):
    def _parse(self, stream, context): return stream.tell()
    def _build(self, obj, stream, context): return stream.tell()


class VarIntType(Construct):
    def _parse(self, stream, context):
        value = 0
        for shift in range(0, 70, 7):
            byte = _read(stream, 1)[0]
            if shift == 63 and byte & 0x7E:
                raise IntegerError("varint exceeds uint64")
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
        raise IntegerError("varint exceeds uint64")

    def _build(self, obj, stream, context):
        value = int(obj)
        if value < 0 or value > (1 << 64) - 1:
            raise IntegerError("varint outside uint64 range")
        data = bytearray()
        while value >= 0x80:
            data.append((value & 0x7F) | 0x80)
            value >>= 7
        data.append(value)
        _write(stream, bytes(data))
        return obj

    def _sizeof(self, context):
        raise SizeofError("varint has no fixed size")


class ZigZagType(Construct):
    def _parse(self, stream, context):
        value = VarInt._parse(stream, context)
        return (value >> 1) ^ -(value & 1)

    def _build(self, obj, stream, context):
        value = int(obj)
        encoded = (value << 1) ^ (value >> 63)
        VarInt._build(encoded, stream, context)
        return obj

    def _sizeof(self, context):
        raise SizeofError("zigzag has no fixed size")


class EnumInteger(int):
    pass


class EnumIntegerString(str):
    def __new__(cls, value, intvalue):
        obj = str.__new__(cls, value)
        obj.intvalue = intvalue
        return obj

    def __int__(self):
        return self.intvalue


class Enum(Construct):
    def __init__(self, subcon, *merge, **mapping):
        self.subcon = subcon
        self.encmapping = {}
        for item in merge:
            self.encmapping.update(item)
        self.encmapping.update(mapping)
        self.decmapping = {value: key for key, value in self.encmapping.items()}

    def _parse(self, stream, context):
        value = self.subcon._parse(stream, context)
        if value in self.decmapping:
            return EnumIntegerString(self.decmapping[value], value)
        return EnumInteger(value)

    def _build(self, obj, stream, context):
        value = self.encmapping.get(obj, obj)
        self.subcon._build(value, stream, context)
        return obj

    def _sizeof(self, context):
        return self.subcon._sizeof(context)


class FlagsEnum(Construct):
    def __init__(self, subcon, *merge, **flags):
        self.subcon = subcon
        self.flags = {}
        for item in merge:
            self.flags.update(item)
        self.flags.update(flags)

    def _parse(self, stream, context):
        value = self.subcon._parse(stream, context)
        result = Container((name, bool(value & mask))
                           for name, mask in self.flags.items())
        result._flagsenum = True
        return result

    def _build(self, obj, stream, context):
        if isinstance(obj, int):
            value = obj
        else:
            value = 0
            for name, mask in self.flags.items():
                if obj.get(name, False):
                    value |= mask
        self.subcon._build(value, stream, context)
        return obj

    def _sizeof(self, context):
        return self.subcon._sizeof(context)


class FixedSized(Construct):
    def __init__(self, length, subcon, pattern=b"\x00"):
        self.length = length
        self.subcon = subcon
        self.pattern = pattern

    def _length(self, context): return int(_eval(self.length, context))

    def _parse(self, stream, context):
        return self.subcon._parse(io.BytesIO(_read(stream, self._length(context))), context)

    def _build(self, obj, stream, context):
        temp = io.BytesIO()
        value = self.subcon._build(obj, temp, context)
        data = temp.getvalue()
        length = self._length(context)
        if len(data) > length:
            raise PaddingError("subconstruct exceeds fixed size")
        padding = (self.pattern * ((length - len(data) + len(self.pattern) - 1)
                                   // len(self.pattern)))[:length - len(data)]
        _write(stream, data + padding)
        return value

    def _sizeof(self, context): return self._length(context)


class Padding(Construct):
    flagbuildnone = True

    def __init__(self, length, pattern=b"\x00"):
        self.length = length
        self.pattern = pattern

    def _length(self, context): return int(_eval(self.length, context))

    def _parse(self, stream, context):
        if isinstance(stream, _BitReader):
            stream.read_bits(self._length(context))
        else:
            _read(stream, self._length(context))

    def _build(self, obj, stream, context):
        length = self._length(context)
        if isinstance(stream, _BitWriter):
            stream.write_bits(0, length)
        else:
            _write(stream, (self.pattern * ((length + len(self.pattern) - 1)
                                             // len(self.pattern)))[:length])

    def _sizeof(self, context): return self._length(context)


class PaddedStringConstruct(Construct):
    def __init__(self, length, encoding):
        self.length = length
        self.encoding = encoding

    def _length(self, context): return int(_eval(self.length, context))

    def _parse(self, stream, context):
        return _read(stream, self._length(context)).rstrip(b"\x00").decode(self.encoding)

    def _build(self, obj, stream, context):
        data = obj.encode(self.encoding)
        length = self._length(context)
        if len(data) > length:
            raise PaddingError("string exceeds padded field")
        _write(stream, data + bytes(length - len(data)))
        return obj

    def _sizeof(self, context): return self._length(context)


class CStringConstruct(Construct):
    def __init__(self, encoding):
        self.encoding = encoding
        self.term = "\x00".encode(encoding)

    def _parse(self, stream, context):
        chunks = bytearray()
        while True:
            unit = _read(stream, len(self.term))
            if unit == self.term:
                return bytes(chunks).decode(self.encoding)
            chunks.extend(unit)

    def _build(self, obj, stream, context):
        _write(stream, obj.encode(self.encoding) + self.term)
        return obj

    def _sizeof(self, context): raise SizeofError("cstring has no fixed size")


class Prefixed(Construct):
    def __init__(self, lengthfield, subcon, includelength=False):
        self.lengthfield = lengthfield
        self.subcon = subcon
        self.includelength = includelength

    def _parse(self, stream, context):
        length = self.lengthfield._parse(stream, context)
        if self.includelength:
            length -= self.lengthfield._sizeof(context)
        return self.subcon._parse(io.BytesIO(_read(stream, length)), context)

    def _build(self, obj, stream, context):
        temp = io.BytesIO()
        value = self.subcon._build(obj, temp, context)
        data = temp.getvalue()
        length = len(data)
        if self.includelength:
            length += self.lengthfield._sizeof(context)
        self.lengthfield._build(length, stream, context)
        _write(stream, data)
        return value

    def _sizeof(self, context): raise SizeofError("prefixed size depends on object")


class PrefixedArray(Construct):
    def __init__(self, countfield, subcon):
        self.countfield = countfield
        self.subcon = subcon

    def _parse(self, stream, context):
        return Array(self.countfield._parse(stream, context), self.subcon)._parse(
            stream, context
        )

    def _build(self, obj, stream, context):
        self.countfield._build(len(obj), stream, context)
        return Array(len(obj), self.subcon)._build(obj, stream, context)

    def _sizeof(self, context): raise SizeofError("prefixed array size varies")


class RepeatUntil(Construct):
    def __init__(self, predicate, subcon, discard=False):
        self.predicate = predicate
        self.subcon = subcon
        self.discard = discard

    def _parse(self, stream, context):
        result = ListContainer()
        while True:
            value = self.subcon._parse(stream, context)
            result.append(value)
            if self.predicate(value, result, context):
                return ListContainer() if self.discard else result

    def _build(self, obj, stream, context):
        result = ListContainer()
        for value in obj:
            self.subcon._build(value, stream, context)
            result.append(value)
            if self.predicate(value, result, context):
                return obj
        raise RangeError("predicate never matched")

    def _sizeof(self, context): raise SizeofError("repeat-until size varies")


class Select(Construct):
    def __init__(self, *subcons, includename=False):
        self.subcons = subcons
        self.includename = includename

    def _parse(self, stream, context):
        for subcon in self.subcons:
            pos = stream.tell()
            try:
                value = subcon._parse(stream, context)
                return (subcon.name, value) if self.includename else value
            except ConstructError:
                stream.seek(pos)
        raise ConstructError("no select branch matched")

    def _build(self, obj, stream, context):
        for subcon in self.subcons:
            temp = io.BytesIO()
            try:
                value = subcon._build(obj, temp, context)
            except ConstructError:
                continue
            _write(stream, temp.getvalue())
            return value
        raise ConstructError("no select branch matched")

    def _sizeof(self, context): raise SizeofError("select size varies")


class Seek(Construct):
    flagbuildnone = True

    def __init__(self, at, whence=0):
        self.at = at
        self.whence = whence

    def _parse(self, stream, context): return stream.seek(_eval(self.at, context), self.whence)
    def _build(self, obj, stream, context): return stream.seek(_eval(self.at, context), self.whence)
    def _sizeof(self, context): return 0


class Pointer(Construct):
    def __init__(self, offset, subcon, stream=None):
        self.offset = offset
        self.subcon = subcon
        self.stream = stream

    def _work(self, fn, obj, stream, context):
        target = _eval(self.stream, context) if self.stream is not None else stream
        pos = target.tell()
        target.seek(_eval(self.offset, context))
        try:
            return fn(obj, target, context) if obj is not _MISSING else fn(target, context)
        finally:
            target.seek(pos)

    def _parse(self, stream, context): return self._work(self.subcon._parse, _MISSING, stream, context)
    def _build(self, obj, stream, context): return self._work(self.subcon._build, obj, stream, context)
    def _sizeof(self, context): return 0


class RawCopy(Subconstruct):
    def _parse(self, stream, context):
        offset1 = stream.tell()
        value = self.subcon._parse(stream, context)
        offset2 = stream.tell()
        stream.seek(offset1)
        data = _read(stream, offset2 - offset1)
        return Container(
            data=data,
            value=value,
            offset1=offset1,
            offset2=offset2,
            length=offset2 - offset1,
        )

    def _build(self, obj, stream, context):
        offset1 = stream.tell()
        if "data" in obj:
            data = bytes(obj["data"])
            _write(stream, data)
            result = Container(obj)
        elif "value" in obj:
            value = self.subcon._build(obj["value"], stream, context)
            offset2 = stream.tell()
            stream.seek(offset1)
            data = _read(stream, offset2 - offset1)
            result = Container(obj)
            result["value"] = value
        else:
            raise RawCopyError("RawCopy needs data or value")
        offset2 = stream.tell()
        result.update(data=data, offset1=offset1, offset2=offset2,
                      length=offset2 - offset1)
        return result


class Checksum(Construct):
    flagbuildnone = True

    def __init__(self, checksumfield, hashfunc, bytesfunc):
        self.checksumfield = checksumfield
        self.hashfunc = hashfunc
        self.bytesfunc = bytesfunc

    def _parse(self, stream, context):
        found = self.checksumfield._parse(stream, context)
        expected = self.hashfunc(_eval(self.bytesfunc, context))
        if found != expected:
            raise ChecksumError(
                f"wrong checksum, read {found!r}, computed {expected!r}"
            )
        return found

    def _build(self, obj, stream, context):
        value = self.hashfunc(_eval(self.bytesfunc, context))
        self.checksumfield._build(value, stream, context)
        return value

    def _sizeof(self, context):
        return self.checksumfield._sizeof(context)


class StringEncoded(Subconstruct):
    def __init__(self, subcon, encoding):
        super().__init__(subcon)
        self.encoding = encoding

    def _parse(self, stream, context):
        try:
            return self.subcon._parse(stream, context).decode(self.encoding)
        except UnicodeError as exc:
            raise StringError(str(exc)) from exc

    def _build(self, obj, stream, context):
        try:
            data = obj.encode(self.encoding)
        except (AttributeError, UnicodeError) as exc:
            raise StringError(str(exc)) from exc
        self.subcon._build(data, stream, context)
        return obj


class NullStripped(Subconstruct):
    def __init__(self, subcon, pad=b"\x00"):
        super().__init__(subcon)
        if not isinstance(pad, bytes) or not pad:
            raise PaddingError("pad must be at least one byte")
        self.pad = pad

    def _parse(self, stream, context):
        data = stream.read()
        if len(self.pad) == 1:
            data = data.rstrip(self.pad)
        else:
            tail = len(data) % len(self.pad)
            end = len(data)
            if tail and data[-tail:] == self.pad[:tail]:
                end -= tail
            while end >= len(self.pad) and data[end-len(self.pad):end] == self.pad:
                end -= len(self.pad)
            data = data[:end]
        return self.subcon._parse(io.BytesIO(data), context)

    def _build(self, obj, stream, context):
        return self.subcon._build(obj, stream, context)

    def _sizeof(self, context):
        raise SizeofError("null-stripped field has no fixed size")


class NullTerminated(Subconstruct):
    def __init__(
        self, subcon, term=b"\x00", include=False, consume=True, require=True
    ):
        super().__init__(subcon)
        if not term:
            raise PaddingError("terminator cannot be empty")
        self.term = term
        self.include = include
        self.consume = consume
        self.require = require

    def _parse(self, stream, context):
        data = bytearray()
        while True:
            unit = stream.read(len(self.term))
            if unit == self.term:
                if self.include:
                    data.extend(unit)
                if not self.consume:
                    stream.seek(-len(unit), 1)
                break
            if len(unit) != len(self.term):
                if self.require:
                    raise StreamError("terminator not found")
                data.extend(unit)
                break
            data.extend(unit)
        return self.subcon._parse(io.BytesIO(bytes(data)), context)

    def _build(self, obj, stream, context):
        value = self.subcon._build(obj, stream, context)
        _write(stream, self.term)
        return value

    def _sizeof(self, context):
        raise SizeofError("null-terminated field has no fixed size")


class BitsInteger(Construct):
    def __init__(self, length, signed=False, swapped=False):
        self.length = length
        self.signed = signed
        self.swapped = swapped

    def _length(self, context): return int(_eval(self.length, context))

    def _parse(self, stream, context):
        if not isinstance(stream, _BitReader):
            raise StreamError("BitsInteger must be inside Bitwise or BitStruct")
        length = self._length(context)
        value = stream.read_bits(length)
        if self.signed and value & (1 << (length - 1)):
            value -= 1 << length
        return value

    def _build(self, obj, stream, context):
        if not isinstance(stream, _BitWriter):
            raise StreamError("BitsInteger must be inside Bitwise or BitStruct")
        length = self._length(context)
        value = int(obj)
        low = -(1 << (length - 1)) if self.signed else 0
        high = (1 << (length - (1 if self.signed else 0))) - 1
        if value < low or value > high:
            raise IntegerError("bit integer outside range")
        stream.write_bits(value & ((1 << length) - 1), length)
        return obj

    def _sizeof(self, context): return self._length(context)


class FlagType(Construct):
    def _parse(self, stream, context):
        if isinstance(stream, _BitReader):
            return bool(stream.read_bits(1))
        return bool(_read(stream, 1)[0])

    def _build(self, obj, stream, context):
        if isinstance(stream, _BitWriter):
            stream.write_bits(bool(obj), 1)
        else:
            _write(stream, bytes([1 if obj else 0]))
        return obj

    def _sizeof(self, context): return 1


class _BitReader:
    def __init__(self, data):
        self.data = data
        self.bitpos = 0

    def read_bits(self, length):
        if self.bitpos + length > len(self.data) * 8:
            raise StreamError("not enough bits")
        value = 0
        for _ in range(length):
            byte, bit = divmod(self.bitpos, 8)
            value = (value << 1) | ((self.data[byte] >> (7 - bit)) & 1)
            self.bitpos += 1
        return value

    def tell(self): return self.bitpos


class _BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.bitpos = 0

    def write_bits(self, value, length):
        for shift in range(length - 1, -1, -1):
            if self.bitpos % 8 == 0:
                self.data.append(0)
            if (value >> shift) & 1:
                self.data[-1] |= 1 << (7 - self.bitpos % 8)
            self.bitpos += 1

    def tell(self): return self.bitpos


class Bitwise(Construct):
    def __init__(self, subcon):
        self.subcon = subcon

    def _bits(self, context):
        bits = self.subcon._sizeof(context)
        if bits % 8:
            raise SizeofError("bitwise field must be byte aligned")
        return bits

    def _parse(self, stream, context):
        bits = self._bits(context)
        return self.subcon._parse(_BitReader(_read(stream, bits // 8)), context)

    def _build(self, obj, stream, context):
        writer = _BitWriter()
        value = self.subcon._build(obj, writer, context)
        if writer.bitpos % 8:
            raise StreamError("bitwise output is not byte aligned")
        _write(stream, bytes(writer.data))
        return value

    def _sizeof(self, context): return self._bits(context) // 8


_MISSING = object()
GreedyBytes = GreedyBytesType()
Pass = PassType()
Tell = TellType()
VarInt = VarIntType()
ZigZag = ZigZagType()
Flag = FlagType()
Bit = BitsInteger(1)
Nibble = BitsInteger(4)
Octet = BitsInteger(8)


def If(condfunc, subcon):
    return IfThenElse(condfunc, subcon, Pass)


def Optional(subcon):
    return Select(subcon, Pass)


def Padded(length, subcon, pattern=b"\x00"):
    return FixedSized(length, subcon, pattern)


def PaddedString(length, encoding):
    return PaddedStringConstruct(length, encoding)


def CString(encoding):
    return CStringConstruct(encoding)


def BitStruct(*subcons, **subconskw):
    return Bitwise(Struct(*subcons, **subconskw))


Byte = Int8ub = FormatField(">", "B")
Int8sb = FormatField(">", "b")
Int8ul = FormatField("<", "B")
Int8sl = FormatField("<", "b")
Int8un = FormatField("=", "B")
Int8sn = FormatField("=", "b")
Int16ub = FormatField(">", "H")
Int16sb = FormatField(">", "h")
Int16ul = FormatField("<", "H")
Int16sl = FormatField("<", "h")
Int16un = FormatField("=", "H")
Int16sn = FormatField("=", "h")
Int24ub = BytesInteger(3, signed=False, swapped=False)
Int24sb = BytesInteger(3, signed=True, swapped=False)
Int24ul = BytesInteger(3, signed=False, swapped=True)
Int24sl = BytesInteger(3, signed=True, swapped=True)
Int32ub = FormatField(">", "L")
Int32sb = FormatField(">", "l")
Int32ul = FormatField("<", "L")
Int32sl = FormatField("<", "l")
Int32un = FormatField("=", "L")
Int32sn = FormatField("=", "l")
Int64ub = FormatField(">", "Q")
Int64sb = FormatField(">", "q")
Int64ul = FormatField("<", "Q")
Int64sl = FormatField("<", "q")
Int64un = FormatField("=", "Q")
Int64sn = FormatField("=", "q")
Float16b = FormatField(">", "e")
Float16l = FormatField("<", "e")
Float32b = FormatField(">", "f")
Float32l = FormatField("<", "f")
Float32n = FormatField("=", "f")
Float64b = FormatField(">", "d")
Float64l = FormatField("<", "d")
Float64n = FormatField("=", "d")
Short = Int16ub
Int = Int32ub
Long = Int64ub
Half = Float16b
Single = Float32b
Singlel = Float32l
Double = Float64b
Doublel = Float64l
