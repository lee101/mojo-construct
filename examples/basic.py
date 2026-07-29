from mojo_construct import Byte, Const, Rebuild, Struct, VarInt, len_, this


Packet = Struct(
    "magic" / Const(b"MC"),
    "count" / Rebuild(Byte, len_(this.values)),
    "values" / VarInt[this.count],
)

blob = Packet.build({"values": [1, 128, 16384]})
assert blob.hex() == "4d4303018001808001"
assert Packet.parse(blob)["values"] == [1, 128, 16384]

print(blob.hex())
