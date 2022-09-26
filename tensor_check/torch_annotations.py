from tensor_check import types


ones_type = types.Function(
    (("size", types.InternalInt()),),
    types.InternalTensor((types.InternalInt(), types.InternalInt())),
)

ones_type.constraints = [
    types.Equal("size", "$ret.shape[0]"),
    types.Equal("size", "$ret.shape[1]"),
]

TorchType = types.Module({"ones": ones_type})
