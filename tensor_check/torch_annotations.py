from tensor_check import tc_types


ones_type = tc_types.Function(
    (("size", tc_types.InternalInt()),),
    tc_types.InternalTensor((tc_types.InternalInt(), tc_types.InternalInt())),
)

ones_type.constraints = [
    tc_types.Equal("size", "$ret.shape[0]"),
    tc_types.Equal("size", "$ret.shape[1]"),
]

TorchType = tc_types.Module({"ones": ones_type})
