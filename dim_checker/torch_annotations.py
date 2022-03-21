from dim_checker.types import Allowed, InternalInt, InternalFloat, InternalTensor


def constructor(args) -> InternalTensor:
    shape_arg = args[0]
    if isinstance(shape_arg, tuple):
        return InternalTensor(shape_arg)
    elif isinstance(shape_arg, int):
        return InternalTensor((shape_arg, ))
    else:
        raise TypeError


def arange_type(args) -> InternalTensor:
    size = int((args[1] - args[0]) / args[2])
    return InternalTensor((size, ))


def range_type(args) -> InternalTensor:
    size = int((args[1] - args[0]) / args[2]) + 1
    return InternalTensor((size, ))


def eye_type(args) -> InternalTensor:
    if args[1] is None:
        return InternalTensor((args[0], args[0]))
    else:
        return InternalTensor((args[0], args[1]))


METHOD_MAP = {
    'zeros': constructor,
    'ones': constructor,
    'arange': arange_type,
    'range': range_type,
    'eye': eye_type,
}