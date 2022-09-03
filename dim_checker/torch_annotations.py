from torch import TensorType
from typing import Iterable
from dim_checker.types import *


def arange_annotate(args):
    pass


def ones_annotate(shape: Union[int, Iterable[int]], *, dtype=None) -> InternalTensor:
    if isinstance(shape, int):
        final_shape = (shape, )
    else:
        final_shape = tuple(shape)  # type: ignore
    return InternalTensor(final_shape)


def zeros_annotate(shape: Union[int, Iterable[int]], *, dtype=None) -> InternalTensor:
    if isinstance(shape, int):
        final_shape = (shape, )
    else:
        final_shape = tuple(shape)  # type: ignore
    return InternalTensor(final_shape)


TorchType = Module({
    'arange': Dependent(arange_annotate),
    'zeros': Dependent(arange_annotate)
})


