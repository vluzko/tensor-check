"""Type definitions.
This would be much nicer with tagged unions, but Python doesn't have them. So we define a class hierarchy instead.
"""
from typing import Any, Callable, Tuple, List, Dict, Union, Optional
from dataclasses import dataclass


@dataclass
class Predicate:
    pass


class Self(Predicate):
    pass


@dataclass
class Equal(Predicate):
    lhs: Any
    rhs: Any


class ChkType:
    attributes: Dict[str, "ChkType"]
    constraints: List[Predicate]

    def __add__(self, other: Any):
        raise NotImplementedError

    def __radd__(self, other: Any):
        raise NotImplementedError

    def __sub__(self, other: Any):
        raise NotImplementedError

    def __mul__(self, other: Any):
        raise NotImplementedError

    def __truediv__(self, other: Any):
        raise NotImplementedError

    def bind(self, variable_map: Dict[str, "ChkType"]):
        raise NotImplementedError


class NoneType(ChkType):
    pass


class Function(ChkType):
    arg_types: Tuple[Tuple[str, ChkType], ...]
    ret_type: ChkType

    def __init__(self, arg_types: Tuple[Tuple[str, ChkType], ...], ret_type: ChkType):
        self.arg_types = arg_types
        self.ret_type = ret_type

    def bind(self, variable_map: Dict[str, "ChkType"]) -> ChkType:
        return self.ret_type.bind(variable_map)


class InternalInt(ChkType):
    def __add__(self, other: ChkType):
        if isinstance(other, InternalInt):
            return InternalInt
        elif isinstance(other, InternalFloat):
            return InternalFloat
        elif isinstance(other, InternalTensor):
            return other
        else:
            raise TypeError

    def __sub__(self, other: ChkType):
        if isinstance(other, InternalInt):
            return InternalInt
        elif isinstance(other, InternalFloat):
            return InternalFloat
        elif isinstance(other, InternalTensor):
            return other
        else:
            raise TypeError

    def __mul__(self, other: ChkType):
        if isinstance(other, InternalInt):
            return InternalInt
        elif isinstance(other, InternalFloat):
            return InternalFloat
        elif isinstance(other, InternalTensor):
            return other
        else:
            raise TypeError

    def __truediv__(self, other: ChkType):
        if isinstance(other, InternalInt):
            return InternalFloat
        elif isinstance(other, InternalFloat):
            return InternalFloat
        elif isinstance(other, InternalTensor):
            return other
        else:
            raise TypeError


class InternalFloat(ChkType):
    def __add__(self, other: ChkType):
        if isinstance(other, InternalFloat):
            return InternalFloat
        elif isinstance(other, InternalInt):
            return InternalFloat
        elif isinstance(other, InternalTensor):
            return other
        else:
            raise TypeError


class InternalTensor(ChkType):
    shape: Tuple[InternalInt, ...]
    dtype: Optional[str] = None
    device: Optional[str] = "cpu"

    def __init__(self, shape: Tuple[InternalInt, ...]):
        self.shape = shape

    def __add__(self, other: "InternalTensor"):
        if self.shape == other.shape:
            return InternalTensor(self.shape)
        else:
            raise TypeError("Tried to add tensors with incompatible shapes.")

    def __sub__(self, other: "InternalTensor"):
        return self.__add__(other)

    def __mul__(self, other: "InternalTensor"):
        # Only implemented for two dimensions
        if (
            len(self.shape) == 2
            and len(other.shape) == 2
            and self.shape[1] == other.shape[0]
        ):
            return InternalTensor((self.shape[0], other.shape[1]))
        else:
            raise TypeError

    def __truediv__(self, other: "InternalTensor"):
        return self.__mul__(other)

    def bind(self, variable_map: Dict[str, "ChkType"]) -> ChkType:
        new_shape = tuple(x.bind(variable_map) for x in self.shape)
        return InternalTensor(new_shape)


class TorchModule(ChkType):
    attributes: Dict[str, ChkType]
    forward: Function


class Module(ChkType):
    attributes: Dict[str, ChkType]

    def __init__(self, attributes: Dict[str, ChkType]):
        self.attributes = attributes


def broadcast(
    type_1: InternalTensor, type_2: InternalTensor
) -> Optional[InternalTensor]:
    # Arrange by length
    if len(type_1.shape) > len(type_2.shape):
        t1, t2 = type_1, type_2
    else:
        t1, t2 = type_2, type_1

    new_shape = []
    for i in range(len(t2.shape)):
        if t2.shape[-i] == t1.shape[-i]:
            new_shape.append(t2.shape[-i])
        elif t2.shape[-i] == 1:
            new_shape.append(t1.shape[-i])
        elif t1.shape[-i] == 1:
            new_shape.append(t2.shape[-i])
        else:
            return None
    for j in range(len(t2.shape), len(t1.shape)):
        new_shape.append(t1.shape[-j])

    final_shape = tuple(reversed(new_shape))
    return InternalTensor(final_shape)


def from_pyre(pyre_type: Any) -> ChkType:
    """Convert a Pyre annotation to a ChkType."""
    if pyre_type == "int":
        return InternalInt()
    else:
        raise NotImplementedError
