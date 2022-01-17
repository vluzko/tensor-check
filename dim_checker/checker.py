import ast
import pdb
from typing import Any, Tuple, Literal, List, Dict, Union, Optional



# TODO: Broadcasting
class TensorType:
    shape: Tuple[int, ...]

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def __add__(self, other: 'TensorType'):
        if self.shape == other.shape:
            return TensorType(self.shape)
        else:
            raise TypeError

    def __sub__(self, other: 'TensorType'):
        return self.__add__(other)

    def __mul__(self, other: 'TensorType'):
        # Only implemented for two dimensions
        if len(self.shape) == 2 and len(other.shape) == 2 and self.shape[1] == other.shape[0]:
            return TensorType((self.shape[0], other.shape[1]))
        else:
            raise TypeError

    def __truediv__(self, other: 'TensorType'):
        return self.__mul__(other)


def broadcast(type_1: TensorType, type_2: TensorType) -> Optional[TensorType]:
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
    return TensorType(final_shape)


AllowedTypes = Union[TensorType, int, float, bool]


class Scope:
    assignments: Dict[str, AllowedTypes]

    def __init__(self, assigments: Dict[str, AllowedTypes] = {}):
        self.assignments = assigments


# TODO: Different scopes
class Context:
    assignments: Dict[str, AllowedTypes]
    scopes: Dict[int, Scope]

    def __init__(self):
        self.assignments = {}

    def __getitem__(self, key: str) -> AllowedTypes:
        return self.assignments[key]

    def __setitem__(self, key, value):
        self.assignments[key] = value

    def new_scope(self):
        self.scopes[len(self.scopes)] = Scope()


# TODO: Imports
class TorchChecker(ast.NodeTransformer):

    def __init__(self):
        super().__init__()
        self.context = Context()

    def get_type(self, node):
        if isinstance(node, ast.Name):
            return self.context[node.id]
        else:
            return node._tensor_type

    def visit_Call(self, node):
        # Check if we're calling torch
        if node.func.value.id != 'torch':
            return node
        else:
            torch_method = node.func.attr
            parsed_args = [get_arg(x) for x in node.args]

            method = METHOD_MAP[torch_method]
            tensor_type = method(parsed_args)
            node._tensor_type = tensor_type
            return node

    def visit_Assign(self, node):
        expr = self.visit(node.value)
        assert len(node.targets) == 1
        name = node.targets[0].id
        self.context[name] = expr._tensor_type
        return node

    def visit_BinOp(self, node):
        left_type = self.get_type(node.left)
        right_type = self.get_type(node.right)

        if isinstance(node.op, ast.Add):
            new_type = left_type + right_type
        elif isinstance(node.op, ast.Sub):
            new_type = left_type - right_type
        elif isinstance(node.op, ast.Mult):
            new_type = left_type * right_type
        elif isinstance(node.op, ast.Div):
            new_type = left_type / right_type
        else:
            # pdb.set_trace()
            raise TypeError

        node._tensor_type = new_type
        return node


def get_arg(arg: ast.expr) -> Any:

    if isinstance(arg, ast.Tuple):
        result = tuple(get_arg(x) for x in arg.elts)
    elif isinstance(arg, ast.Constant):
        result = arg.value

    return result


def constructor(args) -> TensorType:
    shape_arg = args[0]
    if isinstance(shape_arg, tuple):
        return TensorType(shape_arg)
    elif isinstance(shape_arg, int):
        return TensorType((shape_arg, ))
    else:
        raise TypeError


def arange_type(args) -> TensorType:
    size = int((args[1] - args[0]) / args[2])
    return TensorType((size, ))


def range_type(args) -> TensorType:
    size = int((args[1] - args[0]) / args[2]) + 1
    return TensorType((size, ))


def eye_type(args) -> TensorType:
    if args[1] is None:
        return TensorType((args[0], args[0]))
    else:
        return TensorType((args[0], args[1]))

def check(contents: str):
    tree = ast.parse(contents)
    TorchChecker().visit(tree)


METHOD_MAP = {
    'zeros': constructor,
    'ones': constructor,
    'arange': arange_type,
    'range': range_type,
    'eye': eye_type,
}