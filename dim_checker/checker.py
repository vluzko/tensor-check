import ast
import pdb
from typing import Any, Tuple, Literal, List, Dict, Union


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


AllowedTypes = Union[TensorType, int, float, bool]

# TODO: Different scopes
class Context:
    assignments: Dict[str, AllowedTypes]

    def __init__(self):
        self.assignments = {}

    def __getitem__(self, key: str) -> AllowedTypes:
        return self.assignments[key]

    def __setitem__(self, key, value):
        self.assignments[key] = value


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


def check(contents: str):
    tree = ast.parse(contents)
    TorchChecker().visit(tree)


METHOD_MAP = {
    'zeros': constructor,
    'ones': constructor,
}