import ast
import pdb
from typing import Any, Tuple, List, Dict, Union, Optional


class Allowed:
    pass

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError


class InternalInt(Allowed):
    def __add__(self, other: Allowed):
        if isinstance(other, InternalInt):
            return InternalInt
        elif isinstance(other, InternalFloat):
            return InternalFloat
        elif isinstance(other, InternalTensor):
            return other
        else:
            raise TypeError


class InternalFloat(Allowed):

    def __add__(self, other: Allowed):
        if isinstance(other, InternalFloat):
            return InternalFloat
        elif isinstance(other, InternalInt):
            return InternalFloat
        elif isinstance(other, InternalTensor):
            return other
        else:
            raise TypeError


class InternalTensor(Allowed):
    shape: Tuple[int, ...]

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def __add__(self, other: 'InternalTensor'):
        if self.shape == other.shape:
            return InternalTensor(self.shape)
        else:
            raise TypeError

    def __sub__(self, other: 'InternalTensor'):
        return self.__add__(other)

    def __mul__(self, other: 'InternalTensor'):
        # Only implemented for two dimensions
        if len(self.shape) == 2 and len(other.shape) == 2 and self.shape[1] == other.shape[0]:
            return InternalTensor((self.shape[0], other.shape[1]))
        else:
            raise TypeError

    def __truediv__(self, other: 'InternalTensor'):
        return self.__mul__(other)


class Predicate:
    pass


class Refinement(Allowed):

    predicates: List[Predicate]


class Scope:
    parent: Optional['Scope']


# TODO: Different scopes
class Context:
    assignments: Dict[str, Allowed]
    scopes: List[Scope]

    def __init__(self):
        self.assignments = {}
        self.scopes = []

    def __getitem__(self, key: str) -> Allowed:
        return self.assignments[key]

    def __setitem__(self, key, value):
        self.assignments[key] = value


# TODO: Imports
class TorchChecker(ast.NodeTransformer):

    def __init__(self):
        super().__init__()
        self.context = Context()

    def get_type(self, node: ast.AST) -> Allowed:
        if isinstance(node, ast.Name):
            return self.context[node.id]
        elif isinstance(node, ast.Constant):
            try:
                return CONSTANT_TYPE_MAP[type(node.value)]()
            except KeyError:
                return type(node.value)
        else:
            return node._tensor_type

    def visit_Call(self, node: ast.Call) -> ast.Call:
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

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        expr = self.visit(node.value)
        assert len(node.targets) == 1
        name = node.targets[0].id
        self.context[name] = expr._tensor_type
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
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

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        try:
            node._tensor_type = CONSTANT_TYPE_MAP[type(node.value)]()
        except KeyError:
            node._tensor_type = type(node.value)
        return node


def get_arg(arg: ast.expr) -> Any:

    if isinstance(arg, ast.Tuple):
        result = tuple(get_arg(x) for x in arg.elts)
    elif isinstance(arg, ast.Constant):
        result = arg.value

    return result


def constructor(args) -> InternalTensor:
    shape_arg = args[0]
    if isinstance(shape_arg, tuple):
        return InternalTensor(shape_arg)
    elif isinstance(shape_arg, int):
        return InternalTensor((shape_arg, ))
    else:
        raise TypeError


def check(contents: str):
    tree = ast.parse(contents)
    checker = TorchChecker()
    result = checker.visit(tree)
    import pdb
    pdb.set_trace()


CONSTANT_TYPE_MAP = {
    int: InternalInt,
    float: InternalFloat
}


METHOD_MAP = {
    'zeros': constructor,
    'ones': constructor,
}