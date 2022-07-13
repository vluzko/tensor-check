import ast
import operator
from dataclasses import dataclass
from typing import Dict, Optional, Any

from dim_checker.types import ChkType, InternalInt, InternalFloat, InternalTensor, Function, Module, ModuleObj, NoneType
from dim_checker import types

builtin_types = {
    'super': Function([], NoneType())
}


@dataclass
class Scope:
    assignments: Dict[str, ChkType]
    parent: Optional['Scope']

    def __getitem__(self, key: str) -> ChkType:
        return self.assignments[key]

    def __setitem__(self, key, value):
        self.assignments[key] = value

    @staticmethod
    def builtin_scope():
        return Scope(
            builtin_types, None
        )


# TODO: Different scopes
class Context:
    assignments: Dict[str, ChkType]
    scopes: Dict[int, Scope]
    node_types: Dict[ast.AST, ChkType]

    def __init__(self):
        self.imports = []
        self.assignments = {}
        self.scopes = {0: Scope.builtin_scope()}
        self.node_types = {}

    def add_type(self, node: ast.AST, node_type: ChkType):
        self.node_types[node] = node_type

    def get_type(self, node: ast.AST) -> ChkType:
        return self.node_types[node]

    def new_scope(self):
        self.scopes[len(self.scopes)] = Scope()

    def lookup_name(self, name: str, scope_id: int = 0) -> ChkType:
        scope = self.scopes[scope_id]
        return scope.assignments[name]

SCOPE = 0
# TODO: Imports
class TorchChecker(ast.NodeTransformer):

    def __init__(self):
        super().__init__()
        self.context = Context()

    def get_type(self, node: ast.AST) -> ChkType:
        if isinstance(node, ast.Name):
            return self.context.lookup_name(node.id)
        elif isinstance(node, ast.Constant):
            try:
                return CONSTANT_TYPE_MAP[type(node.value)]()
            except KeyError:
                return type(node.value)
        else:
            return self.context.get_type(node)

    def visit_Import(self, node: ast.Import) -> Any:
        # TODO: Import types for all imported modules
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if node.module == 'torch':
            # TODO: Import types for all imported modules
            # TODO: Remap aliases (`node.names[i].asname`)
            self.context.imports.extend(node.names)
        return node

    def visit_Init(self, node: ast.FunctionDef) -> Any:
        attributes = {}
        for stmt in node.body:
            self.visit(stmt)
            if isinstance(stmt, ast.Assign):
                # TODO: Handle multiple targets (probably not)
                if len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    # TODO: Allow other kinds of assignment
                    assert isinstance(target, ast.Attribute)
                    assert isinstance(target.value, ast.Name)
                    if target.value.id == 'self':
                        attributes[target.attr] = NoneType()
        return node, attributes

    def visit_MethodDef(self, node: ast.FunctionDef) -> Any:
        ret_types = []
        for stmt in node.body:
            self.visit(stmt)
            if isinstance(stmt, ast.Return):
                ret_types.append(self.get_type(stmt))
            print(stmt)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        ret_types = []

        # Record arg types.
        arg_types = []
        for arg in node.args.args:
            if hasattr(arg, 'annotation'):
                arg_type = types.read_annotation(arg.annotation.id)  # type: ignore
            else:
                arg_type = NoneType()
            self.context.scopes[SCOPE][arg.arg] = arg_type
            arg_types.append(arg_type)

        # Check function body
        for stmt in node.body:
            self.visit(stmt)
            if isinstance(stmt, ast.Return):
                ret_types.append(self.get_type(stmt))

        func_type = Function(tuple(arg_types), ret_types)
        self.context.add_type(node, func_type)
        self.context.scopes[SCOPE][node.name] = func_type
        return node

    def visit_Return(self, node: ast.Return) -> Any:
        if node.value is not None:
            import pdb
            # pdb.set_trace()
            self.visit(node.value)
            self.context.add_type(node, self.get_type(node.value))
        return node

    # TODO: decorator_list
    # TODO: keywords
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        is_module = False
        for base in node.bases:
            if isinstance(base, ast.AST):
                self.visit(base)
            # TODO: Handle import aliases and shadowing
            if isinstance(base, ast.Attribute) and base.attr == 'Module' and isinstance(base.value, ast.Name) and base.value.id == 'nn':
                is_module = True

        # Check the __init__ type first
        attrs = {}
        for val in node.body:
            if isinstance(val, ast.FunctionDef) and val.name == '__init__':
                init_node, attrs = self.visit_Init(val)

        class_type = Module(attrs, NoneType())
        self.context.add_type(node, class_type)
        self.context['self'] = class_type

        for val in node.body:
            if isinstance(val, ast.FunctionDef):
                # We don't check __init__ again
                if val.name == '__init__':
                    continue
                # For now, no support for decorators. General decorators are unlikely to ever be supported
                # TODO: classmethods (maybe)
                # TODO: staticmethods (maybe)
                elif len(val.decorator_list) > 0:
                    continue
                else:
                    # TODO: Create separate `self` scopes for methods
                    self.visit_MethodDef(val)
                # We hard code the fact that `forward` is called by `__call__` for torch modules.
                if is_module and val.name == 'forward':
                    import pdb
                    # pdb.set_trace()
            else:
                self.visit(val)

        return node

    def visit_Call(self, node) -> ast.Call:
        for arg in node.args:
            self.visit(arg)
        self.visit(node.func)

        func_type = self.get_type(node.func)
        # TODO: Typechecking: check arg types
        arg_types = [self.get_type(arg) for arg in node.args]

        assert isinstance(func_type, Function)
        self.context.add_type(node, func_type.ret_type)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        for target in node.targets:
            self.visit(target)
        expr = self.visit(node.value)
        # TODO: Minor: Handle destructuring
        assert len(node.targets) == 1
        self.context.add_type(node.targets[0], self.get_type(expr))
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            self.context.scopes[0][name] = self.context.get_type(expr)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        self.visit(node.left)
        self.visit(node.right)
        left_type = self.get_type(node.left)
        right_type = self.get_type(node.right)

        new_type = BIN_OP_MAP[type(node.op)](left_type, right_type)

        self.context.add_type(node, new_type)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        try:
            node_type = CONSTANT_TYPE_MAP[type(node.value)]()
            self.context.add_type(node, node_type)
        except KeyError:
            self.context.add_type(node, type(node.value))
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in self.context.assignments:
            self.context.add_type(node, self.context.assignments[node.id])
        return node

    def visit_Attribute(self, node: ast.Attribute):
        super().visit(node.ctx)
        super().visit(node.value)
        try:
            # TODO: This shouldn't be special cased
            if isinstance(node.value, ast.Name):
                obj_type = self.context.lookup_name(node.value)
                attr_type = obj_type.attributes[node.attr]
                self.context.add_type(node, attr_type)
        except KeyError:
            pass
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

def check(contents: str):
    tree = ast.parse(contents)
    checker = TorchChecker()
    result = checker.visit(tree)


CONSTANT_TYPE_MAP = {
    int: InternalInt,
    float: InternalFloat
}


METHOD_MAP = {
    'zeros': constructor,
    'ones': constructor,
    'arange': arange_type,
    'range': range_type,
    'eye': eye_type,
}


BIN_OP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

# if isinstance(node.op, ast.Add):
#             new_type = left_type + right_type
#         elif isinstance(node.op, ast.Sub):
#             new_type = left_type - right_type
#         elif isinstance(node.op, ast.Mult):
#             new_type = left_type * right_type
#         elif isinstance(node.op, ast.Div):
#             new_type = left_type / right_type
#         else:
#             raise TypeError